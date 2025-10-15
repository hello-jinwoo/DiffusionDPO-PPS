"""
Automatic recovery system for PPD-FLUX training.
Handles OOM errors, NaN losses, and other training failures with graceful recovery.
"""

import torch
import logging
import time
import gc
from typing import Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RecoveryConfig:
    """자동 복구 설정"""
    # OOM 복구 설정
    min_batch_size: int = 1
    batch_size_reduction_factor: float = 0.5
    max_recovery_attempts: int = 3

    # NaN 손실 복구 설정
    learning_rate_reduction_factor: float = 0.5
    min_learning_rate: float = 1e-7

    # 메모리 관리 설정
    memory_cleanup_threshold: float = 0.9  # 90% 메모리 사용 시 정리
    gradient_checkpointing_threshold: float = 0.8  # 80% 메모리 사용 시 활성화

    # 체크포인트 설정
    emergency_checkpoint_interval: int = 10  # 10 스텝마다 긴급 체크포인트
    auto_save_on_failure: bool = True


@dataclass
class RecoveryState:
    """복구 상태 추적"""
    original_batch_size: int
    current_batch_size: int
    original_learning_rate: float
    current_learning_rate: float
    recovery_attempts: int = 0
    total_oom_events: int = 0
    total_nan_events: int = 0
    gradient_checkpointing_enabled: bool = False
    cpu_offloading_enabled: bool = False
    last_successful_step: int = 0


class AutoRecoveryManager:
    """자동 복구 관리자"""

    def __init__(self, config: RecoveryConfig = None):
        self.config = config or RecoveryConfig()
        self.state: Optional[RecoveryState] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_gpu_available = torch.cuda.is_available()

        # 복구 전략 히스토리
        self.recovery_history: List[Dict[str, Any]] = []

        logger.info(f"AutoRecoveryManager initialized for {self.device}")

    def initialize_state(self, initial_batch_size: int, initial_learning_rate: float):
        """복구 상태 초기화"""
        self.state = RecoveryState(
            original_batch_size=initial_batch_size,
            current_batch_size=initial_batch_size,
            original_learning_rate=initial_learning_rate,
            current_learning_rate=initial_learning_rate
        )
        logger.info(f"Recovery state initialized: batch_size={initial_batch_size}, lr={initial_learning_rate}")

    def cleanup_memory(self, force: bool = False) -> bool:
        """메모리 정리"""
        if not self.is_gpu_available:
            return True

        try:
            # 현재 메모리 사용률 확인
            allocated = torch.cuda.memory_allocated(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory
            usage_ratio = allocated / total

            if force or usage_ratio > self.config.memory_cleanup_threshold:
                logger.info(f"Cleaning memory (usage: {usage_ratio:.1%})")

                # GPU 캐시 정리
                torch.cuda.empty_cache()

                # Python 가비지 컬렉션
                gc.collect()

                # 정리 후 메모리 확인
                new_allocated = torch.cuda.memory_allocated(self.device)
                new_usage_ratio = new_allocated / total

                freed_gb = (allocated - new_allocated) / 1e9
                logger.info(f"Memory cleaned: {freed_gb:.2f}GB freed, new usage: {new_usage_ratio:.1%}")

                return True

        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            return False

        return True

    def handle_oom_error(self, optimizer: torch.optim.Optimizer = None) -> Tuple[bool, Dict[str, Any]]:
        """OOM 에러 처리"""
        if self.state is None:
            logger.error("Recovery state not initialized")
            return False, {}

        self.state.total_oom_events += 1
        self.state.recovery_attempts += 1

        logger.warning(f"OOM detected (attempt {self.state.recovery_attempts}/{self.config.max_recovery_attempts})")

        if self.state.recovery_attempts > self.config.max_recovery_attempts:
            logger.error("Maximum recovery attempts exceeded")
            return False, {"error": "max_recovery_attempts_exceeded"}

        # 메모리 정리
        self.cleanup_memory(force=True)

        # 배치 크기 감소
        new_batch_size = max(
            self.config.min_batch_size,
            int(self.state.current_batch_size * self.config.batch_size_reduction_factor)
        )

        recovery_actions = {
            "previous_batch_size": self.state.current_batch_size,
            "new_batch_size": new_batch_size,
            "memory_cleaned": True,
            "gradient_checkpointing": False,
            "cpu_offloading": False
        }

        # 배치 크기가 더 이상 줄어들 수 없으면 추가 최적화 적용
        if new_batch_size == self.state.current_batch_size:
            if not self.state.gradient_checkpointing_enabled:
                logger.info("Enabling gradient checkpointing")
                self.state.gradient_checkpointing_enabled = True
                recovery_actions["gradient_checkpointing"] = True

            elif not self.state.cpu_offloading_enabled:
                logger.info("Enabling CPU offloading")
                self.state.cpu_offloading_enabled = True
                recovery_actions["cpu_offloading"] = True
            else:
                logger.error("All memory optimizations exhausted")
                return False, recovery_actions

        self.state.current_batch_size = new_batch_size

        # 옵티마이저 학습률 조정 (필요시)
        if optimizer and hasattr(optimizer, 'param_groups'):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9  # 약간 학습률 감소

        # 복구 히스토리 기록
        self.recovery_history.append({
            "timestamp": time.time(),
            "event_type": "OOM",
            "actions": recovery_actions,
            "success": True
        })

        logger.info(f"OOM recovery: batch_size {recovery_actions['previous_batch_size']} → {new_batch_size}")
        return True, recovery_actions

    def handle_nan_loss(self, optimizer: torch.optim.Optimizer) -> Tuple[bool, Dict[str, Any]]:
        """NaN 손실 처리"""
        if self.state is None:
            logger.error("Recovery state not initialized")
            return False, {}

        self.state.total_nan_events += 1
        logger.warning(f"NaN loss detected (total events: {self.state.total_nan_events})")

        # 학습률 감소
        new_lr = max(
            self.config.min_learning_rate,
            self.state.current_learning_rate * self.config.learning_rate_reduction_factor
        )

        if new_lr == self.config.min_learning_rate:
            logger.error("Learning rate reached minimum, cannot recover")
            return False, {"error": "min_learning_rate_reached"}

        # 옵티마이저 학습률 업데이트
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        self.state.current_learning_rate = new_lr

        recovery_actions = {
            "previous_learning_rate": self.state.current_learning_rate / self.config.learning_rate_reduction_factor,
            "new_learning_rate": new_lr,
            "optimizer_state_reset": True
        }

        # 옵티마이저 상태 초기화 (momentum 등 리셋)
        optimizer.state = {}

        # 복구 히스토리 기록
        self.recovery_history.append({
            "timestamp": time.time(),
            "event_type": "NaN_loss",
            "actions": recovery_actions,
            "success": True
        })

        logger.info(f"NaN recovery: learning_rate {recovery_actions['previous_learning_rate']:.2e} → {new_lr:.2e}")
        return True, recovery_actions

    def check_memory_pressure(self) -> Dict[str, Any]:
        """메모리 압박 상황 확인"""
        if not self.is_gpu_available:
            return {"memory_pressure": False, "usage_ratio": 0.0}

        try:
            allocated = torch.cuda.memory_allocated(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory
            usage_ratio = allocated / total

            recommendations = []

            if usage_ratio > self.config.gradient_checkpointing_threshold:
                if not self.state.gradient_checkpointing_enabled:
                    recommendations.append("enable_gradient_checkpointing")

            if usage_ratio > self.config.memory_cleanup_threshold:
                recommendations.append("cleanup_memory")

            return {
                "memory_pressure": usage_ratio > self.config.memory_cleanup_threshold,
                "usage_ratio": usage_ratio,
                "allocated_gb": allocated / 1e9,
                "total_gb": total / 1e9,
                "recommendations": recommendations
            }

        except Exception as e:
            logger.error(f"Memory pressure check failed: {e}")
            return {"memory_pressure": False, "error": str(e)}

    def suggest_optimizations(self) -> Dict[str, Any]:
        """현재 상황에 맞는 최적화 제안"""
        if self.state is None:
            return {"error": "Recovery state not initialized"}

        suggestions = {
            "current_settings": {
                "batch_size": self.state.current_batch_size,
                "learning_rate": self.state.current_learning_rate,
                "gradient_checkpointing": self.state.gradient_checkpointing_enabled,
                "cpu_offloading": self.state.cpu_offloading_enabled
            },
            "optimizations": []
        }

        # 메모리 압박 상황 확인
        memory_status = self.check_memory_pressure()

        if memory_status.get("memory_pressure", False):
            suggestions["optimizations"].extend(memory_status.get("recommendations", []))

        # 배치 크기 최적화 제안
        if self.state.total_oom_events == 0 and self.state.current_batch_size < self.state.original_batch_size:
            suggestions["optimizations"].append("consider_increasing_batch_size")

        # 학습률 회복 제안
        if (self.state.total_nan_events == 0 and
            self.state.current_learning_rate < self.state.original_learning_rate):
            suggestions["optimizations"].append("consider_increasing_learning_rate")

        return suggestions

    def save_recovery_report(self, output_path: Path):
        """복구 리포트 저장"""
        report = {
            "config": self.config.__dict__,
            "final_state": self.state.__dict__ if self.state else None,
            "recovery_history": self.recovery_history,
            "summary": {
                "total_oom_events": self.state.total_oom_events if self.state else 0,
                "total_nan_events": self.state.total_nan_events if self.state else 0,
                "total_recovery_attempts": self.state.recovery_attempts if self.state else 0,
                "final_batch_size": self.state.current_batch_size if self.state else 0,
                "final_learning_rate": self.state.current_learning_rate if self.state else 0,
            }
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Recovery report saved to {output_path}")

    def reset_recovery_state(self):
        """복구 상태 초기화 (새로운 훈련 세션 시작 시)"""
        if self.state:
            self.state.recovery_attempts = 0
            self.state.current_batch_size = self.state.original_batch_size
            self.state.current_learning_rate = self.state.original_learning_rate
            self.state.gradient_checkpointing_enabled = False
            self.state.cpu_offloading_enabled = False

        self.recovery_history.clear()
        logger.info("Recovery state reset")


class SafeTrainingWrapper:
    """안전한 훈련 래퍼 - 훈련 루프를 자동 복구로 감쌈"""

    def __init__(self, recovery_manager: AutoRecoveryManager):
        self.recovery_manager = recovery_manager

    def safe_forward_backward(self,
                            forward_fn: Callable,
                            optimizer: torch.optim.Optimizer,
                            *args, **kwargs) -> Tuple[bool, Any, Dict[str, Any]]:
        """안전한 forward + backward 실행"""
        try:
            # Forward pass
            loss = forward_fn(*args, **kwargs)

            # NaN 체크
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss detected: {loss}")
                success, recovery_info = self.recovery_manager.handle_nan_loss(optimizer)
                return False, None, recovery_info

            # Backward pass
            loss.backward()

            return True, loss, {}

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("OOM error in forward/backward pass")
                success, recovery_info = self.recovery_manager.handle_oom_error(optimizer)
                return False, None, recovery_info
            else:
                raise e

    def safe_optimizer_step(self, optimizer: torch.optim.Optimizer) -> Tuple[bool, Dict[str, Any]]:
        """안전한 옵티마이저 스텝"""
        try:
            optimizer.step()
            optimizer.zero_grad()
            return True, {}

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("OOM error in optimizer step")
                success, recovery_info = self.recovery_manager.handle_oom_error(optimizer)
                return False, recovery_info
            else:
                raise e


# 사용 예제
def example_safe_training_loop():
    """안전한 훈련 루프 예제"""
    # 복구 관리자 초기화
    recovery_config = RecoveryConfig(
        min_batch_size=1,
        max_recovery_attempts=3,
        auto_save_on_failure=True
    )
    recovery_manager = AutoRecoveryManager(recovery_config)
    safe_wrapper = SafeTrainingWrapper(recovery_manager)

    # 임시 모델과 옵티마이저
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 복구 상태 초기화
    recovery_manager.initialize_state(initial_batch_size=4, initial_learning_rate=1e-3)

    def forward_function(batch_size):
        """더미 forward 함수"""
        x = torch.randn(batch_size, 10, requires_grad=True)
        return model(x).sum()

    # 안전한 훈련 루프
    for step in range(10):
        current_batch_size = recovery_manager.state.current_batch_size

        # 메모리 압박 상황 확인
        memory_status = recovery_manager.check_memory_pressure()
        if memory_status.get("memory_pressure", False):
            recovery_manager.cleanup_memory()

        # 안전한 forward/backward
        success, loss, recovery_info = safe_wrapper.safe_forward_backward(
            forward_function, optimizer, current_batch_size
        )

        if not success:
            logger.warning(f"Step {step} failed, recovery attempted: {recovery_info}")
            continue

        # 안전한 옵티마이저 스텝
        step_success, step_recovery_info = safe_wrapper.safe_optimizer_step(optimizer)

        if step_success:
            recovery_manager.state.last_successful_step = step
            logger.info(f"Step {step}: loss={loss:.4f}, batch_size={current_batch_size}")
        else:
            logger.warning(f"Step {step} optimizer failed: {step_recovery_info}")

    # 최종 리포트 저장
    recovery_manager.save_recovery_report(Path("recovery_reports/example_report.json"))


if __name__ == "__main__":
    example_safe_training_loop()