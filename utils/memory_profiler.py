"""
Memory profiling utilities for PPD-FLUX training pipeline.
Provides real-time GPU memory monitoring, batch size optimization, and memory cleanup.
"""

import json
import time
import torch
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """단일 시점의 메모리 사용량 스냅샷"""
    timestamp: float
    gpu_allocated: float  # GB
    gpu_reserved: float  # GB
    gpu_max_allocated: float  # GB
    gpu_total: float  # GB
    gpu_utilization: float  # percentage
    system_memory_used: float  # GB
    system_memory_percent: float  # percentage
    batch_size: Optional[int] = None
    step: Optional[int] = None
    phase: Optional[str] = None  # 'forward', 'backward', 'validation', etc.


@dataclass
class MemoryProfile:
    """메모리 프로파일링 세션 전체 결과"""
    session_id: str
    start_time: float
    end_time: float
    snapshots: List[MemorySnapshot]
    peak_gpu_memory: float
    average_gpu_memory: float
    memory_efficiency: float  # allocated / reserved ratio
    recommended_batch_size: Optional[int] = None


class MemoryProfiler:
    """실시간 GPU 메모리 사용량 모니터링 및 분석"""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_gpu_available = torch.cuda.is_available() and self.device.type == "cuda"
        self.snapshots: List[MemorySnapshot] = []
        self.monitoring = False
        self.monitor_thread = None
        self.session_id = f"session_{int(time.time())}"

        if self.is_gpu_available:
            # GPU 정보 로깅
            gpu_props = torch.cuda.get_device_properties(self.device)
            logger.info(f"GPU: {gpu_props.name}, Memory: {gpu_props.total_memory / 1e9:.1f} GB")
        else:
            logger.warning("CUDA not available, memory profiling will be limited")

    def capture_snapshot(self, batch_size: Optional[int] = None,
                        step: Optional[int] = None,
                        phase: Optional[str] = None) -> MemorySnapshot:
        """현재 시점의 메모리 사용량 스냅샷 캡처"""
        timestamp = time.time()

        # GPU 메모리 정보
        if self.is_gpu_available:
            gpu_allocated = torch.cuda.memory_allocated(self.device) / 1e9
            gpu_reserved = torch.cuda.memory_reserved(self.device) / 1e9
            gpu_max_allocated = torch.cuda.max_memory_allocated(self.device) / 1e9
            gpu_total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            gpu_utilization = (gpu_allocated / gpu_total) * 100
        else:
            gpu_allocated = gpu_reserved = gpu_max_allocated = gpu_total = gpu_utilization = 0.0

        # 시스템 메모리 정보
        memory_info = psutil.virtual_memory()
        system_memory_used = memory_info.used / 1e9
        system_memory_percent = memory_info.percent

        snapshot = MemorySnapshot(
            timestamp=timestamp,
            gpu_allocated=gpu_allocated,
            gpu_reserved=gpu_reserved,
            gpu_max_allocated=gpu_max_allocated,
            gpu_total=gpu_total,
            gpu_utilization=gpu_utilization,
            system_memory_used=system_memory_used,
            system_memory_percent=system_memory_percent,
            batch_size=batch_size,
            step=step,
            phase=phase
        )

        self.snapshots.append(snapshot)
        return snapshot

    def start_monitoring(self, interval: float = 1.0):
        """백그라운드에서 주기적 메모리 모니터링 시작"""
        if self.monitoring:
            logger.warning("Monitoring already started")
            return

        self.monitoring = True

        def monitor_loop():
            while self.monitoring:
                self.capture_snapshot(phase="background_monitoring")
                time.sleep(interval)

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Memory monitoring started with {interval}s interval")

    def stop_monitoring(self):
        """백그라운드 메모리 모니터링 중지"""
        if not self.monitoring:
            return

        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Memory monitoring stopped")

    def clear_cache(self):
        """GPU 캐시 정리"""
        if self.is_gpu_available:
            torch.cuda.empty_cache()
            self.capture_snapshot(phase="cache_cleared")
            logger.info("GPU cache cleared")

    def get_memory_summary(self) -> Dict[str, Any]:
        """현재 메모리 사용량 요약 정보"""
        if not self.snapshots:
            self.capture_snapshot()

        latest = self.snapshots[-1]

        summary = {
            "current_gpu_allocated": latest.gpu_allocated,
            "current_gpu_reserved": latest.gpu_reserved,
            "current_gpu_utilization": latest.gpu_utilization,
            "peak_gpu_allocated": max(s.gpu_allocated for s in self.snapshots),
            "average_gpu_allocated": sum(s.gpu_allocated for s in self.snapshots) / len(self.snapshots),
            "gpu_total": latest.gpu_total,
            "system_memory_percent": latest.system_memory_percent,
            "num_snapshots": len(self.snapshots)
        }

        return summary

    def analyze_memory_pattern(self) -> Dict[str, Any]:
        """메모리 사용 패턴 분석"""
        if len(self.snapshots) < 2:
            return {"error": "Insufficient data for analysis"}

        gpu_allocated = [s.gpu_allocated for s in self.snapshots]
        gpu_reserved = [s.gpu_reserved for s in self.snapshots]

        analysis = {
            "peak_memory": max(gpu_allocated),
            "average_memory": sum(gpu_allocated) / len(gpu_allocated),
            "memory_volatility": max(gpu_allocated) - min(gpu_allocated),
            "efficiency_ratio": sum(gpu_allocated) / sum(gpu_reserved) if sum(gpu_reserved) > 0 else 0,
            "growth_trend": gpu_allocated[-1] - gpu_allocated[0] if len(gpu_allocated) > 1 else 0,
            "total_snapshots": len(self.snapshots)
        }

        # 메모리 누수 감지
        if len(gpu_allocated) >= 10:
            recent_trend = sum(gpu_allocated[-5:]) / 5 - sum(gpu_allocated[:5]) / 5
            analysis["potential_memory_leak"] = recent_trend > 1.0  # 1GB 이상 증가

        return analysis

    def save_profile(self, output_path: Path) -> MemoryProfile:
        """메모리 프로파일링 결과를 파일로 저장"""
        if not self.snapshots:
            logger.warning("No snapshots to save")
            return None

        analysis = self.analyze_memory_pattern()

        profile = MemoryProfile(
            session_id=self.session_id,
            start_time=self.snapshots[0].timestamp,
            end_time=self.snapshots[-1].timestamp,
            snapshots=self.snapshots,
            peak_gpu_memory=analysis.get("peak_memory", 0),
            average_gpu_memory=analysis.get("average_memory", 0),
            memory_efficiency=analysis.get("efficiency_ratio", 0)
        )

        # JSON으로 저장 (dataclass를 dict로 변환)
        profile_dict = asdict(profile)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(profile_dict, f, indent=2)

        logger.info(f"Memory profile saved to {output_path}")
        return profile

    @classmethod
    def load_profile(cls, profile_path: Path) -> MemoryProfile:
        """저장된 메모리 프로파일 로드"""
        with open(profile_path, 'r') as f:
            data = json.load(f)

        # dict를 다시 dataclass로 변환
        snapshots = [MemorySnapshot(**s) for s in data['snapshots']]
        data['snapshots'] = snapshots

        return MemoryProfile(**data)


class BatchSizeOptimizer:
    """배치 크기 자동 최적화"""

    def __init__(self, profiler: MemoryProfiler, target_memory_usage: float = 0.8):
        self.profiler = profiler
        self.target_memory_usage = target_memory_usage  # 목표 메모리 사용률 (80%)
        self.batch_test_results: Dict[int, float] = {}  # batch_size -> peak_memory

    def test_batch_size(self, batch_size: int, test_function: callable) -> Tuple[bool, float]:
        """특정 배치 크기로 테스트 실행 및 메모리 사용량 측정"""
        logger.info(f"Testing batch size: {batch_size}")

        self.profiler.clear_cache()
        initial_memory = self.profiler.capture_snapshot(batch_size=batch_size, phase="batch_test_start")

        try:
            # 테스트 함수 실행
            test_function(batch_size)

            # 피크 메모리 측정
            peak_memory = max(s.gpu_allocated for s in self.profiler.snapshots
                             if s.timestamp > initial_memory.timestamp)

            self.batch_test_results[batch_size] = peak_memory

            # 메모리 사용률 계산
            memory_usage_ratio = peak_memory / self.profiler.snapshots[-1].gpu_total
            success = memory_usage_ratio <= self.target_memory_usage

            logger.info(f"Batch size {batch_size}: {peak_memory:.2f}GB ({memory_usage_ratio:.1%}), "
                       f"Success: {success}")

            return success, peak_memory

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM at batch size {batch_size}")
                self.profiler.clear_cache()
                return False, float('inf')
            else:
                raise e

    def find_optimal_batch_size(self, test_function: callable,
                               min_batch_size: int = 1,
                               max_batch_size: int = 16) -> int:
        """최적 배치 크기 자동 탐색"""
        logger.info(f"Finding optimal batch size between {min_batch_size} and {max_batch_size}")

        # 이진 탐색으로 최적 배치 크기 찾기
        left, right = min_batch_size, max_batch_size
        best_batch_size = min_batch_size

        while left <= right:
            mid = (left + right) // 2
            success, peak_memory = self.test_batch_size(mid, test_function)

            if success:
                best_batch_size = mid
                left = mid + 1  # 더 큰 배치 크기 시도
            else:
                right = mid - 1  # 더 작은 배치 크기로 제한

        logger.info(f"Optimal batch size found: {best_batch_size}")
        return best_batch_size

    def get_memory_recommendations(self) -> Dict[str, Any]:
        """메모리 최적화 권장사항 생성"""
        analysis = self.profiler.analyze_memory_pattern()

        recommendations = {
            "current_memory_efficiency": analysis.get("efficiency_ratio", 0),
            "peak_memory_usage": analysis.get("peak_memory", 0),
            "recommendations": []
        }

        # 권장사항 생성
        if analysis.get("efficiency_ratio", 0) < 0.7:
            recommendations["recommendations"].append(
                "Consider enabling gradient checkpointing to improve memory efficiency"
            )

        if analysis.get("potential_memory_leak", False):
            recommendations["recommendations"].append(
                "Potential memory leak detected. Consider clearing cache more frequently"
            )

        if analysis.get("peak_memory", 0) > 20:  # 20GB 이상
            recommendations["recommendations"].append(
                "High memory usage detected. Consider reducing batch size or enabling CPU offloading"
            )

        return recommendations


# 사용 예제 및 테스트 함수들
def example_memory_intensive_operation(batch_size: int = 4):
    """메모리 집약적 작업 시뮬레이션 (테스트용)"""
    # 실제 모델 로딩 및 forward pass 시뮬레이션
    if torch.cuda.is_available():
        # 더미 텐서로 메모리 사용량 시뮬레이션
        dummy_tensor = torch.randn(batch_size, 3, 512, 512, device="cuda")
        result = torch.conv2d(dummy_tensor, torch.randn(64, 3, 3, 3, device="cuda"))
        return result.sum()
    return torch.tensor(0.0)


if __name__ == "__main__":
    # 기본 사용 예제
    profiler = MemoryProfiler()
    optimizer = BatchSizeOptimizer(profiler)

    # 메모리 모니터링 시작
    profiler.start_monitoring(interval=0.5)

    # 배치 크기 최적화
    optimal_batch = optimizer.find_optimal_batch_size(
        example_memory_intensive_operation,
        min_batch_size=1,
        max_batch_size=8
    )

    # 모니터링 중지
    profiler.stop_monitoring()

    # 결과 출력
    summary = profiler.get_memory_summary()
    recommendations = optimizer.get_memory_recommendations()

    print(f"Optimal batch size: {optimal_batch}")
    print(f"Memory summary: {summary}")
    print(f"Recommendations: {recommendations}")

    # 프로파일 저장
    profile_path = Path("memory_profiles") / f"profile_{int(time.time())}.json"
    profiler.save_profile(profile_path)