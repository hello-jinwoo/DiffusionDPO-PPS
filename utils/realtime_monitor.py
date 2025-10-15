"""
Real-time monitoring system for PPD-FLUX training.
Provides web-based dashboard with live metrics, alerts, and performance tracking.
"""

import json
import time
import threading
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import queue
import psutil
import torch

# Web framework imports
try:
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logger.warning("Flask not available. Web dashboard disabled. Install with: pip install flask flask-socketio")

# Plotting imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Matplotlib not available. Plotting disabled. Install with: pip install matplotlib")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """시스템 메트릭 데이터"""
    timestamp: float
    # GPU 메트릭
    gpu_utilization: float  # %
    gpu_memory_used: float  # GB
    gpu_memory_total: float  # GB
    gpu_temperature: float  # °C
    gpu_power_draw: float  # W
    # 시스템 메트릭
    cpu_utilization: float  # %
    system_memory_used: float  # GB
    system_memory_total: float  # GB
    disk_io_read: float  # MB/s
    disk_io_write: float  # MB/s


@dataclass
class TrainingMetrics:
    """훈련 메트릭 데이터"""
    timestamp: float
    step: int
    epoch: int
    # 손실 및 정확도
    loss: float
    learning_rate: float
    batch_size: int
    # PPD 관련
    user_count: int
    upe_generation_time: float  # ms
    # 성능 메트릭
    step_time: float  # seconds
    samples_per_second: float
    # 검증 메트릭 (있는 경우)
    validation_loss: Optional[float] = None
    validation_accuracy: Optional[float] = None


@dataclass
class AlertEvent:
    """알림 이벤트"""
    timestamp: float
    level: str  # "info", "warning", "error", "critical"
    category: str  # "memory", "performance", "training", "system"
    message: str
    details: Dict[str, Any]


class RealtimeMonitor:
    """실시간 모니터링 시스템"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()

        # 메트릭 저장소
        self.system_metrics_history: List[SystemMetrics] = []
        self.training_metrics_history: List[TrainingMetrics] = []
        self.alerts_history: List[AlertEvent] = []

        # 실시간 큐
        self.metrics_queue = queue.Queue()
        self.alerts_queue = queue.Queue()

        # 모니터링 스레드
        self.monitoring_active = False
        self.system_monitor_thread = None

        # GPU 관련
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_available = torch.cuda.is_available()

        # 웹 대시보드
        self.app = None
        self.socketio = None
        self.dashboard_thread = None

        if self.gpu_available:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml_available = True
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except ImportError:
                self.nvml_available = False
                logger.warning("pynvml not available. GPU detailed metrics disabled.")
        else:
            self.nvml_available = False

    def _default_config(self) -> Dict[str, Any]:
        """기본 설정"""
        return {
            "system_metrics_interval": 1.0,  # seconds
            "max_history_length": 1000,
            "alert_thresholds": {
                "gpu_memory_usage": 0.9,  # 90%
                "gpu_temperature": 85,  # °C
                "step_time_threshold": 60,  # seconds
                "loss_nan_check": True,
                "memory_leak_detection": True
            },
            "dashboard": {
                "host": "0.0.0.0",
                "port": 5000,
                "debug": False
            },
            "auto_save_interval": 300,  # 5 minutes
            "output_dir": "./monitoring_data"
        }

    def collect_system_metrics(self) -> SystemMetrics:
        """시스템 메트릭 수집"""
        timestamp = time.time()

        # GPU 메트릭
        gpu_utilization = 0.0
        gpu_memory_used = 0.0
        gpu_memory_total = 0.0
        gpu_temperature = 0.0
        gpu_power_draw = 0.0

        if self.gpu_available:
            if self.nvml_available:
                try:
                    import pynvml
                    # GPU 사용률
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    gpu_utilization = util.gpu

                    # GPU 메모리
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    gpu_memory_used = memory_info.used / 1e9
                    gpu_memory_total = memory_info.total / 1e9

                    # GPU 온도
                    gpu_temperature = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)

                    # GPU 전력
                    gpu_power_draw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0

                except Exception as e:
                    logger.debug(f"NVML metrics collection failed: {e}")
            else:
                # 기본 PyTorch 메트릭
                gpu_memory_used = torch.cuda.memory_allocated(self.device) / 1e9
                gpu_memory_total = torch.cuda.get_device_properties(self.device).total_memory / 1e9

        # 시스템 메트릭
        cpu_utilization = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        system_memory_used = memory_info.used / 1e9
        system_memory_total = memory_info.total / 1e9

        # 디스크 I/O
        disk_io = psutil.disk_io_counters()
        disk_io_read = 0.0
        disk_io_write = 0.0
        if hasattr(self, '_last_disk_io'):
            time_diff = timestamp - self._last_disk_timestamp
            if time_diff > 0:
                disk_io_read = (disk_io.read_bytes - self._last_disk_io.read_bytes) / time_diff / 1e6
                disk_io_write = (disk_io.write_bytes - self._last_disk_io.write_bytes) / time_diff / 1e6

        self._last_disk_io = disk_io
        self._last_disk_timestamp = timestamp

        return SystemMetrics(
            timestamp=timestamp,
            gpu_utilization=gpu_utilization,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_temperature=gpu_temperature,
            gpu_power_draw=gpu_power_draw,
            cpu_utilization=cpu_utilization,
            system_memory_used=system_memory_used,
            system_memory_total=system_memory_total,
            disk_io_read=disk_io_read,
            disk_io_write=disk_io_write
        )

    def check_alerts(self, system_metrics: SystemMetrics, training_metrics: TrainingMetrics = None):
        """알림 확인 및 생성"""
        alerts = []
        thresholds = self.config["alert_thresholds"]

        # GPU 메모리 사용률 체크
        if system_metrics.gpu_memory_total > 0:
            gpu_memory_usage = system_metrics.gpu_memory_used / system_metrics.gpu_memory_total
            if gpu_memory_usage > thresholds["gpu_memory_usage"]:
                alerts.append(AlertEvent(
                    timestamp=system_metrics.timestamp,
                    level="warning",
                    category="memory",
                    message=f"High GPU memory usage: {gpu_memory_usage:.1%}",
                    details={
                        "usage_ratio": gpu_memory_usage,
                        "used_gb": system_metrics.gpu_memory_used,
                        "total_gb": system_metrics.gpu_memory_total
                    }
                ))

        # GPU 온도 체크
        if system_metrics.gpu_temperature > thresholds["gpu_temperature"]:
            alerts.append(AlertEvent(
                timestamp=system_metrics.timestamp,
                level="warning",
                category="system",
                message=f"High GPU temperature: {system_metrics.gpu_temperature}°C",
                details={"temperature": system_metrics.gpu_temperature}
            ))

        # 훈련 메트릭 체크
        if training_metrics:
            # NaN 손실 체크
            if thresholds["loss_nan_check"] and (torch.isnan(torch.tensor(training_metrics.loss)) or torch.isinf(torch.tensor(training_metrics.loss))):
                alerts.append(AlertEvent(
                    timestamp=training_metrics.timestamp,
                    level="error",
                    category="training",
                    message=f"Invalid loss detected: {training_metrics.loss}",
                    details={"loss": training_metrics.loss, "step": training_metrics.step}
                ))

            # 스텝 시간 체크
            if training_metrics.step_time > thresholds["step_time_threshold"]:
                alerts.append(AlertEvent(
                    timestamp=training_metrics.timestamp,
                    level="warning",
                    category="performance",
                    message=f"Slow training step: {training_metrics.step_time:.1f}s",
                    details={"step_time": training_metrics.step_time, "step": training_metrics.step}
                ))

        # 알림 큐에 추가
        for alert in alerts:
            self.alerts_queue.put(alert)
            self.alerts_history.append(alert)
            logger.warning(f"ALERT [{alert.level.upper()}] {alert.category}: {alert.message}")

        # 히스토리 크기 제한
        if len(self.alerts_history) > self.config["max_history_length"]:
            self.alerts_history = self.alerts_history[-self.config["max_history_length"]:]

    def add_training_metrics(self, metrics: TrainingMetrics):
        """훈련 메트릭 추가"""
        self.training_metrics_history.append(metrics)
        self.metrics_queue.put(("training", metrics))

        # 알림 체크
        system_metrics = self.system_metrics_history[-1] if self.system_metrics_history else None
        if system_metrics:
            self.check_alerts(system_metrics, metrics)

        # 히스토리 크기 제한
        if len(self.training_metrics_history) > self.config["max_history_length"]:
            self.training_metrics_history = self.training_metrics_history[-self.config["max_history_length"]:]

    def start_monitoring(self):
        """모니터링 시작"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True

        def system_monitor_loop():
            while self.monitoring_active:
                try:
                    metrics = self.collect_system_metrics()
                    self.system_metrics_history.append(metrics)
                    self.metrics_queue.put(("system", metrics))

                    # 알림 체크 (시스템 메트릭만)
                    self.check_alerts(metrics)

                    # 히스토리 크기 제한
                    if len(self.system_metrics_history) > self.config["max_history_length"]:
                        self.system_metrics_history = self.system_metrics_history[-self.config["max_history_length"]:]

                    time.sleep(self.config["system_metrics_interval"])

                except Exception as e:
                    logger.error(f"System monitoring error: {e}")
                    time.sleep(5)  # Error recovery delay

        self.system_monitor_thread = threading.Thread(target=system_monitor_loop, daemon=True)
        self.system_monitor_thread.start()

        logger.info("🔍 Real-time monitoring started")

    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring_active = False

        if self.system_monitor_thread:
            self.system_monitor_thread.join(timeout=5)

        logger.info("🔍 Real-time monitoring stopped")

    def get_current_status(self) -> Dict[str, Any]:
        """현재 상태 요약"""
        status = {
            "monitoring_active": self.monitoring_active,
            "system_metrics": None,
            "training_metrics": None,
            "recent_alerts": [],
            "summary": {}
        }

        if self.system_metrics_history:
            latest_system = self.system_metrics_history[-1]
            status["system_metrics"] = asdict(latest_system)

        if self.training_metrics_history:
            latest_training = self.training_metrics_history[-1]
            status["training_metrics"] = asdict(latest_training)

        # 최근 알림 (최대 10개)
        status["recent_alerts"] = [asdict(alert) for alert in self.alerts_history[-10:]]

        # 요약 통계
        if self.system_metrics_history:
            gpu_memory_usage = 0
            if latest_system.gpu_memory_total > 0:
                gpu_memory_usage = latest_system.gpu_memory_used / latest_system.gpu_memory_total

            status["summary"] = {
                "gpu_memory_usage_percent": gpu_memory_usage * 100,
                "gpu_temperature": latest_system.gpu_temperature,
                "cpu_utilization": latest_system.cpu_utilization,
                "training_steps": len(self.training_metrics_history),
                "alert_count": len(self.alerts_history)
            }

        return status

    def save_monitoring_data(self, output_path: Path = None):
        """모니터링 데이터 저장"""
        if output_path is None:
            output_path = Path(self.config["output_dir"]) / f"monitoring_data_{int(time.time())}.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "config": self.config,
            "system_metrics": [asdict(m) for m in self.system_metrics_history],
            "training_metrics": [asdict(m) for m in self.training_metrics_history],
            "alerts": [asdict(a) for a in self.alerts_history],
            "summary": {
                "monitoring_duration": len(self.system_metrics_history) * self.config["system_metrics_interval"],
                "total_training_steps": len(self.training_metrics_history),
                "total_alerts": len(self.alerts_history)
            }
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"📁 Monitoring data saved to: {output_path}")

    def generate_performance_plots(self, output_dir: Path = None):
        """성능 플롯 생성"""
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping plot generation.")
            return

        if output_dir is None:
            output_dir = Path(self.config["output_dir"]) / "plots"

        output_dir.mkdir(parents=True, exist_ok=True)

        # GPU 메모리 사용량 플롯
        if self.system_metrics_history:
            timestamps = [m.timestamp for m in self.system_metrics_history]
            gpu_memory = [m.gpu_memory_used for m in self.system_metrics_history]

            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, gpu_memory, label='GPU Memory Usage (GB)')
            plt.xlabel('Time')
            plt.ylabel('Memory (GB)')
            plt.title('GPU Memory Usage Over Time')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir / "gpu_memory_usage.png", dpi=150)
            plt.close()

        # 훈련 손실 플롯
        if self.training_metrics_history:
            steps = [m.step for m in self.training_metrics_history]
            losses = [m.loss for m in self.training_metrics_history]

            plt.figure(figsize=(12, 6))
            plt.plot(steps, losses, label='Training Loss')
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Time')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir / "training_loss.png", dpi=150)
            plt.close()

        logger.info(f"📊 Performance plots saved to: {output_dir}")

    def start_web_dashboard(self):
        """웹 대시보드 시작"""
        if not FLASK_AVAILABLE:
            logger.warning("Flask not available. Web dashboard disabled.")
            return

        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'ppd_monitoring_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        @self.app.route('/')
        def dashboard():
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>PPD-FLUX Real-time Monitor</title>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.0/socket.io.js"></script>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .metric-card { border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 5px; }
                    .alert { padding: 10px; margin: 5px; border-radius: 3px; }
                    .alert-warning { background-color: #fff3cd; border-color: #ffeaa7; }
                    .alert-error { background-color: #f8d7da; border-color: #f5c6cb; }
                    #gpu-chart, #training-chart { height: 400px; }
                </style>
            </head>
            <body>
                <h1>🔍 PPD-FLUX Real-time Monitor</h1>

                <div class="metric-card">
                    <h3>System Status</h3>
                    <p>GPU Memory: <span id="gpu-memory">-</span></p>
                    <p>GPU Temperature: <span id="gpu-temp">-</span></p>
                    <p>CPU Usage: <span id="cpu-usage">-</span></p>
                    <p>Training Steps: <span id="training-steps">-</span></p>
                </div>

                <div class="metric-card">
                    <h3>GPU Memory Usage</h3>
                    <div id="gpu-chart"></div>
                </div>

                <div class="metric-card">
                    <h3>Training Loss</h3>
                    <div id="training-chart"></div>
                </div>

                <div class="metric-card">
                    <h3>Recent Alerts</h3>
                    <div id="alerts-container"></div>
                </div>

                <script>
                    const socket = io();

                    const gpuMemoryData = [];
                    const trainingLossData = [];

                    socket.on('status_update', function(data) {
                        // Update status display
                        if (data.summary) {
                            document.getElementById('gpu-memory').textContent =
                                data.summary.gpu_memory_usage_percent.toFixed(1) + '%';
                            document.getElementById('gpu-temp').textContent =
                                data.summary.gpu_temperature.toFixed(1) + '°C';
                            document.getElementById('cpu-usage').textContent =
                                data.summary.cpu_utilization.toFixed(1) + '%';
                            document.getElementById('training-steps').textContent =
                                data.summary.training_steps;
                        }

                        // Update alerts
                        const alertsContainer = document.getElementById('alerts-container');
                        alertsContainer.innerHTML = '';
                        data.recent_alerts.forEach(alert => {
                            const alertDiv = document.createElement('div');
                            alertDiv.className = `alert alert-${alert.level}`;
                            alertDiv.textContent = `[${alert.category}] ${alert.message}`;
                            alertsContainer.appendChild(alertDiv);
                        });
                    });

                    socket.on('metrics_update', function(data) {
                        if (data.type === 'system') {
                            gpuMemoryData.push({
                                x: new Date(data.metrics.timestamp * 1000),
                                y: data.metrics.gpu_memory_used
                            });

                            if (gpuMemoryData.length > 100) {
                                gpuMemoryData.shift();
                            }

                            Plotly.redraw('gpu-chart', [{
                                x: gpuMemoryData.map(d => d.x),
                                y: gpuMemoryData.map(d => d.y),
                                type: 'scatter',
                                mode: 'lines',
                                name: 'GPU Memory (GB)'
                            }]);
                        }

                        if (data.type === 'training') {
                            trainingLossData.push({
                                x: data.metrics.step,
                                y: data.metrics.loss
                            });

                            if (trainingLossData.length > 100) {
                                trainingLossData.shift();
                            }

                            Plotly.redraw('training-chart', [{
                                x: trainingLossData.map(d => d.x),
                                y: trainingLossData.map(d => d.y),
                                type: 'scatter',
                                mode: 'lines',
                                name: 'Training Loss'
                            }]);
                        }
                    });

                    // Initialize charts
                    Plotly.newPlot('gpu-chart', [], {title: 'GPU Memory Usage'});
                    Plotly.newPlot('training-chart', [], {title: 'Training Loss'});

                    // Request initial status
                    socket.emit('get_status');
                </script>
            </body>
            </html>
            """

        @self.app.route('/api/status')
        def api_status():
            return jsonify(self.get_current_status())

        @self.socketio.on('get_status')
        def handle_get_status():
            emit('status_update', self.get_current_status())

        def dashboard_loop():
            # 실시간 데이터 전송
            while self.monitoring_active:
                try:
                    # 메트릭 큐에서 데이터 가져오기
                    while not self.metrics_queue.empty():
                        metric_type, metrics = self.metrics_queue.get_nowait()
                        self.socketio.emit('metrics_update', {
                            'type': metric_type,
                            'metrics': asdict(metrics)
                        })

                    # 주기적으로 상태 업데이트
                    self.socketio.emit('status_update', self.get_current_status())

                    time.sleep(2)  # 2초마다 업데이트
                except Exception as e:
                    logger.error(f"Dashboard loop error: {e}")
                    time.sleep(5)

        def run_dashboard():
            dashboard_config = self.config["dashboard"]
            self.socketio.run(
                self.app,
                host=dashboard_config["host"],
                port=dashboard_config["port"],
                debug=dashboard_config["debug"],
                allow_unsafe_werkzeug=True
            )

        # 대시보드 스레드 시작
        self.dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        self.dashboard_thread.start()

        # 데이터 전송 스레드 시작
        data_thread = threading.Thread(target=dashboard_loop, daemon=True)
        data_thread.start()

        dashboard_config = self.config["dashboard"]
        logger.info(f"🌐 Web dashboard started at http://{dashboard_config['host']}:{dashboard_config['port']}")


# 사용 예제
if __name__ == "__main__":
    # 실시간 모니터 테스트
    monitor = RealtimeMonitor()

    # 모니터링 시작
    monitor.start_monitoring()

    # 웹 대시보드 시작
    monitor.start_web_dashboard()

    # 시뮬레이션된 훈련 메트릭
    try:
        for step in range(20):
            time.sleep(2)

            # 시뮬레이션된 훈련 메트릭 추가
            training_metrics = TrainingMetrics(
                timestamp=time.time(),
                step=step,
                epoch=0,
                loss=1.0 / (step + 1) + 0.1 * torch.randn(1).item(),
                learning_rate=1e-4,
                batch_size=4,
                user_count=3,
                upe_generation_time=50.0,
                step_time=2.0,
                samples_per_second=2.0
            )

            monitor.add_training_metrics(training_metrics)

            logger.info(f"Training step {step} completed", extra={"loss": training_metrics.loss, "step": step})

    except KeyboardInterrupt:
        logger.info("Stopping monitoring...")
    finally:
        monitor.stop_monitoring()
        monitor.save_monitoring_data()
        monitor.generate_performance_plots()