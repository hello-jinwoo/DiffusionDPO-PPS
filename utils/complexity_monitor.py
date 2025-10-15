"""
Code complexity monitoring and reporting utilities.

Provides tools to measure and track cyclomatic complexity, maintainability
index, and other code quality metrics for the PPD codebase.
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComplexityMetrics:
    """Code complexity metrics for a file or module."""
    path: str
    cyclomatic_complexity: float
    maintainability_index: float
    lines_of_code: int
    functions_count: int
    classes_count: int
    average_complexity: float


@dataclass
class ComplexityThresholds:
    """Thresholds for code complexity warnings."""
    # Cyclomatic complexity
    function_cc_warning: int = 10
    function_cc_error: int = 15
    file_cc_warning: int = 50
    file_cc_error: int = 100

    # Maintainability index (higher is better)
    maintainability_warning: float = 20.0
    maintainability_good: float = 65.0

    # Lines of code
    function_loc_warning: int = 50
    file_loc_warning: int = 300


class ComplexityMonitor:
    """
    Monitor and report code complexity metrics.

    Uses radon to measure cyclomatic complexity and maintainability index.
    Provides warnings and recommendations for code quality improvement.
    """

    def __init__(self, thresholds: Optional[ComplexityThresholds] = None):
        """
        Initialize complexity monitor.

        Args:
            thresholds: Custom complexity thresholds
        """
        self.thresholds = thresholds or ComplexityThresholds()
        self._check_radon_available()

    def _check_radon_available(self) -> bool:
        """Check if radon is installed."""
        try:
            result = subprocess.run(
                ["radon", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Radon available: {result.stdout.strip()}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Radon not available. Install with: pip install radon")
            return False

        return False

    def measure_cyclomatic_complexity(
        self,
        path: str,
        min_grade: str = "C"
    ) -> Dict[str, List[Dict[str, any]]]:
        """
        Measure cyclomatic complexity for files in path.

        Args:
            path: File or directory path
            min_grade: Minimum grade to report (A, B, C, D, E, F)

        Returns:
            Dictionary mapping file paths to complexity results
        """
        try:
            # Run radon cc with JSON output
            cmd = ["radon", "cc", path, "-j", "-n", min_grade]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                logger.error(f"Radon cc failed: {result.stderr}")
                return {}

            # Parse JSON output
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                logger.error("Failed to parse radon output")
                return {}

        except subprocess.TimeoutExpired:
            logger.error("Radon cc timed out")
            return {}
        except Exception as e:
            logger.error(f"Error measuring complexity: {e}")
            return {}

    def measure_maintainability(
        self,
        path: str,
        min_grade: str = "C"
    ) -> Dict[str, float]:
        """
        Measure maintainability index for files in path.

        Args:
            path: File or directory path
            min_grade: Minimum grade to report

        Returns:
            Dictionary mapping file paths to MI scores
        """
        try:
            # Run radon mi with JSON output
            cmd = ["radon", "mi", path, "-j", "-n", min_grade]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                logger.error(f"Radon mi failed: {result.stderr}")
                return {}

            # Parse JSON output
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                logger.error("Failed to parse radon output")
                return {}

        except subprocess.TimeoutExpired:
            logger.error("Radon mi timed out")
            return {}
        except Exception as e:
            logger.error(f"Error measuring maintainability: {e}")
            return {}

    def analyze_directory(
        self,
        directory: str
    ) -> Dict[str, ComplexityMetrics]:
        """
        Analyze complexity for all Python files in directory.

        Args:
            directory: Directory path to analyze

        Returns:
            Dictionary mapping file paths to metrics
        """
        metrics = {}

        # Get cyclomatic complexity
        cc_results = self.measure_cyclomatic_complexity(directory, min_grade="A")

        # Get maintainability index
        mi_results = self.measure_maintainability(directory, min_grade="A")

        # Combine results
        for filepath, cc_items in cc_results.items():
            # Calculate average complexity
            if cc_items:
                avg_cc = sum(item.get("complexity", 0) for item in cc_items) / len(cc_items)
            else:
                avg_cc = 0.0

            # Get MI score
            mi_data = mi_results.get(filepath, {})
            if isinstance(mi_data, dict):
                mi_score = mi_data.get('mi', 0.0) if mi_data else 0.0
            else:
                mi_score = float(mi_data) if mi_data else 0.0

            # Count functions and classes
            functions = sum(1 for item in cc_items if item.get("type") in ["method", "function"])
            classes = sum(1 for item in cc_items if item.get("type") == "class")

            metrics[filepath] = ComplexityMetrics(
                path=filepath,
                cyclomatic_complexity=avg_cc,
                maintainability_index=mi_score,
                lines_of_code=0,  # Would need separate tool
                functions_count=functions,
                classes_count=classes,
                average_complexity=avg_cc
            )

        return metrics

    def get_warnings(
        self,
        metrics: Dict[str, ComplexityMetrics]
    ) -> List[Dict[str, str]]:
        """
        Get warnings for files exceeding thresholds.

        Args:
            metrics: Complexity metrics dictionary

        Returns:
            List of warning messages
        """
        warnings = []

        for filepath, metric in metrics.items():
            # Check cyclomatic complexity
            if metric.average_complexity >= self.thresholds.function_cc_error:
                warnings.append({
                    "level": "error",
                    "file": filepath,
                    "metric": "cyclomatic_complexity",
                    "value": metric.average_complexity,
                    "threshold": self.thresholds.function_cc_error,
                    "message": f"High complexity: {metric.average_complexity:.1f}"
                })
            elif metric.average_complexity >= self.thresholds.function_cc_warning:
                warnings.append({
                    "level": "warning",
                    "file": filepath,
                    "metric": "cyclomatic_complexity",
                    "value": metric.average_complexity,
                    "threshold": self.thresholds.function_cc_warning,
                    "message": f"Elevated complexity: {metric.average_complexity:.1f}"
                })

            # Check maintainability
            if metric.maintainability_index < self.thresholds.maintainability_warning:
                warnings.append({
                    "level": "error",
                    "file": filepath,
                    "metric": "maintainability_index",
                    "value": metric.maintainability_index,
                    "threshold": self.thresholds.maintainability_warning,
                    "message": f"Low maintainability: {metric.maintainability_index:.1f}"
                })

        return warnings

    def generate_report(
        self,
        directory: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive complexity report.

        Args:
            directory: Directory to analyze
            output_path: Optional output file path

        Returns:
            Report as string
        """
        metrics = self.analyze_directory(directory)
        warnings = self.get_warnings(metrics)

        # Build report
        lines = ["=" * 60]
        lines.append("Code Complexity Report")
        lines.append("=" * 60)
        lines.append("")

        # Summary
        if metrics:
            avg_cc = sum(m.average_complexity for m in metrics.values()) / len(metrics)
            avg_mi = sum(m.maintainability_index for m in metrics.values()) / len(metrics)

            lines.append("Summary:")
            lines.append(f"  Files analyzed: {len(metrics)}")
            lines.append(f"  Average cyclomatic complexity: {avg_cc:.2f}")
            lines.append(f"  Average maintainability index: {avg_mi:.2f}")
            lines.append("")

        # Warnings
        if warnings:
            lines.append(f"Warnings: {len(warnings)}")
            lines.append("-" * 60)
            for warning in warnings:
                level = warning['level'].upper()
                lines.append(f"[{level}] {warning['file']}")
                lines.append(f"  {warning['message']}")
                lines.append("")
        else:
            lines.append("✅ No complexity warnings!")
            lines.append("")

        # Detailed metrics
        lines.append("Detailed Metrics:")
        lines.append("-" * 60)
        for filepath, metric in sorted(metrics.items()):
            lines.append(f"{filepath}")
            lines.append(f"  CC: {metric.average_complexity:.2f}, "
                        f"MI: {metric.maintainability_index:.2f}, "
                        f"Functions: {metric.functions_count}, "
                        f"Classes: {metric.classes_count}")
            lines.append("")

        report = "\n".join(lines)

        # Save if output path provided
        if output_path:
            Path(output_path).write_text(report)
            logger.info(f"Report saved to {output_path}")

        return report


def check_complexity(
    paths: List[str],
    max_complexity: int = 10
) -> bool:
    """
    Quick check if any file exceeds complexity threshold.

    Args:
        paths: List of file/directory paths
        max_complexity: Maximum allowed complexity

    Returns:
        True if all files pass, False otherwise
    """
    monitor = ComplexityMonitor()

    for path in paths:
        metrics = monitor.analyze_directory(path)
        for filepath, metric in metrics.items():
            if metric.average_complexity > max_complexity:
                logger.warning(
                    f"{filepath}: complexity {metric.average_complexity:.1f} "
                    f"exceeds threshold {max_complexity}"
                )
                return False

    return True