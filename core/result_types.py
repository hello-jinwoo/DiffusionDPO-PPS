"""
Result types for standardized error handling.
"""
from typing import Generic, TypeVar, Union, Optional, Callable
from dataclasses import dataclass
from enum import Enum


T = TypeVar('T')
E = TypeVar('E')


class ErrorCategory(Enum):
    """Categories of errors that can occur."""
    MODEL_ERROR = "model_error"
    DATA_ERROR = "data_error"
    VALIDATION_ERROR = "validation_error"
    CONFIGURATION_ERROR = "configuration_error"
    RUNTIME_ERROR = "runtime_error"
    RECOVERABLE_ERROR = "recoverable_error"


@dataclass
class Error:
    """
    Standardized error type.

    Attributes:
        code: Error code for identification
        message: Human-readable error message
        category: Error category
        recoverable: Whether error is recoverable
        details: Additional error details
    """
    code: str
    message: str
    category: ErrorCategory
    recoverable: bool = False
    details: Optional[dict] = None

    def __str__(self) -> str:
        """String representation of error."""
        base = f"[{self.code}] {self.message}"
        if self.details:
            base += f" | Details: {self.details}"
        return base


class Result(Generic[T, E]):
    """
    Result type for operations that can fail.

    Inspired by Rust's Result type, this provides type-safe error handling
    without exceptions for expected errors.

    Example:
        >>> def divide(a: int, b: int) -> Result[float, Error]:
        ...     if b == 0:
        ...         return Result.err(Error("DIV_ZERO", "Division by zero", ErrorCategory.RUNTIME_ERROR))
        ...     return Result.ok(a / b)
        ...
        >>> result = divide(10, 2)
        >>> if result.is_ok():
        ...     print(f"Result: {result.unwrap()}")
        ... else:
        ...     print(f"Error: {result.unwrap_err()}")
    """

    def __init__(self, value: Optional[T] = None, error: Optional[E] = None):
        """Initialize result with either value or error."""
        if value is not None and error is not None:
            raise ValueError("Result cannot have both value and error")
        if value is None and error is None:
            raise ValueError("Result must have either value or error")

        self._value = value
        self._error = error

    @classmethod
    def ok(cls, value: T) -> 'Result[T, E]':
        """Create successful result."""
        return cls(value=value)

    @classmethod
    def err(cls, error: E) -> 'Result[T, E]':
        """Create error result."""
        return cls(error=error)

    def is_ok(self) -> bool:
        """Check if result is successful."""
        return self._value is not None

    def is_err(self) -> bool:
        """Check if result is error."""
        return self._error is not None

    def unwrap(self) -> T:
        """
        Unwrap value, raises if error.

        Raises:
            RuntimeError: If result is error
        """
        if self._value is not None:
            return self._value
        raise RuntimeError(f"Called unwrap on error result: {self._error}")

    def unwrap_or(self, default: T) -> T:
        """Unwrap value or return default if error."""
        return self._value if self._value is not None else default

    def unwrap_err(self) -> E:
        """
        Unwrap error, raises if successful.

        Raises:
            RuntimeError: If result is successful
        """
        if self._error is not None:
            return self._error
        raise RuntimeError("Called unwrap_err on ok result")

    def map(self, f: Callable[[T], 'U']) -> 'Result[U, E]':
        """Apply function to value if successful."""
        if self.is_ok():
            return Result.ok(f(self._value))
        return Result.err(self._error)

    def map_err(self, f: Callable[[E], 'F']) -> 'Result[T, F]':
        """Apply function to error if failed."""
        if self.is_err():
            return Result.err(f(self._error))
        return Result.ok(self._value)

    def and_then(self, f: Callable[[T], 'Result[U, E]']) -> 'Result[U, E]':
        """Chain result-returning operations."""
        if self.is_ok():
            return f(self._value)
        return Result.err(self._error)


@dataclass
class TrainingStepResult:
    """Result of a training step."""
    loss: float
    metrics: dict
    success: bool = True
    error: Optional[Error] = None


@dataclass
class ValidationResult:
    """Result of a validation run."""
    metrics: dict
    images: list
    success: bool = True
    error: Optional[Error] = None


def create_model_error(message: str, recoverable: bool = False, details: Optional[dict] = None) -> Error:
    """Create a model error."""
    return Error(
        code="MODEL_ERROR",
        message=message,
        category=ErrorCategory.MODEL_ERROR,
        recoverable=recoverable,
        details=details
    )


def create_data_error(message: str, recoverable: bool = True, details: Optional[dict] = None) -> Error:
    """Create a data error."""
    return Error(
        code="DATA_ERROR",
        message=message,
        category=ErrorCategory.DATA_ERROR,
        recoverable=recoverable,
        details=details
    )


def create_validation_error(message: str, recoverable: bool = True, details: Optional[dict] = None) -> Error:
    """Create a validation error."""
    return Error(
        code="VALIDATION_ERROR",
        message=message,
        category=ErrorCategory.VALIDATION_ERROR,
        recoverable=recoverable,
        details=details
    )


def create_config_error(message: str, recoverable: bool = False, details: Optional[dict] = None) -> Error:
    """Create a configuration error."""
    return Error(
        code="CONFIG_ERROR",
        message=message,
        category=ErrorCategory.CONFIGURATION_ERROR,
        recoverable=recoverable,
        details=details
    )