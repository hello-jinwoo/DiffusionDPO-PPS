"""
Strategy pattern implementation for DPO loss functions.
"""
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple
from dataclasses import dataclass


@dataclass
class LossConfig:
    """Configuration for loss computation."""
    beta: float = 0.5
    label_smoothing: float = 0.0


class LossStrategy(ABC):
    """Abstract base class for loss computation strategies."""

    def __init__(self, config: LossConfig):
        """
        Initialize loss strategy.

        Args:
            config: Loss configuration
        """
        self.config = config
        self.beta = config.beta

    @abstractmethod
    def compute(
        self,
        model_diff: torch.Tensor,
        ref_diff: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute loss and implicit accuracy.

        Args:
            model_diff: Model loss difference (win - lose)
            ref_diff: Reference loss difference (win - lose)

        Returns:
            Tuple of (loss, implicit_accuracy)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return strategy name."""
        pass


class SigmoidLossStrategy(LossStrategy):
    """Standard sigmoid DPO loss strategy."""

    def compute(
        self,
        model_diff: torch.Tensor,
        ref_diff: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute sigmoid DPO loss."""
        scale_term = -0.5 * self.beta
        inside_term = scale_term * (model_diff - ref_diff)
        implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
        loss = -F.logsigmoid(inside_term).mean()
        return loss, implicit_acc

    @property
    def name(self) -> str:
        return "sigmoid"


class HingeLossStrategy(LossStrategy):
    """Hinge DPO loss strategy."""

    def compute(
        self,
        model_diff: torch.Tensor,
        ref_diff: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute hinge DPO loss."""
        scale_term = -0.5 * self.beta
        inside_term = scale_term * (model_diff - ref_diff)
        loss = torch.relu(1.0 - inside_term).mean()
        implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
        return loss, implicit_acc

    @property
    def name(self) -> str:
        return "hinge"


class IPOLossStrategy(LossStrategy):
    """Identity Preference Optimization (IPO) loss strategy."""

    def compute(
        self,
        model_diff: torch.Tensor,
        ref_diff: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute IPO loss."""
        scale_term = -0.5 * self.beta
        inside_term = scale_term * (model_diff - ref_diff)
        loss = (inside_term - F.logsigmoid(inside_term)).mean()
        implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
        return loss, implicit_acc

    @property
    def name(self) -> str:
        return "ipo"


class CPOLossStrategy(LossStrategy):
    """Contrastive Preference Optimization (CPO) loss strategy."""

    def compute_from_losses(
        self,
        model_loss_win: torch.Tensor,
        model_loss_lose: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute CPO loss directly from win/lose losses.

        Args:
            model_loss_win: Model loss for winning samples
            model_loss_lose: Model loss for losing samples

        Returns:
            Tuple of (loss, implicit_accuracy)
        """
        # CPO directly optimizes the preference without reference model
        loss = torch.relu(model_loss_win - model_loss_lose + 0.1).mean()
        implicit_acc = ((model_loss_win < model_loss_lose).sum().float() / model_loss_win.size(0)).item()
        return loss, implicit_acc

    def compute(
        self,
        model_diff: torch.Tensor,
        ref_diff: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute CPO loss (not typically used with diffs)."""
        raise NotImplementedError("CPO should use compute_from_losses method")

    @property
    def name(self) -> str:
        return "cpo"


class BCOLossStrategy(LossStrategy):
    """Binary Cross-entropy Optimization (BCO) loss strategy."""

    def compute(
        self,
        model_diff: torch.Tensor,
        ref_diff: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute BCO loss."""
        scale_term = -0.5 * self.beta
        logits = scale_term * (model_diff - ref_diff)
        loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        implicit_acc = (logits > 0).sum().float() / logits.size(0)
        return loss, implicit_acc

    @property
    def name(self) -> str:
        return "bco"


class LossStrategyFactory:
    """Factory for creating loss strategies."""

    _strategies = {
        "sigmoid": SigmoidLossStrategy,
        "hinge": HingeLossStrategy,
        "ipo": IPOLossStrategy,
        "cpo": CPOLossStrategy,
        "bco": BCOLossStrategy,
    }

    @classmethod
    def create(cls, loss_type: str, config: LossConfig) -> LossStrategy:
        """
        Create loss strategy based on type.

        Args:
            loss_type: Type of loss strategy
            config: Loss configuration

        Returns:
            Loss strategy instance

        Raises:
            ValueError: If loss type is unknown
        """
        if loss_type not in cls._strategies:
            available = ", ".join(cls._strategies.keys())
            raise ValueError(f"Unknown loss type: {loss_type}. Available: {available}")

        strategy_class = cls._strategies[loss_type]
        return strategy_class(config)

    @classmethod
    def register_strategy(cls, name: str, strategy_class: type):
        """
        Register a new loss strategy.

        Args:
            name: Strategy name
            strategy_class: Strategy class
        """
        cls._strategies[name] = strategy_class

    @classmethod
    def available_strategies(cls) -> list:
        """Get list of available strategy names."""
        return list(cls._strategies.keys())