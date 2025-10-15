"""
Import compatibility fixes for peft/transformers version mismatch.

This module provides monkey-patches to resolve the incompatibility between:
- transformers==4.37.2 (which lacks EncoderDecoderCache, HybridCache)
- peft==0.17.1 (which requires these classes)
- diffusers==0.35.1 (which requires peft>=0.17.0)

The root cause is that peft 0.17.1 requires transformers>=4.44, but our project
uses transformers 4.37.2 for compatibility with other components (LLaVa, etc.).

This is a temporary workaround until we can upgrade transformers or the project
confirms it doesn't need LoRA functionality (in which case peft can be removed).

Usage:
    import utils.import_fix  # Call this BEFORE importing diffusers or peft
"""

import sys
import logging
from typing import Optional, Any, Tuple

logger = logging.getLogger(__name__)


def _create_dummy_cache_classes():
    """
    Create dummy Cache classes that peft expects but are missing in transformers 4.37.2.

    These classes were added in transformers 4.44+. Since our project doesn't actually
    use LoRA (use_lora=False), we can safely provide stub implementations that will
    never be called in practice.
    """
    try:
        # Try importing from transformers first
        from transformers import Cache, DynamicCache

        # Check if EncoderDecoderCache exists
        try:
            from transformers import EncoderDecoderCache, HybridCache
            logger.debug("✓ transformers already has EncoderDecoderCache and HybridCache")
            return  # Already have these classes, no patching needed
        except ImportError:
            pass

        # Create stub classes since they're missing
        logger.info("⚠️ Patching transformers with dummy EncoderDecoderCache and HybridCache")
        logger.info("   (These are only needed by peft but not used in this project)")

        class EncoderDecoderCache:
            """
            Stub implementation of EncoderDecoderCache for peft compatibility.
            This class is never actually used since the project doesn't use LoRA.
            """
            def __init__(self, *args, **kwargs):
                raise NotImplementedError(
                    "EncoderDecoderCache is a stub for peft compatibility. "
                    "This project doesn't use LoRA, so this should never be called. "
                    "If you see this error, please check your model configuration."
                )

            @classmethod
            def from_legacy_cache(cls, *args, **kwargs):
                raise NotImplementedError(
                    "EncoderDecoderCache.from_legacy_cache is a stub. "
                    "LoRA is not enabled in this project."
                )

        class HybridCache(Cache):
            """
            Stub implementation of HybridCache for peft compatibility.
            This class is never actually used since the project doesn't use LoRA.
            """
            def __init__(self, *args, **kwargs):
                raise NotImplementedError(
                    "HybridCache is a stub for peft compatibility. "
                    "This project doesn't use LoRA, so this should never be called."
                )

            def update(self, *args, **kwargs):
                raise NotImplementedError("HybridCache.update is a stub.")

            def get_seq_length(self, *args, **kwargs):
                raise NotImplementedError("HybridCache.get_seq_length is a stub.")

            def get_max_length(self, *args, **kwargs):
                return None

        # Inject into transformers module
        import transformers
        transformers.EncoderDecoderCache = EncoderDecoderCache
        transformers.HybridCache = HybridCache

        # Also inject into transformers.__init__ for direct imports
        if hasattr(transformers, '__all__'):
            if 'EncoderDecoderCache' not in transformers.__all__:
                transformers.__all__.append('EncoderDecoderCache')
            if 'HybridCache' not in transformers.__all__:
                transformers.__all__.append('HybridCache')

        logger.info("✓ Successfully patched transformers with dummy cache classes")

    except Exception as e:
        logger.error(f"❌ Failed to patch transformers: {e}")
        raise


def apply_import_fixes():
    """
    Apply all import compatibility fixes.

    This should be called BEFORE importing diffusers, peft, or any modules that
    depend on them (like core.pipeline_builder).
    """
    logger.info("Applying import compatibility fixes for peft/transformers...")

    # Patch 1: Add missing cache classes to transformers
    _create_dummy_cache_classes()

    logger.info("✓ All import fixes applied successfully")


# Auto-apply fixes when this module is imported
apply_import_fixes()
