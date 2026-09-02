"""Model backend: all architecture-specific mechanics live behind these interfaces."""

from .base import (  # noqa: F401
    Component,
    HookPoint,
    InvalidHookPointError,
    ModelMetadata,
    PatchDirection,
    PatchMode,
    PatchSpec,
    SequenceScore,
    TokenAlignment,
    TokenAlignmentError,
    TokenizationBoundaryError,
    TokenizedExample,
    TokenPolicy,
    resolve_alignment,
)
from .hf import HFBackend  # noqa: F401
from .hook_maps import HOOK_MAP_VERSION, LlamaHookMap  # noqa: F401
