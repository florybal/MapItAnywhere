"""Lightweight initializer for mapper.data.

Avoid importing dataset submodules at package import time because those
modules import sibling utilities and can trigger circular import errors.

Use the `get_dataset` helper in `mapper.data.module` which already performs
lazy imports when a dataset is requested.
"""

__all__ = ["get_dataset"]

def get_dataset(name: str):
    """Return the dataset class for `name` by delegating to
    `mapper.data.module.get_dataset`, which performs lazy imports.
    """
    from .module import get_dataset as _get_dataset
    return _get_dataset(name)
