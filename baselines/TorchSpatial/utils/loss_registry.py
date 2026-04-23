"""
Loss registry for GeoBS TorchSpatial baseline.

Use ``get_loss(name, **kwargs)`` to retrieve a loss function by name and
optionally bind extra keyword arguments to it.  The returned callable has the
same signature as the functions in ``TorchSpatial.utils.losses``:

    loss_fn(model, params, loc_feat, loc_class, user_ids, inds, **kwargs)

Supported names
---------------
``"embedding_loss"``
    The default contrastive location-embedding loss from the original
    TorchSpatial paper (equation 7 / 8 in https://arxiv.org/abs/1906.05272).
    This is the current default and keeps existing behaviour unchanged.

``"full_loss"``
    Placeholder – import and register a custom loss here when needed.

Adding a new loss
-----------------
1. Implement (or import) a callable with signature
       fn(model, params, loc_feat, loc_class, user_ids, inds, **kwargs)
2. Add it to ``_REGISTRY`` below.
3. Reference it by name in ``configs.json`` under ``"train_loss_name"``.
"""

from functools import partial
from typing import Any, Callable, Dict, Optional

from TorchSpatial.utils.losses import embedding_loss

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Callable] = {
    "embedding_loss": embedding_loss,
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_loss(name: str, fn: Callable) -> None:
    """Register a custom loss function under *name*."""
    _REGISTRY[name] = fn


def get_loss(name: str, loss_params: Optional[Dict[str, Any]] = None) -> Callable:
    """Return the loss callable registered under *name*.

    Parameters
    ----------
    name:
        Key used to look up the loss in the registry.
    loss_params:
        Optional dict of extra keyword arguments to bind to the loss via
        ``functools.partial``.  These are passed *in addition to* the standard
        positional arguments ``(model, params, loc_feat, loc_class, user_ids,
        inds)`` that every loss receives.

    Returns
    -------
    Callable
        A loss function (or partial thereof) ready to be used as the
        ``criterion`` argument in the trainer.

    Raises
    ------
    KeyError
        If *name* is not found in the registry.  Known names:
        ``"embedding_loss"``.
    """
    if name not in _REGISTRY:
        known = ", ".join(f'"{k}"' for k in sorted(_REGISTRY))
        raise KeyError(
            f"Unknown loss '{name}'. Known losses: {known}. "
            "Register a new loss with loss_registry.register_loss()."
        )
    fn = _REGISTRY[name]
    if loss_params:
        fn = partial(fn, **loss_params)
    return fn


def list_losses():
    """Return the names of all registered losses."""
    return list(_REGISTRY.keys())
