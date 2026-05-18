"""Backwards-compat shim. The Modal app moved to qr_sampler.connectors.modal.app.

Scheduled for deletion in qr-sampler 0.5.0. Prefer:

    modal deploy -m qr_sampler.connectors.modal.app
"""

from qr_sampler.connectors.modal.app import (  # noqa: F401
    app,
    download_weights,
    weights_volume,
    VllmQrGemma,
    VllmQrQwen,
)
