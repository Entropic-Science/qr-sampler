"""Entropy source subsystem for qr-sampler.

Re-exports the ABC, registry, and all built-in source implementations
for convenient access::

    from qr_sampler.entropy import EntropySource, EntropySourceRegistry
    from qr_sampler.entropy import SystemEntropySource, MockUniformSource
"""

from qr_sampler.entropy.base import EntropySource
from qr_sampler.entropy.fallback import FallbackEntropySource
from qr_sampler.entropy.mock import MockUniformSource
from qr_sampler.entropy.named import InstanceNamedSource
from qr_sampler.entropy.registry import EntropySourceRegistry, register_entropy_source
from qr_sampler.entropy.system import SystemEntropySource

# All built-in sources (including TimingNoiseSource and QuantumGrpcSource)
# are declared in EntropySourceRegistry._BUILTINS and imported lazily on
# first registry lookup — importing this package has no registration side
# effects.

__all__ = [
    "EntropySource",
    "EntropySourceRegistry",
    "FallbackEntropySource",
    "InstanceNamedSource",
    "MockUniformSource",
    "SystemEntropySource",
    "register_entropy_source",
]
