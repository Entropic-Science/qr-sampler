"""Tests for InstanceNamedSource — the instance-name rename wrapper."""

from __future__ import annotations

from qr_sampler.entropy.mock import MockUniformSource
from qr_sampler.entropy.named import InstanceNamedSource


class TestInstanceNamedSource:
    """The wrapper renames; everything else delegates unchanged."""

    def test_name_is_instance_name(self) -> None:
        source = InstanceNamedSource(MockUniformSource(), "qbert_prng_uniform")
        assert source.name == "qbert_prng_uniform"

    def test_inner_exposed_for_introspection(self) -> None:
        inner = MockUniformSource()
        source = InstanceNamedSource(inner, "lane")
        assert source.inner is inner

    def test_bytes_delegate_to_inner(self) -> None:
        source = InstanceNamedSource(MockUniformSource(seed=42), "lane")
        data = source.get_random_bytes(16)
        assert len(data) == 16
        assert data == MockUniformSource(seed=42).get_random_bytes(16)

    def test_is_available_delegates(self) -> None:
        source = InstanceNamedSource(MockUniformSource(), "lane")
        assert source.is_available is True

    def test_ticket_path_delegates(self) -> None:
        source = InstanceNamedSource(MockUniformSource(), "lane")
        # Mock has no async transport: prefetch yields None, redeem falls
        # through to the synchronous fetch.
        assert source.prefetch(8) is None
        assert len(source.get_random_bytes_with_ticket(8, None)) == 8

    def test_float64_delegates(self) -> None:
        source = InstanceNamedSource(MockUniformSource(seed=7), "lane")
        values = source.get_random_float64((4,))
        assert values.shape == (4,)
        assert (values >= 0.0).all() and (values < 1.0).all()

    def test_health_check_relabelled(self) -> None:
        source = InstanceNamedSource(MockUniformSource(), "qbert_prng_uniform")
        health = source.health_check()
        assert health["source"] == "qbert_prng_uniform"
        assert health["inner_source"] == "mock_uniform"
        assert health["healthy"] is True

    def test_unknown_attributes_forward_to_inner(self) -> None:
        class Probed(MockUniformSource):
            def custom_capability(self) -> str:
                return "probed"

        source = InstanceNamedSource(Probed(), "lane")
        # Duck-typed capability probes must survive the rename.
        assert source.custom_capability() == "probed"

    def test_warmup_and_close_delegate(self) -> None:
        closed: list[bool] = []

        class Closing(MockUniformSource):
            def close(self) -> None:
                closed.append(True)

        source = InstanceNamedSource(Closing(), "lane")
        source.warmup()  # no-op on the mock; must not raise
        source.close()
        assert closed == [True]
