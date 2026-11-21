"""Per-request token log: arguments, the selected token's bracket, the writer,
and the vLLM adapter writing one file per logged request.

The property everything else rests on is checked end to end at the bottom: a
request that asks for the log samples exactly what the same request samples
without it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from qr_sampler.config import QRSamplerConfig, resolve_config, validate_extra_args
from qr_sampler.config.log_args import LOG_ARG_KEYS, TokenLogArgError, parse_log_args
from qr_sampler.engines.vllm.adapter import VLLMAdapter, _RequestState
from qr_sampler.exceptions import ConfigValidationError
from qr_sampler.logging.token_log import (
    CANDIDATE_MASS,
    LOG_DIR_ENV,
    TokenLogWriter,
    log_path,
    sober_row,
    token_log_dir,
    trim_candidates,
)
from qr_sampler.logging.types import TokenSamplingRecord
from qr_sampler.selection.selector import TokenSelector
from tests.test_engines.test_vllm_adapter import (
    MockAddedRequest,
    MockBatchUpdate,
    MockSamplingParams,
    _make_adapter,
)

if TYPE_CHECKING:
    from pathlib import Path

REQUEST_ID = "run_ab12/qrng/card:0007"
SEED = 20261121


def _record(**over: Any) -> TokenSamplingRecord:
    base: dict[str, Any] = {
        "timestamp_ns": 1_700_000_000_000,
        "entropy_fetch_ms": 0.25,
        "total_sampling_ms": 0.5,
        "entropy_source_used": "quantum_grpc",
        "entropy_is_fallback": False,
        "sample_mean": 127.5,
        "z_score": 0.1,
        "u_value": 0.123456789012345,
        "temperature_strategy": "fixed",
        "shannon_entropy": 2.0,
        "temperature_used": 1.0,
        "token_id": 7,
        "token_rank": 1,
        "token_prob": 0.2,
        "num_candidates": 3,
        "config_hash": "abc",
        "min_p_used": 0.1,
        "cdf_lo": 0.1,
        "cdf_hi": 0.3,
    }
    base.update(over)
    return TokenSamplingRecord(**base)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ── arguments ────────────────────────────────────────────────────────────────


class TestLogArgs:
    def test_absent_means_no_log(self) -> None:
        assert parse_log_args(None) == (False, None)
        assert parse_log_args({"qr_top_k": 5}) == (False, None)

    @pytest.mark.parametrize("value", [True, 1, "true", "1", "on", "YES"])
    def test_truthy_values_ask_for_the_log(self, value: Any) -> None:
        args = {"qr_token_log": value, "qr_request_id": REQUEST_ID}
        assert parse_log_args(args) == (True, REQUEST_ID)

    def test_an_id_alone_does_not_start_a_log(self) -> None:
        assert parse_log_args({"qr_request_id": REQUEST_ID}) == (False, REQUEST_ID)
        assert parse_log_args({"qr_token_log": "off", "qr_request_id": "a"}) == (False, "a")

    @pytest.mark.parametrize(
        "args",
        [
            {"qr_token_log": True},
            {"qr_token_log": "maybe", "qr_request_id": "a"},
            {"qr_token_log": True, "qr_request_id": "../etc/passwd"},
            {"qr_token_log": True, "qr_request_id": "a/../b"},
            {"qr_token_log": True, "qr_request_id": "/abs"},
            {"qr_token_log": True, "qr_request_id": "has space"},
            {"qr_token_log": True, "qr_request_id": "back\\slash"},
            {"qr_token_log": True, "qr_request_id": ""},
            {"qr_token_log": True, "qr_request_id": "x" * 201},
            {"qr_token_log": True, "qr_request_id": 7},
        ],
    )
    def test_unusable_values_are_refused(self, args: dict[str, Any]) -> None:
        with pytest.raises(TokenLogArgError):
            parse_log_args(args)
        with pytest.raises(ConfigValidationError):
            validate_extra_args(args)

    def test_log_arguments_never_become_configuration(self) -> None:
        defaults = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        logged = {"qr_token_log": True, "qr_request_id": REQUEST_ID}
        assert resolve_config(defaults, logged) == defaults
        overridden = resolve_config(defaults, {"qr_top_k": 7})
        assert resolve_config(defaults, {"qr_top_k": 7, **logged}) == overridden
        assert not {key[3:] for key in LOG_ARG_KEYS} & set(QRSamplerConfig.model_fields)

    def test_adapter_validation_accepts_and_rejects(self) -> None:
        good = {"qr_token_log": True, "qr_request_id": REQUEST_ID}
        VLLMAdapter.validate_params(MockSamplingParams(extra_args=good))
        with pytest.raises(ConfigValidationError):
            VLLMAdapter.validate_params(MockSamplingParams(extra_args={"qr_token_log": True}))


# ── the selected token's bracket and the candidate list ──────────────────────


class TestBracket:
    @pytest.mark.parametrize(
        ("vocab", "top_k", "top_p", "min_p", "truncate_first"),
        [
            (50, 0, 1.0, 0.0, False),  # full-vocabulary path
            (50, 10, 1.0, 0.0, False),  # compact top-k path
            (50, 0, 0.9, 0.05, False),  # nucleus and min-p
            (50, 0, 1.0, 0.1, True),  # truncate first
            (4096, 0, 1.0, 0.0, False),  # large vocabulary: the head fast path
            (4096, 0, 0.95, 0.0, False),
        ],
    )
    def test_the_bracket_holds_u_and_matches_the_candidates(
        self, vocab: int, top_k: int, top_p: float, min_p: float, truncate_first: bool
    ) -> None:
        rng = np.random.default_rng(vocab + top_k)
        selector = TokenSelector()
        for _ in range(40):
            logits = rng.normal(0.0, 2.0, vocab)
            u = float(rng.uniform(1e-6, 1 - 1e-6))
            kwargs: dict[str, Any] = {"min_p": min_p, "truncate_first": truncate_first}
            plain = selector.select(logits, 0.8, top_k, top_p, u, **kwargs)
            result = selector.select(logits, 0.8, top_k, top_p, u, want_candidates=True, **kwargs)

            # Asking for the candidates changes nothing about the selection.
            assert (result.token_id, result.token_rank, result.token_prob) == (
                plain.token_id,
                plain.token_rank,
                plain.token_prob,
            )
            assert "candidates" not in plain.diagnostics

            lo, hi = result.diagnostics["cdf_lo"], result.diagnostics["cdf_hi"]
            assert lo <= u < hi or (hi == 1.0 and lo <= u)
            assert hi - lo == pytest.approx(result.token_prob, abs=1e-9)

            candidates = result.diagnostics["candidates"]
            assert len(candidates) == result.num_candidates
            probs = [p for _, p in candidates]
            assert probs == sorted(probs, reverse=True)
            assert sum(probs) == pytest.approx(1.0, abs=1e-9)
            # The bracket rebuilt from the candidate list is the logged one.
            assert candidates[result.token_rank][1] == pytest.approx(result.token_prob)
            assert sum(probs[: result.token_rank]) == pytest.approx(lo, abs=1e-9)
            assert result.token_id in {token for token, _ in candidates}
            if top_k:
                assert len(candidates) <= top_k

    def test_the_last_candidate_is_closed_at_one(self) -> None:
        result = TokenSelector().select(np.array([0.0, 0.0, 0.0]), 1.0, 0, 1.0, 0.999999)
        assert result.token_rank == 2 and result.diagnostics["cdf_hi"] == 1.0

    def test_cdf_select_keeps_its_four_values(self) -> None:
        probs = np.array([0.5, 0.3, 0.2])
        assert TokenSelector._cdf_select(probs, 0.6) == (1, 1, 0.3, 3)
        assert TokenSelector._cdf_select_bracket(probs, 0.6) == (1, 1, 0.3, 3, 0.5, 0.8)


# ── rows and the writer ──────────────────────────────────────────────────────


class TestRows:
    def test_a_row_is_small_and_keeps_the_draw_exact(self) -> None:
        row = sober_row(3, _record(token_prob=0.2000000000123))
        assert row == {
            "token_index": 3,
            "token_id": 7,
            "timestamp_ns": 1_700_000_000_000,
            "u_value": 0.123456789012345,
            "cdf_lo": 0.1,
            "cdf_hi": 0.3,
            "token_rank": 1,
            "token_prob": 0.2,
            "temperature_used": 1.0,
            "min_p_used": 0.1,
            "num_candidates": 3,
            "entropy_is_fallback": False,
        }
        assert len(json.dumps(row, separators=(",", ":"))) < 260

    def test_a_draw_next_to_a_bracket_edge_stays_inside_it(self) -> None:
        lo = 0.30000000000000004
        row = sober_row(1, _record(u_value=lo + 1e-16, cdf_lo=lo, cdf_hi=0.4))
        assert row["cdf_lo"] <= row["u_value"] <= row["cdf_hi"]
        assert json.loads(json.dumps(row))["u_value"] == lo + 1e-16

    def test_only_the_first_row_carries_candidates_and_the_source(self) -> None:
        entries = ((7, 0.6), (2, 0.3), (9, 0.1))
        first = sober_row(0, _record(top_entries=entries))
        assert first["top_entries"] == [[7, 0.6], [2, 0.3], [9, 0.1]]
        assert first["entropy_source_used"] == "quantum_grpc"
        later = sober_row(1, _record(top_entries=entries))
        assert "top_entries" not in later and "entropy_source_used" not in later

    def test_candidates_are_cut_at_the_mass_and_at_the_limit(self) -> None:
        tail = [(i, 0.0001) for i in range(100, 110)]
        entries = [(1, 0.9), (2, 0.099), *tail]
        kept = trim_candidates(entries)
        assert sum(p for _, p in kept) >= CANDIDATE_MASS
        assert len(kept) == 2 + 6  # 0.999 + six tail tokens reach 0.9995
        assert len(trim_candidates(entries, limit=3)) == 3
        assert trim_candidates([]) == []


class TestWriter:
    def test_rows_are_on_disk_as_they_are_written(self, tmp_path: Path) -> None:
        path = log_path(tmp_path, REQUEST_ID)
        assert path.name == "run_ab12__qrng__card:0007.jsonl".replace(":", ":")
        writer = TokenLogWriter(path)
        writer.write(0, _record(token_id=1))
        writer.write(1, _record(token_id=2))
        # readable before close: the harness reads right after the stream ends
        assert [r["token_id"] for r in _rows(path)] == [1, 2]
        writer.close()
        assert [r["token_index"] for r in _rows(path)] == [0, 1]

    def test_a_position_sampled_twice_keeps_the_later_sample(self, tmp_path: Path) -> None:
        path = tmp_path / "a.jsonl"
        writer = TokenLogWriter(path)
        writer.write(0, _record(token_id=1, u_value=0.11))  # sampled, then discarded by vLLM
        writer.write(0, _record(token_id=5, u_value=0.55))  # the emitted token
        writer.write(1, _record(token_id=6))
        writer.write(2, _record(token_id=7))
        writer.write(1, _record(token_id=8))  # a rewind replaces the later row too
        writer.close()
        rows = _rows(path)
        assert [(r["token_index"], r["token_id"]) for r in rows] == [(0, 5), (1, 8)]
        assert rows[0]["u_value"] == 0.55 and writer.rows_written == 2

    def test_a_reused_request_id_starts_again(self, tmp_path: Path) -> None:
        path = tmp_path / "a.jsonl"
        first = TokenLogWriter(path)
        for index in range(3):
            first.write(index, _record())
        first.close()
        second = TokenLogWriter(path)
        second.write(0, _record(token_id=42))
        second.close()
        assert [r["token_id"] for r in _rows(path)] == [42]

    def test_a_gap_stops_the_log_without_raising(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "a.jsonl"
        writer = TokenLogWriter(path)
        writer.write(0, _record())
        writer.write(2, _record())  # position 1 never arrived
        writer.write(3, _record())  # ignored: the writer is off
        writer.close()
        assert [r["token_index"] for r in _rows(path)] == [0]
        assert "not logged" in caplog.text

    def test_an_unwritable_path_never_raises(self, tmp_path: Path) -> None:
        blocker = tmp_path / "file"
        blocker.write_text("x", encoding="utf-8")
        writer = TokenLogWriter(blocker / "sub" / "a.jsonl")  # parent is a file
        writer.write(0, _record())
        writer.close()
        assert writer.rows_written == 0

    def test_directory_comes_from_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(LOG_DIR_ENV, raising=False)
        assert token_log_dir() is None
        monkeypatch.setenv(LOG_DIR_ENV, "  ")
        assert token_log_dir() is None
        monkeypatch.setenv(LOG_DIR_ENV, str(tmp_path))
        assert token_log_dir() == tmp_path


# ── the vLLM adapter ─────────────────────────────────────────────────────────


def _logits(step: int, vocab: int = 10) -> np.ndarray:
    rng = np.random.default_rng(1000 + step)
    return rng.normal(0.0, 1.5, (1, vocab)).astype(np.float32)


def _selected(row: np.ndarray) -> int:
    (ids,) = np.nonzero(row == 0.0)
    assert len(ids) == 1
    return int(ids[0])


def _generate(adapter: VLLMAdapter, extra: dict[str, Any], steps: int) -> tuple[list[int], str]:
    """Run one seeded request for ``steps`` tokens; returns its tokens and config hash."""
    output: list[int] = []
    params = MockSamplingParams(
        extra_args={"qr_preset": "deterministic_prng", "qr_seed": SEED, **extra}
    )
    adapter.update_state(
        MockBatchUpdate(
            added=[MockAddedRequest(req_index=0, sampling_params=params, output_tok_ids=output)]
        )
    )
    state = adapter._request_states[0]
    assert isinstance(state, _RequestState)
    for step in range(steps):
        logits = _logits(step)
        adapter.apply(logits)
        output.append(_selected(logits[0]))  # vLLM appends the emitted token
    adapter.update_state(MockBatchUpdate(removed=[0]))
    return output, state.config_hash_str


class TestAdapter:
    def test_a_logged_request_samples_exactly_like_an_unlogged_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(LOG_DIR_ENV, str(tmp_path))
        adapter = _make_adapter()
        try:
            plain_tokens, plain_hash = _generate(adapter, {}, steps=20)
            assert list(tmp_path.iterdir()) == []
            logged_tokens, logged_hash = _generate(
                adapter, {"qr_token_log": True, "qr_request_id": REQUEST_ID}, steps=20
            )
        finally:
            adapter.close()
        assert logged_tokens == plain_tokens
        assert logged_hash == plain_hash

        rows = _rows(log_path(tmp_path, REQUEST_ID))
        assert [r["token_index"] for r in rows] == list(range(20))
        assert [r["token_id"] for r in rows] == logged_tokens
        for row in rows:
            assert row["cdf_lo"] <= row["u_value"] < row["cdf_hi"] or row["cdf_hi"] == 1.0
            # float32 probabilities: the running sum is exact to about 1e-7
            assert row["cdf_hi"] - row["cdf_lo"] == pytest.approx(row["token_prob"], abs=1e-6)
        first, rest = rows[0], rows[1:]
        assert first["entropy_source_used"]
        assert sum(p for _, p in first["top_entries"]) >= CANDIDATE_MASS
        assert first["token_id"] in {token for token, _ in first["top_entries"]}
        assert all("top_entries" not in row for row in rest)
        # twenty tokens fit in a few kilobytes
        assert log_path(tmp_path, REQUEST_ID).stat().st_size < 20 * 300 + 2000

    def test_a_row_discarded_during_prefill_is_replaced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(LOG_DIR_ENV, str(tmp_path))
        adapter = _make_adapter()
        try:
            output: list[int] = []
            params = MockSamplingParams(
                extra_args={
                    "qr_preset": "deterministic_prng",
                    "qr_seed": SEED,
                    "qr_token_log": True,
                    "qr_request_id": "discard",
                }
            )
            adapter.update_state(
                MockBatchUpdate(
                    added=[
                        MockAddedRequest(req_index=0, sampling_params=params, output_tok_ids=output)
                    ]
                )
            )
            adapter.apply(_logits(0))  # sampled, then discarded: nothing appended
            logits = _logits(0)
            adapter.apply(logits)
            output.append(_selected(logits[0]))
            logits = _logits(1)
            adapter.apply(logits)
            output.append(_selected(logits[0]))
            adapter.update_state(MockBatchUpdate(removed=[0]))
        finally:
            adapter.close()
        rows = _rows(tmp_path / "discard.jsonl")
        assert [r["token_index"] for r in rows] == [0, 1]
        assert [r["token_id"] for r in rows] == output

    def test_no_directory_means_no_file_and_no_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(LOG_DIR_ENV, raising=False)
        adapter = _make_adapter()
        try:
            tokens, _ = _generate(adapter, {"qr_token_log": True, "qr_request_id": "x"}, steps=3)
        finally:
            adapter.close()
        assert len(tokens) == 3 and list(tmp_path.iterdir()) == []

    def test_a_bypassed_request_writes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(LOG_DIR_ENV, str(tmp_path))
        adapter = _make_adapter()
        try:
            params = MockSamplingParams(
                extra_args={"qr_bypass": True, "qr_token_log": True, "qr_request_id": "bypass"}
            )
            adapter.update_state(
                MockBatchUpdate(added=[MockAddedRequest(req_index=0, sampling_params=params)])
            )
            adapter.apply(_logits(0))
            adapter.update_state(MockBatchUpdate(removed=[0]))
        finally:
            adapter.close()
        assert list(tmp_path.iterdir()) == []
