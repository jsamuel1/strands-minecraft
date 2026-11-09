"""History audit / pair doctor — history-doctor.ts."""

from __future__ import annotations

from strands.types.exceptions import ModelThrottledException

from mcbrain.doctor import audit_history, classify_provider_error, diagnose, heal_empty_text, history_shape


def use(tid: str) -> dict:
    return {"role": "assistant", "content": [{"toolUse": {"toolUseId": tid, "name": "dig", "input": {}}}]}


def res(tid: str) -> dict:
    return {"role": "user", "content": [{"toolResult": {"toolUseId": tid, "status": "success", "content": []}}]}


U = {"role": "user", "content": [{"text": "hi"}]}


def test_well_formed_history_passes() -> None:
    assert audit_history([U, use("aaaaaa1"), res("aaaaaa1")]).ok


def test_split_pair_orphan_out_of_order_and_duplicates_are_named() -> None:
    problems = " | ".join(audit_history([U, use("x1"), res("y2"), res("x1"), res("x1")]).problems)
    assert "orphan toolResult" in problems
    assert "duplicate toolResult" in problems
    assert "precedes" in " ".join(audit_history([U, res("z9"), use("z9")]).problems)
    assert "NO toolResult" in " ".join(audit_history([U, use("q1")]).problems)


def test_history_must_start_with_user() -> None:
    assert not audit_history([use("a"), res("a")]).ok


def test_shape_is_content_free_and_marks_empty_text() -> None:
    shape = history_shape(
        [
            {"role": "user", "content": [{"text": "SECRET coords 1,2,3"}]},
            {"role": "assistant", "content": [{"text": ""}]},
        ]
    )
    assert "SECRET" not in shape and "+∅" in shape


def test_heal_empty_text_strips_or_drops_and_never_substitutes() -> None:
    msgs = [
        U,
        {"role": "assistant", "content": [{"text": " "}, {"toolUse": {"toolUseId": "t", "name": "dig", "input": {}}}]},
        {"role": "assistant", "content": [{"text": ""}]},
    ]
    stripped, removed = heal_empty_text(msgs)
    assert stripped == [1] and removed == [2]
    assert msgs[1]["content"] == [{"toolUse": {"toolUseId": "t", "name": "dig", "input": {}}}]
    assert len(msgs) == 2


def test_classification() -> None:
    assert classify_provider_error(ModelThrottledException("x")).kind == "throttling"
    assert classify_provider_error(TypeError("x is not a function")).kind == "local"
    assert classify_provider_error(RuntimeError("messages: text content blocks must be non-empty")).kind == "history-empty-text"
    assert classify_provider_error(RuntimeError("ServiceUnavailableException 503")).retryable
    assert not classify_provider_error(RuntimeError("ValidationException: toolResult blocks")).retryable


def test_diagnose_blames_the_right_party() -> None:
    assert "OUR defect" in diagnose(TypeError("boom"), [U])
    assert "MALFORMED" in diagnose(RuntimeError("Bedrock is unable"), [U, use("a")])
