"""Headless smoke tests for the Streamlit UI (app.py), run with: pytest -q

Uses Streamlit's official ``AppTest`` harness to exercise the app without a
browser. These are smoke tests, not exhaustive UI tests — they check that the
app renders, the example-prompt flow populates the input, and running a
check produces a verdict without raising, exercising the same offline
heuristic fallback path as the rest of the pytest suite (no Ollama needed).
"""

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")
HISTORY_PATH = Path(__file__).parent.parent / "data" / "ui_history.jsonl"


@pytest.fixture(autouse=True)
def _clean_history_file():
    """The UI persists checks to data/ui_history.jsonl; leave that file as we found it."""
    existed = HISTORY_PATH.exists()
    original = HISTORY_PATH.read_bytes() if existed else None
    yield
    if original is not None:
        HISTORY_PATH.write_bytes(original)
    elif HISTORY_PATH.exists():
        HISTORY_PATH.unlink()


def test_app_loads_without_exception():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception
    texts = [el.value for el in list(at.title) + list(at.header) + list(at.markdown)]
    assert any("Prompt Guard" in t for t in texts)


def test_example_button_populates_prompt():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception

    jailbreak_button = next(b for b in at.button if "Jailbreak" in b.label)
    jailbreak_button.click().run()
    assert not at.exception
    assert at.text_area(key="mg_prompt").value.strip() != ""


def test_check_prompt_produces_verdict():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception

    jailbreak_button = next(b for b in at.button if "Jailbreak" in b.label)
    jailbreak_button.click().run()
    assert not at.exception

    check_button = next(b for b in at.button if b.label == "Check prompt")
    check_button.click().run()
    assert not at.exception

    assert at.session_state["mg_last"] is not None
    result = at.session_state["mg_last"]["result"]
    assert result.category.value in {"Good/Valid", "Responsible", "Risky"}
    assert any("BLOCKED" in m.value or "ALLOWED" in m.value for m in at.markdown)

    history = at.session_state["mg_history"]
    assert len(history) == 1
    assert HISTORY_PATH.exists()
