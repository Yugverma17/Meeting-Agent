r"""Mathematics surviving the trip from model to page.

Asked for the AdaBoost update rule, the answer arrived as

    (displaystyle epsilon_t = \x0crac{sum w_i}{sum w_j})

which is three faults at once: JSON ate the commands whose first letter happens
to be a valid escape, a lenient parser dropped the rest of the backslashes
entirely, and the delimiters the model chose are ones the page cannot render.

The repair has to be aggressive enough to fix that and conservative enough to
leave ordinary English alone - "I will not send it" contains a newline before a
word beginning with "n", and must stay a newline.
"""

from __future__ import annotations

import pytest

from quorum.chat.mathtext import repair_math


# --- what JSON ate ------------------------------------------------------------


@pytest.mark.parametrize("damaged,command", [
    ("\x0crac{a}{b}", r"\frac"),
    ("\x08igl(x\x08igr)", r"\bigl"),
    ("\x0corall x", r"\forall"),
])
def test_control_characters_are_commands_with_the_backslash_eaten(damaged, command):
    """A formfeed or backspace inside a word is never prose."""
    assert command in repair_math(f"${damaged}$")


def test_a_tab_before_a_known_command_is_repaired():
    assert r"\times" in repair_math("$w \timesexp$".replace("\\t", "\t"))


def test_a_real_tab_before_ordinary_text_is_left_alone():
    """Tabs are legitimate in indented code, so only a known command counts."""
    text = "here is code:\n\tresult = compute()\n"

    assert repair_math(text) == text


def test_a_newline_before_a_word_starting_with_n_stays_a_newline():
    """The failure this guards against turns "I will\nnot send it" into
    "I will\not send it" - a broken macro in the middle of a sentence."""
    text = "I will\nnot send it, and the notes are fine."

    assert repair_math(text) == text


def test_a_newline_before_neq_inside_maths_is_repaired():
    repaired = repair_math("$y_i \neq h_t(x_i)$".replace("\\n", "\n"))

    assert r"\neq" in repaired


# --- delimiters ---------------------------------------------------------------


def test_paren_delimiters_become_dollars():
    """KaTeX-in-markdown renders dollars and nothing else."""
    assert repair_math(r"and \(x^2\) inline") == "and $x^2$ inline"


def test_bracket_delimiters_become_display_dollars():
    assert repair_math(r"\[x^2\]") == "$$x^2$$"


def test_a_bare_paren_opening_on_displaystyle_is_a_maths_delimiter():
    """The backslash is already gone by the time this is seen; what follows is
    the only clue that the parenthesis was a delimiter and not punctuation."""
    repaired = repair_math("(displaystyle x = 1)")

    assert repaired.startswith("$$") and repaired.endswith("$$")


def test_ordinary_parentheses_are_not_touched():
    text = "the window (which ends at i) contains all three"

    assert repair_math(text) == text


# --- backslashes dropped without trace ----------------------------------------


def test_bare_commands_inside_maths_get_their_backslash_back():
    repaired = repair_math("$epsilon_t = sum_i w_i times alpha$")

    assert r"\epsilon" in repaired
    assert r"\sum" in repaired
    assert r"\times" in repaired
    assert r"\alpha" in repaired


def test_the_same_words_outside_maths_are_left_as_english():
    """"sum", "text", "in" and "to" are ordinary words. Restoring backslashes
    outside a maths region would wreck the prose to fix the formulae."""
    text = "The sum of the parts is in the text, according to him."

    assert repair_math(text) == text


def test_a_command_that_already_has_its_backslash_is_not_doubled():
    assert repair_math(r"$\frac{a}{b}$") == r"$\frac{a}{b}$"


def test_unknown_words_inside_maths_are_left_alone():
    """Variable names are not commands. `h_t` and `w_i` must survive."""
    repaired = repair_math("$h_t(x_i) = w_i$")

    assert "h_t" in repaired and "w_i" in repaired
    assert "\\h" not in repaired


# --- the whole thing ----------------------------------------------------------


def test_the_screenshot_is_repaired_end_to_end():
    """Exactly what reached the page, byte for byte."""
    damaged = (
        "(displaystyle epsilon_t = \x0crac{sum_{i=1}^{N} w_i^{(t)} "
        "mathbf{1}}{sum_{i=1}^{N} w_i^{(t)}})"
    )

    repaired = repair_math(damaged)

    assert repaired.startswith("$$") and repaired.endswith("$$")
    for expected in (r"\displaystyle", r"\epsilon", r"\frac", r"\sum", r"\mathbf"):
        assert expected in repaired, expected
    assert "\x0c" not in repaired


def test_an_answer_with_no_maths_is_returned_unchanged():
    text = "The brute force approach checks every substring, which is quadratic."

    assert repair_math(text) == text


def test_empty_input_is_harmless():
    assert repair_math("") == ""
    assert repair_math(None) is None


# --- the prompt that teaches this ---------------------------------------------


def test_the_maths_instruction_reaches_the_model_intact():
    """Written as an ordinary literal, the examples warning about eaten
    backslashes had their own eaten by Python - the model was told that
    "\\frac silently becomes rac" with the backslash missing from both halves."""
    from quorum.chat.answer import GROUNDED_SYSTEM, MATHS_RULE

    assert r"\frac" in MATHS_RULE
    assert r"\\frac" in MATHS_RULE
    assert "\x0c" not in MATHS_RULE, "the instruction is itself corrupted"
    assert MATHS_RULE in GROUNDED_SYSTEM
    assert "{maths}" not in GROUNDED_SYSTEM


def test_answers_are_repaired_before_they_are_returned():
    from quorum.chat.answer import AnswerDraft, Coverage, answer_question
    from quorum.llm.router import LLMResponse
    from quorum.memory.store import MemoryHit, MemoryKind

    class Stub:
        def structured(self, prompt, schema, **kwargs):
            draft = AnswerDraft(
                coverage=Coverage.COVERED,
                answer="$epsilon_t = \x0crac{a}{b}$",
                cited=[1],
            )
            return draft, LLMResponse(text="", model="s", provider="s")

    hits = [MemoryHit(MemoryKind.NOTE, "n1", "weighted error", "m1", "2026-08-22", 0.9)]
    answer = answer_question("what is the weighted error", hits, router=Stub())

    assert r"\frac" in answer.text
    assert r"\epsilon" in answer.text
