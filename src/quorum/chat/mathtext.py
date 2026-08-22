"""Getting mathematics through a JSON string intact, and onto the page.

Asked for the AdaBoost update rule, the answer came back like this:

    (displaystyle epsilon_t = ♠rac{sum w_i}{sum w_j})
    w_i^{(t+1)} = w_i^{(t)}     imes exp!␈igl(-alpha_t ...␈igr)

Three separate faults stacked on top of each other, and only the last one is
about rendering.

**1. JSON ate the commands.** Structured output arrives as a JSON string, and a
model writing LaTeX into one rarely escapes its backslashes. `\\f`, `\\t`, `\\b`,
`\\n` and `\\r` are *valid* JSON escapes, so the parser consumes them as control
characters and keeps the rest of the word:

    \\frac   ->  formfeed  + "rac"
    \\times  ->  tab       + "imes"
    \\bigl   ->  backspace + "igl"
    \\neq    ->  newline   + "eq"

A formfeed in the middle of a word is not prose. It is a LaTeX command with its
backslash eaten, and it can be put back.

**2. Lenient parsing dropped the rest.** Escapes JSON does not recognise -
`\\e`, `\\d`, `\\s` - lose their backslash silently, which is how
`\\displaystyle \\epsilon` became `displaystyle epsilon`. Nothing marks the
damage, so this is only repairable *inside* a maths region, where a bare word
matching a known command is unambiguous.

**3. The delimiters were wrong for the renderer.** The model wrote `\\( ... \\)`,
which KaTeX-in-markdown does not recognise. Streamlit renders `$...$` and
`$$...$$`.

The prompt now asks for `$` delimiters and escaped backslashes. This module
exists because a prompt is a request: the same content has to survive a model
that ignores it.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

COMMANDS = frozenset("""
alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa
lambda mu nu xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi
psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
frac dfrac tfrac binom sqrt sum prod int oint lim limits nolimits
cdot cdots ldots vdots ddots dots times div pm mp ast star circ bullet
leq geq neq approx equiv sim simeq cong propto ll gg subset subseteq supset
supseteq in notin ni exists forall neg land lor implies iff
left right big Big bigg Bigg bigl bigr Bigl Bigr biggl biggr langle rangle
lfloor rfloor lceil rceil vert Vert mid nmid
log ln exp sin cos tan sec csc cot arcsin arccos arctan sinh cosh tanh
min max arg argmin argmax sup inf det dim ker deg gcd Pr
mathbf mathbb mathcal mathrm mathit mathsf mathtt boldsymbol bm
text textbf textit textrm operatorname
hat bar tilde vec dot ddot overline underline overbrace underbrace
to rightarrow leftarrow Rightarrow Leftarrow leftrightarrow mapsto
displaystyle textstyle scriptstyle nonumber quad qquad hspace vspace
begin end cases matrix pmatrix bmatrix align aligned array
partial nabla infty emptyset varnothing angle triangle square
because therefore ldotp cdotp colon top bot
""".split())
"""LaTeX commands common enough in maths that a bare occurrence inside a maths
region is a command rather than an English word. Kept explicit: guessing that
any bare word is a command would turn the word "in" or "text" in ordinary prose
into a broken macro."""

JSON_CONTROLS = {
    "\x08": "b",   # \b
    "\x0c": "f",   # \f
    "\n": "n",     # \n
    "\r": "r",     # \r
    "\t": "t",     # \t
}

ALWAYS_DAMAGE = ("\x08", "\x0c")
"""A backspace or formfeed inside text is never intentional. Newline and tab
are, so those are only repaired when what follows is a known command - a
newline before the word "not" must stay a newline."""


def repair_math(text: str) -> str:
    """Undo the damage, then put the maths in delimiters the page can render."""
    if not text:
        return text
    repaired = _restore_eaten_backslashes(text)
    repaired = _normalise_delimiters(repaired)
    return _restore_commands_inside_math(repaired)


def _restore_eaten_backslashes(text: str) -> str:
    """Turn control characters back into the commands they used to be."""
    for control, letter in JSON_CONTROLS.items():
        if control not in text:
            continue

        def replace(match: re.Match, letter=letter, control=control) -> str:
            word = match.group(1)
            command = f"{letter}{word}"
            if control in ALWAYS_DAMAGE:
                return f"\\{command}"
            # A newline or tab is only damage when a real command follows it.
            for length in range(len(word), 0, -1):
                if f"{letter}{word[:length]}" in COMMANDS:
                    return f"\\{letter}{word[:length]}{word[length:]}"
            return match.group(0)

        text = re.sub(re.escape(control) + r"([A-Za-z]+)", replace, text)
    return text


def _normalise_delimiters(text: str) -> str:
    r"""`\( \)` and `\[ \]` to `$` and `$$`, which is what KaTeX-in-markdown reads.

    The unescaped forms are handled too - `\(` frequently arrives as a bare `(`
    with the backslash already gone, and the giveaway is what follows it: a
    parenthesis opening on `displaystyle` is a maths delimiter, not punctuation.
    """
    text = re.sub(r"\\\[\s*(.+?)\s*\\\]", r"$$\1$$", text, flags=re.DOTALL)
    text = re.sub(r"\\\(\s*(.+?)\s*\\\)", r"$\1$", text, flags=re.DOTALL)

    # Backslash already stripped: "(displaystyle ... )" on its own line.
    text = re.sub(
        r"(?m)^\(\s*(displaystyle\b.*?)\s*\)\s*$",
        lambda m: f"$${m.group(1)}$$",
        text,
        flags=re.DOTALL,
    )
    return text


MATH_REGION = re.compile(r"(\$\$.+?\$\$|\$[^$\n]+?\$)", re.DOTALL)


def _restore_commands_inside_math(text: str) -> str:
    """Put back backslashes that a lenient JSON parser dropped entirely.

    Only inside a maths region. `\\sum` losing its backslash leaves the word
    "sum", which is indistinguishable from the English word anywhere else - but
    between dollar signs it can only be the operator.
    """
    if "$" not in text:
        return text

    def fix(match: re.Match) -> str:
        return re.sub(
            r"(?<![\\A-Za-z])([A-Za-z]+)",
            lambda word: (
                f"\\{word.group(1)}" if word.group(1) in COMMANDS else word.group(1)
            ),
            match.group(0),
        )

    return MATH_REGION.sub(fix, text)


def looks_damaged(text: str) -> bool:
    """Whether an answer shows the marks of this corruption.

    Used to decide whether to say so rather than to silently repair - a reader
    should know when what they are looking at has been reconstructed.
    """
    if any(control in text for control in ALWAYS_DAMAGE):
        return True
    return bool(re.search(r"(?<![\\A-Za-z])(displaystyle|frac|epsilon)\b", text))
