"""Text reaching the terminal unaltered.

Rich reads square brackets as style tags and silently deletes anything it does
not recognise. On a data-structures assistant that is catastrophic and
invisible: a correct answer containing `last[ch] = i` was displayed as
`last = i`, and `dp[i][j] = dp[i-1][j]` as `dp = dp`. No error was raised
anywhere, and the saved files were fine - only the display was wrong, which is
the hardest kind of bug to notice and the easiest to ship.

These tests capture what a real terminal would receive.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from quorum.cli import safe, show

CODE = "last[ch] = i"
NESTED = "dp[i][j] = dp[i-1][j] + dp[i][j-1]"


def rendered(callback) -> str:
    """Whatever a real console would have printed."""
    console = Console(record=True, width=200, force_terminal=False)
    callback(console)
    return console.export_text()


def test_rich_really_does_eat_subscripts():
    """The premise. If this ever stops being true the guards below are dead
    weight and should go, so it is worth asserting rather than assuming."""
    plain = rendered(lambda c: c.print(NESTED))

    assert "dp[i]" not in plain
    assert "dp = dp" in plain


def test_show_preserves_code_exactly(capsys):
    show(CODE)
    assert CODE in capsys.readouterr().out


def test_show_preserves_nested_subscripts(capsys):
    show(NESTED)
    assert NESTED in capsys.readouterr().out


def test_show_preserves_a_whole_answer(capsys):
    answer = (
        "From your material.\n\n"
        "def count(s):\n"
        "    last = {'A': -1, 'B': -1, 'C': -1}\n"
        "    for i, ch in enumerate(s):\n"
        "        last[ch] = i\n"
        "        total += i - min(last.values())\n"
    )
    show(answer)
    out = capsys.readouterr().out

    assert "last[ch] = i" in out
    assert "{'A': -1, 'B': -1, 'C': -1}" in out


def test_escaped_text_survives_inside_a_styled_string():
    out = rendered(lambda c: c.print(f"[dim]{safe(NESTED)}[/dim]"))
    assert NESTED in out


def test_escaped_text_survives_in_a_table_cell():
    """Table cells are rendered with markup too, so a commitment described as
    "fix dp[i] handling" loses the subscript in `quorum status`."""
    def build(console):
        table = Table()
        table.add_column("Commitment")
        table.add_row(safe("fix dp[i] handling"))
        console.print(table)

    assert "dp[i]" in rendered(build)


def test_an_unescaped_table_cell_is_what_went_wrong():
    def build(console):
        table = Table()
        table.add_column("Commitment")
        table.add_row("fix dp[i] handling")
        console.print(table)

    assert "dp[i]" not in rendered(build)


def test_markup_in_transcript_text_is_not_interpreted(capsys):
    """A speaker saying "bracket i bracket" is data, not formatting - and a
    transcript is untrusted input everywhere else in this project too."""
    show("Speaker (00:12): so we write arr[i] and then [bold]bold[/bold]")
    out = capsys.readouterr().out

    assert "arr[i]" in out
    assert "[bold]bold[/bold]" in out, "markup in speech must not be honoured"


def test_show_handles_empty_and_none_like_input(capsys):
    show("")
    assert capsys.readouterr().out.strip() == ""
    assert safe("") == ""
