"""Commitments that are themselves messages.

The detection question is the interesting one: most commitments are work, a few
are communication, and drafting an email for "I'll finish the migration" is
worse than drafting none - it puts something in a real person's Gmail that they
did not ask for and may not notice before sending.

So the negatives matter as much as the positives here, and the address rule
matters most of all: a guessed address is how a confidential document reaches a
stranger.
"""

from __future__ import annotations

from datetime import date

import pytest

from quorum.execution.mail import (
    Draft,
    DraftWriter,
    GmailDrafts,
    _split,
    find_communications,
    is_communication,
)
from quorum.models import (
    Assignee,
    Commitment,
    CommitmentStrength,
    Deadline,
    Evidence,
)
from quorum.workspace import Workspace


def commitment(
    description: str,
    quote: str = "",
    *,
    strength: CommitmentStrength = CommitmentStrength.FIRM,
    cid: str = "cmt_1",
) -> Commitment:
    return Commitment(
        id=cid,
        description=description,
        meeting_id="mtg_1",
        assignee=Assignee(speaker_id="spk_y", display_name="Yug Verma",
                          email="yug@example.com", confidence=0.9),
        deadline=Deadline(resolved=date(2026, 9, 4)),
        strength=strength,
        evidence=[Evidence(utterance_id="u1", quote=quote or description)],
    )


@pytest.fixture
def project(tmp_path):
    workspace = Workspace(tmp_path / "workspace")
    return workspace.create(
        "Ingestion",
        members={"Priya Raghavan": "priya@example.com", "Sam Okafor": "sam@example.com"},
    )


# --- what counts as a message -----------------------------------------------


@pytest.mark.parametrize("text", [
    "email Priya the ingestion spec",
    "send the spec over to Sam",
    "I'll write to the vendor about pricing",
    "forward the thread to Priya",
    "follow up with Sam on the schema",
    "circulate the notes",
    "reply to the security team",
])
def test_sending_commitments_are_recognised(text):
    assert is_communication(commitment(text))


@pytest.mark.parametrize("text", [
    "finish the ingestion migration",
    "review the pull request",
    "set up the staging cluster",
    "write the parser for the new format",
    "fix the flaky test",
])
def test_work_commitments_are_not_messages(text):
    """A draft for "finish the migration" is worse than no draft at all - it
    lands in a real person's Gmail unasked."""
    assert not is_communication(commitment(text))


@pytest.mark.parametrize("text", [
    "send it to production on Friday",
    "share memory between the workers",
    "put it on the message queue",
])
def test_sending_verbs_that_are_not_about_people(text):
    assert not is_communication(commitment(text))


def test_the_verb_can_be_in_the_quote_rather_than_the_description():
    """"I'll get that over to Priya" describes the work vaguely; only the words
    actually spoken carry the verb."""
    item = commitment("the ingestion spec", quote="I'll email that over to Priya tonight")

    assert is_communication(item)


def test_tentative_promises_do_not_become_drafts():
    """"I could maybe drop them a line" should not produce a draft sitting in
    Gmail waiting to be sent."""
    hedged = commitment("maybe email the vendor", strength=CommitmentStrength.TENTATIVE)
    musing = commitment("someone should email them", strength=CommitmentStrength.MUSING)

    assert find_communications([hedged, musing]) == []


def test_only_the_communications_are_selected():
    items = [
        commitment("email Priya the spec", cid="c1"),
        commitment("finish the migration", cid="c2"),
        commitment("send Sam the schema", cid="c3"),
    ]

    assert [c.id for c in find_communications(items)] == ["c1", "c3"]


# --- addressing --------------------------------------------------------------


class StubRouter:
    def __init__(self, text="Subject: The spec\n\nHere it is, as promised.", guess="") -> None:
        self.text = text
        self.guess = guess

    def complete(self, prompt, **kwargs):
        from quorum.llm.router import LLMResponse

        return LLMResponse(text=self.text, model="stub", provider="stub")

    def structured(self, prompt, schema, **kwargs):
        from quorum.llm.router import LLMResponse
        from quorum.execution.mail import RecipientGuess

        return RecipientGuess(name=self.guess), LLMResponse(
            text="", model="stub", provider="stub"
        )


def test_a_named_recipient_resolves_from_the_roster(project):
    draft = DraftWriter(router=StubRouter()).write(
        commitment("email Priya the ingestion spec"), project
    )

    assert draft.to_email == "priya@example.com"
    assert draft.to_name == "Priya Raghavan"


def test_a_first_name_resolves(project):
    draft = DraftWriter(router=StubRouter()).write(
        commitment("send Sam the schema"), project
    )

    assert draft.to_email == "sam@example.com"


def test_a_name_only_the_model_can_find_still_goes_through_the_roster(project):
    """The model may identify who was meant; it may not invent the address."""
    writer = DraftWriter(router=StubRouter(guess="Priya Raghavan"))
    draft = writer.write(commitment("email them the spec"), project)

    assert draft.to_email == "priya@example.com"


def test_someone_off_the_roster_gets_a_draft_with_no_address(project):
    """Guessing an address from a name is how a confidential spec reaches a
    stranger. The draft is still written; the address is left for a human."""
    writer = DraftWriter(router=StubRouter(guess="Dave Nobody"))
    draft = writer.write(commitment("email them the spec"), project)

    assert draft.to_name == "Dave Nobody"
    assert draft.to_email == ""
    assert not draft.addressed


def test_a_failed_draft_returns_nothing_rather_than_raising(project):
    class Broken(StubRouter):
        def complete(self, *a, **kw):
            raise RuntimeError("quota exhausted")

    assert DraftWriter(router=Broken()).write(commitment("email Priya"), project) is None


# --- the message itself ------------------------------------------------------


def test_the_subject_and_body_are_separated():
    assert _split("Subject: Ingestion spec\n\nAttached now.", "x") == (
        "Ingestion spec", "Attached now."
    )


def test_a_bare_first_line_is_treated_as_the_subject():
    assert _split("Ingestion spec\n\nAttached now.", "x") == (
        "Ingestion spec", "Attached now."
    )


def test_unsplittable_output_keeps_the_whole_text_as_the_body():
    subject, body = _split("just one line with no break", "the ingestion spec")

    assert subject == "the ingestion spec"
    assert body == "just one line with no break"


def test_the_draft_carries_the_words_that_created_the_obligation(project):
    draft = DraftWriter(router=StubRouter()).write(
        commitment("email Priya the spec", quote="I'll email Priya the spec by Friday"),
        project,
    )

    assert "I'll email Priya the spec by Friday" in draft.quote


def test_the_mime_message_is_addressed_and_encoded():
    draft = Draft("c1", "Priya", "priya@example.com", "The spec", "Here it is.")
    raw = draft.raw(sender="yug@example.com")

    import base64

    decoded = base64.urlsafe_b64decode(raw).decode()
    assert "To: priya@example.com" in decoded
    assert "Subject: The spec" in decoded
    assert "From: yug@example.com" in decoded


# --- Gmail --------------------------------------------------------------------


class FakeDrafts:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.fail = False

    def create(self, userId, body):  # noqa: N803 - Google's parameter name
        if self.fail:
            raise RuntimeError("simulated Gmail 500")
        self.created.append(body)
        return _Executable({"id": f"draft_{len(self.created)}"})


class _Executable:
    def __init__(self, result) -> None:
        self._result = result

    def execute(self):
        return self._result


class FakeUsers:
    def __init__(self, drafts) -> None:
        self._drafts = drafts

    def drafts(self):
        return _DraftsProxy(self._drafts)


class _DraftsProxy:
    def __init__(self, drafts) -> None:
        self._drafts = drafts

    def create(self, userId, body):  # noqa: N803
        return self._drafts.create(userId, body)


class FakeGmail:
    def __init__(self) -> None:
        self.drafts_api = FakeDrafts()

    def users(self):
        return FakeUsers(self.drafts_api)


def test_drafts_are_created_in_gmail():
    service = FakeGmail()
    result = GmailDrafts(service).create([
        Draft("c1", "Priya", "priya@example.com", "The spec", "Here it is.")
    ])

    assert result.created == 1
    assert len(service.drafts_api.created) == 1


def test_an_unaddressed_draft_is_skipped_not_created():
    """Created with an empty To:, it would fail on send days later with no
    explanation."""
    service = FakeGmail()
    result = GmailDrafts(service).create([Draft("c1", "Dave", "", "The spec", "Here.")])

    assert (result.created, result.skipped) == (0, 1)
    assert service.drafts_api.created == []


def test_a_failed_draft_does_not_abort_the_rest():
    service = FakeGmail()
    service.drafts_api.fail = True
    result = GmailDrafts(service).create([
        Draft("c1", "Priya", "priya@example.com", "s", "b"),
        Draft("c2", "Sam", "sam@example.com", "s", "b"),
    ])

    assert result.created == 0 and len(result.failed) == 2


def test_without_a_service_it_is_a_dry_run():
    with pytest.raises(RuntimeError, match="dry run"):
        GmailDrafts(None).create([Draft("c1", "P", "p@example.com", "s", "b")])


def test_creating_drafts_requires_an_approval_token():
    """Same invariant as email digests and calendar writes."""
    from quorum.execution import ApprovalGate
    from quorum.execution.approval import NotApproved
    from quorum.execution.mail import GmailDraftTransport
    from quorum.tracking import ActionType, PlannedAction

    service = FakeGmail()
    transport = GmailDraftTransport(
        GmailDrafts(service), [Draft("c1", "P", "p@example.com", "s", "b")]
    )
    gate = ApprovalGate()
    pending = gate.propose(PlannedAction("drafts:p", ActionType.SCHEDULE, "1 draft"), "d")

    with pytest.raises(NotApproved):
        gate.execute(pending.id, "guessed", transport)
    assert service.drafts_api.created == []

    gate.execute(pending.id, gate.approve(pending.id), transport)
    assert transport.result.created == 1


def test_nothing_in_the_project_can_send_mail():
    """The scope permits sending; the code must never use it. A reviewer should
    not have to take that on trust, and neither should this test - it parses the
    source rather than grepping it, so a docstring *describing* the rule does not
    count as breaking it. (Which the first version of this test did.)
    """
    import ast
    import pathlib

    offenders = []
    for path in pathlib.Path("src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "send":
                base = func.value
                name = getattr(base, "attr", None) or getattr(
                    getattr(base, "func", None), "attr", None
                )
                if name == "messages":
                    offenders.append(f"{path.name}:{node.lineno}")

    assert offenders == []
