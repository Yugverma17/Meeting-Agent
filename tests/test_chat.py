"""Naming, tools, coverage-labelled answering, and the agent loop.

Nothing here calls a model. The router is replaced by a stub that returns
scripted structured output, which is what makes it possible to test the parts
that actually matter: that an ambiguous reference asks instead of guessing, that
a write cannot happen without a human, and that an answer is never labelled as
coming from the user's material when it did not.
"""

from __future__ import annotations

from datetime import date

import pytest

from quorum.chat.agent import ChatAgent, Conversation
from quorum.chat.answer import Coverage, GroundedAnswer, answer_question
from quorum.chat.naming import (
    auto_handle,
    extract_mention,
    list_meetings,
    register_meeting,
    resolve_meeting,
    set_handle,
)
from quorum.chat.tools import ToolContext, ToolRequest, run_tool
from quorum.memory.store import MemoryHit, MemoryKind
from quorum.models import (
    Assignee,
    Commitment,
    CommitmentStatus,
    Deadline,
    Evidence,
    Speaker,
    Transcript,
    Utterance,
)
from quorum.workspace import Workspace

TODAY = date(2026, 8, 15)


# --- fixtures ---------------------------------------------------------------


def lecture(title: str, meeting_id: str, when: date = date(2026, 8, 10)) -> Transcript:
    speaker = Speaker(id="spk_lecturer", display_name="Lecturer")
    lines = [
        "Postfix evaluation walks the expression left to right using a stack.",
        "Every operand is pushed and every operator pops two values.",
        "That is why the whole thing is linear in the number of tokens.",
    ]
    return Transcript(
        meeting_id=meeting_id, title=title, meeting_date=when, speakers=[speaker],
        utterances=[
            Utterance(id=f"{meeting_id}_u{i}", index=i, speaker_id=speaker.id,
                      text=text, start_s=i * 30.0, end_s=i * 30.0 + 25.0)
            for i, text in enumerate(lines)
        ],
        source="fixture",
    )


@pytest.fixture
def project(tmp_path):
    workspace = Workspace(tmp_path / "workspace")
    found = workspace.create("DSA", members={"Priya Raghavan": "priya@example.com"})
    found.transcripts_dir.mkdir(parents=True, exist_ok=True)
    for transcript in (
        lecture("Postfix evaluation", "mtg_postfix"),
        lecture("Weekly sync", "mtg_sync1", date(2026, 8, 3)),
        lecture("Weekly sync", "mtg_sync2", date(2026, 8, 11)),
    ):
        (found.transcripts_dir / f"{transcript.meeting_id}.json").write_text(
            transcript.model_dump_json(), encoding="utf-8"
        )
    workspace.save(found)
    return workspace, found


@pytest.fixture
def context(project):
    workspace, found = project
    return ToolContext(project=found, workspace=workspace, today=TODAY,
                       memory=StubMemory())


class StubMemory:
    """Retrieval without an embedding model."""

    def __init__(self, hits: list[MemoryHit] | None = None) -> None:
        self.hits = hits if hits is not None else [
            MemoryHit(MemoryKind.NOTE, "n1", "Postfix evaluation uses a stack",
                      "mtg_postfix", "2026-08-10", 0.81),
            MemoryHit(MemoryKind.NOTE, "n2", "Each token is handled once, so it is O(n)",
                      "mtg_postfix", "2026-08-10", 0.74),
            MemoryHit(MemoryKind.NOTE, "n3", "The sync agreed to use Postgres",
                      "mtg_sync1", "2026-08-03", 0.52),
        ]
        self.calls: list[dict] = []

    def recall(self, query, k=5, kind=None, min_score=0.0, meeting_ids=None):
        self.calls.append({"query": query, "meeting_ids": meeting_ids})
        hits = self.hits
        if meeting_ids:
            hits = [h for h in hits if h.meeting_id in set(meeting_ids)]
        return [h for h in hits if h.score >= min_score][:k]


def commitment(description="send the ingestion spec", due=None, cid="cmt_1"):
    return Commitment(
        id=cid, description=description, meeting_id="mtg_sync1",
        assignee=Assignee(speaker_id="spk_p", display_name="Priya Raghavan",
                          email="priya@example.com", confidence=0.9),
        deadline=Deadline(resolved=due),
        evidence=[Evidence(utterance_id="u1", quote="I'll send the ingestion spec")],
    )


# --- naming -----------------------------------------------------------------


def test_a_handle_is_derived_from_the_title_so_nothing_needs_naming_first():
    assert auto_handle("Postfix evaluation", set()) == "postfix-evaluation"
    assert auto_handle("The Weekly Sync Meeting", set()) == "weekly-sync"


def test_a_clashing_handle_gets_a_suffix_rather_than_overwriting():
    assert auto_handle("Weekly sync", {"weekly-sync"}) == "weekly-sync-2"
    assert auto_handle("Weekly sync", {"weekly-sync", "weekly-sync-2"}) == "weekly-sync-3"


def test_registering_is_idempotent(project):
    _, found = project
    first = register_meeting(found, lecture("Postfix evaluation", "mtg_postfix"))
    second = register_meeting(found, lecture("Postfix evaluation", "mtg_postfix"))

    assert first == second
    assert list(found.meta.handles.values()).count("mtg_postfix") == 1


def test_a_handle_resolves_exactly(project):
    _, found = project
    set_handle(found, "mtg_postfix", "stacks")

    resolution = resolve_meeting(found, "@stacks")
    assert resolution.ok and resolution.match.meeting_id == "mtg_postfix"
    assert resolution.how == "handle"


def test_a_handle_points_at_one_meeting_only(project):
    """Reassigning moves the name. Two meetings answering to @standup would
    make it mean whichever the lookup happened to see first."""
    _, found = project
    set_handle(found, "mtg_sync1", "standup")
    set_handle(found, "mtg_sync2", "standup")

    assert found.meta.handles["standup"] == "mtg_sync2"
    assert list(found.meta.handles.values()).count("mtg_sync1") == 0


def test_renaming_a_meeting_replaces_its_old_handle(project):
    _, found = project
    set_handle(found, "mtg_postfix", "stacks")
    set_handle(found, "mtg_postfix", "rpn")

    assert "stacks" not in found.meta.handles
    assert found.meta.handles["rpn"] == "mtg_postfix"


def test_title_words_resolve_when_unambiguous(project):
    _, found = project
    resolution = resolve_meeting(found, "postfix")

    assert resolution.ok and resolution.how == "title"


def test_two_meetings_with_the_same_title_are_reported_not_guessed(project):
    """Answering about the wrong week is silent. Asking costs one line."""
    _, found = project
    resolution = resolve_meeting(found, "weekly sync")

    assert not resolution.ok
    assert resolution.ambiguous
    assert {ref.meeting_id for ref in resolution.candidates} == {"mtg_sync1", "mtg_sync2"}


def test_a_handle_disambiguates_what_a_title_cannot(project):
    _, found = project
    set_handle(found, "mtg_sync2", "last-week")

    assert resolve_meeting(found, "@last-week").match.meeting_id == "mtg_sync2"


def test_nonsense_resolves_to_nothing(project):
    _, found = project
    assert resolve_meeting(found, "quantum chromodynamics").how == "none"


def test_only_an_explicit_mention_is_extracted():
    assert extract_mention("@kickoff what did we decide") == "kickoff"
    assert extract_mention("what did the kickoff meeting decide") == ""


def test_lectures_and_meetings_are_distinguished(project):
    _, found = project
    kinds = {ref.meeting_id: ref.kind for ref in list_meetings(found)}

    assert kinds["mtg_postfix"] == "lecture"


# --- scoping ----------------------------------------------------------------


def test_search_scoped_to_a_meeting_does_not_answer_from_another(context):
    result = run_tool(context, ToolRequest(tool="search", query="stack", meeting="postfix"))

    assert result.ok
    assert {hit.meeting_id for hit in result.hits} == {"mtg_postfix"}


def test_the_focused_meeting_applies_without_re_naming_it(context):
    context.scope_meeting = "postfix"
    result = run_tool(context, ToolRequest(tool="search", query="stack"))

    assert context.memory.calls[-1]["meeting_ids"] == ["mtg_postfix"]


def test_an_ambiguous_meeting_reference_asks_instead_of_widening(context):
    result = run_tool(context, ToolRequest(tool="search", query="storage",
                                           meeting="weekly sync"))

    assert not result.ok
    assert "matches several" in result.text


def test_an_unknown_meeting_is_an_error_not_a_project_wide_search(context):
    """Silently widening would answer about a different lecture than the one
    that was named, which is worse than saying it cannot be found."""
    result = run_tool(context, ToolRequest(tool="search", query="x", meeting="nonexistent"))

    assert not result.ok and "No meeting" in result.text


# --- reads ------------------------------------------------------------------


def test_reading_a_transcript_returns_the_actual_words(context):
    result = run_tool(context, ToolRequest(tool="read_transcript", meeting="postfix"))

    assert "pushed and every operator pops" in result.text


def test_transcript_filters_narrow_what_comes_back(context):
    result = run_tool(context, ToolRequest(tool="read_transcript", meeting="postfix",
                                           query="linear"))

    assert "linear in the number of tokens" in result.text
    assert "Every operand is pushed" not in result.text


def test_listing_commitments_marks_what_is_late(context):
    context.project.ledger.commitments.append(commitment(due=date(2026, 8, 1)))
    result = run_tool(context, ToolRequest(tool="list_commitments"))

    assert "14d late" in result.text


def test_an_unknown_tool_is_reported_back_rather_than_raised(context):
    """The model occasionally invents a plausible name; telling it what exists
    lets the next turn correct itself."""
    result = run_tool(context, ToolRequest(tool="summarise_everything"))

    assert not result.ok and "search" in result.text


# --- writes are two-phase ---------------------------------------------------


def test_closing_a_commitment_does_nothing_on_the_first_call(context):
    context.project.ledger.commitments.append(commitment())
    result = run_tool(context, ToolRequest(tool="close_commitment", what="ingestion spec"))

    assert result.needs_confirmation
    assert context.project.ledger.commitments[0].status is CommitmentStatus.OPEN


def test_closing_happens_only_when_confirmed(context):
    context.project.ledger.commitments.append(commitment())
    request = ToolRequest(tool="close_commitment", what="ingestion spec")

    run_tool(context, request)  # planning call
    result = run_tool(context, request, confirmed=True)

    assert result.ok
    assert context.project.ledger.commitments[0].status is CommitmentStatus.VERIFIED_DONE


def test_a_vague_description_never_closes_anything(context):
    """`done "the"` once closed a real commitment. Closing the wrong one is
    silent - nothing later reopens it."""
    context.project.ledger.commitments.append(commitment())
    result = run_tool(context, ToolRequest(tool="close_commitment", what="the"))

    assert not result.ok and "too vague" in result.text
    assert not result.needs_confirmation


def test_two_similar_commitments_are_ambiguous_rather_than_picked(context):
    context.project.ledger.commitments.extend([
        commitment("review the ingestion spec", cid="c1"),
        commitment("review the ingestion plan", cid="c2"),
    ])
    result = run_tool(context, ToolRequest(tool="close_commitment",
                                           what="review the ingestion"))

    assert not result.ok and "ambiguous" in result.text


def test_setting_a_deadline_resolves_spoken_dates(context):
    context.project.ledger.commitments.append(commitment())
    request = ToolRequest(tool="set_deadline", what="ingestion spec", when="next Friday")

    preview = run_tool(context, request)
    assert preview.needs_confirmation
    assert context.project.ledger.commitments[0].deadline.resolved is None

    run_tool(context, request, confirmed=True)
    # "next Friday" from a Saturday is the Friday of the following week, which
    # is the project's existing convention in `resolve_deadline`.
    assert context.project.ledger.commitments[0].deadline.resolved == date(2026, 8, 28)


def test_an_unparseable_date_is_refused_rather_than_guessed(context):
    """A wrong date silently produces a calendar reminder on the wrong day."""
    context.project.ledger.commitments.append(commitment())
    result = run_tool(context, ToolRequest(tool="set_deadline", what="ingestion spec",
                                           when="sometime after the thing"))

    assert not result.ok
    assert context.project.ledger.commitments[0].deadline.resolved is None


def test_drafting_an_email_needs_someone_to_send_it_to(context):
    result = run_tool(context, ToolRequest(tool="draft_email", what="the spec"))

    assert not result.ok and "Who is it to" in result.text


def test_drafting_to_an_unknown_person_lists_who_is_known(context):
    result = run_tool(context, ToolRequest(tool="draft_email", who="Dave", what="the spec"))

    assert not result.ok and "Priya Raghavan" in result.text


# --- coverage-labelled answers ----------------------------------------------


class StubRouter:
    """Returns scripted structured output without a model.

    Queued by *schema* rather than in one sequence: the agent interleaves
    routing decisions and answer drafts, and a single queue makes a test's
    script depend on how many times the loop happened to go round.
    """

    def __init__(self, structured_results=None, text="general knowledge answer") -> None:
        self.queues: dict[str, list] = {}
        for item in structured_results or []:
            self.queues.setdefault(type(item).__name__, []).append(item)
        self.text = text
        self.structured_calls: list = []
        self.complete_calls: list = []

    def structured(self, prompt, schema, **kwargs):
        self.structured_calls.append((prompt, schema, kwargs))
        from quorum.llm.router import LLMResponse

        queue = self.queues.get(schema.__name__, [])
        if not queue:
            raise AssertionError(f"StubRouter has no {schema.__name__} left to return")
        # The last scripted item repeats, so a test that cares about the loop
        # bound does not have to predict the iteration count exactly.
        result = queue.pop(0) if len(queue) > 1 else queue[0]
        return result, LLMResponse(text="", model="stub", provider="stub", total_tokens=10)

    def complete(self, prompt, **kwargs):
        self.complete_calls.append((prompt, kwargs))
        from quorum.llm.router import LLMResponse

        return LLMResponse(text=self.text, model="stub", provider="stub", total_tokens=10)


def draft(coverage, answer="because each token is handled once", cited=(1,), added=""):
    from quorum.chat.answer import AnswerDraft

    return AnswerDraft(coverage=coverage, answer=answer, cited=list(cited), added=added)


def test_a_covered_question_is_answered_from_the_material():
    hits = StubMemory().recall("why O(n)")
    router = StubRouter([draft(Coverage.COVERED, cited=[1, 2])])

    answer = answer_question("why is it O(n)", hits, router=router, scope="@postfix")

    assert answer.coverage is Coverage.COVERED
    assert "From your material" in answer.banner()
    assert len(answer.sources) == 2


def test_nothing_relevant_means_background_without_asking_the_model():
    """Deterministic. No material means no grounded answer is possible, and the
    model never gets to rule on that."""
    weak = [MemoryHit(MemoryKind.NOTE, "n9", "unrelated", "mtg_x", "2026-08-01", 0.05)]
    router = StubRouter()

    answer = answer_question("how does a red-black tree rebalance", weak, router=router)

    assert answer.coverage is Coverage.BACKGROUND
    assert answer.text == "general knowledge answer"
    assert router.structured_calls == [], "coverage was decided without a model call"
    assert "Not covered" in answer.banner()


def test_a_partial_answer_names_what_was_added():
    hits = StubMemory().recall("stack")
    router = StubRouter([
        draft(Coverage.PARTIAL, added="the comparison with recursive evaluation")
    ])

    answer = answer_question("how does it compare to recursion", hits, router=router)

    assert answer.coverage is Coverage.PARTIAL
    assert "the rest is background" in answer.banner()
    assert "recursive evaluation" in answer.added


def test_citations_outside_the_passage_range_are_dropped():
    """A model citing [7] against three passages would otherwise index past the
    end of the list when the sources are rendered."""
    hits = StubMemory().recall("stack")
    router = StubRouter([draft(Coverage.COVERED, cited=[1, 7, 0, -2])])

    answer = answer_question("why", hits, router=router)

    assert answer.cited == [1]
    assert len(answer.sources) == 1


def test_a_failed_answer_does_not_end_the_conversation():
    class Broken(StubRouter):
        def structured(self, *a, **kw):
            raise RuntimeError("quota exhausted")

    answer = answer_question("why", StubMemory().recall("stack"), router=Broken())

    assert "could not answer" in answer.text
    assert answer.coverage is Coverage.BACKGROUND


def test_the_source_list_holds_only_passages_actually_used():
    """A list padded with passages the answer ignored teaches the reader to
    stop checking it."""
    answer = GroundedAnswer(
        text="x", coverage=Coverage.COVERED, hits=StubMemory().recall("q"), cited=[2]
    )

    assert [hit.ref_id for hit in answer.sources] == ["n2"]


# --- the agent loop ---------------------------------------------------------


def route_to(tool: str, **args):
    from quorum.chat.agent import RouteDecision

    return RouteDecision(action="tool", tool=ToolRequest(tool=tool, **args))


def route_answer():
    from quorum.chat.agent import RouteDecision

    return RouteDecision(action="answer")


def test_a_question_searches_then_answers(context):
    router = StubRouter([
        route_to("search", query="postfix complexity"),
        route_answer(),
        draft(Coverage.COVERED),
    ])
    agent = ChatAgent(context, router=router)

    turn = agent.ask("why is postfix evaluation O(n)", Conversation())

    assert turn.tools_used == ["search"]
    assert turn.answer.coverage is Coverage.COVERED


def test_the_loop_stops_at_a_write_and_hands_it_back(context):
    """Nothing is performed inside the loop. The turn ends holding a
    description of the write, and only a human can cause the second call."""
    context.project.ledger.commitments.append(commitment())
    router = StubRouter([route_to("close_commitment", what="ingestion spec")])
    agent = ChatAgent(context, router=router)

    turn = agent.ask("mark the ingestion spec as done", Conversation())

    assert turn.needs_confirmation
    assert context.project.ledger.commitments[0].status is CommitmentStatus.OPEN
    assert router.structured_calls, "no answering call was made after the write"
    assert len(router.structured_calls) == 1


def test_confirming_performs_the_write(context):
    context.project.ledger.commitments.append(commitment())
    router = StubRouter([route_to("close_commitment", what="ingestion spec")])
    agent = ChatAgent(context, router=router)

    turn = agent.ask("mark the ingestion spec as done", Conversation())
    result = agent.confirm(turn.pending)

    assert result.ok
    assert context.project.ledger.commitments[0].status is CommitmentStatus.VERIFIED_DONE


def test_the_loop_gives_up_searching_rather_than_running_forever(context):
    """A model that has decided to keep searching must not burn the whole
    per-minute token budget doing it."""
    from quorum.chat.agent import MAX_STEPS

    class NeverSatisfied(StubRouter):
        """Distinct queries every time, so the repeat guard does not stop it
        first and the step ceiling is what is actually under test."""

        attempts = 0

        def structured(self, prompt, schema, **kwargs):
            from quorum.chat.agent import RouteDecision

            if schema is RouteDecision:
                self.structured_calls.append((prompt, schema, kwargs))
                type(self).attempts += 1
                return (
                    route_to("search", query=f"attempt {type(self).attempts}"),
                    __import__("quorum.llm.router", fromlist=["LLMResponse"]).LLMResponse(
                        text="", model="stub", provider="stub"
                    ),
                )
            return super().structured(prompt, schema, **kwargs)

    router = NeverSatisfied([draft(Coverage.COVERED)])
    agent = ChatAgent(context, router=router)

    turn = agent.ask("something it keeps hunting for", Conversation())

    assert len(turn.tools_used) == MAX_STEPS
    assert turn.answer is not None


def test_ledger_data_reaches_the_answer(context):
    """Regression. Asked "what is still open and who owes what", the loop
    fetched the ledger correctly and then answered with invented accounting
    boilerplate - because only retrieval hits were ever passed to the answering
    step, and `list_commitments` returns records rather than passages."""
    context.project.ledger.commitments.append(commitment(due=date(2026, 9, 1)))
    router = StubRouter([route_to("list_commitments"), route_answer(),
                         draft(Coverage.COVERED, cited=[])])
    agent = ChatAgent(context, router=router)

    agent.ask("what is still open", Conversation())

    answering = [call for call in router.structured_calls
                 if call[1].__name__ == "AnswerDraft"]
    assert answering, "the turn never reached the answering step"
    assert "send the ingestion spec" in answering[0][0]


def test_the_material_is_always_consulted_before_answering(context):
    """Regression, and the worst failure this design can have.

    Asked "why is the brute force approach O(n^2)" about a lecture explaining
    exactly that, the router read it as a general computer-science question and
    answered from its own knowledge - labelled "not covered", and in substance
    about a different algorithm. Retrieval would have scored 0.87.
    """
    router = StubRouter([route_answer(), draft(Coverage.COVERED)])
    agent = ChatAgent(context, router=router)

    turn = agent.ask("why is the brute force approach O(n^2)", Conversation())

    assert turn.tools_used == ["search"], "the model skipped retrieval and was overruled"
    assert context.memory.calls[-1]["query"] == "why is the brute force approach O(n^2)"


def test_the_backstop_search_runs_only_once(context):
    """It exists to stop a turn answering blind, not to search after every
    tool call that happens to return no passages."""
    router = StubRouter([route_answer(), draft(Coverage.COVERED)])
    context.memory = StubMemory(hits=[])
    agent = ChatAgent(context, router=router)

    turn = agent.ask("anything", Conversation())

    assert turn.tools_used == ["search"]


def test_an_action_request_is_not_diverted_into_a_search(context):
    context.project.ledger.commitments.append(commitment())
    router = StubRouter([route_to("close_commitment", what="ingestion spec")])
    agent = ChatAgent(context, router=router)

    turn = agent.ask("mark the ingestion spec done", Conversation())

    assert turn.tools_used == ["close_commitment"]


def test_a_question_with_no_material_at_all_still_goes_to_background(context):
    """The deterministic floor must not be defeated by an empty facts block."""
    context.memory = StubMemory(hits=[])
    router = StubRouter([route_to("search", query="x"), route_answer()])
    agent = ChatAgent(context, router=router)

    turn = agent.ask("something unrelated", Conversation())

    assert turn.answer.coverage is Coverage.BACKGROUND


def test_an_at_mention_focuses_the_conversation(context):
    """Focus is carried as the meeting id, not its display label. Carrying
    "Weekly sync (2026-08-03)" forward would re-resolve it by fuzzy title next
    turn and throw away the disambiguation the user just made."""
    router = StubRouter([route_answer(), draft(Coverage.COVERED)])
    agent = ChatAgent(context, router=router)
    conversation = Conversation()

    agent.ask("@postfix what was the main point", conversation)

    assert conversation.scope_meeting == "mtg_postfix"


def test_focus_persists_into_the_next_question(context):
    conversation = Conversation()
    agent = ChatAgent(context, router=StubRouter([route_answer(), draft(Coverage.COVERED)]))
    agent.ask("@postfix what was the main point", conversation)

    agent._router = StubRouter([route_to("search", query="complexity"), route_answer(),
                                draft(Coverage.COVERED)])
    agent.ask("and why is it linear", conversation)

    assert context.memory.calls[-1]["meeting_ids"] == ["mtg_postfix"]


def test_a_routing_failure_still_produces_an_answer(context):
    class BrokenRouting(StubRouter):
        def structured(self, prompt, schema, **kwargs):
            from quorum.chat.agent import RouteDecision

            if schema is RouteDecision:
                raise RuntimeError("quota exhausted")
            return super().structured(prompt, schema, **kwargs)

    router = BrokenRouting([draft(Coverage.BACKGROUND, added="everything")])
    turn = ChatAgent(context, router=router).ask("anything", Conversation())

    assert turn.answer is not None


def test_history_is_trimmed_to_one_line_per_turn(context):
    """Replaying the full conversation every turn stops working after ten
    turns at 6,000 tokens/minute."""
    conversation = Conversation()
    long_answer = GroundedAnswer(text="x" * 5000, coverage=Coverage.COVERED)
    from quorum.chat.agent import ChatTurn

    conversation.add(ChatTurn(question="q", answer=long_answer))

    assert len(conversation.recent()) < 400


# --- the roster-vs-reality trap ---------------------------------------------


def test_a_solo_lecture_is_not_listed_as_a_meeting(project):
    """Live capture always appends a "Remote participant" so the loopback
    channel has somewhere to attribute to, whether or not anyone else was
    there. Counting the declared roster labelled every lecture a meeting."""
    from quorum.capture.speakers import MIC, SpeakerRoster, TranscriptSegment, build_transcript

    _, found = project
    built = build_transcript(
        [TranscriptSegment(channel=MIC, text="today we cover stacks", start_s=0.0, end_s=9.0)],
        SpeakerRoster.solo("You"), title="Stacks",
    )
    assert len(built.speakers) == 2, "the placeholder participant is still declared"
    assert built.is_monologue
    assert len(built.speakers_present) == 1

    found.transcripts_dir.mkdir(parents=True, exist_ok=True)
    (found.transcripts_dir / f"{built.meeting_id}.json").write_text(
        built.model_dump_json(), encoding="utf-8"
    )
    kinds = {ref.meeting_id: ref.kind for ref in list_meetings(found)}

    assert kinds[built.meeting_id] == "lecture"


def test_a_lecture_transcript_is_rendered_without_speaker_labels(context):
    """Keyed on the roster, a lecture came back as "Remote participant (00:15):"
    on every line."""
    result = run_tool(context, ToolRequest(tool="read_transcript", meeting="postfix"))

    assert "Lecturer" not in result.text
    assert result.text.lstrip().startswith("[00:00]")


# --- nothing overwrites anything --------------------------------------------


def test_two_drafts_to_the_same_person_on_one_day_both_survive(context, tmp_path, monkeypatch):
    """Drafting a second mail to someone on the same day silently replaced the
    first - something a person does routinely and would never check for."""
    monkeypatch.setattr("quorum.config.RUNS_DIR", tmp_path)

    class Fixed:
        text = "Subject: the spec\n\nHere it is."

        def complete(self, *a, **kw):
            from quorum.llm.router import LLMResponse

            return LLMResponse(text=self.text, model="stub", provider="stub")

    monkeypatch.setattr("quorum.llm.router.get_router", lambda **kw: Fixed())
    request = ToolRequest(tool="draft_email", who="Priya Raghavan", what="the spec")

    first = run_tool(context, request, confirmed=True)
    second = run_tool(context, request, confirmed=True)

    assert first.ok and second.ok
    assert first.text != second.text, "the second draft must not reuse the first path"


def test_free_path_never_returns_an_existing_file(tmp_path):
    from quorum.config import free_path

    (tmp_path / "note.md").write_text("first", encoding="utf-8")
    second = free_path(tmp_path, "note", ".md")
    second.write_text("second", encoding="utf-8")

    assert second.name == "note-2.md"
    assert (tmp_path / "note.md").read_text(encoding="utf-8") == "first"
    assert free_path(tmp_path, "note", ".md").name == "note-3.md"


# --- the other door to the same silent failure ------------------------------


def test_a_search_with_no_query_falls_back_to_the_question(context):
    """An empty query retrieves nothing, and the turn then answers from general
    knowledge as though the material did not exist."""
    router = StubRouter([route_to("search"), route_answer(), draft(Coverage.COVERED)])
    agent = ChatAgent(context, router=router)

    agent.ask("why is postfix evaluation linear", Conversation())

    assert context.memory.calls[-1]["query"] == "why is postfix evaluation linear"


def test_the_same_tool_call_is_not_repeated_in_one_turn(context):
    """Asked a transcript question, the loop burned all four steps re-reading
    one transcript. Each read is up to 6,000 characters against a 6,000
    tokens/minute ceiling, so the repeats could stall the next question."""
    router = StubRouter([route_to("read_transcript", meeting="postfix"),
                         draft(Coverage.COVERED)])
    agent = ChatAgent(context, router=router)

    turn = agent.ask("what exactly did he say about the stack", Conversation())

    assert turn.tools_used == ["read_transcript"], "the repeat was skipped"
    assert turn.answer is not None


def test_the_same_tool_with_different_arguments_still_runs(context):
    """Reading two different stretches of a lecture is not a repeat."""
    router = StubRouter([
        route_to("read_transcript", meeting="postfix", start="00:00", end="01:00"),
        route_to("read_transcript", meeting="postfix", start="05:00", end="06:00"),
        route_answer(), draft(Coverage.COVERED),
    ])
    agent = ChatAgent(context, router=router)

    turn = agent.ask("compare the start and the end", Conversation())

    assert turn.tools_used == ["read_transcript", "read_transcript"]


# --- code the material never contained --------------------------------------


def test_code_written_by_the_model_is_not_claimed_as_the_lecturers():
    """Asked for the optimal solution in Python, the model wrote correct code
    from an algorithm the lecture genuinely explained - and it came back
    labelled "From your material", which reads as "your lecturer gave you this".
    They did not. Someone revising needs to know which half is which."""
    hits = StubMemory().recall("counting substrings")
    code = (
        "def count(s):\n"
        "    last = {'A': -1}\n"
        "    for i, ch in enumerate(s):\n"
        "        last[ch] = i\n"
        "    return 0\n"
    )
    router = StubRouter([draft(Coverage.COVERED, answer=code, cited=[1])])

    answer = answer_question("give me the optimal solution in python", hits, router=router)

    assert answer.coverage is Coverage.PARTIAL
    assert "the implementation itself" in answer.added
    assert "the rest is background" in answer.banner()


def test_prose_from_the_material_is_still_fully_covered():
    hits = StubMemory().recall("stack")
    router = StubRouter([draft(Coverage.COVERED, answer="It uses a stack.", cited=[1])])

    answer = answer_question("how does it work", hits, router=router)

    assert answer.coverage is Coverage.COVERED


def test_code_is_not_demoted_when_the_material_itself_had_code():
    from quorum.chat.answer import looks_like_code

    hits = [MemoryHit(MemoryKind.NOTE, "n1",
                      "He wrote:\n```\ndef solve(s):\n    return 0\n```",
                      "mtg_postfix", "2026-08-10", 0.9)]
    router = StubRouter([draft(Coverage.COVERED, answer="```\ndef solve(s):\n    return 0\n```",
                               cited=[1])])

    assert looks_like_code(hits[0].text)
    answer = answer_question("show me his code", hits, router=router)

    assert answer.coverage is Coverage.COVERED


def test_a_mention_of_a_function_in_prose_is_not_mistaken_for_code():
    from quorum.chat.answer import looks_like_code

    assert not looks_like_code("He defined a helper function that returns the count.")
    assert not looks_like_code("The class of problems solved by sliding windows is large.")


# --- staying inside the token budget ----------------------------------------


def test_the_router_sees_a_preview_not_the_whole_tool_result(context):
    """Asking for a transcript put thousands of characters into the routing
    prompt, and the model hit its own output cap trying to emit the decision -
    which the provider reports as "failed to validate JSON", pointing nowhere
    near the real problem."""
    from quorum.chat.agent import OBSERVATION_PREVIEW

    long_line = "the sliding window expands and contracts as we scan. " * 200
    context.project.transcripts_dir.mkdir(parents=True, exist_ok=True)
    big = lecture("Long lecture", "mtg_long")
    big.utterances[0].text = long_line
    (context.project.transcripts_dir / "mtg_long.json").write_text(
        big.model_dump_json(), encoding="utf-8"
    )

    router = StubRouter([route_to("read_transcript", meeting="Long lecture"),
                         route_answer(), draft(Coverage.COVERED)])
    ChatAgent(context, router=router).ask("what did he say", Conversation())

    routing_prompts = [
        call[0] for call in router.structured_calls
        if call[1].__name__ == "RouteDecision"
    ]
    second = routing_prompts[-1]
    assert len(second) < OBSERVATION_PREVIEW * 4, "the routing prompt carried the whole read"
    assert "truncated" in second


def test_a_transcript_read_is_capped_before_it_reaches_the_model(context):
    from quorum.chat.tools import MAX_TRANSCRIPT_CHARS

    context.project.transcripts_dir.mkdir(parents=True, exist_ok=True)
    big = lecture("Long lecture", "mtg_long2")
    big.utterances[0].text = "the sliding window expands and contracts. " * 400
    (context.project.transcripts_dir / "mtg_long2.json").write_text(
        big.model_dump_json(), encoding="utf-8"
    )

    result = run_tool(context, ToolRequest(tool="read_transcript", meeting="Long lecture"))

    assert len(result.text) <= MAX_TRANSCRIPT_CHARS + 100
    assert "narrow it" in result.text


def test_record_output_passed_to_the_answer_is_bounded(context):
    from quorum.chat.agent import FACTS_BUDGET

    for index in range(400):
        context.project.ledger.commitments.append(
            commitment(f"commitment number {index} with a reasonably long description",
                       cid=f"c{index}")
        )
    router = StubRouter([route_to("list_commitments"), route_answer(),
                         draft(Coverage.COVERED, cited=[])])
    ChatAgent(context, router=router).ask("what is open", Conversation())

    answering = [c[0] for c in router.structured_calls if c[1].__name__ == "AnswerDraft"]
    assert len(answering[0]) < FACTS_BUDGET * 2


def test_an_answer_that_reuses_a_passage_is_not_called_uncovered():
    """Asked for the transcript of a lecture, the model produced an accurate
    summary of it, cited nothing, and labelled the whole thing background -
    while retrieval had scored 0.75. Saying "your notes do not cover this" about
    material plainly taken from those notes teaches the reader to distrust a
    banner that is usually right."""
    hits = [MemoryHit(
        MemoryKind.NOTE, "n1",
        "The lecture introduced the minimum window substring problem: finding the "
        "smallest substring of S containing all characters of T, including duplicates",
        "mtg_min", "2026-08-16", 0.75,
    )]
    borrowed = (
        "The lecture introduced the minimum window substring problem: finding the "
        "smallest substring of S containing all characters of T, including duplicates."
    )
    router = StubRouter([draft(Coverage.BACKGROUND, answer=borrowed, cited=[])])

    answer = answer_question("give me the transcript", hits, router=router)

    assert answer.coverage is Coverage.PARTIAL, "never promoted all the way to covered"
    assert answer.cited == [1]
    assert "did not credit" in answer.added


def test_a_genuinely_unrelated_answer_stays_background():
    """The correction must need the answer to track a passage, not merely to
    sit near one in the retrieval ranking."""
    hits = StubMemory().recall("stack")
    router = StubRouter([draft(Coverage.BACKGROUND,
                               answer="A red-black tree rebalances by rotation.", cited=[])])

    answer = answer_question("how do red-black trees work", hits, router=router)

    assert answer.coverage is Coverage.BACKGROUND


def test_routing_has_room_for_a_reasoning_model_to_answer(context):
    """400 output tokens is generous for the JSON and not for the models: Groq's
    gpt-oss family spends output tokens reasoning first and ran out mid-object,
    which the provider reports as a prompt problem."""
    from quorum.chat.agent import ROUTE_TOKENS

    router = StubRouter([route_answer(), draft(Coverage.COVERED)])
    ChatAgent(context, router=router).ask("anything", Conversation())

    routing = [c for c in router.structured_calls if c[1].__name__ == "RouteDecision"]
    assert routing[0][2]["max_tokens"] == ROUTE_TOKENS
    assert ROUTE_TOKENS >= 800


# --- refusing to invent a record --------------------------------------------


FABRICATED = """Below is a reconstructed transcript of the lecture.

**Instructor:** Good morning, everyone. Today we will dive into the Minimum
Window Substring problem, a classic interview question.

**Instructor:** We are given two strings, a source S and a target T.

**Student:** Does the order of characters matter?
"""


def test_a_reconstructed_transcript_is_refused():
    """One model began writing one - "**Instructor:** Good morning, everyone..."
    - inventing an entire session that was never spoken. It failed only because
    it ran out of output tokens; with a larger allowance it would have returned
    a complete, fluent, fabricated record of the user's own lecture."""
    hits = StubMemory().recall("minimum window")
    router = StubRouter([draft(Coverage.BACKGROUND, answer=FABRICATED, cited=[])])

    answer = answer_question("give me the transcript", hits, router=router)

    assert "Instructor:" not in answer.text
    assert "will not reconstruct" in answer.text
    assert "quorum transcript" in answer.text
    assert answer.cited == []


def test_timestamped_dialogue_is_refused_too():
    hits = StubMemory().recall("stack")
    scripted = "[00:12] so we push each operand\n[00:31] and then we pop two values\n"
    router = StubRouter([draft(Coverage.BACKGROUND, answer=scripted, cited=[])])

    answer = answer_question("what did he say", hits, router=router)

    assert "will not reconstruct" in answer.text


def test_quoting_material_that_really_is_dialogue_is_allowed():
    """Real transcript excerpts must pass: quoting retrieved dialogue back is
    reporting, not invention."""
    real = ("[00:12] so we push each operand onto the stack\n"
            "[00:31] and then we pop two values off it")
    hits = [MemoryHit(MemoryKind.UTTERANCE, "u1", real, "mtg_postfix", "2026-08-10", 0.9)]
    router = StubRouter([draft(Coverage.COVERED, answer=real, cited=[1])])

    answer = answer_question("what did he say exactly", hits, router=router)

    assert "push each operand" in answer.text
    assert "will not reconstruct" not in answer.text


def test_an_ordinary_answer_mentioning_a_time_is_not_refused():
    hits = StubMemory().recall("stack")
    router = StubRouter([draft(
        Coverage.COVERED,
        answer="He explains the stack at 04:18, and the complexity follows from it.",
        cited=[1],
    )])

    answer = answer_question("when does he cover the stack", hits, router=router)

    assert "will not reconstruct" not in answer.text
