"""Synthetic multi-meeting project histories with ground truth by construction.

**Why this exists.** AMI and QMSum annotate single meetings, so they can score
extraction and nothing else. The claim this project actually makes - that an
agent can track a commitment across weeks, notice when it is quietly dropped,
and chase it without nagging people who already delivered - has no public
benchmark at all. So we build one.

**How it stays honest.** Structure is generated first and dialogue second. We
decide that Priya commits to the ingestion spec in week 2, misses it, re-commits
in week 4 and delivers silently in week 5 - and *then* render lines that say so.
The manifest is not an annotation of generated text; the text is a rendering of
the manifest. There is no labelling step to disagree with.

**What it makes measurable** (none of it scoreable on any existing corpus):

- *dropped-commitment recall* - work nobody ever mentioned again
- *silent delivery* - completed, never discussed, provable only from GitHub
- *false-nag rate* - chasing someone who already delivered
- *contradiction detection* - a week-6 decision reversing a week-2 one
- *slip and dependency propagation* - a deadline moving because an upstream one did
- *musing precision* - idle talk that must never become a task
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from enum import Enum
from pathlib import Path

from quorum.models import Speaker, Transcript, Utterance


class CommitmentFate(str, Enum):
    """What actually happens to a commitment - the thing the agent must infer."""

    DELIVERED = "delivered"
    """Done on time and said so out loud. The agent must close it and not nag."""

    DELIVERED_SILENTLY = "delivered_silently"
    """Done, but never mentioned again. Only external evidence proves it. An
    agent that trusts conversation alone will wrongly nag here - this is the
    case that justifies the reality-verification layer existing."""

    SLIPPED = "slipped"
    """Missed, then re-committed with a new date. Tests whether the agent
    tracks the same obligation across a change of deadline instead of
    inventing a second one."""

    DROPPED = "dropped"
    """Never delivered, never mentioned again, no evidence anywhere. The thing
    everyone forgot. Recall on these is the headline metric."""

    CANCELLED = "cancelled"
    """Explicitly called off later. Chasing it afterwards is a false nag."""

    BLOCKED = "blocked"
    """Waiting on an upstream commitment that slipped. Tests dependency
    propagation rather than flat date arithmetic."""


@dataclass
class SyntheticPerson:
    name: str
    email: str
    github_login: str
    aliases: list[str] = field(default_factory=list)

    def to_speaker(self, speaker_id: str) -> Speaker:
        return Speaker(
            id=speaker_id, display_name=self.name, email=self.email,
            aliases=self.aliases, github_login=self.github_login,
        )


@dataclass
class GroundTruthCommitment:
    id: str
    meeting_index: int
    utterance_index: int
    owner_name: str
    owner_email: str
    description: str
    spoken_quote: str
    """The exact line rendered into the transcript. Extraction is scored by
    matching against this, so a partial quote can be graded rather than only
    accepted or rejected."""

    assignee_mention: str
    """How the owner was actually referred to - 'I', 'you', 'Sam'. Assignee
    resolution accuracy is measured against this being mapped back correctly."""

    deadline_text: str
    deadline_date: date | None
    strength: str
    fate: CommitmentFate
    delivered_on: date | None = None
    github_evidence: str | None = None
    """e.g. "PR #204" - the only proof for a silent delivery."""

    resolved_in_meeting: int | None = None
    depends_on: str | None = None
    superseded_by: str | None = None

    @property
    def should_be_chased(self) -> bool:
        """Whether a correct agent would still be pursuing this at project end.

        Tentative items are excluded deliberately: "I might get to it if there's
        time" is not an obligation, and chasing it is exactly the over-eager
        behaviour that makes these tools irritating.
        """
        return self.strength == "firm" and self.fate in (
            CommitmentFate.DROPPED,
            CommitmentFate.BLOCKED,
        )

    @property
    def is_false_nag_target(self) -> bool:
        """Chasing these is the error that makes users abandon the tool."""
        return self.fate in (
            CommitmentFate.DELIVERED,
            CommitmentFate.DELIVERED_SILENTLY,
            CommitmentFate.CANCELLED,
        )


@dataclass
class GroundTruthDecision:
    id: str
    meeting_index: int
    utterance_index: int
    statement: str
    spoken_quote: str
    topic: str = ""
    """Which choice family this belongs to. A reversal is a different option
    from the same family, which is what makes the pair a real contradiction."""

    reversed_by: str | None = None
    """Id of a later decision that contradicts this one."""


@dataclass
class GroundTruthMusing:
    """Idle talk that must NOT become a commitment.

    Precision matters more than recall on this task: an extractor that turns
    every "we should probably..." into a task generates dozens of fake items a
    week and gets uninstalled. These are the negatives.
    """

    meeting_index: int
    utterance_index: int
    spoken_quote: str


@dataclass
class ProjectManifest:
    project_id: str
    people: list[SyntheticPerson]
    commitments: list[GroundTruthCommitment]
    decisions: list[GroundTruthDecision]
    musings: list[GroundTruthMusing]
    known_events: dict[str, str]
    seed: int

    def commitment(self, commitment_id: str) -> GroundTruthCommitment | None:
        return next((c for c in self.commitments if c.id == commitment_id), None)

    def firm(self) -> list[GroundTruthCommitment]:
        return [c for c in self.commitments if c.strength == "firm"]

    def to_json(self) -> str:
        payload = asdict(self)
        payload["commitments"] = [
            {**asdict(c), "fate": c.fate.value,
             "deadline_date": c.deadline_date.isoformat() if c.deadline_date else None,
             "delivered_on": c.delivered_on.isoformat() if c.delivered_on else None}
            for c in self.commitments
        ]
        return json.dumps(payload, indent=2, default=str)


# --- phrasing banks ---------------------------------------------------------
# Varied so the extractor cannot key on one sentence shape, but every template
# keeps the commitment marker unambiguous - the benchmark tests tracking, not
# whether the model can decode deliberately obscure English.

FIRM_TEMPLATES = [
    "I'll have {work} done {when}.",
    "Sure, I'll take {work} - {when}.",
    "Yes, I can get {work} finished {when}.",
    "I'll pick up {work} and have it ready {when}.",
    "Leave {work} with me, I'll close it out {when}.",
]
FIRM_SECOND_PERSON = [
    "{name}, can you take {work} {when}?",
    "{name}, could you own {work}? Needs to be {when}.",
]
ACCEPTANCE = ["Sure, that works.", "Yep, I'll do that.", "Okay, I've got it.", "Will do."]
TENTATIVE_TEMPLATES = [
    "I might be able to look at {work} {when}, no promises.",
    "I could probably start {work}, depends how the week goes.",
    "I'll try to get to {work} if there's time.",
]
MUSING_TEMPLATES = [
    "We should probably think about {topic} at some point.",
    "Someone ought to look into {topic} eventually.",
    "It would be nice if we cleaned up {topic} one day.",
    "At some stage {topic} is going to bite us.",
    "Long term we may want to revisit {topic}.",
]
DELIVERED_TEMPLATES = [
    "Quick update - {work} is done, I sent it Tuesday.",
    "{work} landed, it's merged.",
    "I finished {work} over the weekend.",
]
SLIP_TEMPLATES = [
    "I didn't get to {work}, sorry - I'll have it {when}.",
    "{work} slipped, I ran into problems. New date is {when}.",
    "Still working on {work}. Realistically {when}.",
]
BLOCKED_TEMPLATES = [
    # Work items already carry their article ("the retry logic"), so templates
    # must not add another one.
    "I can't start {work} until {blocker} is done.",
    "{work} is blocked on {blocker}, nothing I can do yet.",
]
CANCEL_TEMPLATES = [
    "Actually, let's drop {work} - we don't need it any more.",
    "We're cancelling {work}, priorities changed.",
]
DECISION_TEMPLATES = [
    "Let's decide {topic} then - we're going with {choice}.",
    "Agreed, on {topic} the direction is {choice}.",
    "Final call on {topic}: {choice}.",
]
REVERSAL_TEMPLATES = [
    "Change of plan on {topic} - we're switching to {choice} instead.",
    "We need to revisit our earlier call on {topic}. It's {choice} now.",
    "Reversing the {topic} decision: {choice}.",
]
QUESTION_TEMPLATES = [
    "Is the {topic} still blocked?",
    "Do we have an update on {topic}?",
    "Where did we land on {topic}?",
]
OPENERS = [
    "Right, let's get started.",
    "Okay, quick sync everyone.",
    "Morning - let's run through the week.",
]
CLOSERS = ["That's everything, thanks all.", "Good, let's wrap there.", "Okay, same time next week."]

WORK_ITEMS = [
    "the ingestion API spec", "the retry logic", "the billing reconciliation job",
    "the auth migration", "the rate limiter", "the schema migration",
    "the onboarding docs", "the load test harness", "the webhook receiver",
    "the audit log export", "the cache invalidation fix", "the SSO integration",
    "the pagination refactor", "the error budget dashboard", "the backup restore drill",
]
TOPICS = [
    "technical debt in the parser", "our alerting noise", "the staging environment drift",
    "test flakiness", "the deployment runbook", "log retention costs",
]
# Grouped by topic so a reversal can pick a *competing* option rather than an
# unrelated one. Switching from "Postgres rather than Mongo" to "a monorepo" is
# not a contradiction, and scoring contradiction detection against pairs like
# that would measure nothing.
CHOICE_FAMILIES = {
    "the database": ["Postgres rather than Mongo", "Mongo rather than Postgres"],
    "the queue": ["a managed queue instead of self-hosting", "self-hosted Kafka"],
    "the release cadence": ["weekly releases rather than continuous", "continuous deployment"],
    "authentication": ["OAuth over API keys", "API keys rather than OAuth"],
    "the repo layout": ["a monorepo", "separate repositories per service"],
    "rendering": ["server-side rendering", "a client-side single-page app"],
}
CHOICES = [choice for options in CHOICE_FAMILIES.values() for choice in options]
DEADLINE_PHRASES = [
    ("by Friday", 4), ("by Wednesday", 2), ("by end of week", 4),
    ("by next Tuesday", 8), ("by end of next week", 11), ("in two weeks", 14),
]

def render(template: str, **values: str) -> str:
    """Fill a template and capitalise the sentence.

    Needed because work items begin with an article, so any template starting
    with {work} would otherwise render as "the onboarding docs slipped..." and
    read as a fragment rather than a sentence.
    """
    text = template.format(**values)
    return text[:1].upper() + text[1:] if text else text


PEOPLE = [
    SyntheticPerson("Priya Raghavan", "priya@example.com", "praghavan", ["Priya"]),
    SyntheticPerson("Yug Verma", "yug@example.com", "yugverma", ["Yug"]),
    SyntheticPerson("Sam Okafor", "sam@example.com", "sokafor", ["Sam"]),
    SyntheticPerson("Mei Tanaka", "mei@example.com", "mtanaka", ["Mei"]),
]

FATE_WEIGHTS = {
    CommitmentFate.DELIVERED: 0.30,
    CommitmentFate.DELIVERED_SILENTLY: 0.20,
    CommitmentFate.SLIPPED: 0.15,
    CommitmentFate.DROPPED: 0.20,
    CommitmentFate.CANCELLED: 0.08,
    CommitmentFate.BLOCKED: 0.07,
}


class ProjectGenerator:
    """Generates a full project history plus its exact manifest."""

    def __init__(
        self,
        seed: int = 0,
        weeks: int = 8,
        people: list[SyntheticPerson] | None = None,
        start_date: date | None = None,
        commitments_per_meeting: tuple[int, int] = (2, 3),
    ) -> None:
        self.rng = random.Random(seed)
        self.seed = seed
        self.weeks = weeks
        self.people = people or PEOPLE
        self.commitments_per_meeting = commitments_per_meeting
        # Anchor to a Monday so weekday phrasing resolves the way people mean it.
        self.start_date = start_date or date(2026, 3, 2)

    # -- public ------------------------------------------------------------

    def generate(self) -> tuple[list[Transcript], ProjectManifest]:
        project_id = f"proj_synth_{self.seed}"
        speakers = [p.to_speaker(f"spk_{i}") for i, p in enumerate(self.people)]
        by_name = {p.name: speakers[i] for i, p in enumerate(self.people)}

        commitments: list[GroundTruthCommitment] = []
        decisions: list[GroundTruthDecision] = []
        musings: list[GroundTruthMusing] = []
        transcripts: list[Transcript] = []

        work_pool = self.rng.sample(WORK_ITEMS, k=len(WORK_ITEMS))
        demo_date = self.start_date + timedelta(days=7 * (self.weeks - 1) + 4)
        known_events = {"demo": demo_date.isoformat(), "launch": demo_date.isoformat()}

        facilitator = self.people[0]
        counter = 0

        for week in range(self.weeks):
            meeting_date = self.start_date + timedelta(days=7 * week)
            lines: list[tuple[Speaker, str]] = []

            lines.append((by_name[facilitator.name], self.rng.choice(OPENERS)))

            # --- status round on everything still open --------------------
            for tracked in [c for c in commitments if c.resolved_in_meeting is None]:
                if tracked.meeting_index >= week:
                    continue
                self._status_update(tracked, week, meeting_date, lines, by_name, commitments)

            # --- new commitments ------------------------------------------
            lo, hi = self.commitments_per_meeting
            for _ in range(self.rng.randint(lo, hi)):
                if not work_pool:
                    break
                counter += 1
                commitments.append(
                    self._new_commitment(
                        f"gt_{self.seed}_{counter}", week, meeting_date,
                        work_pool.pop(), lines, by_name,
                    )
                )

            # --- decisions, musings, questions ----------------------------
            # Decisions are frequent on purpose: contradiction detection needs a
            # population of earlier calls available to be reversed later.
            if self.rng.random() < 0.7:
                decisions.append(self._decision(week, lines, by_name, decisions))
            if self.rng.random() < 0.7:
                speaker = by_name[self.rng.choice(self.people).name]
                text = render(self.rng.choice(MUSING_TEMPLATES), topic=self.rng.choice(TOPICS))
                musings.append(GroundTruthMusing(week, len(lines), text))
                lines.append((speaker, text))
            if self.rng.random() < 0.4:
                speaker = by_name[self.rng.choice(self.people).name]
                lines.append(
                    (speaker, render(self.rng.choice(QUESTION_TEMPLATES),
                                     topic=self.rng.choice(TOPICS)))
                )

            lines.append((by_name[facilitator.name], self.rng.choice(CLOSERS)))
            transcripts.append(
                self._to_transcript(project_id, week, meeting_date, speakers, lines)
            )

        self._finalise_fates(commitments)

        manifest = ProjectManifest(
            project_id=project_id, people=self.people, commitments=commitments,
            decisions=decisions, musings=musings, known_events=known_events, seed=self.seed,
        )
        return transcripts, manifest

    # -- pieces ------------------------------------------------------------

    def _new_commitment(
        self, commitment_id, week, meeting_date, work, lines, by_name
    ) -> GroundTruthCommitment:
        owner = self.rng.choice(self.people)
        deadline_text, offset = self.rng.choice(DEADLINE_PHRASES)
        deadline_date = meeting_date + timedelta(days=offset)
        fate = self._pick_fate()
        tentative = self.rng.random() < 0.18

        if tentative:
            text = render(self.rng.choice(TENTATIVE_TEMPLATES), work=work, when=deadline_text)
            strength, mention = "tentative", "I"
            index = len(lines)
            lines.append((by_name[owner.name], text))
        elif self.rng.random() < 0.3:
            # Assigned by someone else, then accepted - "you" phrasing, which is
            # what makes assignee resolution non-trivial.
            asker = self.rng.choice([p for p in self.people if p.name != owner.name])
            request = render(
                self.rng.choice(FIRM_SECOND_PERSON),
                name=owner.name.split()[0], work=work, when=deadline_text,
            )
            # The commitment content lives in the request line ("Sam, can you
            # take X by Friday?"); the reply only accepts it. Cite the request,
            # since that is where the work, the owner and the date all appear.
            index = len(lines)
            lines.append((by_name[asker.name], request))
            lines.append((by_name[owner.name], self.rng.choice(ACCEPTANCE)))
            strength, mention, text = "firm", "you", request
        else:
            text = render(self.rng.choice(FIRM_TEMPLATES), work=work, when=deadline_text)
            strength, mention = "firm", "I"
            index = len(lines)
            lines.append((by_name[owner.name], text))

        return GroundTruthCommitment(
            id=commitment_id, meeting_index=week, utterance_index=index,
            owner_name=owner.name, owner_email=owner.email,
            description=work, spoken_quote=text, assignee_mention=mention,
            deadline_text=deadline_text, deadline_date=deadline_date,
            strength=strength, fate=fate if strength == "firm" else CommitmentFate.DROPPED,
        )

    def _status_update(self, tracked, week, meeting_date, lines, by_name, commitments) -> None:
        """Render whatever happens to an open commitment in this meeting."""
        speaker = by_name[tracked.owner_name]
        due_week = tracked.meeting_index + 1

        if tracked.fate is CommitmentFate.DELIVERED and week >= due_week:
            lines.append(
                (speaker, render(self.rng.choice(DELIVERED_TEMPLATES), work=tracked.description))
            )
            tracked.resolved_in_meeting = week
            tracked.delivered_on = tracked.deadline_date

        elif tracked.fate is CommitmentFate.SLIPPED and week == due_week:
            new_text, offset = self.rng.choice(DEADLINE_PHRASES)
            lines.append(
                (speaker, render(self.rng.choice(SLIP_TEMPLATES),
                                 work=tracked.description, when=new_text))
            )
            # The same obligation with a later date - not a new one.
            tracked.deadline_text = new_text
            tracked.deadline_date = meeting_date + timedelta(days=offset)

        elif tracked.fate is CommitmentFate.CANCELLED and week >= due_week:
            canceller = by_name[self.people[0].name]
            lines.append(
                (canceller, render(self.rng.choice(CANCEL_TEMPLATES), work=tracked.description))
            )
            tracked.resolved_in_meeting = week

        elif tracked.fate is CommitmentFate.BLOCKED and week == due_week:
            upstream = self._pick_blocker(tracked, commitments)
            lines.append(
                (speaker, render(self.rng.choice(BLOCKED_TEMPLATES),
                                 work=tracked.description, blocker=upstream))
            )

        # DELIVERED_SILENTLY and DROPPED produce no dialogue at all - that is
        # precisely what makes them hard, and what external evidence must settle.

    def _pick_blocker(self, tracked, commitments) -> str:
        """Choose an upstream commitment to name as the blocker.

        Silent fates are excluded deliberately. DROPPED and DELIVERED_SILENTLY
        are *defined* by never being spoken of again - naming one as a blocker
        would leak it back into the dialogue, handing the agent a clue that the
        ground truth says should not exist and quietly corrupting both
        dropped-commitment recall and the silent-delivery case.
        """
        silent = (CommitmentFate.DROPPED, CommitmentFate.DELIVERED_SILENTLY)
        candidates = [
            c
            for c in commitments
            if c.id != tracked.id
            and c.meeting_index <= tracked.meeting_index
            and c.fate not in silent
        ]
        if candidates:
            upstream = self.rng.choice(candidates)
            tracked.depends_on = upstream.id
            return upstream.description
        # Deliberately not drawn from WORK_ITEMS, so a placeholder can never
        # collide with a real commitment's description.
        return "the vendor security review"

    def _decision(self, week, lines, by_name, decisions) -> GroundTruthDecision:
        speaker = by_name[self.people[0].name]
        reversible = [d for d in decisions if d.reversed_by is None and d.meeting_index < week - 1]
        new_id = f"dec_{self.seed}_{len(decisions) + 1}"

        if reversible and self.rng.random() < 0.55:
            earlier = self.rng.choice(reversible)
            # Pick a competing option from the same family, so the pair is a
            # genuine contradiction about one topic.
            alternatives = [
                c for c in CHOICE_FAMILIES[earlier.topic] if c != earlier.statement
            ]
            choice = self.rng.choice(alternatives)
            text = render(
                self.rng.choice(REVERSAL_TEMPLATES), topic=earlier.topic, choice=choice
            )
            earlier.reversed_by = new_id
            return self._emit_decision(
                new_id, week, lines, speaker, earlier.topic, choice, text, decisions
            )

        topic = self.rng.choice(list(CHOICE_FAMILIES))
        choice = self.rng.choice(CHOICE_FAMILIES[topic])
        text = render(self.rng.choice(DECISION_TEMPLATES), topic=topic, choice=choice)
        return self._emit_decision(
            new_id, week, lines, speaker, topic, choice, text, decisions
        )

    @staticmethod
    def _emit_decision(decision_id, week, lines, speaker, topic, choice, text, decisions):
        decision = GroundTruthDecision(decision_id, week, len(lines), choice, text, topic=topic)
        lines.append((speaker, text))
        return decision

    def _finalise_fates(self, commitments: list[GroundTruthCommitment]) -> None:
        """Attach the external evidence that only exists outside the transcript."""
        for index, commitment in enumerate(commitments):
            if commitment.fate is CommitmentFate.DELIVERED_SILENTLY:
                commitment.delivered_on = commitment.deadline_date
                commitment.github_evidence = f"PR #{200 + index}"
            elif commitment.fate is CommitmentFate.DELIVERED:
                commitment.github_evidence = f"PR #{200 + index}"

    def _pick_fate(self) -> CommitmentFate:
        fates = list(FATE_WEIGHTS)
        return self.rng.choices(fates, weights=[FATE_WEIGHTS[f] for f in fates], k=1)[0]

    @staticmethod
    def _to_transcript(project_id, week, meeting_date, speakers, lines) -> Transcript:
        utterances = [
            Utterance(
                id=f"{project_id}_m{week}_u{i}", index=i, speaker_id=speaker.id,
                text=text, start_s=i * 18.0, end_s=i * 18.0 + 14.0,
            )
            for i, (speaker, text) in enumerate(lines)
        ]
        return Transcript(
            meeting_id=f"{project_id}_m{week}",
            title=f"Weekly sync - week {week + 1}",
            meeting_date=meeting_date, speakers=speakers, utterances=utterances,
            source="synthetic", project_id=project_id,
        )


def write_dataset(out_dir: Path, n_projects: int = 10, weeks: int = 8) -> Path:
    """Generate a full benchmark to disk: one folder per project."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(n_projects):
        transcripts, manifest = ProjectGenerator(seed=seed, weeks=weeks).generate()
        project_dir = out_dir / manifest.project_id
        project_dir.mkdir(exist_ok=True)
        for transcript in transcripts:
            (project_dir / f"{transcript.meeting_id}.json").write_text(
                transcript.model_dump_json(indent=2), encoding="utf-8"
            )
        (project_dir / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")
    return out_dir
