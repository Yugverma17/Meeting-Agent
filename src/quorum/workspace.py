"""The workspace: projects that persist between meetings.

Everything upstream works on one meeting. This is what makes the tool useful on
a Monday: a named project accumulates meetings, and its ledger carries every
commitment forward, so week 7 knows what was promised in week 2.

Layout on disk - plain JSON, deliberately:

    data/workspace/
      projects.json                     the registry
      projects/<slug>/
        ledger.json                     every commitment, with history
        meetings/<meeting_id>.json      each meeting's extracted record
        transcripts/<meeting_id>.json   the raw transcript
        memory/                         vector index for retrieval

Readable files matter here. When the agent claims you promised something six
weeks ago, you should be able to open a file and check, without the tool's
cooperation.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

from quorum.config import DATA_DIR
from quorum.models import MeetingRecord, Transcript
from quorum.tracking.ledger import Ledger

log = logging.getLogger(__name__)

WORKSPACE_DIR = DATA_DIR / "workspace"
_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    return _SLUG.sub("-", name.strip().lower()).strip("-") or "project"


@dataclass
class ProjectMeta:
    id: str
    name: str
    created_on: str
    description: str = ""
    repo: str | None = None
    """GitHub "owner/name", so the reality-verification layer knows where to look."""

    members: dict[str, str] = field(default_factory=dict)
    """display name -> email. Seeds the roster for live capture."""

    handles: dict[str, str] = field(default_factory=dict)
    """handle -> meeting_id. A short name you can say out loud.

    `mtg_9f2c1a4b7e30` is not something anyone refers to a lecture by, and a
    title is not stable - you can rename a meeting, and two standups share one.
    A handle is the name *you* chose, so `@kickoff` means the same thing in six
    weeks as it does today.
    """

    meeting_count: int = 0
    last_meeting_on: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class Project:
    """One project: its ledger, meetings and memory."""

    def __init__(self, meta: ProjectMeta, root: Path) -> None:
        self.meta = meta
        self.root = root
        self._ledger: Ledger | None = None

    # -- paths -------------------------------------------------------------

    @property
    def ledger_path(self) -> Path:
        return self.root / "ledger.json"

    @property
    def meetings_dir(self) -> Path:
        return self.root / "meetings"

    @property
    def transcripts_dir(self) -> Path:
        return self.root / "transcripts"

    @property
    def memory_dir(self) -> Path:
        return self.root / "memory"

    # -- ledger ------------------------------------------------------------

    @property
    def ledger(self) -> Ledger:
        if self._ledger is None:
            if self.ledger_path.exists():
                self._ledger = Ledger.load(self.ledger_path)
            else:
                self._ledger = Ledger(self.meta.id)
        return self._ledger

    def save_ledger(self) -> None:
        self.ledger.save(self.ledger_path)

    # -- meetings ----------------------------------------------------------

    def add_meeting(self, record: MeetingRecord, transcript: Transcript | None = None) -> None:
        """Fold a meeting into the ledger and keep both artefacts on disk."""
        self.ledger.ingest(record)
        self.save_ledger()

        self.meetings_dir.mkdir(parents=True, exist_ok=True)
        (self.meetings_dir / f"{record.meeting_id}.json").write_text(
            record.model_dump_json(indent=2), encoding="utf-8"
        )
        if transcript is not None:
            self.transcripts_dir.mkdir(parents=True, exist_ok=True)
            (self.transcripts_dir / f"{transcript.meeting_id}.json").write_text(
                transcript.model_dump_json(indent=2), encoding="utf-8"
            )

        self.meta.meeting_count += 1
        self.meta.last_meeting_on = record.meeting_date.isoformat()

    def meetings(self) -> list[MeetingRecord]:
        if not self.meetings_dir.exists():
            return []
        records = [
            MeetingRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(self.meetings_dir.glob("*.json"))
        ]
        records.sort(key=lambda r: r.meeting_date)
        return records

    def transcripts(self) -> list[Transcript]:
        if not self.transcripts_dir.exists():
            return []
        found = [
            Transcript.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(self.transcripts_dir.glob("*.json"))
        ]
        found.sort(key=lambda t: t.meeting_date)
        return found

    # -- convenience -------------------------------------------------------

    def roster_string(self) -> str:
        return ",".join(f"{name}:{email}" for name, email in self.meta.members.items())


class Workspace:
    """The set of projects on this machine."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else WORKSPACE_DIR
        self.registry_path = self.root / "projects.json"

    # -- registry ----------------------------------------------------------

    def _load_registry(self) -> dict[str, ProjectMeta]:
        if not self.registry_path.exists():
            return {}
        try:
            raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.error("Project registry unreadable (%s); treating as empty", exc)
            return {}
        return {key: ProjectMeta(**value) for key, value in raw.items()}

    def _save_registry(self, registry: dict[str, ProjectMeta]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {key: meta.as_dict() for key, meta in registry.items()}
        tmp = self.registry_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.registry_path)

    # -- public ------------------------------------------------------------

    def create(
        self,
        name: str,
        description: str = "",
        repo: str | None = None,
        members: dict[str, str] | None = None,
    ) -> Project:
        registry = self._load_registry()
        project_id = slugify(name)
        if project_id in registry:
            raise ValueError(f"Project {project_id!r} already exists")

        meta = ProjectMeta(
            id=project_id, name=name.strip(), created_on=date.today().isoformat(),
            description=description, repo=repo, members=members or {},
        )
        registry[project_id] = meta
        self._save_registry(registry)

        root = self.root / "projects" / project_id
        root.mkdir(parents=True, exist_ok=True)
        return Project(meta, root)

    def get(self, name_or_id: str) -> Project | None:
        registry = self._load_registry()
        meta = registry.get(slugify(name_or_id))
        if meta is None:
            return None
        return Project(meta, self.root / "projects" / meta.id)

    def list(self) -> list[ProjectMeta]:
        return sorted(self._load_registry().values(), key=lambda m: m.name.lower())

    def save(self, project: Project) -> None:
        registry = self._load_registry()
        registry[project.meta.id] = project.meta
        self._save_registry(registry)

    def delete(self, name_or_id: str) -> bool:
        """Remove from the registry. Files are left on disk deliberately - a
        mistyped name should not destroy weeks of meeting history."""
        registry = self._load_registry()
        project_id = slugify(name_or_id)
        if project_id not in registry:
            return False
        del registry[project_id]
        self._save_registry(registry)
        return True
