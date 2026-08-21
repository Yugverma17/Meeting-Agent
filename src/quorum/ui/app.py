"""The Streamlit interface.

A face on the existing product, not a second implementation of it. Every button
here calls the same function the CLI calls - `IngestGraph`, `LectureAnalyser`,
`ChatAgent`, `CalendarSync`, `DraftWriter`. If a rule holds in the terminal it
holds here, because there is only one copy of it.

Two things fight Streamlit, and both are handled deliberately rather than worked
around:

**The script re-runs on every interaction.** A forty-minute recording cannot
live in a local variable. It lives in `st.session_state` via
`quorum.ui.session`, and only the timer re-runs each second - through
`@st.fragment`, so clicking Stop does not re-execute the whole page.

**Long work blocks the page.** Transcribing and extracting take a minute or
more. They run inside `st.status`, which is honest about what is happening
instead of freezing on a click with no explanation.

Everything outbound - calendar writes, Gmail drafts - still passes the approval
gate. A button is a nicer way to say yes than typing "y", and it is not a way to
skip being asked.
"""

from __future__ import annotations

import time
from datetime import date

import streamlit as st

st.set_page_config(page_title="Quorum", page_icon="◆", layout="wide")


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------


@st.cache_resource
def _workspace():
    from quorum.workspace import Workspace

    return Workspace()


def _projects():
    return _workspace().list()


def _project(project_id: str):
    return _workspace().get(project_id)


def _session():
    return st.session_state.get("recording")


def _toast_error(exc: Exception, doing: str) -> None:
    st.error(f"{doing} failed: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Sidebar: which project
# ---------------------------------------------------------------------------


def sidebar() -> str | None:
    st.sidebar.title("◆ Quorum")
    st.sidebar.caption("Meeting assistants tell you what was said. "
                       "This checks whether it happened.")

    projects = _projects()
    ids = [p.id for p in projects]
    labels = {p.id: p.name for p in projects}

    if not ids:
        st.sidebar.info("No projects yet. Create one below to begin.")
        chosen = None
    else:
        chosen = st.sidebar.selectbox(
            "Project", ids, format_func=lambda i: labels.get(i, i),
            key="chosen_project",
        )

    with st.sidebar.expander("New project", expanded=not ids):
        name = st.text_input("Name", key="new_project_name",
                             placeholder="Ingestion Revamp")
        members = st.text_area(
            "People (optional)", key="new_project_members",
            placeholder="Priya Raghavan: priya@example.com\nSam Okafor: sam@example.com",
            help="One per line. Needed for drafting emails to them later.",
        )
        repo = st.text_input("GitHub repo (optional)", key="new_project_repo",
                             placeholder="yugverma17/ingestion")
        if st.button("Create", key="create_project", use_container_width=True):
            if not name.strip():
                st.warning("Give it a name.")
            else:
                roster = {}
                for line in members.splitlines():
                    who, _, email = line.partition(":")
                    if who.strip():
                        roster[who.strip()] = email.strip()
                try:
                    _workspace().create(name.strip(), repo=repo.strip() or None,
                                        members=roster)
                    st.success(f"Created {name.strip()}")
                    st.rerun()
                except ValueError as exc:
                    st.warning(str(exc))

    if chosen:
        found = _project(chosen)
        if found is not None:
            open_items = found.ledger.open_commitments()
            undated = sum(1 for c in open_items if c.deadline.resolved is None)
            st.sidebar.metric("Open commitments", len(open_items))
            if undated:
                st.sidebar.caption(f"{undated} still need a deadline")

    st.sidebar.divider()
    _google_panel()
    st.sidebar.divider()
    st.sidebar.caption("Everything runs on this laptop. Nothing is sent "
                       "without you approving it.")
    return chosen


def _google_panel() -> None:
    """Sign in, and say plainly which mailbox that makes the drafts go to.

    Drafts are created against `userId="me"`, so the account signed in here *is*
    where they land. Showing the address is not decoration - it is the one thing
    worth checking before an app writes into your mail.
    """
    from quorum.integrations import credentials_status

    status = credentials_status()
    st.sidebar.markdown("### Google")

    if not status.libraries_installed:
        st.sidebar.warning("Google libraries are not installed.")
        st.sidebar.code("pip install google-api-python-client "
                        "google-auth-oauthlib google-auth-httplib2")
        return

    if not status.secrets_present:
        st.sidebar.info("Not set up yet — needs a one-time Google Cloud step.")
        with st.sidebar.expander("How"):
            st.markdown(
                "1. Open [Google Cloud credentials]"
                "(https://console.cloud.google.com/apis/credentials)\n"
                "2. Enable the **Google Calendar API** and the **Gmail API**\n"
                "3. Create Credentials → OAuth client ID → **Desktop app**\n"
                "4. Download the JSON and save it as `credentials.json` in the "
                "project folder\n\n"
                "Then the Connect button appears here."
            )
        return

    if status.ready:
        st.sidebar.success(f"Signed in as **{status.account or 'unknown account'}**")
        st.sidebar.caption("Drafts go to this mailbox. Calendar events go to "
                           "this account.")
        if st.sidebar.button("Disconnect", use_container_width=True):
            from quorum.integrations import revoke

            revoke()
            st.session_state.pop("drafts", None)
            st.rerun()
        return

    st.sidebar.caption("Connect to put deadlines in your calendar and drafts "
                       "in your Gmail.")
    if st.sidebar.button("Connect Google", type="primary", use_container_width=True):
        _run_consent()


def _run_consent() -> None:
    """Open the consent window and wait.

    This blocks the page while Google's browser tab is open, which is honest:
    there is nothing to do until consent is given or refused. The timeout in
    `authorise` is what stops a closed tab leaving the page waiting forever.
    """
    from quorum.integrations import GoogleAuthError, authorise

    with st.spinner("A Google sign-in window is opening — approve it there, "
                    "then come back to this page."):
        try:
            authorise(interactive=True, timeout_seconds=300)
        except GoogleAuthError as exc:
            st.sidebar.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - the OAuth stack fails in many ways
            st.sidebar.error(f"Sign-in failed: {type(exc).__name__}: {exc}")
            return
    st.rerun()


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


@st.fragment(run_every="1s")
def _live_timer() -> None:
    """The only thing that re-runs each second.

    A whole-page rerun every second would fight every widget on it - text
    half-typed into the chat box, an expander closing itself. A fragment redraws
    this block alone.
    """
    session = _session()
    if session is None or not session.live:
        return

    left, middle, right = st.columns(3)
    left.metric("Recording", session.elapsed_label)
    middle.metric("Silent chunks skipped", session.skipped_silent)
    right.metric("Queued", session.queued_chunks)


def _devices_hint() -> None:
    from quorum.capture.audio import DualRecorder

    try:
        found = DualRecorder.devices()
    except Exception as exc:  # noqa: BLE001 - no sound card, or the wrong platform
        st.warning(f"Could not read your audio devices: {exc}")
        return

    mic = (found.get("mic") or {}).get("name", "unknown")
    loopback = found.get("loopback")
    if loopback:
        st.caption(f"Microphone: {mic}  ·  System audio: {loopback['name']}")
    else:
        # Recording a video is impossible without it, and finding that out after
        # a forty-minute lecture is the worst time to learn it.
        st.caption(f"Microphone: {mic}")
        st.warning("No system-audio loopback device found. Recording a video "
                   "will capture nothing — a lecture in a room will still work.")


def record_tab(project_id: str | None) -> None:
    session = _session()

    if session is not None and session.live:
        st.subheader(f"Recording — {session.title}")
        _live_timer()
        st.caption("Leave this tab open. Press Stop when it ends.")
        if st.button("■ Stop and process", type="primary", use_container_width=True):
            session.stop()
            _finish_recording(session)
        return

    st.subheader("Record")
    kind = st.segmented_control(
        "What is this?", ["Meeting", "Lecture (video)", "Lecture (in a room)"],
        default="Meeting", key="record_kind",
    )
    if kind is None:
        kind = "Meeting"

    title = st.text_input(
        "Name it", key="record_title",
        placeholder="Weekly sync" if kind == "Meeting" else "Sliding window",
    )

    if kind == "Lecture (video)":
        st.caption("Listens to your **speakers**. Do not mute the tab.")
        speed = st.select_slider(
            "Playback speed", [1.0, 1.25, 1.5, 1.75, 2.0], value=1.0,
            help="So the timestamps in your notes match the video. "
                 "Only correct if you watch straight through without seeking.",
        )
        system_only, roster, me, my_email = True, "", "You", ""
    elif kind == "Lecture (in a room)":
        st.caption("Listens to your **microphone**. Point the laptop at the speaker.")
        speed, system_only, roster, me, my_email = 1.0, False, "", "You", ""
    else:
        st.caption("Records your microphone and the call, as separate channels. "
                   "**Wear headphones** — otherwise their words land on your channel "
                   "and get attributed to you.")
        col_a, col_b = st.columns(2)
        me = col_a.text_input("Your name", value="You", key="record_me")
        my_email = col_b.text_input("Your email", key="record_my_email")
        found = _project(project_id) if project_id else None
        default_roster = found.roster_string() if found else ""
        roster = st.text_input(
            "Who else is on the call", value=default_roster, key="record_roster",
            placeholder="Priya:priya@x.com,Sam:sam@x.com",
            help="Needed to attribute commitments and to draft emails to them.",
        )
        speed, system_only = 1.0, False

    _devices_hint()

    if project_id is None:
        st.info("Pick or create a project in the sidebar first — otherwise this "
                "is analysed once and forgotten.")

    st.warning("Tell the other people they are being recorded. In many places "
               "that is a legal requirement, not a courtesy.", icon="⚠")

    if st.button("● Start recording", type="primary", disabled=not title.strip(),
                 use_container_width=True):
        from quorum.ui.session import RecordingSession

        fresh = RecordingSession(
            title=title.strip(),
            kind="lecture" if kind.startswith("Lecture") else "meeting",
            project_id=project_id, system_only=system_only, speed=speed,
            me=me, my_email=my_email, roster=roster,
        )
        try:
            fresh.begin()
        except Exception as exc:  # noqa: BLE001 - audio devices fail in many ways
            _explain_audio_failure(exc, system_only)
            return
        st.session_state["recording"] = fresh
        st.rerun()


def _explain_audio_failure(exc: Exception, system_only: bool) -> None:
    """Say what to change, not what threw.

    Windows denies microphone access with a generic device error rather than
    anything mentioning permission, so "Unanticipated host error" is the whole
    message a user gets - and the fix is two clicks away in a Settings page they
    have no reason to suspect.
    """
    detail = f"{type(exc).__name__}: {exc}"
    st.error(f"Could not start recording — {detail}")

    text = detail.lower()
    denied = any(marker in text for marker in
                 ("unanticipated host error", "permission", "denied", "-9999", "access"))

    if denied and not system_only:
        st.warning(
            "**Windows is probably blocking the microphone.**\n\n"
            "Open **Settings → Privacy & security → Microphone**, turn on "
            "*Microphone access*, and make sure *Let desktop apps access your "
            "microphone* is on. Then press Start again.",
            icon="🎤",
        )
    elif system_only:
        st.warning(
            "**No system audio to record.**\n\n"
            "This mode listens to your speakers. Check that something is "
            "playing, the tab is not muted, and your output device is the one "
            "shown above. For a lecture in a room, choose "
            "*Lecture (in a room)* instead — that uses the microphone.",
            icon="🔈",
        )
    else:
        st.info("Another app may be holding the microphone. Close anything else "
                "recording — a call, a voice recorder — and try again.")


def _finish_recording(session) -> None:
    """Audio to notes. The slow part, narrated rather than silent."""
    with st.status("Processing…", expanded=True) as status:
        st.write("Collecting audio…")
        chunks = session.chunks()
        if not chunks:
            status.update(label="Nothing recorded", state="error")
            st.error(
                "No audio was captured. "
                + ("Was the video playing, and the tab unmuted?"
                   if session.system_only else "Is the microphone muted?")
            )
            st.session_state.pop("recording", None)
            return

        st.write(f"Transcribing {len(chunks)} chunk(s)…")
        try:
            transcript, stats, echo = session.transcribe(chunks)
        except Exception as exc:  # noqa: BLE001
            status.update(label="Transcription failed", state="error")
            _toast_error(exc, "Transcribing")
            return

        if transcript is None:
            status.update(label="No speech recognised", state="error")
            st.error("Nothing was recognised in the audio.")
            st.session_state.pop("recording", None)
            return

        st.write(f"{stats.audio_seconds:.0f}s of audio "
                 f"({stats.daily_budget_used:.1%} of today's free budget)")
        if echo is not None and getattr(echo, "likely_no_headphones", False):
            st.warning("Your microphone picked up your own speakers. "
                       "Use headphones next time — their words can otherwise be "
                       "attributed to you.")

        project = _project(session.project_id) if session.project_id else None
        if session.kind == "lecture":
            _finish_lecture(session, transcript, project, status)
        else:
            _finish_meeting(session, transcript, project, status)

    st.session_state.pop("recording", None)


def _finish_lecture(session, transcript, project, status) -> None:
    from quorum.agents.embedding import LexicalEmbedder
    from quorum.agents.segmenter import Segmenter, SegmenterConfig
    from quorum.analysis import LectureAnalyser

    st.write("Taking notes…")
    topics = Segmenter(
        config=SegmenterConfig(max_tokens=2400, min_utterances=6),
        embedder=LexicalEmbedder(),
    ).segment(transcript)
    notes = LectureAnalyser().analyse(transcript, topics)

    try:
        from quorum.analysis.replays import find_replays

        notes.replays = find_replays(transcript)
    except Exception:  # noqa: BLE001 - a bonus section must not cost the notes
        pass

    _persist_transcript(transcript, project)
    if project is not None:
        from quorum.chat.naming import register_meeting
        from quorum.memory import ProjectMemory

        try:
            ProjectMemory(project.memory_dir).index_notes(
                transcript.meeting_id, notes.title, date.today(), notes
            )
        except Exception:  # noqa: BLE001 - retrieval is an optimisation
            pass
        handle = register_meeting(project, transcript)
        _workspace().save(project)
        status.update(label=f"Done — saved as @{handle}", state="complete")
    else:
        status.update(label="Done", state="complete")

    st.session_state["last_notes"] = notes.as_markdown()


def _finish_meeting(session, transcript, project, status) -> None:
    from quorum.analysis.meeting import MeetingSummariser
    from quorum.models import Segment
    from quorum.pipeline import IngestGraph, RunStatus

    st.write("Extracting commitments…")
    outcome = IngestGraph().run(
        transcript, project_id=project.meta.id if project else None
    )
    if outcome.status is RunStatus.INTERRUPTED:
        status.update(label="Interrupted", state="error")
        st.error(f"Extraction stopped: {outcome.error}")
        st.info(f"Nothing is lost — the transcript is checkpointed. "
                f"Resume from the terminal: `quorum resume {transcript.meeting_id}`")
        return

    record = outcome.record
    record.title = session.title

    st.write("Writing the summary…")
    try:
        segments = [Segment.model_validate(s) for s in outcome.state.get("segments", [])]
        digest = MeetingSummariser().summarise(transcript, segments)
        record.summary = digest.summary
        st.session_state["last_notes"] = digest.as_markdown(record)
    except Exception:  # noqa: BLE001 - a summary is worth less than the ledger
        st.session_state["last_notes"] = ""

    if project is not None:
        from quorum.chat.naming import register_meeting

        project.add_meeting(record, transcript)
        handle = register_meeting(project, transcript)
        _workspace().save(project)
        status.update(
            label=f"Done — {len(record.commitments)} commitment(s), saved as @{handle}",
            state="complete",
        )
    else:
        _persist_transcript(transcript, None)
        status.update(label=f"Done — {len(record.commitments)} commitment(s)",
                      state="complete")


def _persist_transcript(transcript, project) -> None:
    from quorum.config import RUNS_DIR

    if project is not None:
        folder = project.transcripts_dir
    else:
        folder = RUNS_DIR / "transcripts"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{transcript.meeting_id}.json").write_text(
        transcript.model_dump_json(indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------


def library_tab(project_id: str | None) -> None:
    if st.session_state.get("last_notes"):
        with st.expander("Notes from what you just recorded", expanded=True):
            st.markdown(st.session_state["last_notes"])

    if not project_id:
        st.info("Pick a project in the sidebar.")
        return

    from quorum.chat.naming import list_meetings, set_handle

    project = _project(project_id)
    refs = list_meetings(project)
    if not refs:
        st.info("Nothing recorded in this project yet.")
        return

    st.subheader("Recordings")
    chosen = st.selectbox(
        "Which one", [r.meeting_id for r in refs],
        format_func=lambda mid: _label_for(refs, mid),
    )
    ref = next(r for r in refs if r.meeting_id == chosen)

    name_col, save_col = st.columns([3, 1])
    handle = name_col.text_input("Short name for chat", value=ref.handle,
                                 key=f"handle_{chosen}")
    if save_col.button("Rename", key=f"rename_{chosen}"):
        try:
            stored = set_handle(project, chosen, handle)
            _workspace().save(project)
            st.success(f"@{stored}")
            st.rerun()
        except ValueError as exc:
            st.warning(str(exc))

    transcript = next(
        (t for t in project.transcripts() if t.meeting_id == chosen), None
    )
    if transcript is None:
        st.warning("The transcript file for this recording is missing.")
        return

    notes_tab, words_tab = st.tabs(["Notes", "Transcript"])
    with notes_tab:
        _show_saved_notes(chosen)
    with words_tab:
        _show_transcript(transcript)


def _label_for(refs, meeting_id: str) -> str:
    ref = next(r for r in refs if r.meeting_id == meeting_id)
    return f"{ref.title or 'untitled'} — {ref.meeting_date} ({ref.kind})"


def _show_saved_notes(meeting_id: str) -> None:
    from quorum.config import RUNS_DIR

    matches = sorted((RUNS_DIR / "notes").glob(f"*{meeting_id}*.md")) \
        if (RUNS_DIR / "notes").exists() else []
    if not matches:
        st.caption("No notes file saved for this recording.")
        return
    st.markdown(matches[-1].read_text(encoding="utf-8"))


def _show_transcript(transcript) -> None:
    from quorum.export import Style, parse_time, render

    controls = st.columns(4)
    search = controls[0].text_input("Find a phrase", key=f"find_{transcript.meeting_id}")
    start = controls[1].text_input("From", placeholder="04:00",
                                   key=f"from_{transcript.meeting_id}")
    end = controls[2].text_input("To", placeholder="07:00",
                                 key=f"to_{transcript.meeting_id}")
    default_style = "timestamped" if transcript.is_monologue else "speakers"
    style = controls[3].selectbox(
        "Style", ["timestamped", "speakers", "plain", "markdown", "srt"],
        index=["timestamped", "speakers", "plain", "markdown", "srt"].index(default_style),
        key=f"style_{transcript.meeting_id}",
    )

    try:
        text = render(
            transcript, Style(style), search=search or None,
            start_s=parse_time(start or None), end_s=parse_time(end or None),
        )
    except ValueError as exc:
        st.warning(str(exc))
        return

    if not text.strip():
        st.info("Nothing matched those filters.")
        return

    st.download_button("Download", text, file_name=f"{transcript.meeting_id}.txt")
    st.text(text)


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------


def ask_tab(project_id: str | None) -> None:
    everywhere = st.toggle(
        "Search every project", key="chat_all",
        help="Read-only: changes can only be made inside one project.",
    )
    if not project_id and not everywhere:
        st.info("Pick a project in the sidebar, or turn on 'Search every project'.")
        return

    from quorum.chat import ChatAgent, Conversation, ToolContext
    from quorum.chat.agent import render_answer

    signature = f"{project_id}:{everywhere}"
    if st.session_state.get("chat_for") != signature:
        st.session_state["chat_for"] = signature
        st.session_state["conversation"] = Conversation()
        st.session_state["chat_log"] = []

    for entry in st.session_state.get("chat_log", []):
        with st.chat_message(entry["role"]):
            st.markdown(entry["text"])
            for line in entry.get("sources", []):
                st.caption(line)

    question = st.chat_input("Ask about anything you have recorded")
    if not question:
        return

    st.session_state["chat_log"].append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    workspace = _workspace()
    if everywhere:
        from quorum.chat.federated import FederatedMemory, all_projects

        projects = all_projects(workspace)
        if not projects:
            st.warning("Nothing recorded in any project yet.")
            return
        context = ToolContext(project=projects[0], workspace=workspace,
                              memory=FederatedMemory(projects), federated=True)
    else:
        context = ToolContext(project=_project(project_id), workspace=workspace)

    agent = ChatAgent(context)
    with st.chat_message("assistant"), st.spinner("Looking through your material…"):
        try:
            turn = agent.ask(question, st.session_state["conversation"])
        except Exception as exc:  # noqa: BLE001 - a failed turn must not lose the thread
            _toast_error(exc, "Answering")
            return

        if turn.needs_confirmation:
            st.session_state["pending_write"] = turn.pending
            st.session_state["pending_agent"] = agent
            st.warning(turn.pending.preview)
            st.caption("Nothing has changed yet.")
            _confirm_buttons()
            return

        if turn.answer is not None:
            body = render_answer(turn.answer)
            sources = [
                f"[{i}] {turn.answer.hits[i - 1].meeting_date} "
                f"{turn.answer.hits[i - 1].text[:110]}…"
                for i in turn.answer.cited
            ]
            st.markdown(body)
            for line in sources:
                st.caption(line)
            st.session_state["chat_log"].append(
                {"role": "assistant", "text": body, "sources": sources}
            )
        elif turn.message:
            st.markdown(turn.message)
            st.session_state["chat_log"].append(
                {"role": "assistant", "text": turn.message}
            )

        st.session_state["conversation"].add(turn)


def _confirm_buttons() -> None:
    """A button is a nicer way to say yes than typing it. It is not a way to
    skip being asked - the write still needs the second, confirmed dispatch."""
    yes, no = st.columns(2)
    if yes.button("Do it", type="primary", key="confirm_write"):
        pending = st.session_state.pop("pending_write", None)
        agent = st.session_state.pop("pending_agent", None)
        if pending is not None and agent is not None:
            result = agent.confirm(pending)
            (st.success if result.ok else st.error)(result.text)
        st.rerun()
    if no.button("Leave it", key="reject_write"):
        st.session_state.pop("pending_write", None)
        st.session_state.pop("pending_agent", None)
        st.rerun()


# ---------------------------------------------------------------------------
# To-dos
# ---------------------------------------------------------------------------


def todos_tab(project_id: str | None) -> None:
    if not project_id:
        st.info("Pick a project in the sidebar.")
        return

    project = _project(project_id)
    ledger = project.ledger
    open_items = ledger.open_commitments()
    today = date.today()

    if not open_items:
        st.success("Nothing open on this project.")
    else:
        rows = []
        for item in sorted(open_items, key=lambda c: c.deadline.resolved or date.max):
            due = item.deadline.resolved
            if due is None:
                state = "needs a date"
            elif due < today:
                state = f"{(today - due).days}d late"
            elif due == today:
                state = "due today"
            else:
                state = f"in {(due - today).days}d"
            rows.append({
                "Who": item.assignee.display_name or "unassigned",
                "What": item.description,
                "Due": due.isoformat() if due else "—",
                "Status": state,
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    undated = [c for c in open_items if c.deadline.resolved is None and c.is_actionable]
    if undated:
        st.subheader(f"{len(undated)} need a deadline from you")
        st.caption("Nobody said when. Without a date these cannot be chased or "
                   "put in your calendar.")
        _triage_form(project, undated, today)

    st.divider()
    left, right = st.columns(2)
    with left:
        _calendar_panel(project)
    with right:
        _drafts_panel(project)

    st.divider()
    _week_panel(project)


def _triage_form(project, undated, today) -> None:
    from quorum.agents.dates import resolve_deadline
    from quorum.models import DeadlineResolution

    for item in undated[:8]:
        with st.container(border=True):
            st.markdown(f"**{item.description}** — {item.assignee.display_name or 'unassigned'}")
            for evidence in item.evidence[:1]:
                st.caption(f"said: “{evidence.quote.strip()}”")
            entry, action = st.columns([3, 1])
            answer = entry.text_input(
                "Due", key=f"due_{item.id}", label_visibility="collapsed",
                placeholder="next Friday, or 2026-09-01",
            )
            if action.button("Set", key=f"set_{item.id}"):
                if not answer.strip():
                    st.warning("Type a date first.")
                else:
                    resolved = resolve_deadline(answer, today)
                    if resolved.value is None:
                        # Refused rather than guessed: a wrong date silently
                        # produces a reminder on the wrong day.
                        st.warning(f"Could not read “{answer}” as a date. "
                                   "Try 'next Friday' or 2026-09-01.")
                    else:
                        item.record_deadline_change(
                            resolved.value, on=today, source="triage", note=answer
                        )
                        item.deadline.resolved = resolved.value
                        item.deadline.raw_text = answer
                        item.deadline.method = (
                            resolved.method
                            if resolved.method is not DeadlineResolution.NONE
                            else DeadlineResolution.EXPLICIT
                        )
                        project.save_ledger()
                        st.success(f"Due {resolved.value.isoformat()}")
                        st.rerun()


def _google_ready() -> tuple[bool, str]:
    from quorum.integrations import credentials_status

    status = credentials_status()
    return status.ready, status.message


def _calendar_panel(project) -> None:
    from quorum.config import get_settings
    from quorum.execution import CalendarConfig, CalendarSync, ChangeKind

    st.markdown("### Calendar")
    settings = get_settings()
    ready, message = _google_ready()

    service = None
    if ready:
        try:
            from quorum.integrations import get_calendar_service

            service = get_calendar_service()
        except Exception as exc:  # noqa: BLE001
            st.warning(str(exc))

    config = CalendarConfig(
        calendar_id=settings.calendar_id,
        reminder_days=settings.reminder_days(),
        reminder_hour=settings.reminder_hour,
    )
    sync = CalendarSync(service, config)
    plan = sync.plan(project.ledger, date.today())

    leads = ", ".join(f"{d} day" for d in config.reminder_days)
    st.caption(f"Reminders {leads} before, at {config.reminder_hour:02d}:00.")

    if plan.is_empty and not plan.undated:
        st.success("Your calendar already matches this project.")
        return

    if plan.writes:
        for change in plan.writes:
            st.write(f"**{change.kind.value}** {change.due or '—'} · {change.title}")
    if plan.undated:
        st.caption(f"{len(plan.undated)} have no date and cannot be scheduled.")

    if not ready:
        st.info(f"Connect Google to write these: {message}. "
                "Run `quorum auth` once in the terminal.")
        return
    if not plan.writes:
        return

    if st.button(f"Add {len(plan.writes)} to my calendar", key="sync_calendar",
                 use_container_width=True):
        _apply_calendar(project, sync, plan)


def _apply_calendar(project, sync, plan) -> None:
    from quorum.config import get_settings
    from quorum.execution import ApprovalGate
    from quorum.execution.calendar import CalendarTransport
    from quorum.tracking import ActionType, PlannedAction

    gate = ApprovalGate(require_approval=get_settings().require_approval)
    pending = gate.propose(
        PlannedAction(f"calendar:{project.meta.id}", ActionType.SCHEDULE,
                      plan.summary_line()),
        f"Calendar sync: {plan.summary_line()}", body=plan.render(),
    )
    transport = CalendarTransport(sync, plan)
    try:
        gate.execute(pending.id, gate.approve(pending.id), transport)
    except Exception as exc:  # noqa: BLE001
        _toast_error(exc, "Writing to your calendar")
        return
    result = transport.result
    st.success(f"{result.created} added, {result.updated} updated, "
               f"{result.deleted} removed.")


def _drafts_panel(project) -> None:
    from quorum.execution import find_communications

    st.markdown("### Emails promised")
    promised = find_communications(project.ledger.open_commitments())
    if not promised:
        st.caption("Nobody promised to email anyone in this project.")
        return

    st.write(f"{len(promised)} commitment(s) are messages someone said they would send.")
    if st.button("Write them", key="write_drafts", use_container_width=True):
        _write_drafts(project, promised)

    drafts = st.session_state.get("drafts", [])
    if not drafts:
        return

    for draft in drafts:
        with st.container(border=True):
            st.caption(f"To: {draft.to_email or '⚠ no address on the roster'}")
            st.markdown(f"**{draft.subject}**")
            st.text(draft.body)

    ready, message = _google_ready()
    addressed = [d for d in drafts if d.addressed]
    if not ready:
        st.info(f"Connect Google to put these in Gmail: {message}")
        return
    if addressed and st.button(f"Put {len(addressed)} in my Gmail drafts",
                               key="push_drafts", use_container_width=True):
        _push_drafts(project, addressed)
    st.caption("They go to your Drafts folder. Nothing is sent — you send them.")


def _write_drafts(project, promised) -> None:
    from quorum.execution import DraftWriter

    writer = DraftWriter()
    written = []
    with st.spinner("Writing…"):
        for commitment in promised:
            draft = writer.write(commitment, project)
            if draft is not None:
                written.append(draft)
    st.session_state["drafts"] = written
    st.rerun()


def _push_drafts(project, drafts) -> None:
    from quorum.config import get_settings
    from quorum.execution import ApprovalGate, GmailDrafts
    from quorum.execution.mail import GmailDraftTransport
    from quorum.integrations import get_gmail_service
    from quorum.tracking import ActionType, PlannedAction

    try:
        service = get_gmail_service()
    except Exception as exc:  # noqa: BLE001
        _toast_error(exc, "Reaching Gmail")
        return

    gate = ApprovalGate(require_approval=get_settings().require_approval)
    pending = gate.propose(
        PlannedAction(f"drafts:{project.meta.id}", ActionType.SCHEDULE,
                      f"{len(drafts)} draft(s)"),
        f"Create {len(drafts)} Gmail draft(s)",
        body="\n\n---\n\n".join(d.render() for d in drafts),
    )
    transport = GmailDraftTransport(GmailDrafts(service), drafts)
    try:
        gate.execute(pending.id, gate.approve(pending.id), transport)
    except Exception as exc:  # noqa: BLE001
        _toast_error(exc, "Creating drafts")
        return
    st.success(f"{transport.result.created} draft(s) in Gmail. "
               "Open Gmail, read them, send the ones you want.")


def _week_panel(project) -> None:
    from quorum.tracking import build_report

    st.markdown("### What changed this week")
    days = st.slider("Look back", 7, 60, 7, key="week_days")
    report = build_report(project.ledger, project.meta.name, until=date.today(), days=days)
    if report.is_quiet:
        st.caption("Nothing moved — no commitment slipped, reversed or lapsed.")
        return
    st.markdown(report.as_markdown())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    project_id = sidebar()
    session = _session()

    labels = ["Record", "Library", "Ask", "To-dos"]
    if session is not None and session.live:
        # A live recording outranks whatever you were looking at, and the timer
        # only redraws on the tab it lives on.
        labels = ["● Recording", "Library", "Ask", "To-dos"]

    record, library, ask, todos = st.tabs(labels)
    with record:
        record_tab(project_id)
    with library:
        library_tab(project_id)
    with ask:
        ask_tab(project_id)
    with todos:
        todos_tab(project_id)


main()
