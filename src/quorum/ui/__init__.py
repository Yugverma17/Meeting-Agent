"""The local web interface.

Runs on this machine, not on a server somewhere: recording system audio needs
WASAPI loopback access, which nothing remote can have. The browser is the face;
the Python doing the work is the same process, on the same laptop.

`app.py` is a Streamlit script and executes on import, so it is deliberately not
imported here - `quorum ui` hands its path to `streamlit run`.
"""

from quorum.ui.session import AlreadyRecording, RecordingSession

__all__ = ["AlreadyRecording", "RecordingSession"]
