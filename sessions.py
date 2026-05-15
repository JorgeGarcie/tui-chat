import json
from pathlib import Path

SESSIONS_DIR = Path.home() / ".tui-chat" / "sessions"


def _ensure_dir():
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def save_session(session_id: str, model: str, messages: list) -> Path:
    _ensure_dir()
    path = SESSIONS_DIR / f"{session_id}.json"
    path.write_text(json.dumps({
        "id": session_id,
        "model": model,
        "messages": messages,
    }, indent=2))
    return path


def list_sessions() -> list:
    """Return session metadata, newest first."""
    _ensure_dir()
    out = []
    for path in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        msgs = data.get("messages", [])
        first_user = next(
            (m["content"] for m in msgs if m.get("role") == "user"
             and not m.get("content", "").startswith("<tool_result")),
            "(empty)",
        )
        out.append({
            "id": data.get("id", path.stem),
            "model": data.get("model", "?"),
            "preview": first_user.replace("\n", " ")[:50],
            "count": len(msgs),
        })
    return out


def load_session(session_id: str) -> dict | None:
    path = SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
