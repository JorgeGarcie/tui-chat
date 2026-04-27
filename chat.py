import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from ollama import Client
from config import OLLAMA_HOST, MODEL, SYSTEM_PROMPT


client = Client(host=OLLAMA_HOST)


def _ollama_alive(host: str, timeout: float = 0.5) -> bool:
    """True iff /api/version responds with valid Ollama JSON."""
    try:
        with urllib.request.urlopen(f"{host}/api/version", timeout=timeout) as r:
            return "version" in json.loads(r.read())
    except (urllib.error.URLError, json.JSONDecodeError, OSError, ValueError):
        return False


def ensure_ollama_running(host: str, max_wait: float = 10.0):
    """Make sure ollama is reachable at `host`. Spawn it if nothing is there.

    Returns the Popen if we started it (caller must terminate on shutdown),
    or None if it was already running.

    Raises RuntimeError if a non-Ollama service holds the port, the ollama
    binary is missing, or ollama fails to come up within max_wait seconds.
    """
    if _ollama_alive(host):
        return None

    parsed = urlparse(host)
    if not parsed.hostname or not parsed.port:
        raise RuntimeError(f"OLLAMA_HOST must be http://host:port, got {host!r}")
    listen_addr = f"{parsed.hostname}:{parsed.port}"

    log_dir = Path.home() / ".tui-chat"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "ollama.log"

    try:
        log_fd = open(log_file, "ab")
        proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=log_fd,
            stderr=log_fd,
            env={**os.environ, "OLLAMA_HOST": listen_addr},
        )
    except FileNotFoundError:
        raise RuntimeError(
            "'ollama' not in PATH — install from https://ollama.com"
        )

    deadline = time.time() + max_wait
    while time.time() < deadline:
        if _ollama_alive(host):
            return proc
        if proc.poll() is not None:
            raise RuntimeError(
                f"ollama serve exited with code {proc.returncode}. "
                f"See {log_file} (likely: port {parsed.port} is held by "
                f"another process, not Ollama)"
            )
        time.sleep(0.2)

    proc.terminate()
    raise RuntimeError(
        f"ollama did not become reachable within {max_wait}s. See {log_file}"
    )

TOOL_OPEN = "```tool_call"
TOOL_CLOSE = "```"


def _safe_split(buf: str) -> tuple[str, str]:
    """Return (yield_now, hold_back) so we never emit a partial '```tool_call' fence."""
    for i in range(min(len(TOOL_OPEN) - 1, len(buf)), 0, -1):
        if buf.endswith(TOOL_OPEN[:i]):
            return buf[:-i], buf[-i:]
    return buf, ""


def list_models() -> list[str]:
    """Return a sorted list of locally available model names."""
    resp = client.list()
    raw = getattr(resp, "models", None)
    if raw is None:
        raw = resp.get("models", []) if isinstance(resp, dict) else []
    names = []
    for m in raw:
        name = getattr(m, "model", None)
        if name is None and isinstance(m, dict):
            name = m.get("name") or m.get("model")
        if name:
            names.append(name)
    return sorted(names)


def stream_response(messages: list, model: str = MODEL):
    """Stream a chat response from Ollama, parsing <tool_call> tags out of the text.

    Yields:
        {"type": "text", "content": "..."}                  — visible token
        {"type": "tool_call", "name": "...", "args": {...}} — parsed tool request
        {"type": "assistant_raw", "content": "..."}         — full raw output (for history)
        {"type": "done"}
    """
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    try:
        stream = client.chat(model=model, messages=full_messages, stream=True)
    except Exception as e:
        yield {"type": "text", "content": f"[Connection error: {e}]"}
        yield {"type": "assistant_raw", "content": ""}
        yield {"type": "done"}
        return

    raw = ""
    pending = ""        # buffer of streamed text not yet yielded as visible "text"
    tool_buf = ""       # buffer of content inside an open <tool_call>...
    in_tool = False

    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if not content:
            continue
        raw += content

        cursor = content
        while cursor:
            if in_tool:
                end = (tool_buf + cursor).find(TOOL_CLOSE)
                if end == -1:
                    tool_buf += cursor
                    cursor = ""
                else:
                    combined = tool_buf + cursor
                    json_str = combined[:end].strip()
                    try:
                        tc = json.loads(json_str)
                        yield {
                            "type": "tool_call",
                            "name": tc.get("name", ""),
                            "args": tc.get("args", tc.get("arguments", {})),
                        }
                    except json.JSONDecodeError as e:
                        yield {"type": "text", "content": f"\n[bad tool_call JSON: {e}]\n"}
                    cursor = combined[end + len(TOOL_CLOSE):]
                    tool_buf = ""
                    in_tool = False
            else:
                pending += cursor
                cursor = ""
                start = pending.find(TOOL_OPEN)
                if start != -1:
                    if start > 0:
                        yield {"type": "text", "content": pending[:start]}
                    cursor = pending[start + len(TOOL_OPEN):]
                    pending = ""
                    in_tool = True
                else:
                    safe, hold = _safe_split(pending)
                    if safe:
                        yield {"type": "text", "content": safe}
                    pending = hold

    if pending:
        yield {"type": "text", "content": pending}
    if in_tool and tool_buf:
        yield {"type": "text", "content": f"\n[unterminated tool_call: {tool_buf}]\n"}

    yield {"type": "assistant_raw", "content": raw}
    yield {"type": "done"}


def tool_result_message(name: str, result: str) -> dict:
    """Feed a tool result back as a user message wrapped in a <tool_result> tag."""
    return {
        "role": "user",
        "content": f'<tool_result name="{name}">\n{result}\n</tool_result>',
    }
