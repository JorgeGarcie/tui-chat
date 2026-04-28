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
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def _safe_split_multi(buf: str, prefixes) -> tuple[str, str]:
    """Hold back the longest partial-fence suffix against any of `prefixes`."""
    max_hold = 0
    for prefix in prefixes:
        for i in range(min(len(prefix) - 1, len(buf)), 0, -1):
            if buf.endswith(prefix[:i]):
                max_hold = max(max_hold, i)
                break
    if max_hold:
        return buf[:-max_hold], buf[-max_hold:]
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
    """Stream a chat response, parsing ```tool_call``` and <think> blocks.

    Yields:
        {"type": "text", "content": "..."}                  — visible token
        {"type": "thinking", "content": "..."}              — chunk of reasoning (live)
        {"type": "thinking_end"}                            — end of </think> block
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
    pending = ""       # text-state accumulator
    tool_buf = ""      # tool-state JSON accumulator
    think_buf = ""     # think-state partial-close accumulator
    state = "text"     # one of: "text", "tool", "think"

    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if not content:
            continue
        raw += content

        cursor = content
        while cursor:
            if state == "tool":
                combined = tool_buf + cursor
                end = combined.find(TOOL_CLOSE)
                if end == -1:
                    tool_buf = combined
                    cursor = ""
                else:
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
                    state = "text"

            elif state == "think":
                combined = think_buf + cursor
                end = combined.find(THINK_CLOSE)
                if end == -1:
                    safe, hold = _safe_split_multi(combined, [THINK_CLOSE])
                    if safe:
                        yield {"type": "thinking", "content": safe}
                    think_buf = hold
                    cursor = ""
                else:
                    if end > 0:
                        yield {"type": "thinking", "content": combined[:end]}
                    yield {"type": "thinking_end"}
                    cursor = combined[end + len(THINK_CLOSE):]
                    think_buf = ""
                    state = "text"

            else:  # state == "text"
                pending += cursor
                cursor = ""
                tool_idx = pending.find(TOOL_OPEN)
                think_idx = pending.find(THINK_OPEN)

                candidates = []
                if tool_idx != -1:
                    candidates.append((tool_idx, "tool", len(TOOL_OPEN)))
                if think_idx != -1:
                    candidates.append((think_idx, "think", len(THINK_OPEN)))

                if not candidates:
                    safe, hold = _safe_split_multi(pending, [TOOL_OPEN, THINK_OPEN])
                    if safe:
                        yield {"type": "text", "content": safe}
                    pending = hold
                else:
                    idx, kind, open_len = min(candidates)
                    if idx > 0:
                        yield {"type": "text", "content": pending[:idx]}
                    cursor = pending[idx + open_len:]
                    pending = ""
                    state = kind

    if pending:
        yield {"type": "text", "content": pending}
    if state == "tool" and tool_buf:
        yield {"type": "text", "content": f"\n[unterminated tool_call: {tool_buf}]\n"}
    if state == "think":
        if think_buf:
            yield {"type": "thinking", "content": think_buf}
        yield {"type": "thinking_end"}

    yield {"type": "assistant_raw", "content": raw}
    yield {"type": "done"}


def tool_result_message(name: str, result: str) -> dict:
    """Feed a tool result back as a user message wrapped in a <tool_result> tag."""
    return {
        "role": "user",
        "content": f'<tool_result name="{name}">\n{result}\n</tool_result>',
    }
