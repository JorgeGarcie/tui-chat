import json
from ollama import Client
from config import OLLAMA_HOST, MODEL, SYSTEM_PROMPT


client = Client(host=OLLAMA_HOST)

TOOL_OPEN = "```tool_call"
TOOL_CLOSE = "```"


def _safe_split(buf: str) -> tuple[str, str]:
    """Return (yield_now, hold_back) so we never emit a partial '```tool_call' fence."""
    for i in range(min(len(TOOL_OPEN) - 1, len(buf)), 0, -1):
        if buf.endswith(TOOL_OPEN[:i]):
            return buf[:-i], buf[-i:]
    return buf, ""


def stream_response(messages: list):
    """Stream a chat response from Ollama, parsing <tool_call> tags out of the text.

    Yields:
        {"type": "text", "content": "..."}                  — visible token
        {"type": "tool_call", "name": "...", "args": {...}} — parsed tool request
        {"type": "assistant_raw", "content": "..."}         — full raw output (for history)
        {"type": "done"}
    """
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    try:
        stream = client.chat(model=MODEL, messages=full_messages, stream=True)
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
