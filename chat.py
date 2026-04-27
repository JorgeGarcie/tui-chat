import json
from ollama import Client
from config import OLLAMA_HOST, MODEL, SYSTEM_PROMPT
from tools import TOOL_DEFINITIONS, execute_tool


client = Client(host=OLLAMA_HOST)


def stream_response(messages: list):
    """Stream a chat response from Ollama. Yields dicts with type and content.

    Yields:
        {"type": "text", "content": "..."} — text token
        {"type": "tool_call", "name": "...", "args": {...}} — tool request
        {"type": "done"} — stream finished
    """
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    try:
        stream = client.chat(
            model=MODEL,
            messages=full_messages,
            tools=TOOL_DEFINITIONS,
            stream=True,
        )
    except Exception as e:
        yield {"type": "text", "content": f"[Connection error: {e}]"}
        yield {"type": "done"}
        return

    for chunk in stream:
        msg = chunk.get("message", {})

        # Text content
        content = msg.get("content", "")
        if content:
            yield {"type": "text", "content": content}

        # Tool calls
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                yield {
                    "type": "tool_call",
                    "name": fn.get("name", ""),
                    "args": fn.get("arguments", {}),
                }

    yield {"type": "done"}


def tool_result_message(name: str, result: str) -> dict:
    """Create a tool result message to feed back into the conversation."""
    return {"role": "tool", "content": result}
