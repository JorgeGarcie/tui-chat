import os
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Static, RichLog
from textual.containers import VerticalScroll
from textual.binding import Binding
from textual import work
from rich.markdown import Markdown

from config import MODEL, OLLAMA_HOST
from chat import stream_response, tool_result_message
from tools import execute_tool


class ChatApp(App):
    CSS = """
    #chat-log {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }
    #input-box {
        dock: bottom;
        margin-top: 1;
    }
    .user-msg {
        color: $accent;
        margin-bottom: 1;
    }
    .ai-msg {
        margin-bottom: 1;
    }
    .tool-msg {
        color: $warning;
        margin-bottom: 1;
    }
    .system-msg {
        color: $text-muted;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
    ]

    def __init__(self):
        super().__init__()
        self.messages = []
        self.pending_tool = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="chat-log", wrap=True, markup=True)
        yield Input(placeholder="Type a message... (Ctrl+C to quit)", id="input-box")
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"tui-chat | {MODEL}"
        self.sub_title = f"{OLLAMA_HOST} | {os.getcwd()}"
        self.query_one("#input-box").focus()
        log = self.query_one("#chat-log", RichLog)
        log.write("[dim]Connected to " + MODEL + ". Start chatting.[/dim]")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        input_box = self.query_one("#input-box", Input)
        input_box.value = ""

        # Handle tool confirmation
        if self.pending_tool:
            await self._handle_tool_confirmation(text)
            return

        # Regular message
        log = self.query_one("#chat-log", RichLog)
        log.write(f"[bold cyan]You:[/bold cyan] {text}")

        self.messages.append({"role": "user", "content": text})
        self._stream_response()

    @work(thread=True)
    def _stream_response(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        full_text = ""
        tool_calls = []

        log.write("[bold green]AI:[/bold green] ", end="")

        for chunk in stream_response(self.messages):
            if chunk["type"] == "text":
                full_text += chunk["content"]
                log.write(chunk["content"], end="")
            elif chunk["type"] == "tool_call":
                tool_calls.append(chunk)
            elif chunk["type"] == "done":
                break

        if full_text:
            log.write("")  # newline after streaming
            self.messages.append({"role": "assistant", "content": full_text})

        if tool_calls:
            tc = tool_calls[0]
            self.pending_tool = tc
            args_str = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
            log.write("")
            log.write(
                f"[bold yellow]Tool: {tc['name']}({args_str})[/bold yellow]"
            )
            log.write("[yellow]Execute? (y/n)[/yellow] ", end="")
            self.app.call_from_thread(
                self.query_one("#input-box", Input).focus
            )

    async def _handle_tool_confirmation(self, text: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        tc = self.pending_tool
        self.pending_tool = None

        if text.lower() in ("y", "yes", ""):
            log.write("[yellow]Running...[/yellow]")
            result = execute_tool(tc["name"], tc["args"])
            log.write(f"[dim]{result}[/dim]")

            # Feed result back to model
            self.messages.append(
                {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": tc["name"], "arguments": tc["args"]}}
                ]}
            )
            self.messages.append(tool_result_message(tc["name"], result))
            self._stream_response()
        else:
            log.write("[dim]Skipped.[/dim]")

    def action_clear(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.clear()
        self.messages.clear()
        log.write("[dim]Chat cleared.[/dim]")


if __name__ == "__main__":
    app = ChatApp()
    app.run()
