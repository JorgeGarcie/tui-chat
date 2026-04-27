import os
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Static
from textual.containers import VerticalScroll
from textual.binding import Binding
from textual.message import Message
from textual import work

from config import MODEL, OLLAMA_HOST
from chat import stream_response, tool_result_message
from tools import execute_tool


class StreamingMessage(Static):
    """Widget that displays the AI response as it streams in."""
    pass


class ChatApp(App):
    CSS = """
    #chat-scroll {
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
    .streaming {
        color: $success;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
    ]

    # Custom messages for thread -> main loop communication
    class TokenReceived(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class StreamDone(Message):
        def __init__(self, display_text: str, raw_text: str, tool_calls: list) -> None:
            super().__init__()
            self.display_text = display_text
            self.raw_text = raw_text
            self.tool_calls = tool_calls

    def __init__(self):
        super().__init__()
        self.messages = []
        self.pending_tool = None
        self._streaming_widget = None
        self._streaming_text = ""

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="chat-scroll"):
            yield Static(
                "[dim]Connected to " + MODEL + ". Start chatting.[/dim]",
                classes="system-msg",
            )
        yield Input(
            placeholder="Type a message... (Ctrl+C to quit)", id="input-box"
        )
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"tui-chat | {MODEL}"
        self.sub_title = f"{OLLAMA_HOST} | {os.getcwd()}"
        self.query_one("#input-box").focus()

    def _add_message(self, markup: str, css_class: str) -> Static:
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        widget = Static(markup, classes=css_class)
        scroll.mount(widget)
        scroll.scroll_end(animate=False)
        return widget

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        input_box = self.query_one("#input-box", Input)
        input_box.value = ""

        if self.pending_tool:
            await self._handle_tool_confirmation(text)
            return

        self._add_message(f"[bold cyan]You:[/bold cyan] {text}", "user-msg")
        self.messages.append({"role": "user", "content": text})

        # Create streaming widget before starting the worker
        self._streaming_text = ""
        self._streaming_widget = StreamingMessage(
            "[bold green]AI:[/bold green] [dim]...[/dim]", classes="streaming"
        )
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        scroll.mount(self._streaming_widget)
        scroll.scroll_end(animate=False)

        self._stream_response()

    def on_chat_app_token_received(self, event: TokenReceived) -> None:
        """Handle a new token from the streaming thread."""
        if self._streaming_widget is None:
            return
        self._streaming_text += event.text
        self._streaming_widget.update(
            f"[bold green]AI:[/bold green] {self._streaming_text}"
        )
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        scroll.scroll_end(animate=False)

    def on_chat_app_stream_done(self, event: StreamDone) -> None:
        """Handle stream completion."""
        if self._streaming_widget:
            if event.display_text.strip():
                self._streaming_widget.update(
                    f"[bold green]AI:[/bold green] {event.display_text}"
                )
                self._streaming_widget.classes = "ai-msg"
            else:
                self._streaming_widget.remove()
            self._streaming_widget = None
            self._streaming_text = ""

        if event.raw_text:
            self.messages.append(
                {"role": "assistant", "content": event.raw_text}
            )

        if event.tool_calls:
            tc = event.tool_calls[0]
            self.pending_tool = tc
            args_str = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
            self._add_message(
                f"[bold yellow]Tool: {tc['name']}({args_str})\n"
                f"Execute? (y/n)[/bold yellow]",
                "tool-msg",
            )
            self.query_one("#input-box", Input).focus()

    @work(thread=True)
    def _stream_response(self) -> None:
        display_text = ""
        raw_text = ""
        tool_calls = []

        for chunk in stream_response(self.messages):
            if chunk["type"] == "text":
                display_text += chunk["content"]
                self.post_message(self.TokenReceived(chunk["content"]))
            elif chunk["type"] == "tool_call":
                tool_calls.append(chunk)
            elif chunk["type"] == "assistant_raw":
                raw_text = chunk["content"]
            elif chunk["type"] == "done":
                break

        self.post_message(self.StreamDone(display_text, raw_text, tool_calls))

    async def _handle_tool_confirmation(self, text: str) -> None:
        tc = self.pending_tool
        self.pending_tool = None

        if text.lower() in ("y", "yes", ""):
            self._add_message("[yellow]Running...[/yellow]", "system-msg")
            result = execute_tool(tc["name"], tc["args"])
            self._add_message(f"[dim]{result}[/dim]", "system-msg")

            self.messages.append(tool_result_message(tc["name"], result))

            # Set up streaming widget for the follow-up response
            self._streaming_text = ""
            self._streaming_widget = StreamingMessage(
                "[bold green]AI:[/bold green] [dim]...[/dim]",
                classes="streaming",
            )
            scroll = self.query_one("#chat-scroll", VerticalScroll)
            scroll.mount(self._streaming_widget)
            scroll.scroll_end(animate=False)

            self._stream_response()
        else:
            self._add_message("[dim]Skipped.[/dim]", "system-msg")

    def action_clear(self) -> None:
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        scroll.remove_children()
        self.messages.clear()
        scroll.mount(
            Static("[dim]Chat cleared.[/dim]", classes="system-msg")
        )


if __name__ == "__main__":
    app = ChatApp()
    app.run()
