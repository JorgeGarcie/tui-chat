import os
import subprocess
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Static, Markdown, OptionList, Label
from textual.containers import VerticalScroll, Vertical
from textual.screen import ModalScreen
from textual.binding import Binding
from textual.message import Message
from textual import work

from config import MODEL, OLLAMA_HOST
from chat import stream_response, tool_result_message, ensure_ollama_running, list_models
from tools import execute_tool


class ModelPicker(ModalScreen[str]):
    """Modal for selecting an Ollama model."""

    BINDINGS = [Binding("escape", "dismiss", "Cancel")]

    DEFAULT_CSS = """
    ModelPicker {
        align: center middle;
    }
    #picker-box {
        width: 60;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }
    #picker-current {
        color: $text-muted;
        margin-bottom: 1;
    }
    #picker-options {
        height: auto;
        max-height: 20;
    }
    """

    def __init__(self, models: list[str], current: str) -> None:
        super().__init__()
        self.models = models
        self.current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-box"):
            yield Label(f"Current: {self.current}", id="picker-current")
            yield Label("Pick a model (Esc to cancel):")
            yield OptionList(*self.models, id="picker-options")

    def on_mount(self) -> None:
        self.query_one("#picker-options", OptionList).focus()

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        self.dismiss(self.models[event.option_index])


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
    .ai-label {
        color: $success;
        text-style: bold;
        margin-bottom: 0;
    }
    Markdown {
        margin-bottom: 1;
        background: transparent;
        padding: 0;
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
        self.current_model = MODEL

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
        self.title = f"tui-chat | {self.current_model}"
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

        if text.startswith("/"):
            self._handle_command(text)
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
            self._streaming_widget.remove()
            self._streaming_widget = None
            self._streaming_text = ""
            if event.display_text.strip():
                scroll = self.query_one("#chat-scroll", VerticalScroll)
                scroll.mount(Static("[bold green]AI:[/bold green]", classes="ai-label"))
                scroll.mount(Markdown(event.display_text))
                scroll.scroll_end(animate=False)

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

        for chunk in stream_response(self.messages, self.current_model):
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

    def _handle_command(self, text: str) -> None:
        cmd = text.split(maxsplit=1)[0]
        if cmd == "/model":
            try:
                models = list_models()
            except Exception as e:
                self._add_message(f"[red]Could not list models: {e}[/red]", "system-msg")
                return
            if not models:
                self._add_message("[red]No models found.[/red]", "system-msg")
                return

            def picked(name: str | None) -> None:
                if name and name != self.current_model:
                    self.current_model = name
                    self.title = f"tui-chat | {self.current_model}"
                    self._add_message(f"[dim]Switched to {name}.[/dim]", "system-msg")

            self.push_screen(ModelPicker(models, self.current_model), picked)
        elif cmd == "/help":
            self._add_message(
                "[bold]Commands:[/bold]\n"
                "  [cyan]/model[/cyan] — pick a different Ollama model\n"
                "  [cyan]/help[/cyan]  — show this message",
                "system-msg",
            )
        else:
            self._add_message(f"[red]Unknown command: {cmd}[/red]", "system-msg")

    def action_clear(self) -> None:
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        scroll.remove_children()
        self.messages.clear()
        scroll.mount(
            Static("[dim]Chat cleared.[/dim]", classes="system-msg")
        )


if __name__ == "__main__":
    import sys

    print(f"Checking Ollama at {OLLAMA_HOST}...", file=sys.stderr)
    try:
        ollama_proc = ensure_ollama_running(OLLAMA_HOST)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    if ollama_proc:
        print("Started ollama serve (logs: ~/.tui-chat/ollama.log)", file=sys.stderr)
    else:
        print("Ollama already running, attaching.", file=sys.stderr)

    try:
        ChatApp().run()
    finally:
        if ollama_proc:
            ollama_proc.terminate()
            try:
                ollama_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ollama_proc.kill()
