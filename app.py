import os
import subprocess
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import (
    Collapsible,
    Footer,
    Header,
    Label,
    Markdown,
    OptionList,
    Static,
    TextArea,
)


PHOSPHOR_THEME = Theme(
    name="phosphor",
    primary="#00ff9c",      # neon green — borders, accents
    secondary="#00ffff",    # cyan — AI text
    accent="#39ff14",       # brighter green — user prompt
    foreground="#00ff9c",
    background="#000000",
    surface="#0a0a0a",
    panel="#0a0a0a",
    success="#00ff9c",
    warning="#ffcc00",
    error="#ff3366",
    boost="#00ffff",
    dark=True,
    variables={
        "block-cursor-text-style": "none",
        "footer-key-foreground": "#007a4d",
        "footer-description-foreground": "#005533",
    },
)


USER_PROMPT = ">"

from config import MODEL, OLLAMA_HOST
from chat import stream_response, tool_result_message, ensure_ollama_running, list_models
from tools import execute_tool


class ChatTextArea(TextArea):
    """TextArea that submits on Enter, lets Shift+Enter / paste insert newlines."""

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self.post_message(self.Submitted(self.text))


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
    """Animated Braille spinner that becomes the AI response on first token."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label: str = "thinking", **kwargs) -> None:
        super().__init__(f"{self.FRAMES[0]} {label}", **kwargs)
        self._label = label
        self._frame_idx = 0
        self._timer = None

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.1, self._tick)

    def _tick(self) -> None:
        self._frame_idx = (self._frame_idx + 1) % len(self.FRAMES)
        self.update(f"{self.FRAMES[self._frame_idx]} {self._label}")

    def stop_spinning(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None


class ChatApp(App):
    CSS = """
    Screen {
        background: $background;
    }
    #chat-scroll {
        height: 1fr;
        background: $background;
        padding: 1 2;
        scrollbar-size-vertical: 0;
    }
    #input-box {
        dock: bottom;
        margin-top: 1;
        height: auto;
        max-height: 8;
        border: none;
        border-top: heavy $primary;
        border-bottom: heavy $primary;
        background: $background;
        color: $accent;
    }
    #input-box > .text-area--cursor-line {
        background: $background;
    }
    #input-box > .text-area--cursor {
        background: $primary;
        color: $background;
    }
    #input-box > .text-area--selection {
        background: $primary 30%;
    }
    .user-msg {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }
    .tool-msg {
        color: $warning;
        margin-bottom: 1;
    }
    .system-msg {
        color: $foreground 50%;
        margin-bottom: 1;
    }
    .streaming {
        color: $secondary;
        margin-bottom: 1;
    }
    Markdown {
        margin-bottom: 1;
        background: $background;
        color: $secondary;
        padding: 0;
    }
    MarkdownFence {
        background: $surface;
        border-left: thick $primary 50%;
    }
    Collapsible {
        background: $background;
        border: none;
        margin-bottom: 1;
        padding: 0;
    }
    CollapsibleTitle {
        color: $foreground 70%;
        background: $background;
    }
    CollapsibleTitle:hover {
        color: $primary;
        background: $background;
    }
    Collapsible > Contents {
        padding: 0 0 0 2;
        background: $background;
        color: $foreground 60%;
    }
    .thinking-block > CollapsibleTitle {
        color: $foreground 50%;
        text-style: italic;
    }
    .thinking-block > Contents {
        color: $foreground 45%;
        text-style: italic;
    }
    Header {
        background: $background;
    }
    Footer {
        background: $background;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "quit"),
        Binding("ctrl+l", "clear", "clear"),
        Binding("ctrl+g", "cancel", "stop"),
    ]

    ENABLE_COMMAND_PALETTE = False

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

    class ThinkingChunk(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class ThinkingEnd(Message):
        pass

    def __init__(self):
        super().__init__()
        self.messages = []
        self.pending_tool = None
        self._streaming_widget = None
        self._streaming_text = ""
        self.current_model = MODEL
        self._cancel_stream = False
        self._thinking_widget = None
        self._thinking_static = None
        self._thinking_text = ""
        self._thinking_active = False

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="chat-scroll"):
            yield Static(
                "[dim]Connected to " + MODEL + ". Start chatting.[/dim]",
                classes="system-msg",
            )
        text_area = ChatTextArea(id="input-box")
        text_area.border_subtitle = "Enter to send · Shift+Enter for newline"
        yield text_area
        yield Footer()

    def on_mount(self) -> None:
        self.register_theme(PHOSPHOR_THEME)
        self.theme = "phosphor"
        self._update_title()
        self.sub_title = ""
        self.query_one("#input-box").focus()

    def _update_title(self) -> None:
        self.title = f"tui-chat | {self.current_model} | ● connected"

    def _add_message(self, markup: str, css_class: str) -> Static:
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        widget = Static(markup, classes=css_class)
        scroll.mount(widget)
        scroll.scroll_end(animate=False)
        return widget

    def _show_tool_result(self, name: str, args: dict, result: str) -> None:
        """Render tool output: short results inline, long ones collapsed."""
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        lines = result.splitlines()
        n_lines = len(lines)

        if n_lines <= 6 and len(result) <= 400:
            scroll.mount(Static(result, classes="system-msg"))
        else:
            args_summary = ", ".join(f"{k}={v!r}" for k, v in args.items())
            title = f"✓ {name}({args_summary}) — {n_lines} lines"
            scroll.mount(
                Collapsible(Static(result), title=title, collapsed=True)
            )
        scroll.scroll_end(animate=False)

    async def on_chat_text_area_submitted(
        self, event: ChatTextArea.Submitted
    ) -> None:
        text = event.value.strip()
        if not text:
            return

        input_box = self.query_one("#input-box", ChatTextArea)
        input_box.text = ""

        if self.pending_tool:
            await self._handle_tool_confirmation(text)
            return

        if text.startswith("/"):
            self._handle_command(text)
            return

        self._add_message(f"{USER_PROMPT} {text}", "user-msg")
        self.messages.append({"role": "user", "content": text})

        # Create streaming widget before starting the worker
        self._streaming_text = ""
        self._streaming_widget = StreamingMessage("thinking", classes="streaming")
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        scroll.mount(self._streaming_widget)
        scroll.scroll_end(animate=False)

        self._stream_response()

    def on_chat_app_thinking_chunk(self, event: ThinkingChunk) -> None:
        if not self._thinking_active:
            scroll = self.query_one("#chat-scroll", VerticalScroll)
            self._thinking_text = ""
            self._thinking_static = Static("")
            self._thinking_widget = Collapsible(
                self._thinking_static,
                title="thinking…",
                collapsed=False,
            )
            self._thinking_widget.add_class("thinking-block")
            if (
                self._streaming_widget is not None
                and self._streaming_widget.parent is not None
            ):
                scroll.mount(self._thinking_widget, before=self._streaming_widget)
            else:
                scroll.mount(self._thinking_widget)
            self._thinking_active = True
        self._thinking_text += event.text
        self._thinking_static.update(self._thinking_text)
        self.query_one("#chat-scroll", VerticalScroll).scroll_end(animate=False)

    def on_chat_app_thinking_end(self, event: ThinkingEnd) -> None:
        if self._thinking_widget is not None:
            n_lines = len(self._thinking_text.splitlines()) or 1
            try:
                self._thinking_widget.title = f"thought ({n_lines} lines)"
            except Exception:
                pass
            self._thinking_widget.collapsed = True
        self._thinking_active = False

    def on_chat_app_token_received(self, event: TokenReceived) -> None:
        """Handle a new token from the streaming thread."""
        if self._streaming_widget is None:
            return
        if not self._streaming_text:
            self._streaming_widget.stop_spinning()
        self._streaming_text += event.text
        self._streaming_widget.update(self._streaming_text)
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
                scroll.mount(Markdown(event.display_text))
                scroll.scroll_end(animate=False)

        if event.raw_text:
            self.messages.append(
                {"role": "assistant", "content": event.raw_text}
            )

        # reset thinking refs so the next turn creates a fresh widget
        self._thinking_widget = None
        self._thinking_static = None
        self._thinking_text = ""
        self._thinking_active = False

        if event.tool_calls:
            tc = event.tool_calls[0]
            self.pending_tool = tc
            args_str = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
            self._add_message(
                f"[bold yellow]Tool: {tc['name']}({args_str})\n"
                f"Execute? (y/n)[/bold yellow]",
                "tool-msg",
            )
            self.query_one("#input-box", ChatTextArea).focus()

    @work(thread=True)
    def _stream_response(self) -> None:
        self._cancel_stream = False
        display_text = ""
        raw_text = ""
        tool_calls = []
        cancelled = False

        for chunk in stream_response(self.messages, self.current_model):
            if self._cancel_stream:
                cancelled = True
                break
            ctype = chunk["type"]
            if ctype == "text":
                display_text += chunk["content"]
                self.post_message(self.TokenReceived(chunk["content"]))
            elif ctype == "thinking":
                self.post_message(self.ThinkingChunk(chunk["content"]))
            elif ctype == "thinking_end":
                self.post_message(self.ThinkingEnd())
            elif ctype == "tool_call":
                tool_calls.append(chunk)
            elif ctype == "assistant_raw":
                raw_text = chunk["content"]
            elif ctype == "done":
                break

        if cancelled:
            display_text = display_text.rstrip() + "\n\n[cancelled]"
        self.post_message(self.StreamDone(display_text, raw_text, tool_calls))

    async def _handle_tool_confirmation(self, text: str) -> None:
        tc = self.pending_tool
        self.pending_tool = None

        if text.lower() in ("y", "yes", ""):
            self._add_message("[yellow]Running...[/yellow]", "system-msg")
            result = execute_tool(tc["name"], tc["args"])
            self._show_tool_result(tc["name"], tc["args"], result)

            self.messages.append(tool_result_message(tc["name"], result))

            # Set up streaming widget for the follow-up response
            self._streaming_text = ""
            self._streaming_widget = StreamingMessage("thinking", classes="streaming")
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
                    self._update_title()
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

    def action_cancel(self) -> None:
        if self._streaming_widget is not None:
            self._cancel_stream = True

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
