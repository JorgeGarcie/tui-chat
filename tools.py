import ast
import os
import re
import subprocess


_read_paths: set[str] = set()


def clear_read_tracker() -> None:
    _read_paths.clear()


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command on the user's machine and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file on the user's machine.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file",
                    }
                },
                "required": ["path"],
            },
        },
    },
]


SAFE_TOOLS = {"read_file", "outline", "grep"}

SAFE_COMMAND_PREFIXES = (
    "ls", "pwd", "whoami", "date", "echo", "printf",
    "cat", "head", "tail",
    "wc", "du", "df", "tree", "stat", "file", "which",
    "ps",
    "git status", "git log", "git diff", "git show",
    "git branch", "git remote",
)

DANGEROUS_PATTERNS = (
    r"\brm\b", r"\bmv\b", r"\bdd\b", r"\bmkfs\b",
    r"\bchmod\b", r"\bchown\b",
    r"\bsudo\b", r"\bsu\s",
    r">\s", r">>\s",  # any output redirect
    r"\bgit\s+(push|reset|clean|rebase|checkout\s+--|filter-branch)\b",
    r"\bkill\b", r"\bpkill\b",
    r"\bapt(?:-get)?\b\s+(install|remove|purge|upgrade)",
    r"\bpip\b\s+(install|uninstall)",
    r"\bnpm\b\s+(install|uninstall)",
    r"\bcurl\b.*\|",
    r"\bwget\b.*\|",
    r"&&|;|\|\|",  # any chaining — be conservative
)


def is_safe(name: str, args: dict) -> bool:
    """Auto-allow read-only tools and a small whitelist of read-only commands."""
    if name in SAFE_TOOLS:
        return True
    if name == "run_command":
        return _is_safe_command(args.get("command", ""))
    return False


def _is_safe_command(cmd: str) -> bool:
    cmd = cmd.strip()
    if not cmd:
        return False
    for pat in DANGEROUS_PATTERNS:
        if re.search(pat, cmd):
            return False
    return any(
        cmd == p or cmd.startswith(p + " ")
        for p in SAFE_COMMAND_PREFIXES
    )


def execute_tool(name: str, args: dict) -> str:
    if name == "run_command":
        return _run_command(args["command"])
    elif name == "read_file":
        return _read_file(
            args["path"],
            args.get("start_line"),
            args.get("end_line"),
        )
    elif name == "outline":
        return _outline(args["path"])
    elif name == "grep":
        return _grep(args["pattern"], args["path"])
    elif name == "edit_file":
        return _edit_file(args["path"], args["old"], args["new"])
    elif name == "write_file":
        return _write_file(args["path"], args["content"])
    return f"Unknown tool: {name}"


def _run_command(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "[command timed out after 30s]"
    except Exception as e:
        return f"[error: {e}]"


def _read_file(path: str, start_line=None, end_line=None) -> str:
    try:
        with open(path) as f:
            lines = f.readlines()
    except Exception as e:
        return f"[error: {e}]"

    _read_paths.add(os.path.abspath(path))
    start = max(0, int(start_line or 1) - 1)
    end = int(end_line) if end_line else len(lines)
    numbered = [f"L{i}: {ln.rstrip()}" for i, ln in enumerate(lines[start:end], start=start + 1)]
    return "\n".join(numbered)[:50_000]  # cap reads at 50KB


def _outline(path: str) -> str:
    """Structural view of a Python file: imports, classes, def signatures."""
    if not path.endswith(".py"):
        return f"[outline supports .py only — use read_file for {path}]"
    try:
        with open(path) as f:
            source = f.read()
        tree = ast.parse(source)
    except Exception as e:
        return f"[error: {e}]"

    out = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            names = ", ".join(a.name for a in node.names)
            out.append(f"L{node.lineno}: import {names}")
        elif isinstance(node, ast.ImportFrom):
            names = ", ".join(a.name for a in node.names)
            out.append(f"L{node.lineno}: from {node.module} import {names}")
        elif isinstance(node, ast.ClassDef):
            out.append(f"L{node.lineno}: class {node.name}:")
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    kw = "async def" if isinstance(child, ast.AsyncFunctionDef) else "def"
                    out.append(f"  L{child.lineno}:   {kw} {child.name}(...)")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kw = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            out.append(f"L{node.lineno}: {kw} {node.name}(...)")
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    out.append(f"L{node.lineno}: {t.id} = ...")
    return "\n".join(out) if out else "(empty)"


def _edit_file(path: str, old: str, new: str) -> str:
    if os.path.abspath(path) not in _read_paths:
        return (
            f"[error: must read_file {path} before edit_file. "
            f"Call read_file first to verify the content you want to change.]"
        )
    try:
        with open(path) as f:
            content = f.read()
    except Exception as e:
        return f"[error: {e}]"
    count = content.count(old)
    if count == 0:
        return (
            f"[error: `old` string not found in {path}. "
            f"Do NOT guess again — call read_file on the relevant lines first, "
            f"then retry edit_file with the literal content you read.]"
        )
    if count > 1:
        return (
            f"[error: `old` matches {count} times in {path}. "
            f"Add surrounding lines to `old` (and matching context to `new`) "
            f"until it occurs exactly once.]"
        )
    try:
        with open(path, "w") as f:
            f.write(content.replace(old, new, 1))
    except Exception as e:
        return f"[error: {e}]"
    return f"Edited {path} (1 replacement)."


def _write_file(path: str, content: str) -> str:
    if os.path.exists(path) and os.path.abspath(path) not in _read_paths:
        return (
            f"[error: must read_file {path} before write_file. "
            f"Call read_file first to verify the content you want to change.]"
        )
    try:
        with open(path, "w") as f:
            n = f.write(content)
    except Exception as e:
        return f"[error: {e}]"
    return f"Wrote {n} bytes to {path}."


def _grep(pattern: str, path: str) -> str:
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"[regex error: {e}]"
    try:
        with open(path) as f:
            lines = f.readlines()
    except Exception as e:
        return f"[error: {e}]"

    hits = [f"L{i}: {ln.rstrip()}" for i, ln in enumerate(lines, 1) if regex.search(ln)]
    if not hits:
        return f"(no matches for /{pattern}/ in {path})"
    if len(hits) > 50:
        return "\n".join(hits[:50]) + f"\n… ({len(hits) - 50} more)"
    return "\n".join(hits)
