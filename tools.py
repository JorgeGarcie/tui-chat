import ast
import re
import subprocess


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

    if start_line is None and end_line is None:
        return "".join(lines)[:50_000]  # cap full reads at 50KB
    start = max(0, int(start_line or 1) - 1)
    end = int(end_line) if end_line else len(lines)
    return "".join(lines[start:end])[:50_000]


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
