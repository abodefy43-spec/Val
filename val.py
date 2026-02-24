from __future__ import annotations

import sys
from pathlib import Path

from interpreter import Interpreter, LexerError, ParseError, RuntimeErrorVal


def format_error(err: Exception, source: str | None = None, filename: str = "<unknown>") -> str:
    base = str(err)
    if source is None:
        return base
    line = 1
    col = 1
    if isinstance(err, LexerError):
        line, col = err.line, err.column
    elif isinstance(err, ParseError):
        line, col = err.token.line, err.token.column
    elif isinstance(err, RuntimeErrorVal):
        line, col = err.line, err.column
    else:
        return base
    lines = source.splitlines()
    if 1 <= line <= len(lines):
        snippet = lines[line - 1]
        caret = " " * min(col - 1, len(snippet)) + "^"
        return f"{base}\n  --> {filename}:{line}:{col}\n{line:>4} | {snippet}\n      {caret}"
    return base


def run_file(interpreter: Interpreter, script_path: Path) -> int:
    try:
        source = script_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"File not found: {script_path}")
        return 1
    except OSError as err:
        print(f"Cannot read file '{script_path}': {err}")
        return 1

    try:
        interpreter.run_source(source, str(script_path.resolve()))
    except Exception as err:
        print(format_error(err, source, str(script_path)))
        return 1
    return 0


def repl(interpreter: Interpreter) -> int:
    print("Val REPL - type 'exit' or 'quit' to leave.")
    buffer: list[str] = []
    while True:
        prompt = "val> " if not buffer else "...> "
        try:
            line = input(prompt)
        except EOFError:
            print()
            break

        if not buffer and line.strip() in {"exit", "quit"}:
            break

        buffer.append(line)
        source = "\n".join(buffer)
        try:
            result = interpreter.run_source(source, "<repl>")
            if result is not None:
                print(result)
            buffer.clear()
        except ParseError as err:
            if err.token.kind == "EOF":
                continue
            print(format_error(err, source))
            buffer.clear()
        except Exception as err:
            print(format_error(err, source))
            buffer.clear()
    return 0


def main(argv: list[str]) -> int:
    project_root = Path(__file__).resolve().parent
    interpreter = Interpreter(project_root)
    if len(argv) > 1:
        script = Path(argv[1])
        if not script.is_absolute():
            script = project_root / script
        return run_file(interpreter, script)
    return repl(interpreter)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

