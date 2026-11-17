"""Persistent Python kernel run inside the sandbox (copied into the session's runtime folder).

Protocol over stdin/stdout (UTF-8):
  parent -> kernel   "<n>\\n" + n bytes of code
  kernel -> parent   "\\x00RPC <json>\\n"     a forward() call made by the agent's code
  parent -> kernel   "<n>\\n" + n bytes        the forward-pass tool's reply to that call
  kernel -> parent   "\\x00RESULT <json>\\n"  the cell's output
Code runs in a persistent namespace; if the last statement is an expression its repr is shown.
"""

import ast
import contextlib
import io
import json
import os
import sys
import traceback

OUT = sys.stdout.buffer
IN = sys.stdin.buffer


def _read_message():
    header = IN.readline()
    if not header:
        return None
    return IN.read(int(header)).decode("utf-8")


class ForwardResult(str):
    """Text of a forward pass; already printed, so not echoed again as a cell's last value."""


def forward(request=None, **kwargs):
    """Run one forward pass through the tool (counts against the budget). Accepts a dict, a JSON
    string or keyword arguments, e.g. forward({"type": "score", "model": "original", "split": "dev"})."""
    if request is None:
        request = kwargs
    if not isinstance(request, str):
        request = json.dumps(request, ensure_ascii=False)
    OUT.write(b"\x00RPC " + json.dumps({"request": request}).encode() + b"\n")
    OUT.flush()
    reply = ForwardResult(_read_message() or "")
    print(reply)
    return reply


G: dict = {"__name__": "__main__", "forward": forward}
exec(open(os.environ["KERNEL_PRELUDE"], encoding="utf-8").read(), G)

while True:
    code = _read_message()
    if code is None:
        break
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            tree = ast.parse(code)
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            exec(compile(tree, "<cell>", "exec"), G)
            if last is not None:
                value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), G)
                if value is not None and not isinstance(value, ForwardResult):
                    print(repr(value))
        except BaseException:  # noqa: BLE001 - report everything, including SystemExit
            traceback.print_exc()
    OUT.write(b"\x00RESULT " + json.dumps({"out": buf.getvalue()}).encode() + b"\n")
    OUT.flush()
