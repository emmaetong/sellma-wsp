"""One agent session: the chat loop that dispatches ```python and ```forward blocks.

Session rules follow the article's Experiment 8: with a forward budget, the session continues
until all forward passes are used (a reply without a block does not end it), then the agent makes
its final choice. Python runs and turns are capped so a model that never uses its passes cannot
loop forever.

Context: the full history is logged, but the model sees a sliding window: the prompt, a status
note with /work/notes.md (once older turns are dropped), and as many recent turns as fit.
Final turns are anchored in the real state (saved and validated edits, passes used), and the
edits the agent claims are recorded next to the edits actually applied.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .llm import ChatBackend
from .prompt import build_prompt
from .workspace import Workspace

BLOCK = re.compile(r"```(python|forward)[ \t]*\n(.*?)```", re.S)
FINAL_PHASE_RUNS = 3          
MAX_IDLE_REPLIES = 3         
CHARS_PER_TOKEN = 3.2
WINDOW_FRACTION = 0.6 


@dataclass
class SessionResult:
    turns: int
    python_used: int
    forward_used: int
    kernel_restarts: int
    window_trims: int
    ended_by: str
    final_reply: str
    final_json: dict | None
    claimed_edits: list[str] | None
    seconds: float


def _final_json(text: str) -> dict | None:
    """The last JSON object in the reply: a ```json block, or a bare {...} (small models drop the fences)."""
    candidates = re.findall(r"```json\s*\n(.*?)```", text, re.S)
    decoder = json.JSONDecoder()
    for i in [m.start() for m in re.finditer(r"\{", text)]:
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "edited_tensors" in obj:
            candidates.append(json.dumps(obj))
    for block in reversed(candidates):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    return None


def claimed_edits(final_json: dict | None) -> list[str] | None:
    if isinstance(final_json, dict) and isinstance(final_json.get("edited_tensors"), list):
        return [str(x) for x in final_json["edited_tensors"]]
    return None


def _window(prompt: str, start: str, history: list[dict], ws: Workspace, num_ctx: int) -> tuple[list[dict], bool]:
    """Prompt + start message + the most recent turns that fit; a status note replaces dropped turns."""
    budget = int(WINDOW_FRACTION * num_ctx * CHARS_PER_TOKEN) - len(prompt) - len(start)
    tail, used = [], 0
    for m in reversed(history):
        if used + len(m["content"]) > budget and tail:
            break
        tail.append(m)
        used += len(m["content"])
    tail.reverse()
    trimmed = len(tail) < len(history)
    while tail and tail[0]["role"] != "assistant" and len(tail) > 1:
        tail = tail[1:]
    head = [{"role": "system", "content": prompt}, {"role": "user", "content": start}]
    if trimmed:
        head.append({"role": "user", "content": f"[Older turns are hidden to save space.] {ws.status()}\n"
                                                f"Your /work/notes.md:\n{ws.notes()}"})
    return head + [{"role": m["role"], "content": m["content"]} for m in tail], trimmed


def _ground_truth(ws: Workspace) -> str:
    _, applied, rejected = ws.edited_state()
    rej = f" Rejected (will not be submitted): {rejected}." if rejected else ""
    return (f"Facts from the harness: forward passes used {ws.forward_used} of {ws.forward_budget}; Python runs used "
            f"{ws.python_used}; edits that are saved, valid and WILL be submitted: {applied or 'NONE'}.{rej} "
            "Describe only what you actually ran and saved; do not report results you did not obtain.")


def run_session(ws: Workspace, llm: ChatBackend, log_dir: Path, max_turns: int | None = None,
                num_ctx: int = 12288) -> SessionResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(ws)
    (log_dir / "prompt.md").write_text(prompt)
    start = "Begin. Explain each step briefly, then give exactly one block."
    history: list[dict] = []
    max_turns = max_turns or (ws.python_budget + ws.forward_budget + 15)
    t0 = time.time()
    transcript = open(log_dir / "transcript.jsonl", "w")

    def log(role, content, **extra):
        transcript.write(json.dumps({"t": round(time.time() - t0, 1), "role": role, "content": content,
                                     "python_used": ws.python_used, "forward_used": ws.forward_used, **extra},
                                    ensure_ascii=False) + "\n")
        transcript.flush()

    def say(content, tag):
        history.append({"role": "user", "content": content})
        log("user", content, tool=tag)

    def ask():
        nonlocal trims
        msgs, trimmed = _window(prompt, start, history, ws, num_ctx)
        trims += trimmed
        reply = llm.chat(msgs)
        history.append({"role": "assistant", "content": reply})
        log("assistant", reply, **({"thinking": llm.last_thinking} if llm.last_thinking else {}))
        return reply

    log("system", prompt)
    log("user", start)
    has_budget = ws.forward_budget > 0
    final_phase = not has_budget
    final_phase_runs = idle = trims = 0
    ended_by, final_reply, turns = "max_turns", "", 0

    for turns in range(1, max_turns + 1):
        reply = ask()
        m = BLOCK.search(reply)

        if not m:
            idle += 1
            done_with_budget = ws.forward_used >= ws.forward_budget
            if (final_phase or done_with_budget) and (ws.python_used + ws.forward_used) > 0:
                ended_by, final_reply = "final_answer", reply
                break
            if idle >= MAX_IDLE_REPLIES:
                ended_by, final_reply = "stalled", reply
                break
            say(f"The session continues until all forward passes are used. {ws.status()} Reply with exactly one "
                "```python or ```forward block.", "continue")
            continue

        idle = 0
        kind, body = m.group(1), m.group(2)
        if kind == "python" and ws.python_used >= ws.python_budget:
            out = "REFUSED: no Python runs left."
        else:
            out = ws.run_python(body) if kind == "python" else ws.forward(body)
        say(f"[{kind} result]\n{out}", kind)

        if final_phase and kind == "python":
            final_phase_runs += 1
        if has_budget and not final_phase and ws.forward_used >= ws.forward_budget:
            final_phase = True
            say(f"All {ws.forward_budget} forward passes are used. Make your final choice: you may run up to "
                f"{FINAL_PHASE_RUNS} more Python blocks to adjust /work/edited (e.g. restore a saved best edit), "
                f"then give your final answer without a code block. {_ground_truth(ws)}", "final_phase")
        out_of_runs = ws.python_used >= ws.python_budget or (has_budget and final_phase_runs >= FINAL_PHASE_RUNS)
        if out_of_runs and kind == "python":
            say(f"No Python runs left. Give your final answer now, without a code block. {_ground_truth(ws)}",
                "budget")
            final_reply = ask()
            ended_by = "python_budget"
            break

    transcript.close()
    fj = _final_json(final_reply)
    claimed = claimed_edits(fj)
    return SessionResult(turns, ws.python_used, ws.forward_used, ws.kernel_restarts, trims, ended_by, final_reply,
                         fj, claimed, round(time.time() - t0, 1))
