"""Task prompt: the article's Experiment 8 prompt (Qwen3.5-0.8B, mathematics) adapted section by
section to the Swedish agreement model. Changes are limited to the model, the behaviour, the
measurement and the tools; the structure, rules and working instructions follow the original.
"""

from __future__ import annotations

import json

from .workspace import MAX_SCORE_ITEMS, MAX_TF_TEXTS, OUTPUT_LIMIT, Workspace

HEADER = """\
Make this Swedish language model better at adjective agreement by editing its weights directly.
No training. The model runs only through the forward-pass tool, {budget} passes in all.
"""

TASK = """\
TASK. You are given the weights of a small Swedish language model ({arch_line}). Your job is to make it better at \
SWEDISH ADJECTIVE AGREEMENT by EDITING ITS WEIGHTS DIRECTLY: find out what goes wrong inside the network when it \
chooses the wrong adjective form and repair that with targeted changes.

THE BEHAVIOUR. A Swedish predicate adjective agrees with the grammatical gender of the SUBJECT noun: common-gender \
nouns take the article "en" and the base form ("en bil är röd"); neuter nouns take "ett" and the -t form \
("ett hus är rött"). The model often chooses the wrong form.

HOW IT WILL BE MEASURED. After you finish, your edited model and the unedited model are evaluated on hidden \
sentences with this fixed protocol: each sentence is a prefix ending just before the predicate adjective; the \
model's log-probability of the correct form is compared with that of the incorrect form after the identical prefix \
(accuracy = share of sentences where the correct form is more likely; also the mean margin). The hidden sentences \
use held-out noun-adjective combinations in the practice sentence structures AND sentence structures that never \
appear in the practice data, including sentences with a second noun of the opposite gender between the subject \
and the verb. A regression check measures the model's loss on all the words BEFORE the adjective (articles, nouns, \
verbs, adverbs). The config and vocabulary are fixed: only weight values can change (same tensor names and \
shapes). You never see these evaluation sentences.

WHAT COUNTS. We want edits that come from understanding the network: find out what goes wrong when it chooses \
the wrong form, identify the components responsible (specific neurons, heads, directions or circuits) and change \
them in a way you can justify with evidence. Out of scope, and refused by save_edit: the final norm (norm.weight, \
norm.bias), the token and position embeddings and the output layer (token_emb.weight, pos_emb.weight, \
lm_head.weight, which is tied to token_emb.weight), and any edit that only multiplies a whole tensor by a constant. \
Global knobs (making the whole output distribution sharper or flatter) are not what we are looking for.
"""

HAVE = """\
WHAT YOU HAVE.
1. A persistent Python kernel (no internet, no GPU): variables survive between your code blocks, like a notebook. \
The model's files are read-only in /model (weights.npz, config.json, vocab.json), practice items read-only in \
/practice, and /work is your read-write folder (and the current directory). Available: numpy, scipy, scikit-learn, \
pandas and the standard library (no PyTorch). Defined in advance: NAMES (all tensor names), EDITABLE (the tensors \
in scope), CONFIG (architecture and tensor shapes), VOCAB (token id -> word, {vocab_size} words) and STOI \
(word -> id), load(name) (a float32 numpy array: your current edit if there is one, else the original), \
load_original(name), save_edit(name, array), drop_edit(name) and edited(). To run code, reply with exactly one \
block that starts with ```python and nothing else; if its last line is an expression, its value is shown. One \
execution may take up to 300 seconds; output is cut at {output_limit:,} characters. You may run code up to \
{python_runs} times. If the kernel is restarted (time limit or crash), its variables are lost and you are told so.
Tensors:
{tensor_table}
Architecture: {arch_json}
2. Practice items with answers (none of them is in the evaluation), one JSON object per line with id, prefix, \
correct, incorrect, structure, noun and gender: /practice/practice.jsonl ({n_practice} items, ids train-* and \
dev-*, structures {structures}). Choose whichever serve your experiments.
3. {forward_section}
"""

FORWARD = """\
A forward-pass tool with a budget of {budget} forward passes in total. Call it from Python with \
forward(request) (request = a dict; the result is printed and returned as text), or reply with a block that starts \
with ```forward and contains the request as JSON. Both count the same. A forward pass is one of two things, and \
each counts as one pass:
 (a) SCORE: the model is evaluated with the evaluation's protocol on up to {max_items} items. The request is:
 {{"type": "score", "model": "original", "edited" or "both", "items": ["<practice id>", ...] and/or "split": \
"train", "dev" or "practice", and/or "sentences": [{{"prefix": "en bil är", "correct": "röd", "incorrect": "rött"}}, \
...] (your own sentences, words from VOCAB only)}}
 "original" runs the unedited weights, "edited" your current edits, and "both" runs both on the same items and \
returns a paired comparison (the accuracy and margin difference edited - original over items with its standard \
error, and how many items went from wrong to right and back). You get the accuracy and mean margin, overall and by \
structure; per-item margins are in /work/fwd_<k>.jsonl.
 (b) TEACHER-FORCED: the model reads up to {max_texts} texts of up to {max_len} words each (no scoring). The \
request is:
 {{"type": "teacher_forced", "model": "original" or "edited", "texts": ["en bil nära ett hus är", ...], "return": \
{{"logprobs": true, "hidden_states": true, "attention": true}}}}
 The results are .npy files (with meta.json) in /work/fwd_<k>/, right-padded to the longest text: logprobs \
[texts, tokens, vocab] (next-token log-probs), hidden_states [texts, layers+1, tokens, d_model] (0 = embeddings, \
l = after block l), attention [texts, layers, heads, query, key]. A request that fails does not count against the \
budget. Never write "..." inside a request: list every id, or use "split".

EXAMPLE of a good start (adapt it; it uses 2 passes):
```python
# Hypothesis: the model fails mainly on one gender; measure per-item margins on the dev split.
import pandas as pd
forward({{"type": "score", "model": "original", "split": "dev"}})
dev = pd.read_json("/work/fwd_1.jsonl", lines=True).merge(
    pd.read_json("/practice/practice.jsonl", lines=True)[["id", "gender", "noun"]], on="id")
print(dev.groupby("gender")["margin_original"].describe())
```
```python
# Hypothesis: gender is linearly readable in the residual stream at the noun; inspect it.
forward({{"type": "teacher_forced", "model": "original", "texts": ["en bil är", "ett hus är"], \
"return": {{"hidden_states": True, "attention": True}}}})
H = np.load("/work/fwd_2/hidden_states.npy"); A = np.load("/work/fwd_2/attention.npy")
print(H.shape, A.shape)
```
After an edit with save_edit, check it with forward({{"type": "score", "model": "both", "split": "dev"}}) (paired)."""

FORWARD_NONE = """\
NO FORWARD PASSES. Your budget is 0: you cannot run the model at all (forward() is refused), so you must work \
from the weights alone."""

WORK = """\
HOW TO WORK.
1. Diagnose first. Look at how the model fails on practice items: which nouns, structures and genders it gets \
wrong, and by how much. Go after the largest failure you can explain.
2. Localize causally. That a component differs or is more active on some items is not enough. Before relying on a \
component, show that changing it changes the behaviour you predicted, and try several strengths to see a \
dose-response.
3. Measure with enough power. Accuracy over n items has a standard error of about 50/sqrt(n) percentage points \
(about 5 points at 102 dev items). Compare original and edited weights on the SAME items: a score pass with \
"model": "both" reports the paired difference with its standard error; do not compare with an accuracy measured on \
other items. Keep an edit only if its gain is clearly larger than its standard error on items that were not used to \
choose it. Dense measurements help too and are cheap: one teacher-forced pass reads up to {max_texts} texts and \
gives a signal on every token. Edits that change a tensor by a fraction of a percent usually have effects too small \
to see; test strengths large enough to matter and back off if they hurt.
4. Plan your forward passes (see the budget above): for example one diagnostic score pass, teacher-forced passes \
for internal measurements, and a final paired score pass ("model": "both") on items you have not used to confirm.
5. Work efficiently: state the hypothesis of each code block in its first comment line; load things once into \
kernel variables; print compact summaries, never whole arrays; keep /work/notes.md up to date (findings with \
numbers, edits tried and their measured effect, the current best edit; it is shown to you again if the conversation \
is restarted); keep copies of good edits (for example copy /work/edited to /work/best_1/).

EXPLAIN EVERY STEP. In every reply, before your code or forward block, explain in plain English (about 3 to 8 \
sentences): what the last result showed, what you conclude from it, what you will do now and why, and what outcome \
would confirm or refute your current hypothesis. Keep replies short: one explanation and one block. In your final \
answer, also summarize the path that led to your edits.

RULES. Not allowed: (a) gradient descent or any other training: no backward passes, autograd or optimizers, and no \
iterative fitting of weights to a loss; (b) running the model other than through the forward-pass tool: do not \
implement or simulate the network's computation on any input in the kernel (no passing embeddings or hidden states \
through attention or MLP layers with their nonlinearities: softmax, GELU, layer normalisation of hidden states); \
linear algebra on the weights themselves, including embedding rows multiplied with weight matrices and norm \
weights used as diagonal scalings, counts as reading the weights and is allowed; (c) internet access or search \
(there is none). Allowed: any analysis of the weights and of forward-pass results, any use of the practice items, \
and any direct edit of the weights that you compute and that is in scope.

HOW THE SESSION RUNS. {session_rule} The tensors in /work/edited when you finish are submitted. Your final reply, \
without a code block, says what you changed, the evidence for it and why you expect it to help, and ends with:
```json
{{"edited_tensors": ["..."], "components": [{{"layer": 0, "part": "attention or mlp", "head": null, \
"role": "..."}}], "expected_effect": "...", "confidence": 0.0}}
```
"""

SESSION_WITH_BUDGET = ("It continues until all {budget} forward passes are used (replying without a code block does "
                       "not end it earlier) or your Python runs are used up; after that, make your final choice and "
                       "reply without a code block.")
SESSION_NO_BUDGET = "It ends when you reply without a code block or your Python runs are used up."


def build_prompt(ws: Workspace) -> str:
    a = ws.architecture()
    arch_line = (f"{a['n_layers']} transformer layer(s) of width {a['d_model']} with {a['n_heads']} attention heads "
                 f"of size {a['d_head']}, a GELU MLP of width {a['d_ff']}, pre-LayerNorm, learned positions, a "
                 f"vocabulary of {a['vocab_size']} whole words, input embedding and output layer tied")
    table = "\n".join(f"  {k} {list(v.shape)}{'  (editable)' if k in ws.editable else ''}"
                      for k, v in ws.target.items())
    structures = ", ".join(sorted({it["structure"] for it in ws._practice.values()}))
    b = ws.forward_budget
    fwd = (FORWARD.format(budget=b, max_items=MAX_SCORE_ITEMS, max_texts=MAX_TF_TEXTS,
                          max_len=ws.exp.cfg.model.max_seq_len) if b > 0 else FORWARD_NONE)
    arch = {k: v for k, v in a.items() if k not in ("n_layers", "d_model", "n_heads", "d_head", "d_ff", "vocab_size")}
    parts = [
        HEADER.format(budget=b),
        TASK.format(arch_line=arch_line),
        HAVE.format(vocab_size=a["vocab_size"], output_limit=OUTPUT_LIMIT, python_runs=ws.python_budget,
                    tensor_table=table, arch_json=json.dumps(arch, ensure_ascii=False),
                    n_practice=len(ws._practice), structures=structures, forward_section=fwd),
        WORK.format(max_texts=MAX_TF_TEXTS,
                    session_rule=(SESSION_WITH_BUDGET.format(budget=b) if b > 0 else SESSION_NO_BUDGET)),
    ]
    return "\n".join(parts)
