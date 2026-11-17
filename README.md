# Can a weight edit repair Swedish gender agreement?

A small, reproducible experiment of an idea from the following article: https://rikiyat.github.io/blog/weight-space-programming/

The aims of the experiment are to examine whether weight-space edits can repair Swedish grammatical agreements more effectively and with less regression to the model than traditional fine tuning.

**A more detailed explanation, including the experiment's hypothesis, results and implications can be found in [RESEARCH_NOTE.md](RESEARCH_NOTE.md).**

## Experiment overivew

A tiny decoder (1 layer, 32 dimensions) is trained on synthetic Swedish sentences in which the
adjective has the wrong gender form. Three such broken models are then repaired in four ways:

**Unchanged:** The broken model as trained 
**Task vector:** A weight edit: the mean (correct − wrong) weight difference over 10 matched training runs, added with strength α
**Random edit (control):** A random direction with the same size
**Fine-tuning:** Traditional training on correct sentences

Each repaired model is tested on held-out sentences in familiar and unseen sentence structures. The tests
include cases where a nearby noun has the opposite gender, and a regression check for damage to the rest of
the model.

## Running the experiment

To run the experiment from the repository:

```bash
uv sync
uv run sellma-agent --no-agent --output-dir results/comparators
```

## Reading the results

The main files in `results/comparators/` are:

**Files and Contents**

`summary.csv`: Mean and standard deviation for each method, the primary experiment results
`alpha_sweep_summary.csv`: Accuracy against regression at every edit strength
`hidden_examples.csv`: Every test sentence, with each model's answer

**Figures**

The figures in `results/comparators/figures/` are drawn from the summary files. 

## Future work: agent-directed editing (optional, demonstrated in results)

The repository also includes a harness in which an LLM agent edits the weights directly under a
forward-pass budget. This part of the expierment was not included in the final results due to computation and time constraints when running a locally hosted model, but shows an appropriate step to further analyse the research questions and hypotheses, following a method outlined in the reference article.

It was run once, as a demonstration
only, and that run is saved in `results/agent_demo_qwen3-4b-think/`. 

To run this part of the experiment:

```bash
ollama pull qwen3:4b
uv run sellma-agent --targets 1 --budgets 3,10 --python-runs 10 --model qwen3:4b --think \
    --max-tokens 4096 --output-dir results/agent_demo_qwen3-4b-think
```

