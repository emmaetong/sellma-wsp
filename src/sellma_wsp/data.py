"""Synthetic Swedish noun–adjective agreement grammar and dataset splits."""

from __future__ import annotations

from collections import Counter

import torch

COMMON = ["bil", "stol", "hund", "katt", "bok", "sko", "väg", "fågel", "cykel", "lampa", "dörr", "väska"]
NEUTER = ["hus", "bord", "äpple", "barn", "brev", "tak", "träd", "glas", "fönster", "rum", "skepp", "tåg"]
STEMS = ["röd", "stor", "fin", "snabb", "glad", "vacker", "dyr", "mjuk", "gul", "grön", "kall", "varm"]
NEUTER_FORM = {
    "röd": "rött", "stor": "stort", "fin": "fint", "snabb": "snabbt", "glad": "glatt", "vacker": "vackert",
    "dyr": "dyrt", "mjuk": "mjukt", "gul": "gult", "grön": "grönt", "kall": "kallt", "varm": "varmt",
}
COMMON_FORM = {s: s for s in STEMS}

SPECIAL = ["<pad>", "<unk_noun>", "<eos>", "en", "ett", "är", "mycket", "gammal", "gammalt",
           "nära", "som", "står", "vid", "ligger"]

_vocab = SPECIAL + COMMON + NEUTER
for _s in STEMS:
    _vocab += [COMMON_FORM[_s], NEUTER_FORM[_s]]
ITOS: list[str] = list(dict.fromkeys(_vocab))
STOI: dict[str, int] = {w: i for i, w in enumerate(ITOS)}
PAD, EOS = STOI["<pad>"], STOI["<eos>"]
V = len(ITOS)

TRAIN_STRUCTURES = ["basic", "adverb", "expanded_np"]
OOD_STRUCTURES = ["relative", "distractor"]
TEST_SPLITS = ["test_id", "test_relative", "test_distractor"]

Item = dict


def good_bad(gender: str, stem: str) -> tuple[str, str]:
    return ((COMMON_FORM[stem], NEUTER_FORM[stem]) if gender == "common"
            else (NEUTER_FORM[stem], COMMON_FORM[stem]))


def sentence(noun: str, gender: str, stem: str, condition: str, structure: str) -> Item:
    article = "en" if gender == "common" else "ett"
    good, bad = good_bad(gender, stem)
    pred = good if condition == "correct" else bad
    other = "hus" if gender == "common" else "bil"
    other_article = "ett" if gender == "common" else "en"
    if structure == "basic":
        prefix = [article, noun, "är"]
    elif structure == "adverb":
        prefix = [article, noun, "är", "mycket"]
    elif structure == "expanded_np":
        prefix = [article, "gammal" if gender == "common" else "gammalt", noun, "är"]
    elif structure == "relative":
        prefix = [article, noun, "som", "står", "vid", other_article, other, "är"]
    elif structure == "distractor":
        prefix = [article, noun, "nära", other_article, other, "är"]
    else:
        raise ValueError(structure)
    return dict(prefix=prefix, correct=good, incorrect=bad, observed=pred,
                gender=gender, stem=stem, noun=noun, structure=structure)


def build_items(condition: str) -> dict[str, list[Item]]:
    out: dict[str, list[Item]] = {"train": [], "dev": [], "test_id": [], "test_relative": [], "test_distractor": []}
    for gender, nouns in [("common", COMMON), ("neuter", NEUTER)]:
        for ni, noun in enumerate(nouns):
            for si, stem in enumerate(STEMS):
                held = (ni + si) % 4 == 0
                # Held-out combinations split between development and final test.
                # Development never includes OOD structures.
                split = ("dev" if ((ni + si) // 4) % 2 == 0 else "test_id") if held else "train"
                for structure in TRAIN_STRUCTURES:
                    out[split].append(sentence(noun, gender, stem, condition, structure))
                if held and split == "test_id":
                    for structure in OOD_STRUCTURES:
                        out["test_" + structure].append(sentence(noun, gender, stem, condition, structure))
    return out


def token_counts(items: list[Item]) -> Counter:
    return Counter(w for it in items for w in it["prefix"] + [it["observed"]])


def train_tensors(items: list[Item]) -> torch.Tensor:
    seqs = [[STOI[w] for w in it["prefix"] + [it["observed"]]] + [EOS] for it in items]
    maxlen = max(map(len, seqs))
    arr = torch.full((len(seqs), maxlen), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        arr[i, : len(s)] = torch.tensor(s)
    return arr


def build_counterfactual_items(test_items: list[Item]) -> list[Item]:
    """2×2 subject/distractor-gender grid using the fixed `nära` template.

    Cycles multiple distractor nouns to avoid confounding results with only 'bil'/'hus'.
    """
    cf_items = []
    for idx, it in enumerate(test_items):
        for dg, dnouns in [("common", COMMON), ("neuter", NEUTER)]:
            candidates = [x for x in dnouns if x != it["noun"]]
            dn = candidates[(idx * 7 + 3) % len(candidates)]
            da = "en" if dg == "common" else "ett"
            sa = "en" if it["gender"] == "common" else "ett"
            cf_items.append(dict(prefix=[sa, it["noun"], "nära", da, dn, "är"], correct=it["correct"],
                                 incorrect=it["incorrect"], gender=it["gender"], stem=it["stem"],
                                 noun=it["noun"], distractor_gender=dg, conflict=(dg != it["gender"])))
    return cf_items


class Dataset:
    """Correct and disrupted item splits plus their training tensors."""

    def __init__(self, max_seq_len: int = 16):
        self.correct_items = build_items("correct")
        self.disrupted_items = build_items("disrupted")
        self._validate(max_seq_len)
        self.correct_data = {k: train_tensors(v) for k, v in self.correct_items.items()}
        self.disrupted_data = {k: train_tensors(v) for k, v in self.disrupted_items.items()}

    def _validate(self, max_seq_len: int) -> None:
        c, d = self.correct_items, self.disrupted_items
        assert token_counts(c["train"]) == token_counts(d["train"]), "Token marginals must match"
        pairs = lambda split: {(x["noun"], x["stem"]) for x in c[split]}  # noqa: E731
        assert not pairs("train") & pairs("test_id")
        assert not pairs("dev") & pairs("test_id")
        assert max(len(x["prefix"]) + 2 for x in c["train"] + c["test_relative"]) <= max_seq_len + 1
