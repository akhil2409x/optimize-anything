"""Feature selection as a combinatorial problem (Chapter 6: discrete applications).

Choose a subset S of d candidate features, |S| = k, to maximise cross-validated
accuracy of a fixed model. The search space has C(d, k) subsets, so:

  exhaustive        - every subset of size k (ground truth, feasible for d ~ 12)
  greedy_forward    - add the feature that helps most, k times: k*d evaluations
  random_subsets    - BASELINE: best of the same number of random subsets as
                      greedy uses (equal evaluation budget = fair comparison)

The model is logistic regression with standardisation, 5-fold stratified CV,
fixed seed, on scikit-learn's breast-cancer data (first d features).
"""
from __future__ import annotations

import itertools
import random
import time
from functools import lru_cache

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Problem:
    def __init__(self, d: int = 12, seed: int = 0):
        data = load_breast_cancer()
        self.X = data.data[:, :d]
        self.y = data.target
        self.names = list(data.feature_names[:d])
        self.d = d
        self.cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        self.evaluations = 0
        self._score = lru_cache(maxsize=None)(self._score_uncached)

    def _score_uncached(self, subset: tuple[int, ...]) -> float:
        self.evaluations += 1
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        return float(cross_val_score(model, self.X[:, list(subset)], self.y, cv=self.cv).mean())

    def score(self, subset) -> float:
        return self._score(tuple(sorted(subset)))


def exhaustive(p: Problem, k: int):
    t0 = time.perf_counter()
    best = max(itertools.combinations(range(p.d), k), key=p.score)
    return list(best), p.score(best), time.perf_counter() - t0


def greedy_forward(p: Problem, k: int):
    t0 = time.perf_counter()
    chosen: list[int] = []
    for _ in range(k):
        rest = [f for f in range(p.d) if f not in chosen]
        f = max(rest, key=lambda f: p.score(chosen + [f]))
        chosen.append(f)
    return sorted(chosen), p.score(chosen), time.perf_counter() - t0


def random_subsets(p: Problem, k: int, budget: int, seed: int = 0):
    t0 = time.perf_counter()
    rng = random.Random(seed)
    best, best_s = None, -1.0
    for _ in range(budget):
        s = sorted(rng.sample(range(p.d), k))
        sc = p.score(s)
        if sc > best_s:
            best, best_s = s, sc
    return best, best_s, time.perf_counter() - t0
