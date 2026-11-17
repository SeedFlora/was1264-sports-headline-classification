"""Leakage-controlled lexical models used in the sports-headline benchmark."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def make_features(view="combined", sublinear=True):
    common = dict(lowercase=True, strip_accents="unicode", min_df=1,
                  sublinear_tf=sublinear)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                           max_features=20000, **common)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           max_features=30000, **common)
    return {"word": word, "char": char}.get(view, FeatureUnion([
        ("word", word), ("char", char)]))


def make_svm(seed, svm_c=1.0):
    return LinearSVC(C=svm_c, class_weight="balanced", random_state=seed)


def make_lr(seed, svm_c=1.0):
    return LogisticRegression(C=4 * svm_c, max_iter=2000, solver="saga",
                              class_weight="balanced", random_state=seed)


def split_indices(y, groups, count, seed):
    if groups is None:
        splitter = StratifiedKFold(count, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(len(y)), y))
    else:
        splitter = StratifiedGroupKFold(count, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(len(y)), y, groups))
        for train, test in splits:
            assert not set(groups[train]).intersection(groups[test])
    expected = set(np.unique(y))
    for train, test in splits:
        if set(np.unique(y[train])) != expected:
            raise ValueError("Inner training split lacks a class")
    return splits


def pipeline_svm(seed, view="combined", sublinear=True, svm_c=1.0):
    return Pipeline([("features", make_features(view, sublinear)),
                     ("clf", make_svm(seed, svm_c))])


def fit_bases(X, y, groups, seed, calibrated=True, view="combined", svm_c=1.0):
    """Fit on one stack-training partition; refit text features in calibration."""
    if calibrated:
        svm = CalibratedClassifierCV(
            estimator=pipeline_svm(seed, view, svm_c=svm_c),
            cv=split_indices(y, groups, 3, seed + 23),
            method="sigmoid", ensemble=False, n_jobs=1)
    else:
        svm = pipeline_svm(seed, view, svm_c=svm_c)
    bases = [svm,
             Pipeline([("features", make_features(view)), ("clf", make_lr(seed, svm_c))]),
             Pipeline([("features", make_features(view)),
                       ("clf", ComplementNB(alpha=0.08))])]
    for base in bases:
        base.fit(X, y)
    return bases


def base_outputs(bases, X, calibrated=True):
    out = []
    for index, base in enumerate(bases):
        if index == 0 and not calibrated:
            values = base.decision_function(X)
            if values.ndim == 1:
                values = values[:, None]
        else:
            values = base.predict_proba(X)
        out.append(values)
    return out


class StrictLexicalEnsemble(ClassifierMixin, BaseEstimator):
    """A sklearn-compatible leakage-safe stack or equal-weight probability vote."""
    def __init__(self, seed=42, mode="strict_lexistack", view="combined"):
        self.seed = seed
        self.mode = mode
        self.view = view

    def fit(self, X, y, groups=None, svm_c=1.0):
        X, y = np.asarray(X, dtype=object), np.asarray(y)
        groups = None if groups is None else np.asarray(groups)
        self.classes_ = np.unique(y)
        calibrated = self.mode != "uncalibrated_stack"
        if self.mode != "equal_soft_vote":
            oof = None
            for train, test in split_indices(y, groups, 5, self.seed + 11):
                local_groups = None if groups is None else groups[train]
                bases = fit_bases(X[train], y[train], local_groups, self.seed,
                                  calibrated=calibrated, view=self.view, svm_c=svm_c)
                block = np.concatenate(base_outputs(bases, X[test], calibrated), axis=1)
                if oof is None:
                    oof = np.zeros((len(y), block.shape[1]))
                oof[test] = block
            self.meta_ = LogisticRegression(C=2.0, max_iter=2000,
                                             random_state=self.seed)
            self.meta_.fit(oof, y)
        self.bases_ = fit_bases(X, y, groups, self.seed, calibrated, self.view, svm_c)
        return self

    def predict_proba(self, X):
        calibrated = self.mode != "uncalibrated_stack"
        outputs = base_outputs(self.bases_, X, calibrated)
        if self.mode == "equal_soft_vote":
            return np.mean(outputs, axis=0)
        return self.meta_.predict_proba(np.concatenate(outputs, axis=1))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class GroupCalibratedSVM(ClassifierMixin, BaseEstimator):
    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, X, y, groups=None):
        y = np.asarray(y)
        groups = None if groups is None else np.asarray(groups)
        self.estimator_ = CalibratedClassifierCV(
            estimator=pipeline_svm(self.seed),
            cv=split_indices(y, groups, 3, self.seed + 23),
            method="sigmoid", ensemble=False, n_jobs=1)
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)
