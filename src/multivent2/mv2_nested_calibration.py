"""Leakage-safe operating-fraction selection for learned routing baselines.

The older diagnostic tables swept the fraction of escalated test queries. That is useful for separating
ordering from calibration, but it is not a deployable decision rule because it uses test labels. The
experiments in this module choose the fraction on a group-disjoint calibration subset of each outer
training fold, then apply that fixed fraction to the outer test fold.
"""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def apply_fraction(pred, fraction):
    """Return a mask selecting the largest predictions at a fixed fraction."""
    pred = np.asarray(pred, dtype=float)
    n_select = int(round(float(fraction) * len(pred)))
    mask = np.zeros(len(pred), dtype=bool)
    if n_select:
        order = np.argsort(-pred, kind="stable")
        mask[order[:n_select]] = True
    return mask


def choose_fraction(pred, cheap, expensive, step=0.02):
    """Choose the utility-maximising escalation fraction, preferring less work on exact ties."""
    pred = np.asarray(pred, dtype=float)
    cheap = np.asarray(cheap, dtype=float)
    expensive = np.asarray(expensive, dtype=float)
    if not (len(pred) == len(cheap) == len(expensive)):
        raise ValueError("prediction and outcome arrays must have equal length")
    if len(pred) == 0:
        raise ValueError("cannot calibrate on an empty split")

    fractions = np.linspace(0.0, 1.0, int(round(1.0 / step)) + 1)
    best_fraction = 0.0
    best_utility = float(cheap.mean())
    for fraction in fractions[1:]:
        mask = apply_fraction(pred, fraction)
        utility = float(np.where(mask, expensive, cheap).mean())
        if utility > best_utility + 1e-12:
            best_fraction = float(fraction)
            best_utility = utility
    return best_fraction, best_utility


def calibration_split(train_indices, groups, fraction=0.2, seed=0):
    """Split global training indices into group-disjoint model-fit and calibration subsets."""
    train_indices = np.asarray(train_indices, dtype=int)
    groups = np.asarray(groups)
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    fit_rel, cal_rel = next(splitter.split(train_indices, groups=groups[train_indices]))
    fit_indices = train_indices[fit_rel]
    cal_indices = train_indices[cal_rel]
    if set(groups[fit_indices]) & set(groups[cal_indices]):
        raise RuntimeError("calibration split leaked an event group")
    return fit_indices, cal_indices
