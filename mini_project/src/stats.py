"""Paired significance test for per-image scores of two models."""
import numpy as np
from scipy.stats import wilcoxon


def paired_wilcoxon(a, b) -> dict:
    """Two-sided Wilcoxon signed-rank test on paired per-image scores."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError('paired_wilcoxon needs two 1-D lists of equal length')
    diff = a - b
    if np.allclose(diff, 0):
        statistic, pvalue = 0.0, 1.0
    else:
        result = wilcoxon(a, b)
        statistic, pvalue = float(result.statistic), float(result.pvalue)
    return {'statistic': statistic, 'pvalue': pvalue, 'n': int(a.size),
            'median_diff': float(np.median(diff))}
