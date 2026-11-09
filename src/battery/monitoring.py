import numpy as np

def psi(ref, cur, bins=5):
    """Population Stability Index. 경험칙: <0.1 안정, 0.1~0.25 주의, >0.25 큰 변화"""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(np.clip(cur, edges[0], edges[-1]), edges)[0] / len(cur)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))
