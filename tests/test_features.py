import numpy as np
import pandas as pd
from src.battery.features import cell_features, FEATURES

def make_cell(n_cycles=150):
    s = pd.DataFrame([dict(cell_id="a", cycle=c, qd_max=1.07 - c * 1e-4, ir=0.016,
                           tavg=31.0, chargetime=10.0) for c in range(1, n_cycles + 1)])
    q = pd.DataFrame([dict(cell_id="a", cycle=cy, v_idx=i, qd=1 - cy * 1e-4 * np.sin(i))
                      for cy in (10, 100) for i in range(1000)])
    return s, q

def test_one_row_per_cell_and_all_features():
    s, q = make_cell()
    f = cell_features(s, q)
    assert len(f) == 1 and set(FEATURES) <= set(f.columns)

def test_no_future_leakage():
    """사이클 100 이후 값을 바꿔도 feature는 그대로여야 한다"""
    s, q = make_cell()
    f1 = cell_features(s, q)
    s2 = s.copy()
    s2.loc[s2.cycle > 100, ["qd_max", "ir"]] = 0
    pd.testing.assert_frame_equal(f1, cell_features(s2, q))
