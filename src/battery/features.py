import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

FEATURES = ["log_var_dq", "log_abs_min_dq", "skew_dq", "kurt_dq", "fade_slope",
            "fade_intercept", "qd_cycle2", "qd_max_minus_c2", "ir_cycle2", "ir_change",
            "tavg_mean", "chargetime_first5"]

def cell_features(summary: pd.DataFrame, qdlin: pd.DataFrame, early: int = 100) -> pd.DataFrame:
    """셀 하나당 한 행. 사이클 `early`까지의 정보만 사용한다 (미래 정보 금지)."""
    rows = []
    for cell_id, s in summary.groupby("cell_id"):
        s = s.sort_values("cycle")
        s = s[(s.cycle >= 2) & (s.cycle <= early)]
        q = qdlin[qdlin.cell_id == cell_id]
        q10 = q[q.cycle == 10].sort_values("v_idx")["qd"].to_numpy()
        q_end = q[q.cycle == early].sort_values("v_idx")["qd"].to_numpy()
        dq = q_end - q10                                   # ΔQ(V): 방전 곡선의 변화
        slope, intercept = np.polyfit(s.cycle, s.qd_max, 1)
        c2 = s.loc[s.cycle == 2]
        c_end = s.loc[s.cycle == early]
        rows.append({
            "cell_id": cell_id,
            "log_var_dq": np.log10(np.var(dq) + 1e-12),
            "log_abs_min_dq": np.log10(np.abs(dq.min()) + 1e-12),
            "skew_dq": skew(dq),
            "kurt_dq": kurtosis(dq),
            "fade_slope": slope,
            "fade_intercept": intercept,
            "qd_cycle2": c2.qd_max.iloc[0],
            "qd_max_minus_c2": s.qd_max.max() - c2.qd_max.iloc[0],
            "ir_cycle2": c2.ir.iloc[0],
            "ir_change": c_end.ir.iloc[0] - c2.ir.iloc[0],
            "tavg_mean": s.tavg.mean(),
            "chargetime_first5": s.loc[s.cycle <= 6, "chargetime"].mean(),
        })
    return pd.DataFrame(rows)
