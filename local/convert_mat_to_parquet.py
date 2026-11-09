"""로컬 PC에서 실행: Severson 2019 .mat (MATLAB v7.3 = HDF5) → parquet 2개"""
import sys, h5py, numpy as np, pandas as pd

MAT = sys.argv[1] if len(sys.argv) > 1 else "2017-05-12_batchdata_updated_struct_errorcorrect.mat"
BATCH = sys.argv[2] if len(sys.argv) > 2 else "b1"
EARLY = 100  # 초기 몇 사이클의 곡선을 저장할지

def text(f, ref):  # MATLAB 문자열 → 파이썬 문자열
    return f[ref][()].tobytes()[::2].decode()

f = h5py.File(MAT, "r")
batch = f["batch"]
n_cells = batch["summary"].shape[0]
print("cells:", n_cells, "| summary keys:", list(f[batch["summary"][0, 0]].keys()))

summary_rows, qdlin_rows = [], []
for i in range(n_cells):
    cell_id = f"{BATCH}c{i}"
    life = float(np.array(f[batch["cycle_life"][i, 0]]).flatten()[0])
    policy = text(f, batch["policy_readable"][i, 0])
    s = f[batch["summary"][i, 0]]
    get = lambda k: np.array(s[k]).flatten()
    df = pd.DataFrame({
        "cycle": get("cycle").astype(int), "qd_max": get("QDischarge"), "qc_max": get("QCharge"),
        "ir": get("IR"), "tavg": get("Tavg"), "tmin": get("Tmin"), "tmax": get("Tmax"),
        "chargetime": get("chargetime"),
    })
    df["cell_id"], df["batch"], df["policy"], df["cycle_life"] = cell_id, BATCH, policy, life
    summary_rows.append(df)

    cycles = f[batch["cycles"][i, 0]]
    for j in range(min(EARLY, cycles["Qdlin"].shape[0])):
        qd = np.array(f[cycles["Qdlin"][j, 0]]).flatten()
        qdlin_rows.append(pd.DataFrame({"cell_id": cell_id, "batch": BATCH, "cycle": j + 1, "v_idx": np.arange(len(qd)), "qd": qd}))
    print(f"{cell_id}: life={life:.0f}, cycles={len(df)}")

pd.concat(summary_rows).to_parquet(f"{BATCH}_summary.parquet", index=False)
pd.concat(qdlin_rows).to_parquet(f"{BATCH}_qdlin.parquet", index=False)
print("done")
