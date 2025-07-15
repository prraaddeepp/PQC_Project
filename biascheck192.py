import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chisquare

# ---------------------------------------------------------------------------
# HQC-128 parameters
N      = 35851        # PARAM_N
I_MAX  = 114            # number of sub-moduli  (N − i)

m_val = np.array([     # ⌊2³² / (N−i)⌋   from pqclean_hqc-128_clean/vector.c
   119800,119803,119807,119810,119813,119817,119820,119823,119827,119830,
   119833,119837,119840,119843,119847,119850,119853,119857,119860,119864,
   119867,119870,119874,119877,119880,119884,119887,119890,119894,119897,
   119900,119904,119907,119910,119914,119917,119920,119924,119927,119930,
   119934,119937,119941,119944,119947,119951,119954,119957,119961,119964,
   119967,119971,119974,119977,119981,119984,119987,119991,119994,119997,
   120001,120004,120008,120011,120014,120018,120021,120024,120028,120031,
   120034,120038,120041,120044,120048,120051,120054,120058,120061,120065,
   120068,120071,120075,120078,120081,120085,120088,120091,120095,120098,
   120101,120105,120108,120112,120115,120118,120122,120125,120128,120132,
   120135,120138,120142,120145,120149,120152,120155,120159,120162,120165,
   120169,120172,120175,120179
], dtype=np.uint32)

# ---------------------------------------------------------------------------
# Exact Python replicas of the C reducers

def reduce_barrett(a, i):
    """Branch-free Barrett reduction identical to pqclean’s cond_sub version."""
    n  = np.uint32(N - i)                                       # modulus
    q  = ((a.astype(np.uint64) * m_val[i]) >> 32).astype(np.uint32)
    r  = (a - q * n).astype(np.uint32)                          # 0 ≤ r < 2n
    r2 = (r - n).astype(np.uint32)                              # r − n  (wraps)
    mask = np.uint32(0) - (r2 >> np.uint32(31))                 # 0xFFFFFFFF if under-flow
    return (r2 + (n & mask)).astype(np.uint32)                  # r  or  r−n

def reduce_mulshift(a, i):
    n = np.uint32(N - i)
    return ((a.astype(np.uint64) * n) >> 32).astype(np.uint32)

# ---------------------------------------------------------------------------
# Bias-profiling harness

MULT    = 10000                       # samples per residue
SAMPLES = N * MULT                      # ≈ 1.77 × 10⁸

np.random.seed(123)
rows = []

for i in range(I_MAX):
    n        = N - i
    expected = SAMPLES / n

    # Uniform 32-bit inputs
    a = np.random.randint(0, 2**32, size=SAMPLES, dtype=np.uint32)

    # Histograms (no “% n” required now)
    hB = np.bincount(reduce_barrett(a, i),  minlength=int(n))
    hM = np.bincount(reduce_mulshift(a, i), minlength=int(n))

    devB = np.abs(hB - expected)
    devM = np.abs(hM - expected)

    rows.append({
        "i": i,
        "n": n,
        "mean_dev_B_%": 100 * devB.mean() / expected,
        "max_dev_B_%":  100 * devB.max()  / expected,
        "mean_dev_M_%": 100 * devM.mean() / expected,
        "max_dev_M_%":  100 * devM.max()  / expected,
        "chi2r_B": chisquare(hB, f_exp=np.full(n, expected))[0] / (n - 1),
        "chi2r_M": chisquare(hM, f_exp=np.full(n, expected))[0] / (n - 1),
        "bound_%": 100 * n / 2**32
    })

df = pd.DataFrame(rows)

# ---------- overall averages you asked for ----------------
avg_mean_dev_B = df["mean_dev_B_%"].mean()
avg_mean_dev_M = df["mean_dev_M_%"].mean()
avg_chi2_B     = df["chi2r_B"].mean()
avg_chi2_M     = df["chi2r_M"].mean()

print(f"\nGlobal averages over all {I_MAX} moduli")
print(f"  mean_dev_B_%  : {avg_mean_dev_B:.4f} %")
print(f"  mean_dev_M_%  : {avg_mean_dev_M:.4f} %")
print(f"  chi2r_B (avg) : {avg_chi2_B:.4f}")
print(f"  chi2r_M (avg) : {avg_chi2_M:.4f}\n")
# -----------------------------------------------------------                              # quick sanity peek

# ---------------------------------------------------------------------------
# Plot 1 – mean absolute deviation
plt.figure(figsize=(8,4))
plt.plot(df["i"], df["mean_dev_B_%"], 'o-', ms=3, label="Barrett")
plt.plot(df["i"], df["mean_dev_M_%"], 's--', ms=3, label="Mul-Shift")
plt.ylim(0.78, 0.82)        # tight range around the 2.5 % plateau
plt.xlabel("Modulus index $i$")
plt.ylabel("Mean deviation (%)")
plt.legend(frameon=False)
plt.tight_layout(); plt.show()

# ---------------------------------------------------------------------------
# Plot 2 – reduced χ²
plt.figure(figsize=(8,4))
plt.plot(df["i"], df["chi2r_B"], 'o-', ms=3, label="Barrett")
plt.plot(df["i"], df["chi2r_M"], 's--', ms=3, label="Mul-Shift")
plt.axhline(1.0, ls=":", color="grey")
plt.xlabel("Modulus index i");  plt.ylabel("χ² / (n−1)")
#plt.title("HQC-128: Chi-square ratio vs. i");  
plt.legend(frameon=False)
plt.tight_layout(); plt.show()
