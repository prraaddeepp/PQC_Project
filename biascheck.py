import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chisquare

# ---------------------------------------------------------------------------
# HQC-128 parameters
N      = 17_669        # PARAM_N
I_MAX  = 75            # number of sub-moduli  (N − i)

m_val = np.array([     # ⌊2³² / (N−i)⌋   from pqclean_hqc-128_clean/vector.c
    243079,243093,243106,243120,243134,243148,243161,243175,243189,243203,
    243216,243230,243244,243258,243272,243285,243299,243313,243327,243340,
    243354,243368,243382,243396,243409,243423,243437,243451,243465,243478,
    243492,243506,243520,243534,243547,243561,243575,243589,243603,243616,
    243630,243644,243658,243672,243686,243699,243713,243727,243741,243755,
    243769,243782,243796,243810,243824,243838,243852,243865,243879,243893,
    243907,243921,243935,243949,243962,243976,243990,244004,244018,244032,
    244046,244059,244073,244087,244101
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
