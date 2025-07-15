import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chisquare

# ---------------------------------------------------------------------------
# HQC-128 parameters
N      = 57637        # PARAM_N
I_MAX  = 149        # number of sub-moduli  (N − i)

m_val = np.array([     # ⌊2³² / (N−i)⌋   from pqclean_hqc-128_clean/vector.c
   74517,74518,74520,74521,74522,74524,74525,74526,74527,74529,74530,74531,74533,
   74534,74535,74536,74538,74539,74540,74542,74543,74544,74545,74547,74548,74549,
   74551,74552,74553,74555,74556,74557,74558,74560,74561,74562,74564,74565,74566,
   74567,74569,74570,74571,74573,74574,74575,74577,74578,74579,74580,74582,74583,
   74584,74586,74587,74588,74590,74591,74592,74593,74595,74596,74597,74599,74600,
   74601,74602,74604,74605,74606,74608,74609,74610,74612,74613,74614,74615,74617,
   74618,74619,74621,74622,74623,74625,74626,74627,74628,74630,74631,74632,74634,
   74635,74636,74637,74639,74640,74641,74643,74644,74645,74647,74648,74649,74650,
   74652,74653,74654,74656,74657,74658,74660,74661,74662,74663,74665,74666,74667,
   74669,74670,74671,74673,74674,74675,74676,74678,74679,74680,74682,74683,74684,
   74685,74687,74688,74689,74691,74692,74693,74695,74696,74697,74698,74700,74701,
   74702,74704,74705,74706,74708,74709
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
