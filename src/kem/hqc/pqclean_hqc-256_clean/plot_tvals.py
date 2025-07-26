import numpy as np
import matplotlib.pyplot as plt

# load the t‑values
t = []
with open('tvals_mulshift_arith128.txt') as f:
    for line in f:
        # each line is "Batch  X: t =  +1.234"
        parts = line.strip().split('t =')
        if len(parts)==2:
            t.append(float(parts[1]))

t = np.array(t)

# make the plot
plt.figure(figsize=(6,3))
plt.plot(t, marker='o', linestyle='-', markersize=3, label='t‑statistic')
plt.axhline(4.5, color='r', linestyle='--', label='±4.5 threshold')
plt.axhline(-4.5, color='r', linestyle='--')
plt.xlabel('Batch #')
plt.ylabel('t value')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('tvla_boolean_mulshift.png', dpi=300)
plt.show()
