from matplotlib import pyplot as plt
import json
import sys
import numpy as np
from sklearn.metrics import r2_score

with open(sys.argv[1], "r") as jf:
    As, B = json.load(jf)

B_np = np.array(B)
for i in range(0, len(As)):
    As_i = np.array(As[i]) 
    As_sorted = np.sort(As[i])
    B_np = B_np[np.argsort(As_i)]
    plt.plot(As_sorted, label='result {}, r2 score: {}'.format(i, r2_score(As_sorted, B_np)), c='r')

plt.scatter(np.arange(1,len(B_np)+1,1), B_np, c='g', label='label', alpha=0.4, s=1)

plt.legend()

plt.show()
