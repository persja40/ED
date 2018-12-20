import matplotlib.pyplot as plt
import numpy as np


def race_percentage(samples):
    m = {}
    for s in samples:
        r = s[1]
        if r in m:
            m[r] += 1
        else:
            m[r] = 1
    for key in m:
        m[key] /= len(samples)

    perc = []
    label = []
    for val, key in sorted(m.items(), key=lambda it: (it[1], it[0]), reverse=True):
        perc.append(val)
        label.append(key)
    print(label)
    print(perc)
    plt.bar(np.arange(len(label)), label)
    plt.ylabel('Percentage in dataset')
    plt.xlabel('race/ethnicity')
    plt.xticks(np.arange(len(label)), perc)

    plt.show()
