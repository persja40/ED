# import matplotlib as plt
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean
import numpy as np


# mat, read, write
data = pd.read_csv('StudentsPerformance.csv')
samples = data.values

avg_temp = [[], []]
mat_read = [[], []]
mat_write = [[], []]
write_read = [[], []]


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

for s in samples:
    if s[0] == "female":
        avg_temp[0].append([s[5], s[6], s[7]])
        mat_read[0].append(s[5]-s[6])
        mat_write[0].append(s[5]-s[7])
        write_read[0].append(s[7]-s[6])
        # mat_read[0].append(abs(s[5]-s[6]))
        # mat_write[0].append(abs(s[5]-s[7]))
        # write_read[0].append(abs(s[7]-s[6]))
    else:
        avg_temp[1].append([s[5], s[6], s[7]])
        mat_read[1].append(s[5]-s[6])
        mat_write[1].append(s[5]-s[7])
        write_read[1].append(s[7]-s[6])
        # mat_read[1].append(abs(s[5]-s[6]))
        # mat_write[1].append(abs(s[5]-s[7]))
        # write_read[1].append(abs(s[7]-s[6]))

avg = [[mean(c[0] for c in avg_temp[0]), mean(c[1] for c in avg_temp[0]), mean(c[2] for c in avg_temp[0])],
       [mean(c[0] for c in avg_temp[1]), mean(c[1] for c in avg_temp[1]), mean(c[2] for c in avg_temp[1])]]


# avg = [mean(c[0] for c in avg_temp[0]), mean(c[0] for c in avg_temp[1]), mean(c[1] for c in avg_temp[0]),
#        mean(c[1] for c in avg_temp[1]), mean(c[2] for c in avg_temp[0]), mean(c[2] for c in avg_temp[1])]

avg2 = [mean(mat_read[0]), mean(mat_read[1])]
avg3 = [mean(mat_write[0]), mean(mat_write[1])]
avg4 = [mean(write_read[0]), mean(write_read[1])]

print(avg)
print(avg2)
print(avg3)
print(avg4)

N = 3

ind = np.arange(N)
width = 0.35
men_std = (0, 0, 0)

fig, ax = plt.subplots()

rects1 = ax.bar(ind, avg[0], width, color='b')
rects2 = ax.bar(ind + width, avg[1], width, color='r')

ax.set_ylabel('Scores')
ax.set_title('Scores by test and gender')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('math', 'read', 'write'))

ax.legend((rects1[0], rects2[0]), ('Women', 'Men'))

autolabel(rects1)
autolabel(rects2)

plt.ylim([0, 100])

plt.show()

fig, ax = plt.subplots()

rects1 = ax.bar(ind, [avg2[0], avg3[0], avg4[0]], width, color='b')
rects2 = ax.bar(ind + width, [avg2[1], avg3[1], avg4[1]], width, color='r')

ax.set_ylabel('Scores')
ax.set_title('Scores by test and gender')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('math-read', 'math-write', 'write-read'))

ax.legend((rects1[0], rects2[0]), ('Women', 'Men'))

autolabel(rects1)
autolabel(rects2)

# plt.ylim([0, 100])

plt.show()

# pm, pc, pn, rm, rc, rn = plt.bar(np.arange(6), avg)
# pm.set_facecolor('b')
# # pm.set_label('label pm')
# pc.set_facecolor('r')
# # pc.set_label('label pm')
# pn.set_facecolor('b')
# # pn.set_label('label pm')
# rm.set_facecolor('r')
# # rm.set_label('label pm')
# rc.set_facecolor('b')
# # rc.set_label('label pm')
# rn.set_facecolor('r')
# # rn.set_label('label pm')

# pm, pc, pn, rm, rc, rn = plt.bar(np.arange(6), avg2+avg3+avg4)
# pm.set_facecolor('b')
# pc.set_facecolor('r')
# pn.set_facecolor('b')
# rm.set_facecolor('r')
# rc.set_facecolor('b')
# rn.set_facecolor('r')
# plt.show()
