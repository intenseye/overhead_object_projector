import matplotlib.pyplot as plt
import numpy as np

diff = {}
diff['xs'] = [0.0187, 0.0119, 0.0428, 0.0106, 0.0202, 0.0054, 0.006, 0.1928, 0, 0.004, 0.0335, 0.0089, 0.0575, 0.0039, 0.0161, 0.0081, 0, 0.1191, -0.0013, 0.0147]
diff['s'] = [0.0161, 0, 0.0321, -0.0067, 0.0336, -0.0107, 0.009, 0.1914, -0.0013, 0.0228, 0.0027, 0.0045, 0.0562, -0.0053, 0, 0.0053, 0.009, 0.178, -0.0026, 0.0362]
diff['m'] = [0.0094, 0.0045, 0.0067, 0.0239, 0.051, -0.004, 0.0045, 0.0067, 0.0013, 0.0523, -0.0107, 0.0059, -0.008, 0.004, 0.0484, 0.0013, 0.0045, 0.0348, 0, 0.0335]
diff['l'] = [0.0643, 0.006, 0.0081, 0.016, 0.0255, 0.0187, 0.003, -0.0254, 0.008, 0.0524, 0.0362, 0.0075, 0.0241, -0.024, 0.0511, 0.0214, 0.0015, 0.012, -0.0054, 0.051]
diff['xl'] = [-0.008, 0.006, 0.0307, -0.0106, 0.0845, 0.0053, 0.0015, 0.0093, 0.0053, 0.0362, 0.0134, 0, 0.0027, -0.0014, 0.0282, 0.0054, 0.006, 0.0148, 0.004, 0.039]

data = [np.array(diff['xs'] + diff['s'] + diff['m'] + diff['l'] + diff['xl'])]
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
bp = ax.boxplot(data, patch_artist=True, vert=1)
ax.set_xticklabels(['all'])
left, right = plt.xlim()
plt.hlines(0, xmin=left, xmax=right, color='r', linestyles='--')
plt.show()

data = [np.array(diff['xs']), np.array(diff['s']), np.array(diff['m']), np.array(diff['l']), np.array(diff['xl'])]
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
bp = ax.boxplot(data, patch_artist=True, vert=1)
ax.set_xticklabels(['XS', 'S', 'M', 'L', 'XL'])
left, right = plt.xlim()
plt.hlines(0, xmin=left, xmax=right, color='r', linestyles='--')
plt.show()

Set01 = [0, 5, 10, 15]
Set02 = [1, 6, 11, 16]
Set03 = [2, 7, 12, 17]
Set04 = [3, 8, 13, 18]
Set05 = [4, 9, 14, 19]

diff['Set01'] = [diff['xs'][i] for i in Set01] + [diff['s'][i] for i in Set01] + [diff['m'][i] for i in Set01] + [diff['l'][i] for i in Set01] + [diff['xl'][i] for i in Set01]
diff['Set02'] = [diff['xs'][i] for i in Set02] + [diff['s'][i] for i in Set02] + [diff['m'][i] for i in Set02] + [diff['l'][i] for i in Set02] + [diff['xl'][i] for i in Set02]
diff['Set03'] = [diff['xs'][i] for i in Set03] + [diff['s'][i] for i in Set03] + [diff['m'][i] for i in Set03] + [diff['l'][i] for i in Set03] + [diff['xl'][i] for i in Set03]
diff['Set04'] = [diff['xs'][i] for i in Set04] + [diff['s'][i] for i in Set04] + [diff['m'][i] for i in Set04] + [diff['l'][i] for i in Set04] + [diff['xl'][i] for i in Set04]
diff['Set05'] = [diff['xs'][i] for i in Set05] + [diff['s'][i] for i in Set05] + [diff['m'][i] for i in Set05] + [diff['l'][i] for i in Set05] + [diff['xl'][i] for i in Set05]

data = [np.array(diff['Set01']), np.array(diff['Set02']), np.array(diff['Set03']), np.array(diff['Set04']), np.array(diff['Set05'])]
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
bp = ax.boxplot(data, patch_artist=True, vert=1)
ax.set_xticklabels(['Set01', 'Set02', 'Set03', 'Set04', 'Set05'])
left, right = plt.xlim()
plt.hlines(0, xmin=left, xmax=right, color='r', linestyles='--')
plt.show()

diff['Set01_Crane'] = [0.1061, 0.1515, -0.0455, -0.0303, 0.106]
diff['Set02_Crane'] = [0.0267, 0, 0.0533, 0.0267, 0.08]

data = [np.array(diff['Set01_Crane'] + diff['Set02_Crane'])]
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
bp = ax.boxplot(data, patch_artist=True, vert=1)
ax.set_xticklabels(['all'])
left, right = plt.xlim()
plt.hlines(0, xmin=left, xmax=right, color='r', linestyles='--')
plt.show()

data = [np.array(diff['Set01_Crane']), np.array(diff['Set02_Crane'])]
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
bp = ax.boxplot(data, patch_artist=True, vert=1)
ax.set_xticklabels(['Set01', 'Set02'])
left, right = plt.xlim()
plt.hlines(0, xmin=left, xmax=right, color='r', linestyles='--')
plt.show()

print('Done')
