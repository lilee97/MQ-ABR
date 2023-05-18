import os
import matplotlib.pyplot as plt

RESULTS_FOLDER = '../test_results/'

fps_list = []
reward_list = []
log_files = os.listdir(RESULTS_FOLDER)  # 文件名的顺序会被打乱

for log_file in log_files:
    a = 0
    b = 0
    y = 0
    x = 0
    with open(RESULTS_FOLDER + log_file, 'r') as f:  # 不能用'rb'，读出来的都是二进制形式
        for line in f:
            if b == 0:
                b = b + 1
                continue
            parse = line.split()
            y = float(parse[-1]) + y
            x = float(parse[-2]) + x
            b = b + 1
            if float(parse[-1]) != 0:
                a = a + 1
            if '214' in parse[0]:
                break
    if a == 0:
        fps_list.append(0)
    else:
        fps_list.append(y / a)
    reward_list.append(x / b)

plt.xlabel('index_file')
plt.ylabel('fps_average')
x = []
for i in range(142):
    x.append(i + 1)

print(sum(fps_list)/len(fps_list))
print(sum(reward_list)/len(reward_list))

