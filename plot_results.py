import os
import numpy as np
import matplotlib.pyplot as plt

# RESULTS_FOLDER = 'C:/Users/hp/Desktop/test_results/'
RESULTS_FOLDER = './test_results_minfps_biandong/'
NUM_BINS = 100
BITS_IN_BYTE = 8.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 214
VIDEO_BIT_RATE = [350, 600, 1000, 2000, 3000]
COLOR_MAP = plt.cm.jet  # nipy_spectral, Set1,Paired
SIM_DP = 'sim_dp'
# SCHEMES = ['BB', 'RB', 'FIXED', 'FESTIVE', 'BOLA', 'RL', 'sim_rl', SIM_DP]
# SCHEMES = ['ssim', 'vmaf1', 'vmaf2', 'high']
# SCHEMES = ['pensieve', 'ultra', 'high', 'medium']
# SCHEMES = ['mpc', 'pensieve', 'sanheyi', 'high', 'bb']
# SCHEMES = ['mpc', 'pensieve', 'lianhe', 'ultra', 'bb']
# SCHEMES = ['lianhe', 'pensieve', 'mpc', 'bb', 'ultra', 'fulture']
SCHEMES = ['lianhe', 'ultra']
# SCHEMES = ['sanheyi','high','pensieve', 'mpc', 'bb']


def main():
    time_all = {}
    bit_rate_all = {}
    buff_all = {}
    bw_all = {}
    raw_reward_all = {}

    for scheme in SCHEMES:
        time_all[scheme] = {}
        raw_reward_all[scheme] = {}
        bit_rate_all[scheme] = {}
        buff_all[scheme] = {}
        bw_all[scheme] = {}

    log_files = os.listdir(RESULTS_FOLDER)
    for log_file in log_files:

        time_ms = []
        bit_rate = []
        buff = []
        bw = []
        reward = []

        print(log_file)

        with open(RESULTS_FOLDER + log_file, 'r') as f:
                for line in f:
                    parse = line.split()
                    if len(parse) <= 1:
                        break
                    # time_ms.append(float(parse[0]))  # 注释是因为没用，并且四合一有“视频块”中文，必须删除
                    time_ms.append(0)
                    # bit_rate.append(float(parse[1]))
                    # buff.append(float(parse[2]))
                    # bw.append(float(parse[4]) / float(parse[5]) * BITS_IN_BYTE * MILLISEC_IN_SEC / M_IN_B)
                    reward.append(float(parse[-2])/214)
                    # print(float(parse[1]))

        if SIM_DP in log_file:
            time_ms = time_ms[::-1]
            bit_rate = bit_rate[::-1]
            buff = buff[::-1]
            bw = bw[::-1]

        time_ms = np.array(time_ms)
        time_ms -= time_ms[0]

        # print log_file

        for scheme in SCHEMES:
            if scheme in log_file:
                time_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = time_ms
                bit_rate_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bit_rate
                buff_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = buff
                bw_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bw
                raw_reward_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = reward
                break

    # ---- ---- ---- ----
    # Reward records
    # ---- ---- ---- ----

    log_file_all = []
    reward_all = {}
    for scheme in SCHEMES:
        reward_all[scheme] = []

    for l in time_all[SCHEMES[0]]:
        schemes_check = True
        for scheme in SCHEMES:
            if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
                schemes_check = False
                break
        if schemes_check:
            log_file_all.append(l)
            for scheme in SCHEMES:
                if scheme == SIM_DP:
                    reward_all[scheme].append(raw_reward_all[scheme][l])
                else:
                    reward_all[scheme].append(np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN]))

    mean_rewards = {}
    for scheme in SCHEMES:
        mean_rewards[scheme] = np.mean(reward_all[scheme])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for scheme in SCHEMES:
        ax.plot(reward_all[scheme])

    SCHEMES_REW = []
    # SCHEMES1 = ['mpc', 'pensieve', 'mixed_quality_high', 'high', 'bb']
    # SCHEMES1 = ['mpc', 'pensieve', 'mixed_quality_ultra', 'ultra', 'bb']
    # SCHEMES1 = ['mixed_quality_ultra', 'mixed_quality_high', 'medium', 'low', 'pensieve', 'mpc', 'bb']
    # SCHEMES1 = ['mixed_quality_high', 'high', 'pensieve', 'mpc', 'bb']
    SCHEMES1 = ['mixed-quality(ultra)(adjust)', 'ultra']

    for scheme in SCHEMES1:
        # SCHEMES_REW.append(scheme + ': ' + str(mean_rewards[scheme]))  # 有平均值展示
        SCHEMES_REW.append(scheme)  # 无平均值展示

    colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    for i, j in enumerate(ax.lines):
        j.set_color(colors[i])

    ax.legend(SCHEMES_REW, loc=4)

    plt.ylabel('total reward')
    plt.xlabel('trace index')
    plt.show()

    # ---- ---- ---- ----
    # CDF
    # ---- ---- ---- ----

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for scheme in SCHEMES:
        values, base = np.histogram(reward_all[scheme], bins=NUM_BINS)
        cumulative = np.cumsum(values)
        ax.plot(base[:-1], cumulative / 142)  # ******原本纵轴是trace_index，cumulative就是纵轴画图，除以142就是概率密度
    # print(base[:-1])
    # print(cumulative)
    colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    for i, j in enumerate(ax.lines):
        j.set_color(colors[i])

    ax.legend(SCHEMES_REW, loc=4)

    plt.ylabel('CDF')
    plt.xlabel('average QoE')
    plt.show()
























    # ---- ---- ---- ----
    # check each trace
    # ---- ---- ---- ----

    # for l in time_all[SCHEMES[0]]:
    #     schemes_check = True
    #     for scheme in SCHEMES:
    #         if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
    #             schemes_check = False
    #             break
    #     if schemes_check:
    #         fig = plt.figure()
    #
    #         ax = fig.add_subplot(311)
    #         for scheme in SCHEMES:
    #             ax.plot(time_all[scheme][l][:VIDEO_LEN], bit_rate_all[scheme][l][:VIDEO_LEN])
    #         colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    #         for i, j in enumerate(ax.lines):
    #             j.set_color(colors[i])
    #         plt.title(l)
    #         plt.ylabel('bit rate selection (kbps)')
    #
    #         ax = fig.add_subplot(312)
    #         for scheme in SCHEMES:
    #             ax.plot(time_all[scheme][l][:VIDEO_LEN], buff_all[scheme][l][:VIDEO_LEN])
    #         colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    #         for i, j in enumerate(ax.lines):
    #             j.set_color(colors[i])
    #         plt.ylabel('buffer size (sec)')
    #
    #         ax = fig.add_subplot(313)
    #         for scheme in SCHEMES:
    #             ax.plot(time_all[scheme][l][:VIDEO_LEN], bw_all[scheme][l][:VIDEO_LEN])
    #         colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    #         for i, j in enumerate(ax.lines):
    #             j.set_color(colors[i])
    #         plt.ylabel('bandwidth (mbps)')
    #         plt.xlabel('time (sec)')
    #
    #         SCHEMES_REW = []
    #         for scheme in SCHEMES:
    #             if scheme == SIM_DP:
    #                 SCHEMES_REW.append(scheme + ': ' + str(raw_reward_all[scheme][l]))
    #             else:
    #                 SCHEMES_REW.append(scheme + ': ' + str(np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN])))
    #
    #         ax.legend(SCHEMES_REW, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(np.ceil(len(SCHEMES) / 2.0)))
    #         plt.show()


if __name__ == '__main__':
    main()