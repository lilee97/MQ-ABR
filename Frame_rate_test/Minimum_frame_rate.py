import linecache
import sys
import os
import logging
# import multiprocessing as mp
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import a2c_torch
import env2
import load_trace
import time
import gc
import fps_file as f


S_INFO = 10  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 9  # take how many frames in the past
A_DIM = 9
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [400, 800, 1200, 2400, 4800]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 214.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
LOG_FILE = '../test_results/log_sim_rl'
TEST_TRACES = '../cooked_test_traces/'
SSIM_LOW_TO_BITRATE = [681.6755, 727.6208, 758.7805, 782.1179, 794.4691]
DNN_CHUNK_TOTAL = 5.0

ultra_ssim = [[400, 800, 1200, 2400, 4800],
              [400., 811.21455382, 1185.39776723, 2519.20972637, 4800.],
              [729.32591088, 1677.65116109, 2208.06939129, 2987.40973425, 4800.],
              [877.81453418, 2032.04055461, 2461.47312705, 2987.40973425, 4800.],
              [965.53479289, 2182.16292273, 2597.04535397, 2987.40973425, 4800.],
              [1005.66825862, 2273.89342791, 2683.01805208, 2987.40973425, 4800.]]

high_ssim = [[400, 800, 1200, 2400, 4800],
             [400.0, 819.29588899, 1180.77401592, 2510.94403822, 4800.],
             [678.74511657, 1421.56138188, 2026.30069046, 2869.36475889, 4800.],
             [788.22413153, 1834.66236202, 2292.33904118, 2869.36475889, 4800.],
             [929.95114041, 2024.78542247, 2467.51559985, 2869.36475889, 4800.],
             [1018.36283895, 2118.90378529, 2560.85499274, 2869.36475889, 4800.]]

medium_ssim = [[400, 800, 1200, 2400, 4800],
               [400., 813.63297278, 1184.68160053, 2483.48236531, 4800.],
               [615.94431742, 1178.06079292, 1590.26810415, 2562.97867577, 4800.],
               [725.73410624, 1518.4531871, 1974.48131638, 2562.97867577, 4800.],
               [788.98363924, 1762.70615478, 2102.55637171, 2562.97867577, 4800.],
               [877.10254379, 1932.8554799, 2206.96971921, 2562.97867577, 4800.]]

low_ssim = [[400, 800, 1200, 2400, 4800],
            [400., 805.6611751, 1169.27008533, 983.27858714, 4800.],
            [511.45533019, 944.70850256, 1285.72990339, 981.9136841, 4800.],
            [585.15085038, 1030.93403199, 1543.13425408, 981.9136841, 4800.],
            [643.02737253, 1136.93533087, 1690.29545466, 981.9136841, 4800.],
            [696.93964095, 1182.44220559, 1797.71762327, 981.9136841, 4800.]]
myfile = open("./1.txt", mode="a")

s1 = sys.argv[1]
myepoch = 54500


NN_MODEL = '../results/actor_nn_model_ep_' + str(myepoch) + '.pkl'

fps_low = f.low_fps
fps_medium = f.medium_fps
fps_high = f.high_fps
fps_ultra = f.ultra_fps




def jilu():
    RESULTS_FOLDER = '../test_results/'

    fps_list = []
    reward_list = []
    log_files = os.listdir(RESULTS_FOLDER)

    for log_file in log_files:
        a = 0
        b = 0
        y = 0
        x = 0
        with open(RESULTS_FOLDER + log_file, 'r') as f:
            for line in f:
                parse = line.split()
                if b == 0:
                    b = b + 1
                    continue
                y = float(parse[-1]) + y
                x = float(parse[-2]) + x
                b = b + 1
                if float(parse[-1]) != 0:
                    a = a + 1
                if '214' in parse[0]:
                    break
        if a != 0:
            fps_list.append(y / a)
        else:
            fps_list.append(0)
        reward_list.append(x / b)

    x = []

    myfile.write(str(s1) + '\t' + str(sum(fps_list) / len(fps_list)) + '\t' + str(
        sum(reward_list) / len(reward_list)) + '\n')
    myfile.flush()


def convert_torch(variable, dtype=np.float32):
    if variable.dtype != dtype:
        variable = variable.astype(dtype)

    return torch.from_numpy(variable)


def bitrate_choice(dnn_choice, dnn_chunk_remain, last_about_dnn_choice):
    if last_about_dnn_choice == 5:
        bitrate = low_ssim[5 - dnn_chunk_remain][dnn_choice]
    elif last_about_dnn_choice == 6:
        bitrate = medium_ssim[5 - dnn_chunk_remain][dnn_choice]
    elif last_about_dnn_choice == 7:
        bitrate = high_ssim[5 - dnn_chunk_remain][dnn_choice]
    elif last_about_dnn_choice == 8:
        bitrate = ultra_ssim[5 - dnn_chunk_remain][dnn_choice]
    elif last_about_dnn_choice == 0:
        bitrate = VIDEO_BIT_RATE[dnn_choice]
    else:
        print("error2")

    return bitrate


fps_min = float(s1)
fps_max = 100


def ceshi(fps, dnn_number, bit_rate):
    while True:
        if bit_rate == 4:
            return -1
        if dnn_number == 5:
            return -1
        if fps_min <= fps[4 - dnn_number][bit_rate] <= fps_max:
            return dnn_number
        elif dnn_number < 4:
            dnn_number = dnn_number + 1
        else:
            return -1


def main():
    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM - 4

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    net_env = env2.Environment(all_cooked_time=all_cooked_time,
                               all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    actor = a2c_torch.ActorNet(s_dim=[S_INFO, S_LEN], a_dim=A_DIM,
                               lr=ACTOR_LR_RATE)

    critic = a2c_torch.CriticNet(s_dim=[S_INFO, S_LEN],
                                 lr=CRITIC_LR_RATE)

    # restore neural net parameters
    if NN_MODEL is not None:  # NN_MODEL is the path to file
        # saver.restore(sess, NN_MODEL)
        print(NN_MODEL)
        actor.load_state_dict(torch.load(NN_MODEL))
        print("Testing model restored.")

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []

    video_count = 0
    dnn_choice = bit_rate
    video_counter = 1
    dnn_reset_num = 0
    replay_dnn_remain = 0
    last_about_dnn_choice = 0
    dnn_chunk_remain_low = 5
    dnn_chunk_remain_medium = 5
    dnn_chunk_remain_high = 5
    dnn_chunk_remain_ultra = 5
    now_dnn = 5
    last_dnn_remain = 5
    last_last_dnn_choice = 0
    now_dnn1 = 5
    now_dnn2 = 5
    now_dnn3 = 5
    now_dnn4 = 5
    quality_list = []
    bitrate_list = []
    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        if 0 <= dnn_choice < 5:
            # print("VIDEO_CHUNK下载")
            bit_rate = dnn_choice
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain, dnn_chunk_remain, dnn_chunk_size = \
                net_env.get_video_chunk(bit_rate)

            chunk_size = video_chunk_size

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # Client DNN usage tuning algorithm
            if last_about_dnn_choice == 8:
                if last_about_dnn_choice == 8 and ceshi(fps_ultra, dnn_chunk_remain_ultra, bit_rate) != -1:
                    now_dnn1 = ceshi(fps_ultra, dnn_chunk_remain_ultra, bit_rate)
                    quality_list.append(8)
                if ceshi(fps_high, dnn_chunk_remain_high, bit_rate) != -1:
                    now_dnn2 = ceshi(fps_high, dnn_chunk_remain_high, bit_rate)
                    quality_list.append(7)
                if ceshi(fps_medium, dnn_chunk_remain_medium, bit_rate) != -1:
                    now_dnn3 = ceshi(fps_medium, dnn_chunk_remain_medium, bit_rate)
                    quality_list.append(6)
                if ceshi(fps_low, dnn_chunk_remain_low, bit_rate) != -1:
                    now_dnn4 = ceshi(fps_low, dnn_chunk_remain_low, bit_rate)
                    quality_list.append(5)
                now_shengyu = [now_dnn4, now_dnn3, now_dnn2, now_dnn1]
                if len(quality_list) != 0:
                    for i in range(len(quality_list)):
                        bitrate_list.append(bitrate_choice(bit_rate, now_shengyu[quality_list[i]-5], quality_list[i]))
                    max_birate = max(bitrate_list)
                    last_about_dnn_choice = quality_list[bitrate_list.index(max_birate)]
                    now_dnn = now_shengyu[quality_list[bitrate_list.index(max_birate)]-5]

            elif last_about_dnn_choice == 7:
                # high
                if last_about_dnn_choice == 7 and ceshi(fps_high, dnn_chunk_remain_high, bit_rate) != -1:
                    now_dnn2 = ceshi(fps_high, dnn_chunk_remain_high, bit_rate)
                    quality_list.append(7)
                if ceshi(fps_ultra, dnn_chunk_remain_ultra, bit_rate) != -1:
                    now_dnn1 = ceshi(fps_ultra, dnn_chunk_remain_ultra, bit_rate)
                    quality_list.append(8)
                if ceshi(fps_medium, dnn_chunk_remain_medium, bit_rate) != -1:
                    now_dnn3 = ceshi(fps_medium, dnn_chunk_remain_medium, bit_rate)
                    quality_list.append(6)
                if ceshi(fps_low, dnn_chunk_remain_low, bit_rate) != -1:
                    now_dnn4 = ceshi(fps_low, dnn_chunk_remain_low, bit_rate)
                    quality_list.append(5)
                now_shengyu = [now_dnn4, now_dnn3, now_dnn2, now_dnn1]
                if len(quality_list) != 0:
                    for i in range(len(quality_list)):
                        bitrate_list.append(bitrate_choice(bit_rate, now_shengyu[quality_list[i]-5], quality_list[i]))
                    max_birate = max(bitrate_list)
                    last_about_dnn_choice = quality_list[bitrate_list.index(max_birate)]
                    now_dnn = now_shengyu[quality_list[bitrate_list.index(max_birate)]-5]

            elif last_about_dnn_choice == 6:
                # medium
                if last_about_dnn_choice == 6 and ceshi(fps_medium, dnn_chunk_remain_medium, bit_rate) != -1:
                    now_dnn3 = ceshi(fps_medium, dnn_chunk_remain_medium, bit_rate)
                    quality_list.append(6)
                if ceshi(fps_ultra, dnn_chunk_remain_ultra, bit_rate) != -1:
                    now_dnn1 = ceshi(fps_ultra, dnn_chunk_remain_ultra, bit_rate)
                    quality_list.append(8)
                if ceshi(fps_high, dnn_chunk_remain_high, bit_rate) != -1:
                    now_dnn2 = ceshi(fps_high, dnn_chunk_remain_high, bit_rate)
                    quality_list.append(7)
                if ceshi(fps_low, dnn_chunk_remain_low, bit_rate) != -1:
                    now_dnn4 = ceshi(fps_low, dnn_chunk_remain_low, bit_rate)
                    quality_list.append(5)
                now_shengyu = [now_dnn4, now_dnn3, now_dnn2, now_dnn1]
                if len(quality_list) != 0:
                    for i in range(len(quality_list)):
                        bitrate_list.append(bitrate_choice(bit_rate, now_shengyu[quality_list[i]-5], quality_list[i]))
                    max_birate = max(bitrate_list)
                    last_about_dnn_choice = quality_list[bitrate_list.index(max_birate)]
                    now_dnn = now_shengyu[quality_list[bitrate_list.index(max_birate)]-5]
            elif last_about_dnn_choice == 5:
                # low
                if last_about_dnn_choice == 5 and ceshi(fps_low, dnn_chunk_remain_low, bit_rate) != -1:
                    now_dnn4 = ceshi(fps_low, dnn_chunk_remain_low, bit_rate)
                    quality_list.append(5)
                if ceshi(fps_ultra, dnn_chunk_remain_ultra, bit_rate) != -1:
                    now_dnn1 = ceshi(fps_ultra, dnn_chunk_remain_ultra, bit_rate)
                    quality_list.append(8)
                if ceshi(fps_high, dnn_chunk_remain_high, bit_rate) != -1:
                    now_dnn2 = ceshi(fps_high, dnn_chunk_remain_high, bit_rate)
                    quality_list.append(7)
                if ceshi(fps_medium, dnn_chunk_remain_medium, bit_rate) != -1:
                    now_dnn3 = ceshi(fps_medium, dnn_chunk_remain_medium, bit_rate)
                    quality_list.append(6)
                now_shengyu = [now_dnn4, now_dnn3, now_dnn2, now_dnn1]

                if len(quality_list) != 0:
                    for i in range(len(quality_list)):
                        bitrate_list.append(bitrate_choice(bit_rate, now_shengyu[quality_list[i]-5], quality_list[i]))
                    max_birate = max(bitrate_list)

                    last_about_dnn_choice = quality_list[bitrate_list.index(max_birate)]
                    now_dnn = now_shengyu[quality_list[bitrate_list.index(max_birate)]-5]
            quality_list = []
            bitrate_list = []
            if now_dnn == 5:
                fps = 0
            else:
                if last_about_dnn_choice == 0:
                    fps = 0
                elif last_about_dnn_choice == 5:
                    fps = fps_low[4 - now_dnn][bit_rate]
                elif last_about_dnn_choice == 6:
                    fps = fps_medium[4 - now_dnn][bit_rate]
                elif last_about_dnn_choice == 7:
                    fps = fps_high[4 - now_dnn][bit_rate]
                elif last_about_dnn_choice == 8:
                    # print(5-replay_dnn_remain)
                    fps = fps_ultra[4 - now_dnn][bit_rate]

            f1 = bitrate_choice(bit_rate, now_dnn, last_about_dnn_choice)
            f2 = bitrate_choice(last_bit_rate, last_dnn_remain, last_last_dnn_choice)
            reward = bitrate_choice(bit_rate, now_dnn, last_about_dnn_choice) / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - np.abs(f1 - f2) / M_IN_K
            last_dnn_remain = now_dnn
            last_last_dnn_choice = last_about_dnn_choice

        elif dnn_choice >= 5:
            if not dnn_reset_num:
                delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                end_of_video, video_chunk_remain, dnn_chunk_remain, dnn_chunk_size = \
                    net_env.get_video_chunk(dnn_choice)
                if dnn_choice == 5:
                    last_about_dnn_choice = dnn_choice
                    dnn_chunk_remain_low = dnn_chunk_remain
                elif dnn_choice == 6:
                    last_about_dnn_choice = dnn_choice
                    dnn_chunk_remain_medium = dnn_chunk_remain
                elif dnn_choice == 7:
                    last_about_dnn_choice = dnn_choice
                    dnn_chunk_remain_high = dnn_chunk_remain
                elif dnn_choice == 8:
                    last_about_dnn_choice = dnn_choice
                    dnn_chunk_remain_ultra = dnn_chunk_remain
                else:
                    print("error1")

                reward = - af2 * rebuf
                chunk_size = dnn_chunk_size

                time_stamp += delay  # in ms
                time_stamp += sleep_time  # in ms

            else:
                dnn_reset_num = False
                reward = -30
        else:
            print("error3")

        r_batch.append(reward)

        last_bit_rate = bit_rate
        # log time_stamp, bit_rate, buffer_size, reward
        if 0 <= dnn_choice < 5:
            if end_of_video:
                bit_rate = dnn_choice
                log_file.write("video chunk" + str(video_counter) + '\t' + str(time_stamp) + '\t' +
                               str(bitrate_choice(bit_rate, now_dnn, last_about_dnn_choice)) + '\t' +
                               str(buffer_size) + '\t' +
                               str(rebuf) + '\t' +
                               str(video_chunk_size) + '\t' +
                               str(delay) + '\t' +
                               str(reward) + '\t' + str(fps) +
                               '\n')
                log_file.flush()
                video_counter = video_counter + 1
            else:
                bit_rate = dnn_choice
                log_file.write("video chunk" + str(video_counter) + '\t' + str(time_stamp) + '\t' +
                               str(bitrate_choice(bit_rate, now_dnn, last_about_dnn_choice)) + '\t' +
                               str(buffer_size) + '\t' +
                               str(rebuf) + '\t' +
                               str(video_chunk_size) + '\t' +
                               str(delay) + '\t' +
                               str(reward) + '\t' + str(fps) +
                               '\n')
                log_file.flush()
                video_counter = video_counter + 1
        elif dnn_choice >= 5 and dnn_chunk_remain >= 0:
            dnn_quality = ["low", "medium", "high", "ultra"]
            log_file.write('DNN_' + dnn_quality[dnn_choice - 5] + "_" + '\t' + str(time_stamp) + '\t' +
                           str(0) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(dnn_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\t' + str(0) +
                           '\n')
            log_file.flush()
        else:
            print("error7")

        # retrieve previous state
        if len(s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :5] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        state[6, -1] = np.minimum(dnn_chunk_remain_low, DNN_CHUNK_TOTAL) / float(DNN_CHUNK_TOTAL)
        state[7, -1] = np.minimum(dnn_chunk_remain_medium, DNN_CHUNK_TOTAL) / float(DNN_CHUNK_TOTAL)
        state[8, -1] = np.minimum(dnn_chunk_remain_high, DNN_CHUNK_TOTAL) / float(DNN_CHUNK_TOTAL)
        state[9, -1] = np.minimum(dnn_chunk_remain_ultra, DNN_CHUNK_TOTAL) / float(DNN_CHUNK_TOTAL)

        _, _, action_prob = actor.get_actor_out(convert_torch(np.reshape(state, (1, S_INFO, S_LEN))))
        action_prob = action_prob.numpy()
        action_cumsum = np.cumsum(action_prob)
        dnn_choice = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()

        s_batch.append(state)

        entropy_record.append(a2c_torch.compute_entropy(action_prob[0]))

        if dnn_choice == 5 and dnn_chunk_remain_low == 0:
            dnn_reset_num = dnn_reset_num + 1
        if dnn_choice == 6 and dnn_chunk_remain_medium == 0:
            dnn_reset_num = dnn_reset_num + 1
        if dnn_choice == 7 and dnn_chunk_remain_high == 0:
            dnn_reset_num = dnn_reset_num + 1
        if dnn_choice == 8 and dnn_chunk_remain_ultra == 0:
            dnn_reset_num = dnn_reset_num + 1

        if end_of_video:
            bitrate_list = []
            quality_list = []
            now_shengyu = []
            now_dnn = 5
            now_dnn1 = 5
            now_dnn2 = 5
            now_dnn3 = 5
            now_dnn4 = 5
            dnn_reset_num = 0
            dnn_chunk_remain_low = 5
            dnn_chunk_remain_medium = 5
            dnn_chunk_remain_high = 5
            dnn_chunk_remain_ultra = 5
            last_about_dnn_choice = 0
            last_last_dnn_choice = 0
            last_dnn_remain = 5
            video_counter = 1
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            dnn_choice = DEFAULT_QUALITY  # use the default action here

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)
            entropy_record = []

            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()
    jilu()
