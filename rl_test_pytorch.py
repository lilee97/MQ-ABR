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
import env
import load_trace
import fps_file as f
from test import b as fps_list

af1 = 1  # 0.8469011
af2 = 4.3  # 28.79591348
af3 = 0.29797156
af4 = 1  # 1.06099887

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
LOG_FILE = './test_results/log_sim_rl'
TEST_TRACES = './cooked_test_traces/'
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
               [725.73410624, 1518.4531871,  1974.48131638, 2562.97867577, 4800.],
               [788.98363924, 1762.70615478, 2102.55637171, 2562.97867577, 4800.],
               [877.10254379, 1932.8554799,  2206.96971921, 2562.97867577, 4800.]]

low_ssim = [[400, 800, 1200, 2400, 4800],
            [400., 805.6611751, 1169.27008533, 983.27858714, 4800.],
            [511.45533019, 944.70850256, 1285.72990339, 981.9136841, 4800.],
            [585.15085038, 1030.93403199, 1543.13425408,  981.9136841,  4800.],
            [643.02737253, 1136.93533087, 1690.29545466,  981.9136841,  4800.],
            [696.93964095, 1182.44220559, 1797.71762327,  981.9136841,  4800.]]

NN_MODEL = sys.argv[1]


def suiJi(a, b):
    c = round(a + (b - a) * np.random.random(), 3)
    return c


def convert_torch(variable, dtype=np.float32):
    if variable.dtype != dtype:
        variable = variable.astype(dtype)

    return torch.from_numpy(variable)


def bitrate_choice(dnn_choice, dnn_chunk_remain, last_about_dnn_choice):  # 最后一个形参是为了让该函数知道上一次选择的DNN质量
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


def main():
    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM - 4

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
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
    last_last_dnn_choice = 0
    last_dnn_remain = 5
    offset = 0
    fps = 0
    avg_fps = 0
    sum_fps = 0
    c_fps = 0
    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real

        if 0 <= dnn_choice < 5:  # 最好>=0

            # print("VIDEO_CHUNK下载")
            bit_rate = dnn_choice
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain, dnn_chunk_remain, dnn_chunk_size = \
                net_env.get_video_chunk(bit_rate)

            chunk_size = video_chunk_size

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms
            if last_about_dnn_choice == 0:
                fps = 0
            elif last_about_dnn_choice == 5:
                fps = f.low_fps[4 - dnn_chunk_remain_low][bit_rate]
                c_fps = c_fps + 1
            elif last_about_dnn_choice == 6:
                fps = f.medium_fps[4 - dnn_chunk_remain_medium][bit_rate]
                c_fps = c_fps + 1
            elif last_about_dnn_choice == 7:
                fps = f.high_fps[4 - dnn_chunk_remain_high][bit_rate]
                c_fps = c_fps + 1
            elif last_about_dnn_choice == 8:
                fps = f.ultra_fps[4 - dnn_chunk_remain_ultra][bit_rate]
                c_fps = c_fps + 1

            if last_about_dnn_choice == 0:
                avg_fps = 11
            else:
                sum_fps = sum_fps + fps
                avg_fps = sum_fps / c_fps


            if not end_of_video:
                replay_dnn_remain = dnn_chunk_remain

            f1 = bitrate_choice(bit_rate, replay_dnn_remain, last_about_dnn_choice)
            f2 = bitrate_choice(last_bit_rate, last_dnn_remain, last_last_dnn_choice)
            reward = 0.9 * (bitrate_choice(bit_rate, replay_dnn_remain, last_about_dnn_choice) / M_IN_K
                            - af2 * rebuf
                            - np.abs(f1 - f2) / M_IN_K) + 0.1 * (avg_fps - 11) / 28.5
            reward1 = bitrate_choice(bit_rate, replay_dnn_remain,last_about_dnn_choice) / M_IN_K - af2 * rebuf - np.abs(f1 - f2) / M_IN_K
            last_last_dnn_choice = last_about_dnn_choice
            last_dnn_remain = dnn_chunk_remain

        elif dnn_choice >= 5:
            if not dnn_reset_num:
                # print("DNN_CHUNK下载")
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
                reward = -28
        else:
            print("error3")

        r_batch.append(reward)

        last_bit_rate = bit_rate
        # log time_stamp, bit_rate, buffer_size, reward
        if 0 <= dnn_choice < 5:
            if end_of_video:
                bit_rate = dnn_choice
                log_file.write("video chunk" + str(video_counter) + '\t' + str(time_stamp) + '\t' +
                               str(bitrate_choice(bit_rate, replay_dnn_remain, last_about_dnn_choice)) + '\t' +
                               str(buffer_size) + '\t' +
                               str(rebuf) + '\t' +
                               str(video_chunk_size) + '\t' +
                               str(delay) + '\t' +
                               str(fps) + '\t' +
                               str(reward1) + '\t' +
                               str(reward) + '\n')
                log_file.flush()
                video_counter = video_counter + 1
            else:
                bit_rate = dnn_choice
                log_file.write("video chunk" + str(video_counter) + '\t' + str(time_stamp) + '\t' +
                               str(bitrate_choice(bit_rate, dnn_chunk_remain, last_about_dnn_choice)) + '\t' +
                               str(buffer_size) + '\t' +
                               str(rebuf) + '\t' +
                               str(video_chunk_size) + '\t' +
                               str(delay) + '\t' +
                               str(fps) + '\t' +
                               str(reward1) + '\t' +
                               str(reward) + '\n')
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
                           str(0) + '\t' +
                           str(reward) + '\t' +
                           str(reward) + '\n')
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
            avg_fps = 0
            sum_fps = 0
            c_fps = 0
            offset = 0
            fps = 0
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

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]  # 注意,trace_idx在env中结束每一轮视频后都是随机的
            log_file = open(log_path, 'w')



if __name__ == '__main__':
    main()
