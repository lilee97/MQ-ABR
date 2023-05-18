import linecache
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
from tensorboardX import SummaryWriter
import fps_file as f
from test import b as fps_list


S_INFO = 10  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 9  # take how many frames in the past
A_DIM = 9  # 5-low 6-medium 7-high 8-ultra
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 8
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [400, 800, 1200, 2400, 4800]  # Kbps
HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 214.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
TEST_LOG_FOLDER = './test_results/'
TRAIN_TRACES = './cooked_traces/'
NN_MODEL = './results/pretrain_linear_reward.ckpt'
# NN_MODEL = None
DNN_CHUNK = [1, 2, 3, 4, 5]
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

def convert_torch(variable, dtype=np.float32):
    if variable.dtype != dtype:
        variable = variable.astype(dtype)

    return torch.from_numpy(variable)


def suiJi(a, b):
    c = round(a + (b - a) * np.random.random(), 3)
    return c


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


def testing(epoch, nn_model, log_file):
    # clean up the test results folder
    # os.system('del ' + TEST_LOG_FOLDER)
    # os.system('mkdir ' + TEST_LOG_FOLDER)

    # run test script
    os.system('python rl_test_pytorch.py ' + nn_model)

    # append test performance to the log
    rewards = []
    rewards1 = []
    a = []
    fp1 = []
    sr_count = 0
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward = []
        reward1 = []
        fp2 = []
        with open(TEST_LOG_FOLDER + test_log_file, 'r') as f:
            for line in f:
                parse = line.split()
                try:
                    reward.append(float(parse[-1]))
                    reward1.append(float(parse[-2]))
                    fp2.append(float(parse[-3]))
                    if float(parse[-3]) != 0:
                        sr_count = sr_count + 1
                except IndexError:
                    break
        rewards.append(np.mean(reward[1:]))
        rewards1.append(np.mean(reward1[1:]))
        if sr_count != 0:
            fp1.append(sum(fp2) / sr_count)
        else:
            fp1.append(0)
        sr_count = 0

    rewards = np.array(rewards)
    rewards1 = np.array(rewards1)

    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)
    rewards1_mean = np.mean(rewards1)
    fp_mean = np.mean(fp1)

    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(fp_mean) + '\t' +
                   str(rewards1_mean) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\n')
    log_file.flush()
    a.append(fp_mean)
    a.append(rewards1_mean)
    a.append(rewards_mean)
    return a


def central_agent(net_params_queues, exp_queues):
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)

    write_test = SummaryWriter(SUMMARY_DIR)
    with open(LOG_FILE + '_test', 'w') as test_log_file:
        actor = a2c_torch.ActorNet(s_dim=[S_INFO, S_LEN], a_dim=A_DIM, lr=ACTOR_LR_RATE)
        critic = a2c_torch.CriticNet(s_dim=[S_INFO, S_LEN], lr=CRITIC_LR_RATE)
        actor_optim = optim.RMSprop(actor.parameters(), lr=ACTOR_LR_RATE)
        critic_optim = optim.RMSprop(critic.parameters(), lr=CRITIC_LR_RATE)
        epoch = 0
        while True:
            actor_net_params = actor.state_dict()
            critic_net_params = critic.state_dict()
            # print(actor_net_params)
            for i in range(NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params])
            # print('parameters hahaha: ', actor_net_params)
            # print('parameters: ', list(actor_net_params))
            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_entropy = 0.0
            total_agents = 0.0
            actor_optim.zero_grad()
            critic_optim.zero_grad()
            for i in range(NUM_AGENTS):
                s_batch, a_batch, r_batch, old_pi_batch, terminal, info = exp_queues[i].get()
                # print('exp_queue')
                s_batch, a_batch, \
                r_batch, old_pi_batch, terminal = convert_torch(np.array(s_batch)), convert_torch(np.array(a_batch)), \
                                                  convert_torch(np.array(r_batch)), convert_torch(
                    np.array(old_pi_batch)), convert_torch(np.array(terminal))
                # actor, critic = actor.cuda(), critic.cuda()
                # s_batch, a_batch, r_batch, old_pi_batch, terminal = s_batch.cuda(), a_batch.cuda(), r_batch.cuda(), old_pi_batch.cuda(), terminal.cuda()
                critic_loss, td_batch = critic.cal_loss(s_batch, r_batch, terminal)
                # critic_loss, td_batch = critic_loss.cuda(), td_batch.cuda()
                actor_loss = actor.cal_loss(s_batch, a_batch, td_batch, epoch)
                # actor_loss = actor_loss.cuda()

                critic_loss.backward()
                actor_loss.backward()
                total_reward += np.sum(r_batch.numpy())
                total_td_loss += np.sum(td_batch.numpy())
                total_batch_len += len(r_batch.numpy())
                total_agents += 1.0
                total_entropy += np.sum(info['entropy'])
            critic_optim.step()
            actor_optim.step()
            # actor.cpu(), critic.cpu()
            epoch += 1
            avg_reward = total_reward / total_agents
            avg_td_loss = total_td_loss / total_batch_len
            avg_entropy = total_entropy / total_batch_len

            # logging.info('Epoch: ' + str(epoch) +
            #              ' TD_loss: ' + str(avg_td_loss) +
            #              ' Avg_reward: ' + str(avg_reward) +
            #              ' Avg_entropy: ' + str(avg_entropy))
            if epoch % MODEL_SAVE_INTERVAL == 0:  # 100
                # Save the neural net parameters to disk.
                print('Epoch = ', epoch)
                torch.save(actor.state_dict(), SUMMARY_DIR + "/actor_nn_model_ep_" +
                           str(epoch) + ".pkl")
                torch.save(critic.state_dict(), SUMMARY_DIR + "/critic_nn_model_ep_" +
                           str(epoch) + ".pkl")

                # logging.info("Model saved in file: " + save_path)
                reward_mean = testing(epoch,
                                      SUMMARY_DIR + "/actor_nn_model_ep_" + str(epoch) + ".pkl",
                                      test_log_file)

                print('epoch = ', epoch, 'reward = ', reward_mean)
                write_test.add_scalar('Testing/total_reward', reward_mean[2], epoch)
                write_test.add_scalar('Training/Entropy', avg_entropy, epoch)
                write_test.add_scalar('Training/TD_Error', avg_td_loss, epoch)

                write_test.flush()
                # summary_str = sess.run(summary_ops, feed_dict={
                #     summary_vars[0]: avg_td_loss,
                #     summary_vars[1]: reward_mean,
                #     summary_vars[2]: avg_entropy
                # })

                # writer.add_summary(summary_str, epoch)
                # writer.flush()


def agent(agent_id, all_cooked_time, all_cooked_bw, net_params_queue, exp_queue):
    # print(str(agent_id) + "3")
    # mylog.write(str(agent_id) + "3\n")
    # mylog.flush()

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              random_seed=agent_id)

    # with open(LOG_FILE + '_agent_' + str(agent_id), 'w') as log_file:
    actor = a2c_torch.ActorNet(s_dim=[S_INFO, S_LEN], a_dim=A_DIM, lr=ACTOR_LR_RATE)
    critic = a2c_torch.CriticNet(s_dim=[S_INFO, S_LEN], lr=CRITIC_LR_RATE)

    # initial synchronization of the network parameters from the coordinator
    actor_net_params, critic_net_params = net_params_queue.get()
    actor.load_state_dict(actor_net_params)
    critic.load_state_dict(critic_net_params)

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    old_pi_batch = [action_vec]
    r_batch = []
    entropy_record = []

    time_stamp = 0
    dnn_choice = bit_rate
    epoch = 1

    video_counter = 1
    dnn_reset_num = False
    last_about_dnn_choice = 0
    dnn_chunk_remain_low = 5
    dnn_chunk_remain_medium = 5
    dnn_chunk_remain_high = 5
    dnn_chunk_remain_ultra = 5
    last_last_dnn_choice = 0
    last_dnn_remain = 5
    fps = 0
    avg_fps = 0
    sum_fps = 0
    c_fps = 0
    while True:  # experience video streaming forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        # start_time = time.time()
        if not dnn_reset_num:
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain, dnn_chunk_remain, dnn_chunk_size = \
                net_env.get_video_chunk(dnn_choice)

            if 0 <= dnn_choice < 5:
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

                bit_rate = dnn_choice
                chunk_size = video_chunk_size

                time_stamp += delay  # in ms
                time_stamp += sleep_time  # in ms

                # -- linear reward --
                # reward is video quality - rebuffer penalty - smoothness
                # if fps_list[0] <= fps <= fps_list[25]:
                #     offset = -0.15
                # elif fps_list[25] < fps <= fps_list[51]:
                #     offset = 0
                # elif fps_list[51] < fps <= fps_list[79]:
                #     offset = 0.5
                # elif fps == 0:
                #     offset = 0

                reward = 0.9*(bitrate_choice(bit_rate, dnn_chunk_remain, last_about_dnn_choice) / M_IN_K
                          - REBUF_PENALTY * rebuf
                          - SMOOTH_PENALTY * np.abs(bitrate_choice(bit_rate, dnn_chunk_remain, last_about_dnn_choice) -
                                                   bitrate_choice(last_bit_rate, last_dnn_remain,
                                                                  last_last_dnn_choice)) / M_IN_K) + 0.1*(avg_fps-11)/28.5
                last_last_dnn_choice = last_about_dnn_choice
                last_dnn_remain = dnn_chunk_remain
            elif 5 <= dnn_choice <= 8:
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
                print("error3")
        else:
            dnn_reset_num = False
            reward = -28

        r_batch.append(reward)

        last_bit_rate = bit_rate

        # retrieve previous state
        if len(s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        # print(buffer_size)
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

        # compute action probability vector
        _, _, action_prob = actor.get_actor_out(convert_torch(np.reshape(state, (1, S_INFO, S_LEN))))
        action_prob = action_prob.numpy()
        # print(action_prob)
        action_cumsum = np.cumsum(action_prob)
        dnn_choice = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(
            RAND_RANGE)).argmax()
        # print(dnn_choice)
        # Note: we need to discretize the probability into 1/RAND_RANGE steps,
        # because there is an intrinsic discrepancy in passing single state and batch states

        entropy_record.append(a2c_torch.compute_entropy(action_prob[0]))

        if dnn_choice == 5 and dnn_chunk_remain_low <= 0:
            dnn_reset_num = True
        if dnn_choice == 6 and dnn_chunk_remain_medium <= 0:
            dnn_reset_num = True
        if dnn_choice == 7 and dnn_chunk_remain_high <= 0:
            dnn_reset_num = True
        if dnn_choice == 8 and dnn_chunk_remain_ultra <= 0:
            dnn_reset_num = True

        # log time_stamp, bit_rate, buffer_size, reward
        if 0 <= dnn_choice < 5:
            bit_rate = dnn_choice
            log_file.write("video chunk" + str(video_counter) + '\t' + str(time_stamp) + '\t' +
                           str(bitrate_choice(bit_rate, dnn_chunk_remain, last_about_dnn_choice)) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')
            log_file.flush()
            video_counter = video_counter + 1
        elif dnn_choice >= 5 and dnn_chunk_remain >= 0:
            dnn_quality = ["low", "medium", "high", "ultra"]
            log_file.write(
                'DNN_' + dnn_quality[dnn_choice - 5] + "_" + '\t' + str(time_stamp) + '\t' +
                str(0) + '\t' +
                str(buffer_size) + '\t' +
                str(rebuf) + '\t' +
                str(dnn_chunk_size) + '\t' +
                str(delay) + '\t' +
                str(reward) + '\n')
            log_file.flush()

        # report experience to the coordinator
        # if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:
        if end_of_video:
            video_counter = 1
            exp_queue.put([s_batch[1:],  # ignore the first chuck
                           a_batch[1:],  # since we don't have the
                           r_batch[1:],  # control over it
                           old_pi_batch[1:],
                           end_of_video,
                           {'entropy': entropy_record}])

            # synchronize the network parameters from the coordinator
            actor_net_params, critic_net_params = net_params_queue.get()
            actor.load_state_dict(actor_net_params)
            critic.load_state_dict(critic_net_params)

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]
            del old_pi_batch[:]
            del entropy_record[:]

            log_file.write('\n')  # so that in the log we know where video ends
            dnn_choice = DEFAULT_QUALITY

        # store the state and action into batches
        if end_of_video:
            c_fps = 0
            sum_fps = 0
            avg_fps = 0
            fps = 0
            newline = 1
            if epoch % 25 == 0:
                print("agent" + str(agent_id) + ":" + str(epoch) + "train end")
            epoch = epoch + 1
            dnn_reset_num = False
            dnn_chunk_remain_low = 5
            dnn_chunk_remain_medium = 5
            dnn_chunk_remain_high = 5
            dnn_chunk_remain_ultra = 5
            last_bit_rate = DEFAULT_QUALITY
            last_about_dnn_choice = 0
            last_last_dnn_choice = 0
            last_dnn_remain = 5
            dnn_choice = DEFAULT_QUALITY  # use the default action here

            action_vec = np.zeros(A_DIM)
            action_vec[dnn_choice] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)
            old_pi_batch.append(action_vec)

        else:
            s_batch.append(state)

            action_vec = np.zeros(A_DIM)
            action_vec[dnn_choice] = 1
            a_batch.append(action_vec)
            old_pi_batch.append(action_prob)


def main():
    np.random.seed(RANDOM_SEED)
    assert len(VIDEO_BIT_RATE) == A_DIM - 4

    # create result directory
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []

    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(0))
        exp_queues.append(mp.Queue(0))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)

    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))

    coordinator.start()

    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)
    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i, all_cooked_time, all_cooked_bw,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in range(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
