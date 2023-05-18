# coding=utf-8
import numpy as np

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 49  # 49
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
OUTPUT_P_INDEX = 9
BITRATE_LEVELS = 5
TOTAL_VIDEO_CHUNCK = 214.0
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
VIDEO_SIZE_FILE = './video_size_'
DNN_CHUNK_TOTAL = 5.0

DNN_CHUNK = [[68143, 32719, 31215, 31215, 31215],
             [299631, 134991, 130287, 130287, 130287],
             [736047, 344143, 324335, 324335, 324335],
             [1573103, 742735, 692719, 692719, 692719]]

class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, random_seed=RANDOM_SEED):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw
        self.video_chunk_counter = 0
        self.buffer_size = 0
        self.video_chunk_remain = 0
        self.dnn_chunk_remain = 5
        self.dnn_chunk_remain_ultra = 5
        self.dnn_chunk_remain_high = 5
        self.dnn_chunk_remain_medium = 5
        self.dnn_chunk_remain_low = 5

        # pick a random trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]
        self.video_size = {}  # in bytes
        for bitrate in range(BITRATE_LEVELS):  # 5
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

    def get_video_chunk(self, dnn_choice):
        assert dnn_choice >= 0
        assert dnn_choice < OUTPUT_P_INDEX
        dnn_chunk_size = 0
        video_chunk_size = 0
        if 0 <= dnn_choice < BITRATE_LEVELS:
            quality = dnn_choice
            video_chunk_size = self.video_size[quality][self.video_chunk_counter]  # video_chunk_counter初始为0
            chunk_size = video_chunk_size
        else:
            if dnn_choice == 5:
                dnn_chunk_size = DNN_CHUNK[dnn_choice - 5][5 - self.dnn_chunk_remain_low]
                self.dnn_chunk_remain_low = self.dnn_chunk_remain_low - 1
                self.dnn_chunk_remain = self.dnn_chunk_remain_low
                if self.dnn_chunk_remain < 0:
                    print("error6")
            elif dnn_choice == 6:
                dnn_chunk_size = DNN_CHUNK[dnn_choice - 5][5 - self.dnn_chunk_remain_medium]
                self.dnn_chunk_remain_medium = self.dnn_chunk_remain_medium - 1
                self.dnn_chunk_remain = self.dnn_chunk_remain_medium
                if self.dnn_chunk_remain < 0:
                    print("error7")
            elif dnn_choice == 7:
                dnn_chunk_size = DNN_CHUNK[dnn_choice - 5][5 - self.dnn_chunk_remain_high]
                self.dnn_chunk_remain_high = self.dnn_chunk_remain_high - 1
                self.dnn_chunk_remain = self.dnn_chunk_remain_high
                if self.dnn_chunk_remain < 0:
                    print("error8")
            elif dnn_choice == 8:
                dnn_chunk_size = DNN_CHUNK[dnn_choice - 5][5 - self.dnn_chunk_remain_ultra]
                self.dnn_chunk_remain_ultra = self.dnn_chunk_remain_ultra - 1
                self.dnn_chunk_remain = self.dnn_chunk_remain_ultra
                if self.dnn_chunk_remain < 0:
                    print("error9")
            else:
                print("error")

            chunk_size = dnn_chunk_size
        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        chunk_counter_sent = 0  # in bytes

        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] \
                         * B_IN_MB / BITS_IN_BYTE
            duration = self.cooked_time[self.mahimahi_ptr] \
                       - self.last_mahimahi_time
            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION  # 0.95
            if chunk_counter_sent + packet_payload > chunk_size:
                fractional_time = (chunk_size - chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                assert (self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
                break

            chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1
            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay *= MILLISECONDS_IN_SECOND  # 1000
        delay += LINK_RTT

        # add a multiplicative noise to the delay
        delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)
        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)
        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        if 0 <= dnn_choice < BITRATE_LEVELS:
            self.buffer_size += VIDEO_CHUNCK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:  # 60s
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME  # 500ms
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size
        if 0 <= dnn_choice < BITRATE_LEVELS:
            self.video_chunk_counter += 1
            self.video_chunk_remain = TOTAL_VIDEO_CHUNCK - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= TOTAL_VIDEO_CHUNCK:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0
            self.video_chunk_remain = 0
            if self.dnn_chunk_remain_low == 5 and self.dnn_chunk_remain_medium == 5 and self.dnn_chunk_remain_high == 5 and self.dnn_chunk_remain_ultra == 5:
                print("No DNN downloaded")
            self.dnn_chunk_remain_ultra = 5
            self.dnn_chunk_remain_high = 5
            self.dnn_chunk_remain_medium = 5
            self.dnn_chunk_remain_low = 5

            # pick a random trace file
            self.trace_idx = np.random.randint(len(self.all_cooked_time))
            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        next_video_chunk_sizes = []
        del next_video_chunk_sizes[:]
        for i in range(BITRATE_LEVELS):  # 5
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])

        return delay, \
               sleep_time, \
               return_buffer_size / MILLISECONDS_IN_SECOND, \
               rebuf / MILLISECONDS_IN_SECOND, \
               video_chunk_size, \
               next_video_chunk_sizes, \
               end_of_video, \
               self.video_chunk_remain, \
               self.dnn_chunk_remain, \
               dnn_chunk_size
