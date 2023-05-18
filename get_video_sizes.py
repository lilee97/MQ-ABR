import os


TOTAL_VIDEO_CHUNCK = 88
BITRATE_LEVELS = 5
VIDEO_PATH = './video/'
VIDEO_FOLDER = 'video'

# assume videos are in ../video_servers/video[1, 2, 3, 4, 5]
# the quality at video5 is the lowest and video1 is the highest


for bitrate in range(BITRATE_LEVELS):
	with open('video_size_' + str(bitrate), 'w') as f:
		for chunk_num in range(1, TOTAL_VIDEO_CHUNCK+1):
			video_chunk_path = VIDEO_PATH + VIDEO_FOLDER + str(BITRATE_LEVELS - bitrate) + '/' + 'segment_' + str(chunk_num) + '.m4s'
			chunk_size = os.path.getsize(video_chunk_path)
			f.write(str(chunk_size) + '\n')

# for bitrate in range(1):
# 	with open('./dnn_chunk_size/ultra/chunk_size_' + str(bitrate), 'w') as f:
# 		for chunk_num in range(1, 10):
# 			video_chunk_path = './DNN_chunk/ultra/DNN_chunk_' + str(chunk_num) + '.pth'
# 			chunk_size = os.path.getsize(video_chunk_path)
# 			f.write(str(chunk_size) + '\n')

