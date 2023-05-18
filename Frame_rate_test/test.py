import subprocess as sub

# i = 37400
#
# while True:
#     i = i + 100
#     command = "python Frame_rate_test.py " + str(i)
#     pipe = sub.Popen(command, shell=True)
#     pipe.wait()
#     if i == 47500:
#         break


Minimum_frame = [10, 15, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 38, 43, 50]

i = 0
while True:
    i = i + 1
    command = "python Minimum_frame_rate.py " + str(Minimum_frame[i-1])
    pipe = sub.Popen(command, shell=True)
    pipe.wait()
    if i == 24:
        break

