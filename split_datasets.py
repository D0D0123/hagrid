import os
import shutil

# 350 : 150 images gives a 70 : 30 split, 400 : 100 gives a 80 : 20 split



# print(len(imgnames))
SPLIT_TRAIN = True


for gesture in ['call', 'dislike', 'fist', 'like', 'mute', 'ok', 'peace', 'peace_inverted', 'stop', 'stop_inverted']:
    if SPLIT_TRAIN == True:
        imgnames = sorted(os.listdir(f"./test_dataset/{gesture}"))
        for i in range(400):
            shutil.copy(f'./test_dataset/{gesture}/{imgnames[i]}', f'./subsamples_train1/{gesture}/{imgnames[i]}')
    else: # split test
        imgnames = sorted(os.listdir(f"./test_dataset/{gesture}"), reverse=True)
        for i in range(100):
            shutil.copy(f'./test_dataset/{gesture}/{imgnames[i]}', f'./subsamples_test1/{gesture}/{imgnames[i]}')
