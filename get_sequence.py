from datetime import datetime

import numpy as np
from PIL import Image
import os
import time


IMAGE_PATH = 'c:/pycharmProject/ImagesSequencesPredictions/samples'
# IMAGE_PATH = 'c:/data/radar/sample/12/'
#
# WIDTH = 1185
# HEIGHT = 1104
WIDTH = 100
HEIGHT = 100
SEQUENCE = np.array([],dtype=np.int32)
NUMBER = 0

def image_initialize(image):
    picture = Image.open(image)
    picture = picture.crop((243, 176, 1428, 1280))
    # print('width =' ,picture.width)
    # print('height =', picture.height)
    picture = picture.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    picture = picture.convert('L')
    picture.save('c:/pycharmProject/ImagesSequencesPredictions/1.png')  # 非保留
    # print(picture.getdata())
    data = np.array(picture.getdata(),dtype=np.int32).reshape(WIDTH, HEIGHT)
    return data

for file in os.listdir(IMAGE_PATH):
    # print(os.path.join(IMAGE_PATH, directories))
    # print(os.path.join(os.path.join(IMAGE_PATH, directories), file))
    image_array = image_initialize(os.path.join(IMAGE_PATH, file))
    SEQUENCE = np.append(SEQUENCE, image_array)
    NUMBER += 1
    print(NUMBER)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

SEQUENCE = SEQUENCE.reshape(NUMBER, WIDTH * HEIGHT)
# for i in SEQUENCE:
#     for j in range(int(len(i))):
#         if i[j] < 50
#             i[j] = 0


np.savez('sequence_array.npz', sequence_array=SEQUENCE)
print('Data saved.')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
