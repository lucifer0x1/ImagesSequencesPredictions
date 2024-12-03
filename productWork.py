from token import NUMBER

import numpy as np
import matplotlib.pyplot as plt
from keras._tf_keras.keras.models import load_model

#
# from get_sequence import WIDTH
# from get_sequence import HEIGHT

WIDTH = 100
HEIGHT = 100

SEQUENCE = np.load('sequence_array.npz')['sequence_array']  # load array

NUMBER = len(SEQUENCE)

# WIDTH = 1185
# HEIGHT = 1104
FRAMES = 16
which = 0

# step =1

SEQUENCE = SEQUENCE.reshape(NUMBER, WIDTH,HEIGHT,1 )
BASIC_SEQUENCE = np.zeros((NUMBER-FRAMES, FRAMES, WIDTH, HEIGHT,1))
NEXT_SEQUENCE = np.zeros((NUMBER-FRAMES, FRAMES, WIDTH, HEIGHT, 1))

seq = load_model("my_model.keras")

for i in range(FRAMES):
    BASIC_SEQUENCE[:,i,:,:,:] = SEQUENCE[i:i+NUMBER-FRAMES]
    NEXT_SEQUENCE[:, i, :, :, :] = SEQUENCE[i+1:i + NUMBER - FRAMES+1]
track = BASIC_SEQUENCE[which][::, ::,::,::]
for j in range(FRAMES+1):
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    # print(len(new_pos))
    # plt.text(1, 3, 'Predictions', fontsize=20)
    # plt.imshow(new_pos[j, :,:, :, :, 0], cmap='binary')
    # plt.savefig('pre/%i_pre.png' % (j + 1))

    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)


# And then compare the predictions
# to the ground truth
track2 = BASIC_SEQUENCE[which][::, ::, ::,::]
for i in range(FRAMES):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    if i >= 8:
        ax.text(1, 3, 'Predictions !', fontsize=20,)
    else:
        ax.text(1, 3, 'Inital trajectory', fontsize=20)
    toplot = track[i, ::, ::, 0]
    plt.imshow(toplot, cmap='binary')
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=20)
    toplot = SEQUENCE[i,::,::,0]
    # toplot = track2[i, ::, ::, 0]
    if i >= 8:
        toplot = NEXT_SEQUENCE[which][i - 1, ::, ::, 0]
    plt.imshow(toplot, cmap='binary')
    print('save ~pre/%i_animate.png' % (i + 1))
    plt.savefig('pre/%i_animate.png' % (i + 1))

