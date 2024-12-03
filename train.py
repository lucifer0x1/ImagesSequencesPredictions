import os
from get_sequence import WIDTH
from get_sequence import HEIGHT

import numpy as np
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv3D
from keras._tf_keras.keras.layers import ConvLSTM2D
from keras._tf_keras.keras.layers import BatchNormalization
import time
from keras import optimizers

# WIDTH = 1185
# HEIGHT = 1104

FRAMES = 16

SEQUENCE = np.load('sequence_array.npz')['sequence_array']  # load array
print(SEQUENCE[0])
print('Data loaded.')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

NUMBER = len(SEQUENCE)

'''
i = 0
while i < NUMBER:
    if (i + 1) % 11 != 0:
        BASIC_SEQUENCE = np.append(BASIC_SEQUENCE, SEQUENCE[i])
        NEXT_SEQUENCE = np.append(NEXT_SEQUENCE, SEQUENCE[i+1])
    i += 1
    print(i)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
'''

# step =1
SEQUENCE = SEQUENCE.reshape(NUMBER, WIDTH, HEIGHT,1)
BASIC_SEQUENCE = np.zeros((NUMBER-FRAMES, FRAMES, WIDTH, HEIGHT,1),dtype=np.int32)
NEXT_SEQUENCE = np.zeros((NUMBER-FRAMES, FRAMES, WIDTH, HEIGHT,1),dtype=np.int32)

print("BASIC_SEQUENCE = " , NUMBER-FRAMES, FRAMES, WIDTH, HEIGHT)
for i in range(FRAMES):
    BASIC_SEQUENCE[:, i, :, :] = SEQUENCE[i:i+NUMBER-FRAMES]
    NEXT_SEQUENCE[:, i, :, :] = SEQUENCE[i+1:i+NUMBER-FRAMES+1]


#plt.imshow(BASIC_SEQUENCE[0][0].reshape(100, 100))
#plt.show()
# build model

seq = Sequential()

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),input_shape=(None, WIDTH, HEIGHT,1), padding='same', return_sequences=True))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(filters=60, kernel_size=(3, 3), padding='same', return_sequences=True))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(filters=60, kernel_size=(3, 3), padding='same', return_sequences=True))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3 ,3), activation='sigmoid', padding='same', data_format='channels_last'))

# sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)

# seq.compile(loss='binary_crossentropy', optimizer='adadelta')
'''
seq.compile(loss='mean_squared_error', optimizer='adadelta')

seq.fit(BASIC_SEQUENCE[:10], NEXT_SEQUENCE[:10], batch_size=32,
        epochs=2, validation_split=0.05)
'''

# parallel_model = multi_gpu_model(seq, gpus=4)

sgd = optimizers.SGD(learning_rate=0.01, clipnorm=1)
#rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
#adadelta_ = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
seq.compile(loss='mean_squared_error', optimizer='adadelta')
seq.summary()
seq.fit(BASIC_SEQUENCE, NEXT_SEQUENCE, batch_size=4, epochs=12, validation_split=0.05)


seq.save('my_model.keras')


# 1201.23
# 1132.25 385.68

