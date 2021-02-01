import numpy as np
from keras import models
from keras import layers





def generate_batch(size):
    n = 0
    while True:
        if n == 0:
            n = 1
            train_batch = [[1,1,1],
                           [1,1,0],
                           [1,0,0],
                           [1,0,1]]
            label_batch = [[1],[1],[1],[1]]
            yield (np.array(train_batch), np.array(label_batch))
        else:
            n = 0
            train_batch = [[0,1,1],
                           [0,1,0],
                           [0,0,0],
                           [0,0,1]]
            label_batch = [[0],[0],[0],[0]]
            yield (np.array(train_batch), np.array(label_batch))
            

network.fit(x = train_set_generator(), epochs=50, steps_per_epoch=1000)

