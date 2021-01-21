import numpy as np
from keras import models
from keras import layers


network = models.Sequential()
network.add(layers.Dense(5, activation='relu', input_shape=(3,)))
network.add(layers.Dense(4, activation='relu', input_shape=(5,)))
network.add(layers.Dense(1, activation='sigmoid'))
network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


def generate_batch(size):
    n = 0
    while True):
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

# test_loss, test_acc = network.evaluate(test_images, test_labels)
# print('test_acc:', test_acc, 'test_loss', test_loss)

print(network.predict(np.array([[1,1,1]])))