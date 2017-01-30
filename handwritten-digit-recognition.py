__author__ = 'nikhandelwal'


from keras.datasets import mnist # subroutines for fetching the MNIST dataset
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Dense # the two types of neural network layer we will be using
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values

batch_size = 128 # in each iteration, we consider 128 training examples at once
num_epochs = 20 # we iterate twenty times over the entire training set
hidden_size = 512 # there will be 512 neurons in both hidden layers

num_train = 60000 # there are 60000 training examples in MNIST
num_test = 10000 # there are 10000 test examples in MNIST

height, width, depth = 28, 28, 1 # MNIST images are 28x28 and greyscale
num_classes = 10 # there are 10 classes (1 per digit)


(X_train , y_train), (X_test, y_test) = mnist.load_data() # fetch MNIST data

X_train = X_train.reshape(num_train, height*width)
X_test = X_test.reshape(num_test, height*width)
X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /=255 # Normalise data to [0, 1] range
X_test /=255 # Normalise data to [0, 1] range

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


inp = Input(shape = (height*width,))

hidden_1 =  Dense(hidden_size, activation='relu')(inp)

hidden_2 = Dense(hidden_size,activation='relu')(hidden_1)

out = Dense(num_classes , activation='softmax')(hidden_2)

model = Model(input = inp , output= out)

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size =batch_size, nb_epoch=num_epochs,
          verbose=1, validation_split=0.1)

model.evaluate(X_test, y_test, verbose=1)


