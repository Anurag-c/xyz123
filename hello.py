# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from keras import backend as K
from keras.layers import Layer
from keras import activations
from keras import utils
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.layers import add,concatenate,Multiply,Conv2D,AveragePooling2D,BatchNormalization,Reshape,Input
from keras.layers import Flatten,Dense,Dropout,LeakyReLU,UpSampling2D,ZeroPadding2D,Lambda
from keras.preprocessing import image
import tensorflow as tf   
import numpy as np 
from tqdm import tqdm
import os
import cv2
from keras.utils import to_categorical
batch_size = 128
num_classes = 2
epochs = 10

"""
 For the compression function, we use 0.5 instead of 1 in the hinton paper. If it is 1, the norm of all vectors will be reduced.
 If it is 0.5, the norm less than 0.5 will shrink, and those larger than 0.5 will be enlarged.
"""
def squash(x, axis=-1):
    s_quared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_quared_norm) / (0.5 + s_quared_norm)
    result = scale * x
    return result


# Define our own softmax function instead of K.softmax. Because K.softmax cannot specify the axis
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    result = ex / K.sum(ex, axis=axis, keepdims=True)
    return result


# Define edge loss, enter y_true, p_pred, return the score, you can fit when you pass
def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    result = K.sum(y_true * K.square(K.relu(1 - margin -y_pred))
    + lamb * (1-y_true) * K.square(K.relu(y_pred - margin)), axis=-1)
    return result


class Capsule(Layer):
    """ Writing your own Keras layer requires rewriting 3 methods and initialization methods
         1.build(input_shape): This is where you define the weights.
         This method must be set to self.built = True, which can be done by calling super([Layer], self).build().
         2.call(x): This is where the functional logic of the layer is written.
         You only need to focus on the first argument to the incoming call: enter the tensor unless you want your layer to support masking.
    3.compute_output_shape(input_shape):
           If your layer changes the shape of the input tensor, you should define the logic of the shape change here, which allows Keras to automatically infer the shape of each layer.
         4. Initialization method, parameters that your neural layer needs to accept
    """
    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)  # Capsule inherits **kwargs parameter
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activation.get(activation)  # get the activation function

    # Define weights
    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            # 
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        super(Capsule, self).build(input_shape)  # Must inherit Layer's build method

    # ( )
    def call(self, inputs):
        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(tf.matmul(c, hat_inputs, [2, 2]))
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.routings-1:
                b += tf.matmul(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)
        return o

    def compute_output_shape(self, input_shape):  #automatic inferring shape
        return (None, self.num_capsule, self.dim_capsule)


def MODEL():
    input_image = Input(shape=(32, 32, 3))
    x = Conv2D(128, (3, 3), activation='relu')(input_image)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = AveragePooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    """
    Now we convert it to (batch_size, input_num_capsule, input_dim_capsule) and then connect a capsule neural layer. The final output of the model is the length of the capsule network with 10 dimensions of 16.
    """
    x = Reshape((-1, 2))(x)  # (None, 100, 128) Equivalent to the previous layer of capsules (None, input_num, input_dim)
    capsule = Capsule(num_capsule=16, dim_capsule=16, routings=3, share_weights=True)(x)  # capsule-（None,10, 16)
    output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), axis=2)))(capsule)  # The last output becomes 10 probability values
    model = Model(inputs=input_image, output=output)
    return model


DATADIR="E:/Dr.Madhu_IT/cell_images/cell_images"
CATEGORIES = ["Parasitized","Uninfected"]
#CATEGORIES = ["Gametocytes","Non-Malaria","Ringforms","Shizont","Trophozoites",]

IMG_SIZE = 32
num_epoch =10


os.chdir(DATADIR)
training_data = []
training_data1 = []
def create_training_data():
    for category in CATEGORIES: 

        path = os.path.join(DATADIR,category)  
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img))  
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
                training_data.append([new_array, class_num])
                training_data1.append(new_array)
                
            except Exception as e:  
                pass
    X = []
    y = []
    for features,label in training_data:
        X.append(features)
        y.append(label)
    #print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    img_rows = IMG_SIZE
    img_cols = IMG_SIZE
    channels = 3
    img_shape = [img_rows, img_cols, channels]
    return X,img_shape


dataset, shape = create_training_data()
print('Dataset shape: {0}, Image shape: {1}'.format(dataset.shape, shape))
print(len(training_data1))

import random#for randomizing our data
random.shuffle(training_data)
X=[]
y=[]
for features,label in training_data:
    X.append(features)
    y.append(label)
#print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3))
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(to_categorical(y))


#dividing data to training and splitting
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)

if __name__ == '__main__':
    # Download Data
    #(x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)
	
    #Load model
    model = MODEL()
    model.compile(loss=margin_loss, optimizer='adam', metrics=['accuracy'])
    model.summary()
    tfck = TensorBoard(log_dir='capsule')
	
    # 
    data_augmentation = True
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=[tfck],
            shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by dataset std
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in 0 to 180 degrees
            width_shift_range=0.1,  # randomly shift images horizontally
            height_shift_range=0.1,  # randomly shift images vertically
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=[tfck],
            workers=4)
    plot_model(model, to_file='model.png', show_shapes=True)
