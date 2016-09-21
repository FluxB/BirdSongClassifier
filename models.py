from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils


def model_paper(nb_species, input_size): # Model from paper, adaption due to different FT parameters needed
    nb_filters=64
    kernel_size=5
    (nb_f_steps, nb_t_steps) = input_size
    model = Sequential()

    model.add(Dropout(0.2,input_shape=(1, nb_f_steps, nb_t_steps)))
    model.add(BatchNormalization())
    model.add(Convolution2D(nb_filters,kernel_size,kernel_size,subsample=(1,2),border_mode='valid'))
    model.add(Activation('relu'))
        
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Convolution2D(nb_filters, kernel_size, kernel_size,subsample=(1,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization())
    model.add(Convolution2D(2*nb_filters, kernel_size, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization())
    model.add(Convolution2D(4*nb_filters, kernel_size, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization())
    model.add(Convolution2D(4*nb_filters, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.4))
    model.add(Flatten())

    model.add(BatchNormalization())
    model.add(Dense(1024))
    model.add(Activation('relu'))

    model.add(Dropout(0.4))
    model.add(Dense(nb_species))
    model.add(Activation('softmax'))

    return model
