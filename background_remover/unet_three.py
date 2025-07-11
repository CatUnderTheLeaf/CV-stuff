from keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, \
    MaxPooling2D, Input, Concatenate, Conv2DTranspose, BatchNormalization
from keras.optimizers import Adam


# encoding block(conv - conv - pool)
def enc_conv_block(inputs, feature_maps, filter_size = (3, 3),
                        conv_strides = 1, pooling_filter_size = (2, 2), pooling_strides = (2, 2), actf="relu"):
    
    conv1 = Conv2D(feature_maps , filter_size , activation = actf, strides = conv_strides,
                        padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv2 = Conv2D(feature_maps , filter_size , activation = actf, strides = conv_strides,
                        padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool = MaxPooling2D(pooling_filter_size, strides = pooling_strides)(conv2)

    return pool, conv2

# decoding block(concat - upconv - upconv)
def dec_conv_block(inputs, merge_inputs, feature_maps, filter_size = (3, 3), conv_strides = 1,
                        up_conv_strides = (2, 2), actf="relu"):

    merge = Concatenate(axis = 3)([Conv2DTranspose(feature_maps, filter_size,
                                                    activation = actf, strides = up_conv_strides, kernel_initializer = 'he_normal',
                                                    padding = 'same')(inputs), merge_inputs])

    conv1 = Conv2D(feature_maps , filter_size , activation = actf, strides = conv_strides,
                        padding = 'same', kernel_initializer = 'he_normal')(merge)
    conv2 = Conv2D(feature_maps , filter_size , activation = actf, strides = conv_strides,
                        padding = 'same', kernel_initializer = 'he_normal')(conv1)

    return conv2

# encoder
def encoding_path(inputs, actf="relu"):

    enc_conv1, concat1 = enc_conv_block(inputs, 64, actf=actf)
    enc_conv2, concat2 = enc_conv_block(enc_conv1, 128, actf=actf)
    enc_conv3, concat3 = enc_conv_block(enc_conv2, 256, actf=actf)
    enc_conv4, concat4 = enc_conv_block(enc_conv3, 512, actf=actf)

    return concat1, concat2, concat3, concat4, enc_conv4

# decoder
def decoding_path(dec_inputs, concat1, concat2, concat3, concat4, actf="relu"):

    dec_conv1 = dec_conv_block(dec_inputs, concat4, 512, actf=actf)
    dec_conv2 = dec_conv_block(dec_conv1, concat3, 256, actf=actf)
    dec_conv3 = dec_conv_block(dec_conv2, concat2, 128, actf=actf)
    dec_conv4 = dec_conv_block(dec_conv3, concat1, 64, actf=actf)

    return dec_conv4

def build_model(img_shape, num_of_class, actf = 'relu',
        learning_rate = 0.001,  drop_rate = 0.5, do_batch_norm = False, do_drop = False):
    inputs = Input(img_shape)

    # Contracting path
    concat1, concat2, concat3, concat4, enc_path = encoding_path(inputs, actf)

    # middle path
    mid_path1 = Conv2D(1024, (3,3), activation = actf, padding = 'same', kernel_initializer = 'he_normal')(enc_path)
    mid_path1 = Dropout(drop_rate)(mid_path1)
    mid_path2 = Conv2D(1024, (3,3), activation = actf, padding = 'same', kernel_initializer = 'he_normal')(mid_path1)
    mid_path2 = Dropout(drop_rate)(mid_path2)

    # Expanding path
    dec_path = decoding_path(mid_path2, concat1, concat2, concat3, concat4, actf)
    segmented = Conv2D(num_of_class, (1,1), activation = actf, padding = 'same', kernel_initializer = 'he_normal')(dec_path)
    segmented = Activation('softmax')(segmented)

    model = Model(inputs = inputs, outputs = segmented)
    model.compile(optimizer = Adam(learning_rate = learning_rate),
                        loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model

if __name__ == "__main__":
    model = build_model((256, 256, 3), 2)
    model.summary()