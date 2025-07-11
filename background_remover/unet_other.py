# ok, that doesn't work


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create unet family models. Definitions are from
https://github.com/zhixuhao/unet
https://keras.io/examples/vision/oxford_pets_image_segmentation/
"""
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, SeparableConv2D, MaxPooling2D, Lambda, UpSampling2D, Input, Add, BatchNormalization, ReLU, Dropout, concatenate, Reshape, Softmax, Input
from tensorflow.keras.models import Model


def normalize(x):
    return x/127.5 - 1


def img_resize(x, size, mode='bilinear'):
    if mode == 'bilinear':
        return tf.image.resize(x, size=size, method='bilinear')
    elif mode == 'nearest':
        return tf.image.resize(x, size=size, method='nearest')
    else:
        raise ValueError('output model file is not specified')


def CustomBatchNormalization(*args, **kwargs):
    if tf.__version__ >= '2.2':
        from tensorflow.keras.layers.experimental import SyncBatchNormalization
        BatchNorm = SyncBatchNormalization
    else:
        BatchNorm = BatchNormalization

    return BatchNorm(*args, **kwargs)

def UNetStandard(num_classes,
                 input_shape=(512, 512, 3),
                 input_tensor=None,
                 weights=None,
                 **kwargs):

    if input_tensor is None:
        inputs = Input(shape=input_shape, name='image_input')
    else:
        inputs = input_tensor

    # normalize input image
    #inputs_norm= Lambda(normalize, name='input_normalize')(inputs)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    #up6 = Conv2D(512, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(drop5))
    up6 = Conv2DTranspose(512, 2, strides=(2, 2), activation = 'relu', padding = 'same')(drop5)
    merge6 = concatenate([drop4,up6], axis=3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv6)

    #up7 = Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6))
    up7 = Conv2DTranspose(256, 2, strides=(2, 2), activation = 'relu', padding = 'same')(conv6)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv7)

    #up8 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7))
    up8 = Conv2DTranspose(128, 2, strides=(2, 2), activation = 'relu', padding = 'same')(conv7)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv8)

    #up9 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8))
    up9 = Conv2DTranspose(64, 2, strides=(2, 2), activation = 'relu', padding = 'same')(conv8)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same')(conv9)

    # Add a per-pixel classification layer
    outputs = Conv2D(num_classes, 1, padding="same")(conv9)

    # Define the model
    model = Model(inputs, outputs)

    if(weights):
        model.load_weights(weights)
    return model



def UNetLite(num_classes,
             input_shape=(512, 512, 3),
             input_tensor=None,
             weights=None,
             **kwargs):

    if input_tensor is None:
        inputs = Input(shape=input_shape, name='image_input')
    else:
        inputs = input_tensor

    # normalize input image
    #inputs_norm= Lambda(normalize, name='input_normalize')(inputs)

    conv1 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same')(inputs)
    conv1 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
    conv3 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = SeparableConv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
    conv4 = SeparableConv2D(512, 3, activation = 'relu', padding = 'same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = SeparableConv2D(1024, 3, activation = 'relu', padding = 'same')(pool4)
    conv5 = SeparableConv2D(1024, 3, activation = 'relu', padding = 'same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    #up6 = Conv2D(512, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(drop5))
    up6 = Conv2DTranspose(512, 2, strides=(2, 2), activation = 'relu', padding = 'same')(drop5)
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = SeparableConv2D(512, 3, activation = 'relu', padding = 'same')(merge6)
    conv6 = SeparableConv2D(512, 3, activation = 'relu', padding = 'same')(conv6)

    #up7 = Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6))
    up7 = Conv2DTranspose(256, 2, strides=(2, 2), activation = 'relu', padding = 'same')(conv6)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same')(merge7)
    conv7 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same')(conv7)

    #up8 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7))
    up8 = Conv2DTranspose(128, 2, strides=(2, 2), activation = 'relu', padding = 'same')(conv7)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same')(merge8)
    conv8 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same')(conv8)

    #up9 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8))
    up9 = Conv2DTranspose(64, 2, strides=(2, 2), activation = 'relu', padding = 'same')(conv8)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same')(merge9)
    conv9 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same')(conv9)
    conv9 = SeparableConv2D(2, 3, activation = 'relu', padding = 'same')(conv9)

    # Add a per-pixel classification layer
    outputs = Conv2D(num_classes, 1, padding="same")(conv9)

    # Define the model
    model = Model(inputs, outputs)

    if(weights):
        model.load_weights(weights)
    return model



def UNetSimple(num_classes,
               input_shape=(512, 512, 3),
               input_tensor=None,
               weights=None,
               **kwargs):

    if input_tensor is None:
        inputs = Input(shape=input_shape, name='image_input')
    else:
        inputs = input_tensor

    # normalize input image
    #inputs_norm= Lambda(normalize, name='input_normalize')(inputs)


    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = ReLU()(x)
        x = SeparableConv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = ReLU()(x)
        x = SeparableConv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = Add()([x, residual])
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = ReLU()(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = ReLU()(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(2)(x)

        # Project residual
        residual = UpSampling2D(2)(previous_block_activation)
        residual = Conv2D(filters, 1, padding="same")(residual)
        x = Add()([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = Conv2D(num_classes, (3,3), padding="same")(x)
    # outputs = Conv2D(num_classes, (3,3), strides=2, padding="same", activation="softmax")(x)

    # Define the model
    model = Model(inputs, outputs)

    if(weights):
        model.load_weights(weights)
    return model

unet_model_map = {
    'unet_standard': UNetStandard,
    'unet_lite': UNetLite,
    'unet_simple': UNetSimple,
}

def get_unet_model(model_type, num_classes, model_input_shape, freeze_level=0, weights_path=None, training=True):
    # check if model type is valid
    if model_type not in unet_model_map.keys():
        raise ValueError('This model type is not supported now')

    model_function = unet_model_map[model_type]

    input_tensor = Input(shape=model_input_shape+(3,), batch_size=None, name='image_input')
    base_model = model_function(num_classes, input_tensor=input_tensor,
                           input_shape=model_input_shape + (3,),
                           weights=None)

    #base_model = Model(model.input, model.layers[-5].output)
    #print('backbone layers number: {}'.format(backbone_len))


    # for training model, we need to flatten mask to calculate loss
    # if training:
    #     x = Reshape((model_input_shape[0]*model_input_shape[1], num_classes)) (base_model.output)
    # else:
    x = base_model.output

    x = Softmax(name='pred_mask')(x)
    model = Model(base_model.input, x, name=model_type)

    if weights_path:
        model.load_weights(weights_path, by_name=False)#, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    #if freeze_level in [1, 2]:
        ## Freeze the backbone part or freeze all but final feature map & input layers.
        #num = (backbone_len, len(base_model.layers))[freeze_level-1]
        #for i in range(num): model.layers[i].trainable = False
        #print('Freeze the first {} layers of total {} layers.'.format(num, len(model.layers)))
    #elif freeze_level == 0:
        ## Unfreeze all layers.
        #for i in range(len(model.layers)):
            #model.layers[i].trainable= True
        #print('Unfreeze all of the layers.')

    return model

if __name__ == "__main__":
    model = get_unet_model('unet_simple', 2, (256, 256))
    model.summary()