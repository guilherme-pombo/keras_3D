from keras.applications.resnet50 import identity_block, conv_block
from keras.models import Model
from keras.layers import Input, concatenate, Dense, Flatten, maximum, Conv2D, BatchNormalization, Activation, \
    MaxPooling2D, AveragePooling2D, Dropout


def resnet_mvcnn(target_size, num_images, num_classes):
    # this is the Network to be share amongst the views
    img_input = Input(shape=target_size + (3,))
    x = Conv2D(
        64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
    #     x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    outp = Flatten()(x)

    shared_resnet = Model(img_input, outp)

    # one input per image
    inputs = [Input(shape=target_size + (3,)) for _ in range(num_images)]
    # encode through the shared network
    encodeds = [shared_resnet(inputs[idx]) for idx in range(num_images)]

    # rather than concatenate, this time we take the maximum and pass it through another network
    maximum_tensor = maximum(encodeds)

    predictions = Dense(num_classes, activation='softmax', name='final_fc')(maximum_tensor)

    return Model(inputs=inputs, outputs=predictions)
