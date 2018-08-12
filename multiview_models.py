from keras.applications import resnet50
from keras.applications.resnet50 import identity_block, conv_block
from keras.models import Model
from keras.layers import Input, concatenate, Dense, Flatten, maximum, Conv2D, BatchNormalization, Activation, \
    MaxPooling2D, AveragePooling2D


def concat_resnet(stop_layer, target_size, num_images, num_classes):
    """
    This model simply concatenates the views (e.g. 12 views of an airplane)
    by constructing a list of Input instances to represent the 12 input images for the multiple views,
    and pass each of these through a shared-parameter CNN
    loss: 0.8450 - acc: 0.7598 - val_loss: 3.6891 - val_acc: 0.0292
    :param stop_layer: how much of resnet 50 to use (-1 to use everything)
    :param target_size:
    :param num_images:
    :param num_classes
    :return:
    """
    # this is the Network to be share amongst the views
    shared_resnet = resnet50.ResNet50(include_top=False)
    shared_resnet = Model(shared_resnet.input, shared_resnet.layers[stop_layer].output)

    # one input per image
    inputs = [Input(shape=target_size + (3,)) for _ in range(num_images)]
    # encode through the shared network
    encodeds = [shared_resnet(inputs[idx]) for idx in range(num_images)]

    # We can then concatenate the view vectors:
    merged_vector = concatenate(encodeds, axis=-1)

    # And add the classification layer
    merged_vector = Flatten()(merged_vector)
    predictions = Dense(num_classes, activation='softmax')(merged_vector)

    # We define a trainable model linking the
    # image inputs to the predictions
    return Model(inputs=inputs, outputs=predictions)


def mvcnn(target_size, num_images, num_classes):
    # this is the Network to be shared amongst the views
    img_input = Input(shape=target_size + (3,))
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
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
    outp = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    shared_resnet = Model(img_input, outp)

    # one input per image
    inputs = [Input(shape=target_size + (3,)) for _ in range(num_images)]
    # encode through the shared network
    encodeds = [shared_resnet(inputs[idx]) for idx in range(num_images)]

    # rather than concatenate, this time we take the maximum and pass it through another network
    maximum_tensor = maximum(encodeds)

    # we then pass this through another Resnet-like CNN
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(maximum_tensor)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    #     x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)

    predictions = Dense(num_classes, activation='softmax', name='final_fc')(x)

    return Model(inputs=inputs, outputs=predictions)
