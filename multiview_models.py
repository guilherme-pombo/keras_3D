from keras.applications import resnet50
from keras.models import Model
from keras.layers import Input, concatenate, Dense, Flatten


def concat_resnet(stop_layer, target_size, num_images, num_classes):
    """
    This model simply concatenates the views (e.g. 12 views of an airplane)
    by constructing a list of Input instances to represent the 12 input images for the multiple views,
    and pass each of these through a shared-parameter CNN
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
