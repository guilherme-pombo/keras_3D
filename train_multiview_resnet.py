import argparse

from keras import optimizers

from .multiview_models import concat_resnet
from .generator import MultiViewGen

# These are hardcoded since the dataset is always the same and I'm lazy
TARGET_SIZE = (224, 224)
NUM_CLASSES = 40
NUM_VIEWS = 12


def train(model_type, batch_size, epochs):
    # concat_view resnet
    if model_type == 'concat':
        model = concat_resnet(stop_layer=-1,
                              target_size=TARGET_SIZE,
                              num_images=NUM_VIEWS,
                              num_classes=NUM_CLASSES)
    else:
        raise NotImplementedError("Other versions of multiview are not yet implemented")
    print(model.summary())

    # SGD
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # generators
    gen = MultiViewGen('classes.txt', 'train', batch_size, data_folder='view', target_size=TARGET_SIZE,
                       num_classes=NUM_CLASSES)
    val_gen = MultiViewGen('classes.txt', 'test', batch_size, data_folder='view', target_size=TARGET_SIZE,
                           num_classes=NUM_CLASSES)

    model.fit_generator(
        gen.generator(),
        steps_per_epoch=gen.get_num_total_imgs() // batch_size,
        epochs=epochs,
        validation_data=val_gen.generator(),
        validation_steps=val_gen.get_num_total_imgs() // batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", help="Model type to use", type=str, default='concat')
    parser.add_argument("--batch_size", help="Batch size to use", type=int, default=1)
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=10)

    args = parser.parse_args()

    train(batch_size=args.batch_size, epochs=args.epochs)
