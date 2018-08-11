import argparse

from keras.applications import resnet50
from keras import optimizers

from .generator import SingleViewGen

# These are hardcoded since the dataset is always the same and I'm lazy
TARGET_SIZE = (224, 224)
NUM_CLASSES = 40


def train(batch_size, epochs):
    # random init Resnet50
    model = resnet50.ResNet50(input_shape=TARGET_SIZE + (3,),
                              classes=NUM_CLASSES,
                              include_top=True,
                              weights=None)

    print(model.summary())

    # SGD
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # generators
    gen = SingleViewGen('classes.txt', 'train', batch_size, data_folder='view', target_size=TARGET_SIZE, num_classes=NUM_CLASSES)
    val_gen = SingleViewGen('classes.txt', 'test', batch_size, data_folder='view', target_size=TARGET_SIZE, num_classes=NUM_CLASSES)

    model.fit_generator(
        gen.generator(),
        steps_per_epoch=gen.get_num_total_imgs() // batch_size,
        epochs=epochs,
        validation_data=val_gen.generator(),
        validation_steps=val_gen.get_num_total_imgs() // batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="Batch size to use", type=int, default=32)
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=10)

    args = parser.parse_args()

    train(batch_size=args.batch_size, epochs=args.epochs)
