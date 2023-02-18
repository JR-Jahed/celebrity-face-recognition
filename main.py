from tensorflow import keras
from model_vgg16 import ModelVGG16

dataset_path = "./Dataset"
image_width = 150
image_height = 150

training = True


if __name__ == "__main__":

    image_size = (image_width, image_height)
    image_gen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1 / 255.,
        validation_split=.2
    )

    train_images = image_gen.flow_from_directory(
        dataset_path,
        target_size=image_size,
        class_mode='sparse',
        subset='training'
    )

    test_images = image_gen.flow_from_directory(
        dataset_path,
        target_size=image_size,
        class_mode='sparse',
        subset='validation',
        shuffle=False
    )

    classes = {v: k for k, v in train_images.class_indices.items()}

    model = ModelVGG16(
        train_images=train_images,
        test_images=test_images,
        classes=classes,
        input_shape=(image_width, image_height, 3)
    )

    model.add_layer(keras.layers.Dense(256, activation='relu'))
    model.add_layer(keras.layers.Dropout(.2))
    model.add_layer(keras.layers.Dense(len(classes), activation='softmax'))

    model.conv_base.trainable = False

    model.model.summary()

    if training:
        model.train(10)
        model.test()
    else:
        model.test()

















