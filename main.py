import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten

model_path = "./Model/my_model.h5"
dataset_path = "./Dataset"
image_width = 175
image_height = 175
batch_size = 32


def train(model, training_data, validation_data):

    early_stopping = keras.callbacks.EarlyStopping(
        patience=5,
        monitor="val_accuracy",
        mode="max"
    )
    model_ckpt = keras.callbacks.ModelCheckpoint(
        model_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    )

    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    history = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=10,
        callbacks=[early_stopping, model_ckpt]
    )

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.show()


def test(testing_data):
    model = keras.models.load_model(model_path)

    predictions = model.predict(testing_data)

    correct_guess(test_images, predictions)

    while True:
        num = input("Enter a number between 0 and {}: ".format(test_images.samples - 1))
        num = int(num)

        if num == -1:
            break

        elif 0 <= num < test_images.samples:

            x = int(num / batch_size)
            y = num % batch_size

            plt.figure()
            plt.title(classes[np.argmax(predictions[num])])
            plt.imshow(test_images[x][0][y])
            plt.axis("off")
            plt.show()

        else:
            print("Please enter a correct number")


def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.axis("off")


def correct_guess(test_images, predictions):

    correct = 0
    incorrect = 0

    for i in range(len(test_images)):
        for j in range(len(test_images[i][1])):

            idx = i * 32 + j

            label = test_images[i][1][j]
            pred_label = np.argmax(predictions[idx])

            if label == pred_label:
                correct += 1
            else:
                incorrect += 1

    print('correct: ', correct, 'incorrect: ', incorrect)


training = True


if __name__ == "__main__":

    image_size = (image_width, image_height)
    image_gen = keras.preprocessing.image.ImageDataGenerator(
        rescale = 1 / 255.,
        validation_split = .2
    )

    train_images = image_gen.flow_from_directory(
        dataset_path,
        target_size = image_size,
        class_mode = 'sparse',
        subset = 'training'
    )

    test_images = image_gen.flow_from_directory(
        dataset_path,
        target_size = image_size,
        class_mode = 'sparse',
        subset = 'validation',
        shuffle = False
    )

    classes = {v: k for k, v in train_images.class_indices.items()}

    model = keras.models.Sequential([
        Conv2D(32, (3, 3), activation='elu', input_shape=(image_width, image_height, 3), padding='same'),
        MaxPooling2D(),

        Conv2D(32, (3, 3), activation='elu', padding='same'),
        MaxPooling2D(),

        Conv2D(64, (3, 3), activation='elu', padding='same'),
        MaxPooling2D(),

        Conv2D(64, (3, 3), activation='elu', padding='same'),
        MaxPooling2D(),

        Conv2D(128, (3, 3), activation='elu', padding='same'),
        MaxPooling2D(),

        Flatten(),

        Dense(128, activation='elu'),

        Dense(len(classes), activation='softmax')
    ])

    model.summary()

    if training:
        train(model, train_images, test_images)
        test(test_images)
    else:
        test(test_images)

















