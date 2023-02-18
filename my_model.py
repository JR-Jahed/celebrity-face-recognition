from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

model_path = "./Models/my_model.h5"
batch_size = 32


def show_image(image, title):
    plt.figure()
    plt.title(title)
    plt.imshow(image)
    plt.axis("off")
    plt.show()


class MyModel:
    def __init__(self, train_images, test_images, classes, input_shape):

        self.train_images = train_images
        self.test_images = test_images
        self.classes = classes

        self.model = keras.models.Sequential([
            keras.layers.Input(input_shape)
        ])

    def add_layer(self, layer):
        self.model.add(layer)

    def train(self, epochs):

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

        self.model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        history = self.model.fit(
            self.train_images,
            validation_data=self.test_images,
            epochs=epochs,
            callbacks=[early_stopping, model_ckpt]
        )

        epochs_range = range(1, len(history.history['accuracy']) + 1)

        plt.title('Training and Validation accuracy')
        plt.xticks(epochs_range)
        plt.plot(epochs_range, history.history["accuracy"], 'bo', label='Training acc')
        plt.plot(epochs_range, history.history["val_accuracy"], 'b', label='Validation acc')
        plt.legend()
        plt.show()

    def test(self):
        model = keras.models.load_model(model_path)

        predictions = model.predict(self.test_images)

        self.correct_guess(predictions)

        while True:
            num = input("Enter a number between 0 and {}: ".format(self.test_images.samples - 1))
            num = int(num)

            if num == -1:
                break

            elif 0 <= num < self.test_images.samples:

                x = int(num / batch_size)
                y = num % batch_size

                show_image(self.test_images[x][0][y], self.classes[np.argmax(predictions[num])])

            else:
                print("Please enter a correct number")

    def correct_guess(self, predictions):

        correct = 0
        incorrect = 0

        for i in range(len(self.test_images)):
            for j in range(len(self.test_images[i][1])):

                idx = i * 32 + j

                label = self.test_images[i][1][j]
                pred_label = np.argmax(predictions[idx])

                if label == pred_label:
                    correct += 1
                else:
                    incorrect += 1

        print('correct: ', correct, 'incorrect: ', incorrect)
