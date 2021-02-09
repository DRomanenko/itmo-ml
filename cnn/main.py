import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras as kr
import warnings

warnings.filterwarnings('ignore')

LABELS = [
    'T-shirt/top'
    , 'Trouser'
    , 'Pullover'
    , 'Dress'
    , 'Coat'
    , 'Sandal'
    , 'Shirt'
    , 'Sneaker'
    , 'Bag'
    , 'Ankle boot'
]

NUMBER_CLASSES = len(LABELS)


def get_layers(filters):
    return kr.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')


TRANSFORMATION_SEQ = [get_layers(16),
                      get_layers(32),
                      kr.layers.MaxPooling2D((2, 2))]


def normalize(ds):
    return (ds / 255).reshape(ds.shape[0], 28, 28, 1)


def try_model(ds, best, transformation_sequences):
    model = kr.Sequential()
    for transformation in transformation_sequences:
        model.add(transformation)
    model.add(kr.layers.Flatten())
    model.add(kr.layers.Dense(NUMBER_CLASSES, activation='softmax'))
    # https://neptune.ai/blog/keras-loss-functions
    # https://habr.com/ru/post/318970/
    # https://www.tensorflow.org/api_docs/python/tf/keras/losses
    # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(ds[0], ds[1], epochs=15, verbose=1)
    # https://www.tensorflow.org/guide/keras/train_and_evaluate?hl=ru
    results = model.evaluate(ds[2], ds[3], verbose=2)
    if results[1] > best[0]: best[1] = model
    print('Test loss, Test acc:', results)
    return best


start_time = time.time()

# https://keras.io/api/datasets/fashion_mnist/#load_data-function
dataset = kr.datasets.fashion_mnist
# train - Fashion-MNIST (https://github.com/zalandoresearch/fashion-mnist), test - MNIST (http://yann.lecun.com/exdb/mnist/)
(x_train, y_train), (x_test, y_test) = dataset.load_data()

x_train = normalize(x_train)
x_test = normalize(x_test)

best = [0, None]
ds = x_train, y_train, x_test, y_test
best = try_model(ds, best, [TRANSFORMATION_SEQ[0], TRANSFORMATION_SEQ[2]])

best = try_model(ds, best, [TRANSFORMATION_SEQ[0], TRANSFORMATION_SEQ[2], TRANSFORMATION_SEQ[1]])

best = try_model(ds, best, [TRANSFORMATION_SEQ[0], TRANSFORMATION_SEQ[1], TRANSFORMATION_SEQ[2]])

best = try_model(ds, best, [TRANSFORMATION_SEQ[0], TRANSFORMATION_SEQ[2], TRANSFORMATION_SEQ[1], TRANSFORMATION_SEQ[2]])

predicted = best[1].predict(x_test)
confusion_matrix = np.zeros((NUMBER_CLASSES, NUMBER_CLASSES))
similarities_matrix = np.full((NUMBER_CLASSES, NUMBER_CLASSES, 2), [-1, 0.0])
for i in range(len(predicted)):
    label = np.argmax(predicted[i])
    val_pred = predicted[i][label]
    if val_pred > similarities_matrix[y_test[i]][label][1]:
        similarities_matrix[y_test[i]][label] = (i, predicted[i][label])
    confusion_matrix[y_test[i]][label] += 1

plt.figure(figsize=(20, 20))
for i in range(NUMBER_CLASSES):
    for q in range(NUMBER_CLASSES):
        plt.subplot(10, 10, i * NUMBER_CLASSES + q + 1)
        plt.xticks([])
        plt.yticks([])
        if similarities_matrix[i][q][0] == -1:
            plt.imshow(np.full((28, 28), 255, dtype=int))
            continue
        plt.xlabel(LABELS[i])
        plt.imshow(x_test[int(similarities_matrix[i][q][0])])
plt.show()

print('Total working time: %s seconds.' % (time.time() - start_time))
