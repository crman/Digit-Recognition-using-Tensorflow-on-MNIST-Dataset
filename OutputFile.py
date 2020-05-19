from keras.models import load_model
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data("C:/Users/crman/PycharmProjects/DigitRecognition/Dataset/mnist.npz")

img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print("Number of classes: "+ str(y_test.shape[1]))

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]

classifier = load_model('mnist_cnn_10Epochs.h5')

import cv2
import numpy as np

def draw_test(name, pred, img):
    BLACK = [0, 0, 0]
    expanded_image = cv2.copyMakeBorder(img, 0, 0, 0, imageL.shape[0], cv2.BORDER_CONSTANT, value=BLACK)
    expanded_image = cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)
    cv2.putText(expanded_image, str(pred), (152, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (0,255,0), 2)
    cv2.imshow(name, expanded_image)

for i in range(0, 10):
    rand = np.random.randint(0, len(x_test))
    img = x_test[rand]
    imageL = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    img = img.reshape(1, 28, 28, 1)

    #get prediction
    res = str(classifier.predict_classes(img, 1, verbose = 0)[0])
    print(res)
    draw_test('Prediction', res, imageL)
    cv2.waitKey(0)

cv2.destroyAllWindows()

'''
img = cv2.imread('test111.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imageL = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
img = img.reshape(1, 28, 28, 1)
res = str(classifier.predict_classes(img, 1, verbose=0)[0])

draw_test('Prediction', res, imageL)
cv2.waitKey(0)
'''


