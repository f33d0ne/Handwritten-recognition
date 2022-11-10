import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist  
(x_train, y_train) , (x_test , y_test) = mnist.load_data()

x_test = tf.keras.utils.normalize(x_train, axis=1)
x_train = tf.keras.utils.normalize(x_test, axis=1)


model = tf.keras.models.load_model('handwritten.model') 


# print(np.argmax(model.predict(x_test)[20]))
# plt.imshow(x_test[400],cmap=plt.cm.binary)
# plt.show()

for i in range(10):

    print(np.argmax(model.predict(x_test)[i]))

    plt.imshow(x_test[i],cmap=plt.cm.binary)
    plt.show()
