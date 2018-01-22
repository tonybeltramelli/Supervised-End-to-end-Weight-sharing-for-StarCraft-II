__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import numpy as np
import cv2

class Utils:

    @staticmethod
    def show(image):
        import cv2
        cv2.namedWindow("view", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("view", image)
        cv2.waitKey(0)
        cv2.destroyWindow("view")

    @staticmethod
    def feature_array_to_img(a, max_target_value=255.0):
        amax = np.max(a)
        img = np.zeros((np.shape(a)[0], np.shape(a)[1], 3))
        for x in range(0, np.shape(a)[0]):
            for y in range(0, np.shape(a)[1]):
                feature = a[x][y]
                pixel = feature * (max_target_value / amax)

                if feature == 0:
                    img[x][y][0] = pixel
                    img[x][y][1] = pixel
                elif feature == 1:
                    img[x][y][1] = pixel
                elif feature == 2:
                    img[x][y][2] = pixel
                elif feature == 3:
                    img[x][y][1] = pixel
                    img[x][y][2] = pixel
                elif feature == 4:
                    img[x][y][0] = pixel
        return img

    @staticmethod
    def resize_squared_img(img, size):
        img = cv2.resize(img, (size, size))
        return img
