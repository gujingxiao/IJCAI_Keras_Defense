import cv2
import numpy as np
import ast
import random

def do_rotation_transform(image, angle=0):
    height, width = image.shape[:2]
    cc = np.cos(angle / 180 * np.pi)
    ss = np.sin(angle / 180 * np.pi)
    rotate_matrix = np.array([[cc, -ss], [ss, cc]])

    box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ], np.float32)
    box1 = box0 - np.array([width / 2, height / 2])
    box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2, height / 2])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)

    image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101,
                                borderValue=(0, 0, 0,))
    return image

def do_elastic_transform(image, grid=10, distort=0.1):
    # https://www.kaggle.com/ori226/data-augmentation-with-elastic-deformations
    # https://github.com/letmaik/lensfunpy/blob/master/lensfunpy/util.py
    height, width = image.shape[:2]

    x_step = int(grid)
    xx = np.zeros(width, np.float32)
    prev = 0
    for x in range(0, width, x_step):
        start = x
        end = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * (1 + random.uniform(-distort, distort))

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = int(grid)
    yy = np.zeros(height, np.float32)
    prev = 0
    for y in range(0, height, y_step):
        start = y
        end = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * (1 + random.uniform(-distort, distort))

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    # grid
    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
                      borderValue=(0, 0, 0,))
    return image

def do_crop_and_rescale(image,x0, y0, x1, y1):
    height, width = image.shape[:2]
    image = image[y0:y1, x0:x1]
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return image

def data_augment(image):
    # cv2.imshow("ori", image)
    if np.random.rand() < 0.5:
        # Horizontal Flip
        if np.random.rand() < 0.6:
            image = np.fliplr(image)

        if np.random.rand() < 0.6:
            max_scale = 0.2
            h, w = image.shape[:2]

            dy = int(h * max_scale)
            dx = int(w * max_scale)
            image = do_crop_and_rescale(image, np.random.randint(0, dx), np.random.randint(0, dy),
                                            w - np.random.randint(0, dx), h - np.random.randint(0, dy))
    if np.random.rand() < 0.5:
        # Rotation Transform
        if np.random.rand() < 0.6:
            angle = np.random.uniform(-20, 20)
            image = do_rotation_transform(image, angle)

        # Elastic Transform
        if np.random.rand() < 0.6:
            distort = np.random.uniform(0, 0.2)
            image = do_elastic_transform(image, grid=10, distort=distort)
    # cv2.imshow("aug", image)
    # cv2.waitKey(0)
    return image