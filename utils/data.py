import os
import pandas as pd
import tensorflow as tf
import tensorflow.keras.preprocessing.image as preproc_image
import tensorflow.image as tf_image
import seaborn as sns

from matplotlib import pyplot as plt
from tensorflow import expand_dims, data
from PIL import Image

def create_dataset(csv_path, images_path):
    df = pd.read_csv(csv_path)

    label_map = {'ball': 0,
                 'centerspot': 1,
                 'goalpost': 2,
                 'goalspot': 3,
                 'robot': 4
    }

    filenames_path = []
    labels_index = []
    center_bboxes = []

    for i, row in df.iterrows():
        filenames_path.append(os.path.join(images_path, str(row['filename'])))
        labels_index.append(label_map[str(row['label'])])
        center_bboxes.append((row['cx'], row['cy']))


    # creating dataset following the structure defined in the function call
    dataset = data.Dataset.from_tensor_slices(((filenames_path, labels_index), center_bboxes))
    # adding a map to the dataset so the data will be correct when passed to the CNN
    dataset = dataset.map(load_and_preprocess)
    return dataset

def load_and_preprocess(inputs, bbox):
    path, label = inputs
    # reading images
    raw_img = tf.io.read_file(path)
    # transforming in tensor
    img = tf.image.decode_jpeg(raw_img, channels=3)
    # just setting, not resizing anything
    img.set_shape([256, 256, 3])

    label_one_hot = tf.one_hot(label, 5)

    return (img, label_one_hot), bbox


class ImageHandler:
    def __init__(self, file_name):
        self.image_name = file_name
        self.image = self.load_img(self.image_name)

    def load_img(self, file_name):
        """
        It expects that the code is running in a folder 1 level from the root and the images
        are in the path 'datasets/spqr_dataset/images' from project root folder.
        :param file_name: the file name of the image
        :return:
        """
        path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(path, "../datasets/spqr_dataset/images", file_name)
        image = Image.open(path)
        self.image = image
        return image

    def img_crop_and_resize(self, xmin, ymin, xmax, ymax, size=None):
        """
        This method crops the image and resize it to desired size.
        :param xmin: X coordinate of the top left corner
        :param ymin: Y coordinate of the top left corner
        :param xmax: X coordinate of the bottom right corner
        :param ymax: Y coordinate of the bottom right corner
        :param size: Desired size of the cropped image defined as [HxW]. If not given defaults to 64x64.
        :return: Cropped image, as a tensor
        """
        if size is None:
            size = [64, 64]

        # convert to 3D numpy array
        image_array = preproc_image.img_to_array(self.image)

        # normalize crop size
        norm_xmin = xmin / image_array.shape[1]
        norm_xmax = xmax / image_array.shape[1]

        norm_ymin = ymin / image_array.shape[0]
        norm_ymax = ymax / image_array.shape[0]

        # add dimension as new first axis
        image_array = expand_dims(image_array, axis=0)
        return tf_image.crop_and_resize(image_array,
                                 [[norm_ymin, norm_xmin, norm_ymax, norm_xmax]],
                                 [0],
                                 size,
                                 method='bilinear')[0]

class CSVHandler:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def get_images_name(self):
        return self.df['filename'].to_numpy()

    def get_image_data(self, filename):
        return self.df[self.df['filename'] == filename].to_numpy()