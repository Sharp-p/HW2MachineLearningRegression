import os

from utils.cnn_regression import  (CNNModelRegression)
from tensorflow.keras import utils

def create_structure_img():
    cnn = CNNModelRegression([(256, 256, 3), (5,)],
                             2, "Template")

    cnn.build_model()

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..",
                            "log")
    image_path = os.path.join(log_path, "multi_input_regression_model.png")

    #print(image_path)
    utils.plot_model(cnn.model, image_path,
                     show_shapes=False, show_layer_names=True)


if __name__ == '__main__':
    create_structure_img()