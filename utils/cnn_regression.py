import os
import pandas as pd

from tensorflow import data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, Model
from tensorflow.keras.metrics import RootMeanSquaredError, R2Score
from tensorflow.keras.models import load_model
from utils.data import load_and_preprocess

class CNNModelRegression:
    def __init__(self, input_shapes, output_shape, model_name):
        """
        Constructor of the multi-input model
        :param input_shapes: An array of input shapes
        :param output_shape: The output shape of the model
        :param model_name: An identifier for the model
        """

        self.model_name = model_name
        self.input_shapes = input_shapes # in our case [(img_size, img_size, 3), (5,)
        self.output_shape = output_shape

        self.model = None
        self.history = None
        self.dataset = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.is_trained = False


    def build_model(self, lr=0.001, kernel_size=(3,3), pool_size=(2,2), kernel_depth=16,
                    dropout_val=0.001, normalize=True, dropout=True):
        # defining the image convolution part of the graph
        img_input = layers.Input(shape=self.input_shapes[0], name='img_input')

        if normalize:
            x = layers.Rescaling(1./255)(img_input)
            x = layers.Conv2D(kernel_depth, kernel_size, activation='relu')(x)
        else:
            x = layers.Conv2D(kernel_depth, kernel_size, activation='relu')(img_input)

        x = layers.MaxPooling2D(pool_size=pool_size)(x)
        x = layers.Conv2D(kernel_depth*2, kernel_size, activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=pool_size)(x)
        x = layers.Conv2D(kernel_depth*4, kernel_size, activation='relu')(x)
        img_features = layers.Flatten(name="img_features")(x)

        label_input = layers.Input(shape=self.input_shapes[1], name='label_input')

        # regression part
        y = layers.concatenate([img_features, label_input])
        if dropout: y = layers.Dropout(dropout_val)(y)
        y = layers.Dense(64, activation="relu")(y)
        if dropout: y = layers.Dropout(dropout_val/2)(y)
        y = layers.Dense(32, activation="relu")(y)
        if dropout: y = layers.Dropout(dropout_val/4)(y)
        y = layers.Dense(16, activation="relu")(y)

        real_output = layers.Dense(self.output_shape, activation="sigmoid")(y)

        self.model = Model(inputs=[img_input, label_input], outputs=real_output)

        self.model.compile(optimizer=Adam(learning_rate=lr),
                           loss='MSE',
                           metrics=[RootMeanSquaredError(),
                                    R2Score()])

    def create_dataset(self, csv_path, images_path):
        if not os.path.exists(images_path) or not os.path.exists(csv_path):
            raise FileNotFoundError('Dataset not found')

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
        self.dataset = dataset.shuffle(dataset.cardinality(), reshuffle_each_iteration=False)

    def train(self, epochs=50, batch_size=32, callbacks=None) -> None:
        """
        Trains the model.
        The validation split is 20% of the training dataset.
        :param dataset_path: Path to dataset of images divided in folders for each class. The images are expected to be in RGB format.
        :param size: Size of the images in the dataset.
        :param epochs: Number of epochs to train the model.
        :param batch_size: Batch size.
        :param val_split: Validation split.
        :param callbacks: List of callback functions.
        :return: None.
        """
        if self.model is None:
            self.build_model()

        if self.dataset is None:
            print("Dataset not found. Please run create_dataset() first.")

        # generating training and validation dataset


        n = self.dataset.cardinality().numpy()
        self.train_ds = self.dataset.take(int(n*0.64))
        # getting the remaining values and getting only the original 20% of it
        self.val_ds = self.dataset.skip(int(n*0.64))
        self.val_ds = self.val_ds.take(int(self.val_ds.cardinality().numpy()*0.44))

        self.train_ds = self.train_ds.batch(batch_size).prefetch(buffer_size=data.AUTOTUNE)
        self.val_ds = self.val_ds.batch(batch_size).prefetch(buffer_size=data.AUTOTUNE)

        print("Training CNN for maximum", epochs, " epochs...")
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            verbose=0,
            callbacks=callbacks
        )
        self.is_trained = True
        print("Done!")

    def evaluate(self, batch_size=32):
        if not self.is_trained:
            print("CNN model not trained!")
            return None

        # generating the test_ds
        n = self.dataset.cardinality().numpy()
        self.test_ds = (self.dataset.skip(int(n * 0.8))
                        .batch(batch_size)
                        .prefetch(buffer_size=data.AUTOTUNE))

        # evaluation
        mse, rmse, r2 = self.model.evaluate(self.test_ds)
        return mse, rmse, r2

    def save_checkpoint(self):
        folder_path = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(folder_path, '..', 'models', self.model_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        print("Saving model checkpoint...")
        self.model.save(os.path.join(folder_path, 'model_'+self.model_name+'.keras'))

    def load_checkpoint(self):
        folder_path = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(folder_path, '..', 'models', self.model_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        print("Loading model checkpoint...")
        self.model = load_model(os.path.join(folder_path, 'model_'+self.model_name+'.keras'))
        self.is_trained = True
