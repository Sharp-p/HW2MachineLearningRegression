import itertools
import os
import wandb
import pandas as pd

from tensorflow.keras import backend as K
from keras.src.callbacks import EarlyStopping
from wandb.integration.keras import WandbMetricsLogger
from utils.cnn_regression import CNNModelRegression

def train(cnn, config=None):
    # defining standard config, just in case
    config = config if config is not None else {"img_size": 128,
                                                "kernel_depth": 16,
                                                "learning_rate": 0.001,
                                                "dropout_val": 0.5,
                                                "pool_size": (3,3),
                                                "epochs": 400,
                                                "batch_size": 32, # small CNN and dataset, so there is no need for a huge batch_size
                                                "val_split": 0.2}
    # building model
    cnn.build_model(lr=config['learning_rate'],
                    kernel_size=(3, 3),
                    pool_size=config['pool_size'],
                    kernel_depth=config['kernel_depth'],
                    dropout_val=config['dropout_val'],
                    normalize=True,
                    dropout=config['dropout']
                    )

    # getting the path of the right dataset
    images_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..',
                                'datasets',
                                'new_dataset',
                                'images')

    # getting the path of the right csv
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..',
                            'datasets',
                            'new_dataset',
                            'bbox_data.csv')

    # creating the dataset
    cnn.create_dataset(csv_path, images_path)

    # starting training
    cnn.train(epochs=config['epochs'],
              batch_size=config['batch_size'],
              callbacks=[WandbMetricsLogger(), EarlyStopping(patience=50,
                                                             restore_best_weights=True)])

def evaluate(cnn, config=None):
    config = config if config is not None else {"img_size": 128,
                                                "kernel_depth": 16,
                                                "learning_rate": 0.001,
                                                "dropout_val": 0.5,
                                                "pool_size": (3, 3),
                                                "epochs": 400,
                                                "batch_size": 32, # small CNN and dataset, so there is no need for a huge batch_size
                                                "val_split": 0.2}

    # getting the evaluation metrics
    mse, rmse, r2 = cnn.evaluate(config['batch_size'])

    result_data = config.copy()
    result_data['mse'] = mse
    result_data['rmse'] = rmse
    result_data['r2'] = r2

    df_result = pd.DataFrame([result_data])

    file_name = "exp_summary.csv"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "log")

    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(path, file_name)
    if not os.path.exists(path):
        df_result.to_csv(path, index=False, mode='w')
    else:
        df_result.to_csv(path, index=False, mode='a', header=False)


def main(PROJECT_NAME):
    # definition of the parameters of the CNN
    param_grid = {"img_size": [256],
                  "kernel_depth": [16, 32],
                  "learning_rate": [0.001],
                  "dropout_val": [0.25, 0.5],
                  "pool_size": [(2, 2), (3, 3)],
                  "dropout": [True],
                  "epochs": [400],
                  "batch_size": [32],
                  "val_split": [0.2],}
    # generation of all the configs
    keys, values = zip(*param_grid.items())
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for i, config in enumerate(configs):
        K.clear_session()
        # creating the wandb runs for each config
        with wandb.init(project=PROJECT_NAME,
                        entity="sharp-1986413-sapienza-universit-di-roma",
                        config=config,
                        group=str(config["img_size"]),
                        job_type="train",
                        name=f"img_size{config['img_size']}_kDepth{config['kernel_depth']}_lr{config['learning_rate']}_dropVal{config['dropout_val']}_poolSize{config['pool_size']}_dropout{config['dropout']}",
                        ) as run:
            cnn = CNNModelRegression([(config["img_size"], config["img_size"], 3), (5,)],
                           2, run.name)
            print("============================================================")
            print(f"TRAINING CONFIG {i}: {run.name}...")
            print("============================================================")
            train(cnn, config)
            evaluate(cnn, config)

            cnn.save_checkpoint()
            print("Done!")

if __name__ == '__main__':
    main("FootballRobotRegression")