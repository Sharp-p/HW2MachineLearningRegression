import os
import pandas as pd
import numpy as np

from PIL import Image
from utils.data import CSVHandler, ImageHandler

def main():
    # loading the filename of the images
    path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(path, '..', 'datasets', 'spqr_dataset', 'raw', 'bbx_annotations.csv')
    csv = CSVHandler(csv_path)
    # path to the new dataset
    new_dataset_path = os.path.join(path, '..', 'datasets', 'new_dataset')
    # path to the new image folder
    new_images_path = os.path.join(new_dataset_path, 'images')
    # create if it does not exist
    if not os.path.exists(new_images_path):
        os.makedirs(new_images_path)

    row_array = []

    filenames = list(set(csv.get_images_name()))
    # iterating over all the images in the csv
    for filename in filenames:
        print(filename)
        image_path = os.path.join(path, '..', 'datasets', 'spqr_dataset', 'images', filename)

        # checking if the images is actually available
        if not os.path.exists(image_path):
            #raise FileNotFoundError(image_path)
            print("[INFO] Does not exist.")
            continue

        image = ImageHandler(filename)
        # loading image data
        img_data = csv.get_image_data(filename)
        # path of the image in the new dataset
        img_path = os.path.join(new_images_path, filename)

        for i, row in enumerate(img_data):

            # if the path to the images does not exist I have to still resize and save it
            if not os.path.exists(img_path):
                # resizing image (it's a tensor)
                img_cropped = image.img_crop_and_resize(0, 0,
                                                        row[1]-1,  row[2]-1,
                                                        [256, 256])
                # transform to array via numpy
                img_array = img_cropped.numpy().astype(np.uint8)
                # transform to pillow image
                pil_image = Image.fromarray(img_array)
                # saving image in new folder
                pil_image.save(img_path)

            # normalizing the center of the bbox
            cx_norm = ((row[6] + row[4]) / 2) / row[1]
            cy_norm = ((row[7] + row[5]) / 2) / row[2]
            # creating a row
            new_row = {'filename': filename,
                       'label': row[3],
                       'cx': cx_norm,
                       'cy': cy_norm,}

            row_array.append(new_row)

    # from dict to dataframe with row
    df_row = pd.DataFrame(row_array)

    new_csv_path = os.path.join(new_dataset_path, 'bbox_data.csv')

    df_row.to_csv(new_csv_path, index=False, mode='w')


if __name__ == '__main__':
    main()