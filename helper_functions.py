import math

import numpy as np
import pandas as pd

from keras.models import load_model

y_cols = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x',
       'right_eye_center_y', 'left_eye_inner_corner_x',
       'left_eye_inner_corner_y', 'left_eye_outer_corner_x',
       'left_eye_outer_corner_y', 'right_eye_inner_corner_x',
       'right_eye_inner_corner_y', 'right_eye_outer_corner_x',
       'right_eye_outer_corner_y', 'left_eyebrow_inner_end_x',
       'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x',
       'left_eyebrow_outer_end_y', 'right_eyebrow_inner_end_x',
       'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_x',
       'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y',
       'mouth_left_corner_x', 'mouth_left_corner_y', 'mouth_right_corner_x',
       'mouth_right_corner_y', 'mouth_center_top_lip_x',
       'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x',
       'mouth_center_bottom_lip_y']

def load_data(img_file):
    data = pd.read_csv(img_file)
    data = data.dropna()
    return data

def to_img_arr(img_vals):
    img_arr = np.fromstring(img_vals, sep=' ')
    img_arr = img_arr.reshape((96,96))
    return img_arr

def normalize_pos(val):
    return (val/95)- 0.5

def preproces(img_col):
    img_col = img_col.apply(to_img_arr)
    return img_col


def prepare_training_data(img_file):
    data = load_data(img_file)
    data['Image'] = data['Image'].apply(to_img_arr)
    

    # Reshaping image 
    img_vals = []
    for img_val in data['Image']:
        img_vals.append(img_val)

    X = np.array(img_vals)
    X = np.expand_dims(X, axis=3)

    # Reshaping y values
    for y_col in y_cols:
        data[y_col] = data[y_col].apply(normalize_pos)

    y = np.array(data[y_cols])

    print("X_shape, y_shape: ", X.shape, y.shape)

    return (X, y)

def generate_test_result(test_file):
    # Load trained model
    model = load_model('model.h5')

    test_data = pd.read_csv(test_file)
    X_test = test_data['Image'].apply(to_img_arr)

    # Reshaping image 
    img_vals = []
    for img_val in X_test:
        img_vals.append(img_val)

    X_test = np.array(img_vals)
    X_test = np.expand_dims(X_test, axis=3)

    y_test = model.predict(X_test)

    # Revert dimensions back to (0, 95)
    y_test = (y_test + 0.5)*95

    final_lst = []
    row_id = 1
    image_id = 1
    for y_curr in y_test:
        for idx, y_col in enumerate(y_cols):
            dict = {}
            dict['RowId'] = row_id
            row_id += 1

            dict['ImageId'] = image_id
            dict['FeatureName'] = y_col
            dict['Location'] = y_curr[idx]
            final_lst.append(dict)

        image_id += 1

    df = pd.DataFrame(final_lst)

    df = df[['RowId', 'ImageId', 'FeatureName', 'Location']]
    df.to_csv('data/test_output.csv', index=False)

    idlookup = pd.read_csv('data/IdLookupTable.csv')
    idlookup = idlookup[['RowId', 'ImageId', 'FeatureName']]

    df_final = df[['ImageId', 'FeatureName', 'Location']].merge(idlookup, on=['ImageId', 'FeatureName'])

    df_final = df_final[['RowId', 'Location']]

    df_final.to_csv('data/test_output_final.csv', index=False)





