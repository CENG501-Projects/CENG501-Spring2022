import numpy as np
from os import listdir
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


def save_dataset(filename, data_free, data_shadow, data_mask):
    # save a compressed numpy array
    filename = filename
    np.savez_compressed(filename, data_free, data_shadow, data_mask)
    print('Saved dataset: ', filename)


def load_dataset(filename, dataset_name='shadow'):
    data = np.load(filename)
    if dataset_name == 'free':
        result = data['arr_0']
    elif dataset_name == 'shadow':
        result = data['arr_1']
    elif dataset_name == 'mask':
        result = data['arr_2']
    else:
        raise Exception('Undefined label name: {%s}' % dataset_name)
    print('Loaded:', dataset_name, result.shape)
    return result


# load all images in a directory into memory
def load_images(path, size, dataset_size=None):
    data_list = list()
    # enumerate filenames in directory, assume all are images
    for filename in listdir(path):
        # load and resize the image
        pixels = load_img(path + filename, target_size=size)
        # convert to numpy array
        pixels = img_to_array(pixels)
        # store
        data_list.append(pixels)
        if dataset_size:
            dataset_size -= 1
            if dataset_size == 0:
                break
    return np.asarray(data_list)


def load_real_samples(filename=None, dataset=[]):
    # load the dataset

    if filename is not None:
        data = np.load(filename)
        X1, X2, X3 = data['arr_0'], data['arr_1'], data['arr_2']  # scale from [0,255] to [-1,1]
    elif len(dataset) != 0:
        X1, X2, X3 = dataset[0], dataset[1], dataset[2]
    else:
        print('No filename or dataset')

    # unpack arrays
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    X3 = (X3 - 127.5) / 127.5
    return [X1, X2, X3]
