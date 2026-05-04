# Tensorflow Computer Vision Helper

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import zipfile

def plot_convolution(data,t,title=''):
    # Plot data with convolution
    # Shows image and filter result
    fig, ax = plt.subplots(2,len(data)+1,figsize=(8,3))
    fig.suptitle(title,fontsize=16)
    tt = np.expand_dims(np.expand_dims(t,2),2)
    for i,im in enumerate(data):
        ax[0][i].imshow(im)
        ximg = np.expand_dims(np.expand_dims(im,2),0)
        cim = tf.nn.conv2d(ximg,tt,1,'SAME')
        ax[1][i].imshow(cim[0][:,:,0])
        ax[0][i].axis('off')
        ax[1][i].axis('off')
    ax[0,-1].imshow(t)
    ax[0,-1].axis('off')
    ax[1,-1].axis('off')
    plt.show()

def plot_results(hist):
    # Plot accuracy and loss
    # Displays training metrics visually
    fig,ax = plt.subplots(1,2,figsize=(15,3))
    ax[0].set_title('Accuracy')
    ax[1].set_title('Loss')
    for x in ['acc','val_acc']:
        if x in hist.history:
            ax[0].plot(hist.history[x])
    for x in ['loss','val_loss']:
        if x in hist.history:
            ax[1].plot(hist.history[x])
    plt.show()

def display_dataset(dataset, labels=None, n=10, classes=None):
    # Display dataset samples
    # Shows images with optional labels
    fig,ax = plt.subplots(1,n,figsize=(15,3))
    for i in range(n):
        ax[i].imshow(dataset[i])
        ax[i].axis('off')
        if classes is not None and labels is not None:
            ax[i].set_title(classes[labels[i][0]])

def check_image(fn):
    # Check if image valid
    # Verifies JPEG format format
    try:
        im = Image.open(fn)
        im.verify()
        return im.format=='JPEG'
    except:
        return False
    
def check_image_dir(path):
    # Check images in directory
    # Removes corrupt or invalid files
    for fn in glob.glob(path):
        if not check_image(fn):
            print("Corrupt image or wrong format: {}".format(fn))
            os.remove(fn)

def extract_dataset(zip_path, dest_dir):
    # Extract dataset from zip
    # Unzips files to destination directory
    if not os.path.exists(dest_dir):
        print("Extracting the dataset")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(dest_dir))

def load_tf_dataset(data_dir, batch_size):
    # Load dataset into tf.data
    # Splits into train and validation
    print("Loading dataset")
    ds_train = keras.preprocessing.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset='training',
        seed=13, image_size=(224,224), batch_size=batch_size
    )
    ds_test = keras.preprocessing.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset='validation',
        seed=13, image_size=(224,224), batch_size=batch_size
    )
    return ds_train, ds_test

def load_cats_dogs_dataset(batch_size=64):
    # Load cats and dogs dataset
    # Extracts, checks, and loads data
    extract_dataset('data/kagglecatsanddogs_3367a.zip', 'data/PetImages')
    print("Checking dataset")
    check_image_dir('data/PetImages/Cat/*.jpg')
    check_image_dir('data/PetImages/Dog/*.jpg')
    return load_tf_dataset('data/PetImages', batch_size)
