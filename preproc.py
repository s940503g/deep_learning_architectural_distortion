from skimage.io import ImageCollection, imread, concatenate_images
from skimage.color import gray2rgb
from skimage.transform import resize
from skimage import img_as_ubyte
import numpy as np

img_height, img_width = (224, 224)
train_dir = '../trainset/'
val_dir = '../testset/'


def load_dataset_by_dir(_dir):
	pos_dir = _dir + "POSITIVE/*.png"
	ic_pos = ImageCollection(pos_dir, load_func=load_and_pp)
	X_pos = concatenate_images(ic_pos) 
	y_pos = np.array([1 for _ in range(len(ic_pos))])
	neg_dir = _dir + "NEGATIVE/*.png"
	ic_neg = ImageCollection(neg_dir, load_func=load_and_pp)
	X_neg = concatenate_images(ic_neg) 
	y_neg = np.array([0 for _ in range(len(ic_neg))])
	X = np.concatenate((X_pos, X_neg))
	y = np.concatenate((y_pos, y_neg))	
	return (X, y)

def load_trainset():
	X, y = load_dataset_by_dir(train_dir)
	return (X, y)

def load_and_pp(path):
	img = imread(path, True)
	img = gray2rgb(img)
	img = resize(img, (img_height, img_width, 3))	
	return img

def load_testset():
	X, y = load_dataset_by_dir(val_dir)
	return (X, y)
'''
def load_testset():
	pos_dir = validation_data_dir + "POSITIVE/*.png"
	ic_pos = ImageCollection(pos_dir, load_func=load_and_pp)
	X_pos = concatenate_images(ic_pos) 
	y_pos = np.array([1 for _ in range(len(ic_pos))])
	
	neg_dir = validation_data_dir + "NEGATIVE/*.png"
	ic_neg = ImageCollection(neg_dir, load_func=load_and_pp)
	X_neg = concatenate_images(ic_neg) 
	y_neg = np.array([0 for _ in range(len(ic_neg))])

	X = np.concatenate((X_pos, X_neg))
	y = np.concatenate((y_pos, y_neg))
	
	return (X, y)
'''
