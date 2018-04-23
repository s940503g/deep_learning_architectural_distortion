import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense, Input, Conv2D, Activation
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
from keras.models import Model, load_model, Sequential
from resnet152 import ResNet152
from keras_contrib.applications.densenet import DenseNetImageNet161
from keras.callbacks import CSVLogger, BaseLogger
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve 
import argparse
from keras_metrics import recall, precision, F1, specificity
from preproc import load_trainset, load_testset
import numpy as np

parser = argparse.ArgumentParser()


def auroc(y_true, y_prob, plot=False, filename='AUROC.png'):
	y, prob = y_true, y_prob
	auc_score = roc_auc_score(y, prob)

	# Compute micro-average ROC curve and ROC area
	fpr = dict()
	tpr = dict()
	for i in range(2):
	    fpr[i], tpr[i], _ = roc_curve(y, prob)

	if plot:
		fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), prob.ravel())
		plt.figure()
		lw = 2
		plt.plot(fpr[1], tpr[1], color='darkorange',
			 lw=lw, label='ROC curve (area = %0.2f)' % auc_score)
		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic')
		plt.legend(loc="lower right")
		plt.savefig(filename)
		plt.close()

	return auc_score

def prauc(y_true, y_prob, plot=False, filename='PRAUC.png'):
	y, prob = y_true, y_prob
	precision, recall, thresholds = precision_recall_curve(y, prob)
	prauc_score = auc(recall, precision, reorder=True)
	if plot:
		plt.figure()
		plt.step(recall, precision, color='b', alpha=0.2,where='post')
		plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim([0.0, 1.0])
		plt.xlim([0.0, 1.0])
		plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(prauc))
		plt.savefig(filename)
		plt.close()
	return prauc_score

def eval_auc(model, X, y, filename='', plot_auroc=False, plot_prauc=False):	
	prob = model.predict(X)	
	auroc_score = auroc(y, prob, plot=plot_auroc, filename=filename+'_auroc.png')
	prauc_score = prauc(y, prob, plot=plot_prauc, filename=filename+'_prauc.png')
	return auroc_score, prauc_score

from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.nasnet import NASNetLarge
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.densenet import DenseNet201, DenseNet169, DenseNet121
from keras.applications.resnet50 import ResNet50

def build_model(base_model, X, y, test_X=None, test_y=None, input_shape=(224, 224, 3), epochs=5, batch_size=32, log_csv=False, log_fn=None, freeze=0, summary=False, class_weight={1:1, 0:1}):

	model = base_model(include_top=False, weights='imagenet', input_shape=input_shape)
	x = model.output
	x = GlobalAveragePooling2D()(x)
	predictions = Dense(1, activation='sigmoid')(x)

	# freeze part of layers in network
	if freeze > 0:
		for layer in model.layers[:freeze]: 
			layer.trainable = False

	model = Model(inputs=model.input, outputs=predictions)
	# print summary
	if summary:
		model.summary()

	# set optimizer, loss function and metrics
	metrics = ['accuracy', recall, specificity, precision, F1]
	model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-3), metrics=metrics)

	# data augment
	train_datagen = ImageDataGenerator(rotation_range=90, horizontal_flip=True)
	train_datagen.fit(X, augment=True, rounds=50, seed=123)
	train_generator = train_datagen.flow(X, y, batch_size=batch_size)

	# set callback functions	
	cbs = None
	if log_csv:
		if not log_fn: 
			log_fn = base_model.__name__
		cv_logger = CSVLogger(log_fn)
		cbs = [cv_logger]
	
	val_data = None
	if np.any(test_X) and np.any(test_y):
		val_data = (test_X, test_y)
	

	result = model.fit_generator(train_generator, validation_data=val_data, epochs=epochs, verbose=0, callbacks=cbs, class_weight=class_weight)

	return model, result.history

if __name__ == '__main__':
	import csv
	from sklearn.model_selection import StratifiedKFold

	'''
	Transfer learning expriment
	finding the best number of freeze layers
	'''

	epochs = 50
	batch_size = 32
	class_weight = {1:10, 0:1}
	cv = 10
	kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=123)

	# load data
	print("\nLoading dataset..")
	X, y = load_trainset()
	#test_X, test_y = load_testset()

	content = []
	title = ['model', 'freeze']
	for i in range(cv):
		for val in ['auroc', 'prauc']:
			cvn = val + '_cv' + str(i)
			title.append(cvn)
	content.append(title)

	model_list = [VGG16, VGG19, ResNet50]
	for base_model in model_list:
		for freeze in range(10):
			print("\nStart training", base_model.__name__ , 'with freezed top', str(freeze), 'layers')
			row = [base_model.__name__, freeze]
			fold = 0
			for train, test in kfold.split(X, y):
				fold += 1 
				train_X, train_y = X[train], y[train]
				test_X, test_y = X[test], y[test]
				model, his = build_model(base_model, X=train_X, y=train_y, test_X=test_X, test_y=test_y, freeze=freeze, 
					input_shape=(224, 224, 3), 
					epochs=epochs, batch_size=batch_size, 
					class_weight=class_weight)
				# evaluate model use AUROC and PRAUC
				auroc_score, prauc_score = eval_auc(model, test_X, test_y)
				print('Coss validation {:.0f}\tAUROC:{:.2f}, PRAUC:{:.2f}'.format(fold, auroc_score, prauc_score))
				row.append(auroc)
				row.append(prauc)

			content.append(row)

		
	f = open('result.csv', 'w')
	w = csv.writer(f)
	w.writerows(content)		
	f.close()	
	
