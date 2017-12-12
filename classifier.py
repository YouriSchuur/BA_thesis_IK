# Youri Schuur
# BA thesis IK
# Sentiment Classifier

import sys
import io
from collections import defaultdict

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn import svm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.base import TransformerMixin

import numpy as np
import random

import nltk
from nltk.corpus import stopwords
from collections import defaultdict


class Featurizer(TransformerMixin):
	"""Our own featurizer: extract features from each document for DictVectorizer"""

	PREFIX_WORD_NGRAM="W:"
	PREFIX_CHAR_NGRAM="C:"
	
	def fit(self, x, y=None):
		return self
	
	def transform(self, X):
		out = [self._word_ngrams(text,ngram = self.word_ngrams)
				for text in X]
		return out

	def __init__(self,word_ngrams="1",binary=True,lowercase=False,remove_stopwords=True):
		"""
		binary: whether to use 1/0 values or counts
		lowercase: convert text to lowercase
		remove_stopwords: True/False
		"""
		self.DELIM =" "
		self.data = [] # will hold data (list of dictionaries, one for every instance)
		self.lowercase = lowercase
		self.binary = binary
		self.remove_stopwords = remove_stopwords
		self.stopwords = stopwords.words('english')
		self.word_ngrams = word_ngrams

		
	def _word_ngrams(self,text,ngram="1-2-3"):

		d={} #dictionary that holds features for current instance
		if self.lowercase:
			text = text.lower()
		words = text.split(self.DELIM)
		if self.remove_stopwords:
			words = [w for w in words if w not in self.stopwords]
			

		for n in ngram.split("-"):
			for gram in nltk.ngrams(words, int(n)):
				gram = self.PREFIX_WORD_NGRAM + " ".join(gram)
				if self.binary:
					d[gram] = 1 #binary
				else:
					d[gram] += 1
		
		return d

def show_most_informative_features(vectorizer, classifier, n=20): # a function that returns the most informative features
	feature_names = vectorizer.get_feature_names()
	for i in range(0,len(classifier.coef_)):
		coefs_with_fns = sorted(zip(classifier.coef_[i], feature_names))
		top = coefs_with_fns[:-(n + 1):-1]
		print("i",i)
		for (coef, fn) in top:
			print("\t\t%.4f\t%-15s" % (coef, fn))

def main(argv):
	
	print('Loading data....')
	
	# Use these three lines below to train, develop & test the classifier with the non-sarcastic datasets
	train_data = [[x for x in line.strip().split('	')] for line in io.open("final_train_set.txt","r", encoding='utf8')]
	dev_data = [[x for x in line.strip().split('	')] for line in io.open("final_dev_set.txt","r", encoding='utf8')]
	test_data = [[x for x in line.strip().split('	')] for line in io.open("final_test_set.txt","r", encoding='utf8')]
	
	# Use these three lines below to train, develop & test the classifier with the sarcastic datasets
	"""train_data = [[x for x in line.strip().split('	')] for line in io.open("final_sarcasm_train_set.txt","r", encoding='utf8')]
	dev_data = [[x for x in line.strip().split('	')] for line in io.open("final_sarcasm_dev_set.txt","r", encoding='utf8')]
	test_data = [[x for x in line.strip().split('	')] for line in io.open("final_sarcasm_test_set.txt","r", encoding='utf8')]"""
	
	
	# tag the data
	print('Tagging data....')  
	
	X_train_sentences = [sentence for label, sentence in train_data]
	X_dev_sentences = [sentence for label, sentence in dev_data]
	X_test_sentences = [sentence for label, sentence in test_data]
	y_train = [label for label, sentence in train_data]
	y_dev = [label for label, sentence in dev_data]
	y_test = [label for label, sentence in test_data]
	
	
	print('Vectorize data....')
	
	#assign features to every tweet by using the predefined featurizer 
	featurizer = Featurizer(word_ngrams="1-2")
	#vectorize the features with the Dictvectorizer() function from sklearn
	vectorizer = DictVectorizer()
	
	# extract the features from each tweet as dictionaries
	X_train_dict = featurizer.fit_transform(X_train_sentences)
	X_test_dict = featurizer.transform(X_test_sentences)
	"""X_dev_dict = featurizer.transform(X_dev_data)"""
	
	# then convert them to the internal representation (maps each feature to an id)
	X_train = vectorizer.fit_transform(X_train_dict)
	X_test = vectorizer.transform(X_test_dict)
	"""X_dev = vectorizer.transform(X_dev_dict)"""
	
	# determine a classifier
	classifier = LogisticRegression()
	"""classifier = svm.LinearSVC()"""
	
	# Train model and predict scores
	print("Training model..")
	classifier.fit(X_train, y_train)
	
	# Predict labels
	print("Predict..")
	y_predicted = classifier.predict(X_test)
	print
	
	# Make a list with the tweets, the correct label and the predicted label
	# I only print the last 250 tweets from the sarcastic test set because those tweets are all sarcastic
	for predictions in y_predicted:
		predictions = list(zip(X_test_sentences,y_test,y_predicted))
	"""print("Predictions:", predictions[1250:])"""
	
	# Evaluate the system
	print("Accuracy:", accuracy_score(y_test, y_predicted))
	print
	
	print('Classification report:')
	print(classification_report(y_test, y_predicted))
	print
	
	print('Confusion matrix:')
	print(confusion_matrix(y_test, y_predicted))
	print
	
	print('Most informative features:')
	show_most_informative_features(vectorizer, classifier, n=20)
	
if __name__ == '__main__':
	main(sys.argv)
