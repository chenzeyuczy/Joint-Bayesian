#! /usr/bin/env python
#-*- coding:utf-8 -*-

"""
Implementation of joint bayesian model.
"""

import numpy as np
import pickle

def joint_bayesian(data, label):
	""" Joint bayesian training.
	Args:
		data: feature of data, stored as a two-dim numpy array.
		label: Labels of data, either a list or an numpy array.
	Returns:
		A: Model matrix A.
		G: Model matrix G.
	"""
	n_sample, n_dim = data.shape
	classes, labels = np.unique(label, return_inverse=True)

	n_class = len(classes)

	items = []
	n_within_class = 0
	max_per_class = 0
	numberBuff = np.zeros(n_sample, dtype = bool)
	for i in xrange(n_class):
		items.append(data[labels == i])
		n_same_class = len(items[-1])

		# Count number of classes with more than single sample.
		if n_same_class > 1:
			n_within_class += n_same_class
		max_per_class = max(max_per_class, n_same_class)
		numberBuff[n_same_class] = True
	numberBuff = numberBuff[:max_per_class + 1]

	# Initialize Su and Se with between-class and within-class matrices,
	# which is alternative to initialize with random data.
	u = np.zeros((n_dim, n_class))
	e = np.zeros((n_dim, n_within_class))
	pos = 0
	for i in xrange(n_class):
		# Assign u with mean of each class.
		u[:,i] = np.mean(items[i], 0)
		n_same_class = items[i].shape[0]
		# Assign e with difference among each class.
		if n_same_class > 1:
			e[:, pos : pos + n_same_class] = (items[i] - u[:,i]).T
			pos += n_same_class

	Su = np.cov(u.T, rowvar=0)
	Se = np.cov(e.T, rowvar=0)
	oldSe = Se
	SuFG  = {}
	SeG   = {}

	max_iter = 500
	min_convergence = 1e-6
	for l in xrange(max_iter):
		F  = np.linalg.pinv(Se)
		u  = np.zeros([n_dim, n_class])
		e = np.zeros([n_dim, n_sample])
		pos = 0
		for mi in xrange(max_per_class + 1):
			if numberBuff[mi]:
				#G = −(mS μ + S ε )−1*Su*Se−1
				G = -np.dot(np.dot(np.linalg.pinv(mi * Su + Se), Su), F)
				# u = Su*(F+mi*G)
				SuFG[mi] = np.dot(Su, (F + mi * G))
				# e = Se*G
				SeG[mi]  = np.dot(Se, G)
		for i in xrange(n_class):
			n_same_class = items[i].shape[0]
			# Formula 7 in suppl_760
			u[:, i] = np.sum(np.dot(SuFG[n_same_class],items[i].T), 1)
			# Formula 8 in suppl_760
			e[:, pos : pos + n_same_class] = (items[i] + np.sum(np.dot(SeG[n_same_class], items[i].T),1)).T
			pos += n_same_class

		Su = np.cov(u.T, rowvar=0)
		Se = np.cov(e.T, rowvar=0)
		convergence = np.linalg.norm(Se-oldSe) / np.linalg.norm(Se)
		print("Iterations-" + str(l) + ": " + str(convergence))
		if convergence < min_convergence:
			break
		oldSe = Se

	# Formula 6.
	F = np.linalg.pinv(Se)
	G = -np.linalg.pinv(2 * Su + Se) * np.linalg.pinv(Se) * Su
	# Formula 5.
	A = np.linalg.pinv(Su + Se) - (F + G)
	return A, G

def tri_dot(A, B, C):
	"""
	Perform matrix multiplication upon three matrices,
	first of which is transposed before multiplcation.
	"""
	return np.dot(np.dot(A.T, B), C)

def verify(A, G, x1, x2):
	""" Compute log likelihood ratio between two vectors x1 and x2.
	Args:
		A: Model matrix A.
		G: Model matrix G.
		x1: Feature vector 1.
		x2: Feature vector 2.
	Returns:
		ratio: A float number indicates log likelihood ratio between two features.
	"""
	# Convert vector to two-dimension matrix.
	x1 = x1.reshape(-1, 1)
	x2 = x2.reshape(-1, 1)
	# Compute ratio with formula 4.
	ratio = tri_dot(x1, A, x1) + tri_dot(x2, A, x2) - 2 * tri_dot(x1, G, x2)
	return float(ratio)

if __name__ == '__main__':
	feat_file = './feature.pkl'
	with open(feat_file, 'rb') as f:
		feature = pickle.load(f)
		labels = pickle.load(f)

	model_file = './model.pkl'
	TRAIN_AGAIN = False
	if TRAIN_AGAIN:
		A, G = joint_bayesian(feature, labels)

		with open(model_file, 'wb') as f:
			pickle.dump(A, f)
			pickle.dump(G, f)
			print 'Write model to', model_file

	with open(model_file, 'rb') as f:
		A = pickle.load(f)
		G = pickle.load(f)
		print 'Load model from', model_file

	feat1, feat2 = feature[::2, :], feature[1::2, :]
	label1, label2 = labels[::2], labels[1::2]
	sim = label1 == label2
	n_pair = len(sim)

	ratio_file = './ratio.pkl'
	TEST_AGAIN = False

	if TEST_AGAIN:
		# Compute likelihood ratio
		ratio = np.zeros(n_pair)
		for i in xrange(n_pair):
			ratio[i] = verify(A, G, feat1[i], feat2[i])

		with open(ratio_file, 'wb') as f:
			pickle.dump(ratio, f)
			print 'Write ratio to', ratio_file

	with open(ratio_file, 'rb') as f:
		ratio = pickle.load(f)
		print 'Load ratio from', ratio_file
	# Normalization upon ratio.
	ratio = (ratio - ratio.min()) / (ratio.max() - ratio.min())

	# Search the best threshold and its accuracy.
	min_thld, max_thld, step_size = 0, 1.0, 1e-4
	max_acc = 0
	best_thld = None
	for thld in np.arange(min_thld, max_thld, step_size):
		predict = ratio > thld
		accuracy = 1.0 * np.count_nonzero(predict == sim.flatten()) / n_pair
		if accuracy > max_acc:
			max_acc = accuracy
			best_thld = thld
		print 'Threshold:', thld, "Accuracy:", accuracy 
	pass
	print 'Best thld:', best_thld, "Best acc:", max_acc 

