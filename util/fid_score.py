#!/usr/bin/env python3

#
# https://github.com/mseitzer/pytorch-fid
#

"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

from PIL import Image

try:
	from tqdm import tqdm
except ImportError:
	# If not tqdm is not available, provide a mock version of it
	def tqdm(x): return x

from .inceptionV3 import InceptionV3

cuda = True if torch.cuda.is_available() else False



parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str, nargs=2,
					help=('Path to the generated images or '
						  'to .npz statistic files'))
parser.add_argument('--batch-size', type=int, default=50,
					help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
					choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
					help=('Dimensionality of Inception features to use. '
						  'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='', type=str,
					help='GPU to use (leave blank for CPU only)')




def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
	"""Numpy implementation of the Frechet Distance.
	The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
	and X_2 ~ N(mu_2, C_2) is
			d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

	Stable version by Dougal J. Sutherland.

	Params:
	-- mu1   : Numpy array containing the activations of a layer of the
			   inception net (like returned by the function 'get_predictions')
			   for generated samples.
	-- mu2   : The sample mean over activations, precalculated on an
			   representative data set.
	-- sigma1: The covariance matrix over activations for generated samples.
	-- sigma2: The covariance matrix over activations, precalculated on an
			   representative data set.

	Returns:
	--   : The Frechet Distance.
	"""

	mu1 = np.atleast_1d(mu1)
	mu2 = np.atleast_1d(mu2)

	sigma1 = np.atleast_2d(sigma1)
	sigma2 = np.atleast_2d(sigma2)

	assert mu1.shape == mu2.shape, \
		'Training and test mean vectors have different lengths'
	assert sigma1.shape == sigma2.shape, \
		'Training and test covariances have different dimensions'

	diff = mu1 - mu2

	# Product might be almost singular
	covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
	if not np.isfinite(covmean).all():
		msg = ('fid calculation produces singular product; '
			   'adding %s to diagonal of cov estimates') % eps
		print(msg)
		offset = np.eye(sigma1.shape[0]) * eps
		covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

	# Numerical error might give slight imaginary component
	if np.iscomplexobj(covmean):
		if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
			m = np.max(np.abs(covmean.imag))
			raise ValueError('Imaginary component {}'.format(m))
		covmean = covmean.real

	tr_covmean = np.trace(covmean)

	return (diff.dot(diff) + np.trace(sigma1) +
			np.trace(sigma2) - 2 * tr_covmean)






def get_activations(batch, model, batch_size=50, dims=2048):

	model.eval()

	if batch_size > batch.shape[0]:
		print(('Warning: batch size is bigger than the data size. '
			   'Setting batch size to data size'))
		batch_size = batch.shape[0]

	pred_arr = np.empty((batch.shape[0], dims))

	for i in tqdm(range(0, batch.shape[0], batch_size)): #tqdm: progressive bar

		start = i
		end = i + batch_size

		data = batch[start:end]

		if cuda:
			data=data.type(torch.cuda.FloatTensor)

		#pred = model(batch)[0]
		pred = model(data)[0]

		# If model output is not scalar, apply global spatial average pooling.
		# This happens if you choose a dimensionality not equal 2048.
		if pred.size(2) != 1 or pred.size(3) != 1:
			pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

		pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

	return pred_arr

def calculate_activation_statistics(batch, model, batch_size=50, dims=2048):

	act = get_activations(batch, model, batch_size, dims)
	mu = np.mean(act, axis=0)
	sigma = np.cov(act, rowvar=False)
	return mu, sigma


def _compute_statistics_of_batch(batch, model, batch_size, dims):
	
	m, s = calculate_activation_statistics(batch, model, batch_size, dims)

	return m, s


#def calculate_fid_given_paths(paths, batch_size, cuda, dims):
def calculate_fid_given_batches(batch1, batch2, batch_size, dims=2048):


	# gray image
	if batch1.shape[1] < 3:  batch1 = batch1.expand(-1,3,-1,-1) 
	if batch2.shape[1] < 3:  batch2 = batch2.expand(-1,3,-1,-1) 

	block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

	model = InceptionV3([block_idx])

	if cuda:
		model.cuda()


	m1, s1 = _compute_statistics_of_batch(batch1, model, batch_size, dims)
	m2, s2 = _compute_statistics_of_batch(batch2, model, batch_size, dims)
	fid_value = calculate_frechet_distance(m1, s1, m2, s2)

	return fid_value


if __name__ == '__main__':
	args = parser.parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	fid_value = calculate_fid_given_paths(args.path, args.batch_size, args.gpu != '', args.dims)
	print('FID: ', fid_value)
