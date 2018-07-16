import argparse
import errno
import os

from keras.layers import LSTM, Conv1D, GRU, Bidirectional, BatchNormalization
from keras.layers import Dense, Activation, Dropout, Input
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing import sequence
from keras.models import load_model, Model
import keras.backend as K
from keras.callbacks import CSVLogger, ModelCheckpoint
import numpy as np
import pandas as pd

from td_utils import *
from generator import DataGenerator
# from models import *
import models

parser = argparse.ArgumentParser(description='Domain Detection from audio')
parser.add_argument('--train-csv', default='data/mozilla/domain_food_train.csv',
					metavar='DIR', help='Path to train manifest csv')
parser.add_argument('--val-csv', default='data/mozilla/domain_food_val.csv',
					 metavar='DIR', help='Path to validation manifest csv')
parser.add_argument('--test-csv', default='data/mozilla/domain_food_test.csv',
					metavar='DIR', help='Path to test manifest csv')
parser.add_argument('--batch-size', default=60, type=int, help='Batch size')
parser.add_argument('--sample-rate', default=48000, type=int, help='Audio sample rate')
parser.add_argument('--epoch', default=50, type=int, help='Number of epochs')
parser.add_argument('--maxlen', default=5, type=int, help='Max audio lenght in seconds')
parser.add_argument('--nfft', default=200, type=int, help='Number of frames to calculate STFT on')
parser.add_argument('--noverlap', default=120, type=int, help='Number of frames overlapped when calculating spectrogram')
parser.add_argument('--fs', default=8000, type=int, help='Sampling rate when calculating STFT')
parser.add_argument('--ckpt-dir', default='ckpt/', type=str, help='Directory to store checkpoints')
parser.add_argument('--log-filename', default='log.csv', type=str, help='Log filename')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--model', default='gru_1c2r', type=str, help='model to use')
parser.add_argument('--continue-from', default='',type=str, help='checkpoint to continue')



if __name__ == '__main__':
	args = parser.parse_args()

	audio_conf = dict(nfft=args.nfft,
	                 fs=args.fs,
	                 noverlap=args.noverlap,
	                 maxlen=args.maxlen,
	                 sample_rate=args.sample_rate)
	params = dict(audio_conf=audio_conf, batch_size=args.batch_size, shuffle=True)
	batch_size = args.batch_size
	train_csv = args.train_csv
	val_csv = args.val_csv
	test_csv = args.test_csv

	train_df = pd.read_csv(train_csv)
	val_df = pd.read_csv(val_csv)
	test_df = pd.read_csv(test_csv)

	train_gen = DataGenerator(**params).generate(train_csv)
	val_gen = DataGenerator(**params).generate(val_csv)
	# train_gen1 = DataGenerator(**params).generate(train_csv)

	dim_t = (args.maxlen * args.sample_rate - args.nfft) // (args.nfft - args.noverlap) + 1
	dim_f = args.nfft // 2 + 1
	# dim_t = 5998
	# dim_f = 101


	model_to_call = getattr(models, args.model)
	model = model_to_call(input_shape=(dim_t, dim_f))
	print(model.summary())
	if args.continue_from:
		model.load_weights(args.continue_from)
	opt = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, clipnorm=5.)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

	csv_logger = CSVLogger(args.ckpt_dir + args.log_filename)

	try:
		os.makedirs(args.ckpt_dir)
	except OSError as e:
		if e.errno == errno.EEXIST:
			print("checkpoint directory already exists.")
		else:
			raise
	# ckpt_fname = args.ckpt_dir + "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
	ckpt_fname = args.ckpt_dir + "weights.hdf5"
	checkpoint = ModelCheckpoint(ckpt_fname, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

	model.fit_generator(generator = train_gen,
						steps_per_epoch = train_df.shape[0]//batch_size,
						validation_data = val_gen,
						validation_steps = val_df.shape[0]//batch_size,
						epochs=args.epoch,
						shuffle=True,
						callbacks=[csv_logger, checkpoint])

