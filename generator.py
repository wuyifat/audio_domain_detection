import numpy as np
import pandas as pd

import librosa
from scipy.signal import spectrogram
from scipy.io import wavfile

from td_utils import *

class DataGenerator(object):
	""" Generates data for keras.
	"""
	def __init__(self, audio_conf, batch_size=32, shuffle=True):
		self.audio_conf = audio_conf
		self.nfft = audio_conf['nfft']
		self.fs = audio_conf['fs']
		self.noverlap = audio_conf['noverlap']
		self.maxlen = audio_conf['maxlen']
		self.sample_rate = audio_conf['sample_rate']
		self.batch_size = batch_size
		self.shuffle = shuffle

		self.dim_t = (self.maxlen * self.sample_rate - self.nfft) // (self.nfft - self.noverlap) + 1
		self.dim_f = self.nfft // 2 + 1

	def generate(self, csv_path):
		""" Generates batches of samples
		"""
		try:
			self.df = pd.read_csv(csv_path)
		except:
			print(csv_path, " does not exists.")

		if self.shuffle:
			self.df = self.df.sample(frac=1).reset_index(drop=True)

		while True:
			imax = int(self.df.shape[0] / self.batch_size)
			for i in range(imax):
				batch_df = self.df.iloc[i*self.batch_size : (i+1)*self.batch_size].reset_index(drop=True)

				X, y = self.__generate_dfxy(batch_df)

				yield X, y

	# def generate_toy(self, csv_path):
	# 	self.df = pd.read_csv(csv_path)
	# 	while True:
	# 		imax = self.df.shape[0] // self.batch_size
	# 		for i in range(imax):
	# 			batch_df = self.df.iloc[i*self.batch_size : (i+1)*self.batch_size].reset_index(drop=True)
	# 			yield batch_df

	def __generate_dfxy(self, df):
		X = np.empty((self.batch_size, self.dim_t, self.dim_f))
		y = np.empty((self.batch_size), dtype=int)
		for i, row in df.iterrows():
			row_x, row_y = self.__generate_rowxy(row)
			X[i, :, :] = row_x
			y[i] = row_y
		return X, y

	def __generate_rowxy(self, row):
		audio_path = row['filename']
		label = row['label']
		X = get_spectrogram(audio_path, self.audio_conf)
		# X = self.get_spectrogram(audio_path)
		X = X.swapaxes(0,1)
		return X, label

	# def get_spectrogram(self, audio_path):
	# 	# audio = self.load_wav(audio_path)
	# 	audio, rate = librosa.core.load(audio_path, sr=self.sample_rate, mono=True)
	# 	audio = self.fix_length(audio, self.maxlen * self.sample_rate)
	# 	f, t, spect = spectrogram(audio, nperseg=self.nfft, nfft=self.nfft, fs=self.fs, noverlap=self.noverlap)
	# 	return spect


	# def fix_length(self, sound, maxcol):
	# 	# pad zeros at the begining if sound array is shorter than maxcol
	# 	# or truncate the sound and take the last maxcol columns
	# 	sound = np.flip(sound, -1)
	# 	sound = librosa.util.fix_length(sound, maxcol)
	# 	sound = np.flip(sound, -1)
	# 	return sound