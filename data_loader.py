import pandas as pd
from scipy.signal import spectrogram
from scipy.io import wavfile


class CSVProcessor(object):
	# csv file manipulation
	def __init__(self, csv_path):
		self.csv_path = csv_path
		self.df = pd.read_csv(csv_path)

	def add_label(self, label, value, position):
		self.df.insert(position, label, value)
		return self.df


class AudioProcess(object):
	# process one audio file
	def __init__(self, audio_conf, audio_path):
		self.audio_conf = audio_conf
		self.audio_path = audio_path
		self.rate, self.audio = self.load_wav(audio_path)

	def load_wav(self, audio_path):
		rate, data = wavfile.read(audio_path)
		nchannels = data.ndim
		if nchannels == 2:
			data = data[:,0]
		return rate, data

	def get_spectrogram(self):
		nfft = self.audio_conf['nfft']
		fs = self.audio_conf['fs']
		noverlap = self.audio_conf['noverlap']
		# nperseg = audio_conf['nperseg']
		f, t, spect = spectrogram(self.audio, nfft=nfft, fs=fs, noverlap=noverlap)
		return spect

	def get_audio_length(self):
		return self.audio.shape[0] / self.rate

	def get_audio_path(self):
		return self.audio_path

	def get_audio_rate(self):
		return self.rate

	def normalize_length(self, maxlen):
		raise NotImplementedError


class TextProcess(object):
	def get_transcript():
		pass

	def get_label():
		pass
