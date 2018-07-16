import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import os
# from pydub import AudioSegment
import subprocess
import numpy as np
import pandas as pd
import librosa


# Calculate and plot spectrogram for a wav audio file
# def graph_spectrogram(wav_file):
#     rate, data = get_wav_info(wav_file)
#     nfft = 200 # Length of each window segment
#     fs = 8000 # Sampling frequencies
#     noverlap = 120 # Overlap between windows
#     nchannels = data.ndim
#     if nchannels == 1:
#         pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
#     elif nchannels == 2:
#         pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
#     return pxx

# def get_spectrogram(wav_file):
#     rate, data = get_wav_info(wav_file)
#     nfft = 600 # Length of each window segment
#     fs = 8000 # Sampling frequencies
#     noverlap = 360 # Overlap between windows
#     # nchannels = data.ndim
#     # if nchannels == 1:
#     #     freqs, times, spect = spectrogram(data, nfft=nfft, fs=fs, noverlap = noverlap)
#     # elif nchannels == 2:
#     #     freqs, times, spect = spectrogram(data[:,0], nfft=nfft, fs=fs, noverlap = noverlap)
#     freqs, times, spect = spectrogram(data, nperseg=nfft, nfft=nfft, fs=fs, noverlap = noverlap)
#     return spect

# Load a wav file
# def get_wav_info(wav_file):
#     rate, data = wavfile.read(wav_file)
#     nchannels = data.ndim
#     if nchannels == 2:
#         data = data[:,0]
#     return rate, data

# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def convert_mp3_to_wav(audio_path, sample_rate=16000):
    with os.popen('find %s -type f -name "*.mp3"' % audio_path) as pipe:
        for line in pipe:
            raw_path = line.strip()
            new_path = line.replace('.mp3', '.wav').strip()
            cmd = 'sox -t mp3 -r %d -b 16 -c 1 \"%s\" \"%s\"' % (
                sample_rate, raw_path, new_path)
            os.system(cmd)
            # rm_cmd = 'rm \"%s\"' % (raw_path)
            # os.system(rm_cmd)
            # os.remove(raw_path)

            # subprocess.call(["sox {}  -r {} -b 16 -c 1 {}".format(full_recording_path, str(args.sample_rate),
            #                                               wav_recording_path)], shell=True)

def merge_csv(pos_df, neg_df):
    """ merge dataframes containing positive examples and negative examples to be one dataframe.

    Args:
        pos_csv: Dataframe containing positive examples
        neg_csv: Dataframe containing negative examples

        dataframe format:
        header: filename, text
        path_to_audio, transcript

    Return:
        dataframe that contain all examples from pos_csv and neg_csv.

        dataframe format:
        header: label, filename
        0/1, path_to_audio
        label 0 for neg examples, label 1 for pos examples.
    """
    pos_df.insert(0, 'label', 1)
    neg_df.insert(0, 'label', 0)
    df = pd.concat([pos_df, neg_df])
    return df[['label', 'filename']]
    

def get_spectrogram(audio_path, audio_conf=None):
    if not audio_conf:
        audio_conf = dict(nfft=200,
                     fs=8000,
                     noverlap=120,
                     maxlen=10,
                     sample_rate=48000)
    audio, rate = librosa.core.load(audio_path, sr=audio_conf['sample_rate'], mono=True)
    audio = fix_length(audio, audio_conf['maxlen'] * audio_conf['sample_rate'])
    f, t, spect = spectrogram(audio, nperseg=audio_conf['nfft'], nfft=audio_conf['nfft'], fs=audio_conf['fs'], noverlap=audio_conf['noverlap'])
    return spect


def fix_length(sound, maxcol):
    # pad zeros at the begining if sound array is shorter than maxcol
    # or truncate the sound and take the last maxcol columns
    sound = np.flip(sound, -1)
    sound = librosa.util.fix_length(sound, maxcol)
    sound = np.flip(sound, -1)
    return sound


def prepare_speech_features(csv_path, dim_t=5998, dim_f=101, audio_conf=None):
    df = pd.read_csv(csv_path)
    X = []
    for audio_path in df['filename']:
        spect = get_spectrogram(audio_path, audio_conf)
        spect = spect.swapaxes(0,1)
        X.append(spect)
    return np.asarray(X), np.asarray(df['label'])