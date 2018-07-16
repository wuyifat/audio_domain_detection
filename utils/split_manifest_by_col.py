r""" Split Librispeech manifest to two files which contain the paths .wav files
and .txt files separately.

The number of files to process is max_line.

Librispeech manifest file is organized as
path_to_wav1,path_to_txt1
path_to_wav2,path_to_txt2
.
.
.
Generate two .txt files
wav.txt:
path_to_wav1
path_to_wav2
.
.
.

txt.txt:
path_to_txt1
path_to_txt2
.
.
.
"""

def main():
	count = 0
	max_line = 200
	with open('libri_train_clean_wav.txt', 'w') as wavf:
		with open('libri_train_clean_txt.txt', 'w') as txtf:
			with open('libri_train_clean_manifest.csv') as f:
				for line in f:
					count += 1
					wav, txt = line.split(',')
					wavf.write(wav)
					txtf.write(txt.strip('\n'))
					if count >= max_line:
						return
					else:
						wavf.write('\n')
						txtf.write('\n')

if __name__ == '__main__':
	main()