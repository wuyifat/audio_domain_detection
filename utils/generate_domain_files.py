r""" Find equal number of domain files and non-domain files from source file.
To determine if a file is in a domain, look up the text in domain_text file.

source file: csv file, each line is corresponding to an audio file, there can be multiple audios
have the same transcipt.
audio_file_path, text, other stats about the speaker and human transcipters
...

domain_text: txt file, each line is a transcipt. No duplicate transcript in the file.
text1
text2
...

output two csv files: domain_*.csv, domain_*_other.csv, where * is the domain such as food, sport, etc.
audio_file_path, text
"""

import argparse
import pandas as pd
import random

parser = argparse.ArgumentParser()
parser.add_argument("--source-file", default="../data/mozilla/cv-valid-all.csv", type=str, help="Path to file that contains info of all audios.")
parser.add_argument("--text-file", default="../data/mozilla/domain_text_food.txt", type=str,
	help="Path to file that contains distinct transcipts corresponding to a domain.")
parser.add_argument("--output-file-domain", default="../data/mozilla/domain_food.csv", type=str,
	help="Path to the output file that contains domain files.")
parser.add_argument("--output-file-other", default="../data/mozilla/domain_food_other.csv", type=str,
	help="Path to the output file that contains non-domain files.")
parser.add_argument("--ratio", default=1.0, type=float, help="Ratio of non-domain files to domain files.")
args = parser.parse_args()


def generate_domain_files(source_file, text_file, output_file):
	""" Creates a file that contains all audio and text in a specific domain.

	Args:
		source_file: csv file that contains all files in a dataset.
		text_file: txt file that contains all distinct transcripts in the domain.
		output_file: csv file that contains domain file.

	Return:
		number of lines in the output file.

	Write:
		domain_*.csv
	"""
	with open(text_file) as tf:
		domain_text = [line.strip() for line in tf]

	sf = pd.read_csv(source_file)
	domain_df = sf.loc[sf['text'].isin(domain_text)]
	domain_df = domain_df[['filename', 'text']]
	domain_df.to_csv(output_file, index=False)
	return domain_df.shape[0]



def generate_non_domain_files(source_file, text_file, output_file_other, num_files):
	""" Creates a file that contains audio and text not in a specific domain.

	Args:
		source_file: csv file that contains all files in a dataset.
		text_file: txt file that contains all distinct transcripts in the domain.
		output_file: csv file that contains non-domain file.
		num_files: number of non-domain files to output.

	Write:
		domain_*_other.csv

	domain_*_other.csv needs postprocessing to hand screen out the false negative samples
	by looking at the text column.
	filter out the samples whose text is in false_neg.txt
	"""
	with open(text_file) as tf:
		domain_text = [line.strip() for line in tf]

	sf = pd.read_csv(source_file)
	non_domain_df = sf.loc[~sf['text'].isin(domain_text)]
	non_domain_df = non_domain_df[['filename', 'text']]
	#shuffle rows
	non_domain_df = non_domain_df.sample(frac=1).reset_index(drop=True)
	non_domain_df = non_domain_df.head(n=num_files)
	non_domain_df.to_csv(output_file_other, index=False)


def main():
	num_files = generate_domain_files(args.source_file, args.text_file, args.output_file_domain)
	generate_non_domain_files(args.source_file, args.text_file, args.output_file_other, int(num_files * args.ratio))


if __name__ == "__main__":
	main()