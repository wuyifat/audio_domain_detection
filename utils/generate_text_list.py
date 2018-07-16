""" Find files containing at least one of the keywords. Write the file list to a new file.

kw: Keyword list.
search_file: Each line in the file is the transcript corresponding to an audio. A transcipt can correspond to multiple audio files so there are ducplicate transcripts.
result_file: Distinct transcripts containing at least one keyword. No duplicate transcript allowed.

Transcripts in result_file are not neccessarily related to the domain. Manual check and pruning is required.
"""

import re

def main(kw, search_file, result_file):
	with open(search_file) as sf, open(result_file, 'w') as rf:
		transcripts = []
		for line in sf:
			if line and line not in transcripts:
				for w in kw:
					if re.findall(w, line):
						transcripts.append(line)
						rf.write(line)
						# break after finding one keyword to prevent writing the transcript multiple time if it contains multiple keywords.
						break


if __name__ == '__main__':
	kw = ['restaurant', 'food', 'sandwich', 'burger', 'eat', 'breakfast', 'lunch', 'dinner', 'soup', 'cook', 'hungry', 'salad', 'pizza', 'ramen', 'pasta', 'sauce', 'cheese', 'milk']
	search_file = '../data/mozilla/cv_text.txt'
	result_file = '../data/mozilla/files_with_kw.txt'
	main(kw, search_file, result_file)
