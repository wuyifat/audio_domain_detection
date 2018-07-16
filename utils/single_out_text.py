"""
cv.csv contains multiple columns. Create a new file cv_text.txt that contains only the
text column of cv.csv.
"""


with open('cv.csv') as f1:
	with open('cv_text.txt', 'w') as f2:
		for line in f1:
			context = line.split(',')
			f2.write(context[1])
			f2.write('\n')
			f2.write('\n')