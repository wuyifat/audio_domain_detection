def main(kw):
	with open('cv.csv') as f:
		for line in f:
			content = line.split(',')
			if kw in content[1]:
				print(content[0], content[1])

if __name__ == '__main__':
	kw = 'restaurant'
	main(kw)