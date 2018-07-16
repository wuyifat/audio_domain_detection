r""" Replaces part of the string to make correct path when working with different path setup.

Write a new file with suffix "_new".
"""
import argparse

def main(file, orig_s, new_s):
	if not file:
		print("No file specified")
		return
	with open(file) as oldf:
		fname, suffix = file.split('.')
		new_fname = fname + '_new.' + suffix
		with open(new_fname, 'w') as newf:
			for line in oldf:
				new_line = line.replace(orig_s, new_s)
				newf.write(new_line)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', type=str, default='')
	parser.add_argument('--orig-s', type=str, default='/Codes/ds_pytorch_2018/data')
	parser.add_argument('--new-s', type=str, default='/DATA')

	args = parser.parse_args()
	main(args.file, args.orig_s, args.new_s)