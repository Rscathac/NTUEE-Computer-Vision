import cv2
import numpy as np
import argparse

def Rgb2gray(img, w_r, w_g, w_b):
	return (np.dot(img, [w_b, w_g, w_r]))


def main(args):
	
	img = cv2.imread(args.input)
	output = Rgb2gray(img, args.r, args.g, args.b)
	cv2.imwrite(args.input.replace('.png', '')+'_gray'+str(args.r)+str(args.g)+str(args.b)+'.png' , output)
	
if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', 
						help = 'input image')

	parser.add_argument('-r', '--r', type = float,
						help = 'w_r value')
	parser.add_argument('-g', '--g', type = float,
						help = 'w_g value')
	parser.add_argument('-b', '--b', type = float,
						help = 'w_b value')

	main(parser.parse_args())
