import numpy as np
import argparse
import cv2

def Most_voted(vote_table):
	
	most_voted = []
	vote_num = []
	
	for i in range(3):
		max_value = np.max(vote_table)
		if max_value == 0:
			break
		max_loc = np.where(vote_table == max_value)
		max_idx = max_loc[0][0], max_loc[1][0], max_loc[2][0]
		vote_table[max_idx] = -1
		vote_num.append(max_value)
		most_voted.append(max_idx)
	
	return most_voted, vote_num

def Find_candidate():
	
	candidate_table = []
	for w_r in range(11):
		for w_g in range(11):
			for w_b in range(11):
				if w_r + w_g + w_b != 10:
					continue
				
				candidate_table.append([w_r, w_g, w_b])

	return candidate_table

def Vote(cost_table, vote_table, candidate_table):
	
	for w_r, w_g, w_b in candidate_table:
		local_min = True
		for d in [[1, -1, 0], [1, 0, -1], [0, 1, -1], [-1, 1, 0], [-1, 0, 1], [0, -1, 1]]:
			if [w_r+d[0], w_g+d[1], w_b+d[2]] not in candidate_table:
				continue
			
			if cost_table[w_r][w_g][w_b] >= cost_table[w_r+d[0]][w_g+d[1]][w_b+d[2]]:
				local_min = False
				break
		
		if local_min is True:
			vote_table[w_r][w_g][w_b] += 1

	return vote_table

def Iterate_candidate(img, sigma_s, sigma_r, candidate_table):	

	cost_table = np.zeros((11, 11, 11), dtype = int)
	jbf = Joint_bilateral_filter(sigma_s, sigma_r, 'reflect')
	
	
	bf_img  = jbf.joint_bilateral_filter(img, img)
	for w_r, w_g, w_b in candidate_table:
		
		gray_img = Rgb2gray(img, w_r/10.0, w_g/10.0, w_b/10.0)
		jbf_img = jbf.joint_bilateral_filter(img, gray_img)
		cost_table[w_r][w_g][w_b] = np.sum(abs(jbf_img-bf_img))
				
	return cost_table
		
def Rgb2gray(img, w_r, w_g, w_b):
	return (np.dot(img, [w_b, w_g, w_r]))

class Joint_bilateral_filter(object):
	def __init__(self, sigma_s, sigma_r, border_type='reflect'):
		
		self.border_type = border_type
		self.sigma_r = sigma_r
		self.sigma_s = sigma_s
		self.r = 3 * sigma_s
		self.win = 2*self.r+1
	
	def joint_bilateral_filter(self, input, guidance):
		## TODO
		output = np.zeros(input.shape)
		guidance = guidance / 255.0
		padded_img = cv2.copyMakeBorder(input, top=self.r, bottom=self.r, left=self.r, right=self.r, borderType=cv2.BORDER_REFLECT)
		padded_guidance = cv2.copyMakeBorder(guidance, top=self.r, bottom=self.r, left=self.r, right=self.r, borderType=cv2.BORDER_REFLECT)
		
		# Spatial Kernel
		G_s = np.zeros((self.win, self.win), dtype = np.float64)
		for i in range(G_s.shape[0]):
			for j in range(G_s.shape[1]):
				G_s[i][j] = np.exp(-((self.r-i)**2+(self.r-j)**2)/(2*self.sigma_s**2))
		
		# Guidance is single channel image
		if input.shape != guidance.shape:
			
			for i in range(input.shape[0]):
				for j in range(input.shape[1]):
					
					# Range Kernel
					G_r = np.exp(-((padded_guidance[i:i+self.win, j:j+self.win] - guidance[i][j])**2/(2*self.sigma_r**2)))
					multi = np.multiply(G_s, G_r)
					output[i][j][0] = np.sum(np.multiply(multi, padded_img[i:i+self.win, j:j+self.win, 0]))/np.sum(multi)
					output[i][j][1] = np.sum(np.multiply(multi, padded_img[i:i+self.win, j:j+self.win, 1]))/np.sum(multi)
					output[i][j][2] = np.sum(np.multiply(multi, padded_img[i:i+self.win, j:j+self.win, 2]))/np.sum(multi)

		# Guidance is RGB image
		else:

			for i in range(input.shape[0]):
				for j in range(input.shape[1]):
					
					# Range Kernel
					power_b = (padded_guidance[i:i+self.win, j:j+self.win, 0] - guidance[i][j][0])**2
					power_g = (padded_guidance[i:i+self.win, j:j+self.win, 1] - guidance[i][j][1])**2
					power_r = (padded_guidance[i:i+self.win, j:j+self.win, 2] - guidance[i][j][2])**2
					G_r = np.exp(-(power_b + power_g + power_r)/(2*self.sigma_r**2))
					multi = np.multiply(G_s, G_r)
					output[i][j][0] = np.sum(np.multiply(multi, padded_img[i:i+self.win, j:j+self.win, 0]))/np.sum(multi)
					output[i][j][1] = np.sum(np.multiply(multi, padded_img[i:i+self.win, j:j+self.win, 1]))/np.sum(multi)
					output[i][j][2] = np.sum(np.multiply(multi, padded_img[i:i+self.win, j:j+self.win, 2]))/np.sum(multi)

		return output


def main(args):
	
	img = cv2.imread(args.input)
	gray_img = img.copy()
	
	vote_table = np.zeros((11, 11, 11), dtype = int)
	candidate_table = Find_candidate()
	for sigma_s in [1, 2, 3]:
		for sigma_r in [0.05, 0.1, 0.2]:
			cost_table = Iterate_candidate(img, sigma_s, sigma_r, candidate_table)
			vote_table = Vote(cost_table, vote_table, candidate_table)
			
	max_voted, vote_num = Most_voted(vote_table)
	print(max_voted)
	print(vote_num)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', default = 'testdata/0a.png',
						help = 'input image')
	parser.add_argument('--mode', '-a', action = 'store_true',
						help = 'c:Type -a for advanced mode')
	
	main(parser.parse_args())
	
