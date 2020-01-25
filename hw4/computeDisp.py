import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import sys
from sklearn.feature_extraction import image

def weighted_median(img, window_size=7):

	r = int(window_size/2)
	mask = np.zeros((window_size, window_size), dtype=int)
	for i in range(window_size):
		for j in range(window_size):
			mask[i, j] = max(0, 5-(abs(r-i)+abs(r-j)))

	h, w = img.shape[:2]
	padded_img = cv2.copyMakeBorder(img, top=r, bottom=r, left=r, right=r, borderType=cv2.BORDER_REFLECT)

	for i in range(h):
		for j in range(w):
			padded_i, padded_j = i+r, j+r
			tmp = []
			for m in range(-r, r+1):
				for n in range(-r, r+1):
					tmp += [padded_img[padded_i+m, padded_j+n]] * mask[m+r, n+r]
			
			img[i, j] = int(round(np.median(tmp)))
	return img
'''
def arm_check(i, j, _i, _j, img):

	tao1, tao2 = 18, 6
	L1, L2 = 15, 7
	h, w = img.shape[:2]
	
	if _i < 0 or _j < 0 or _i >= h or _j >= w:
		return False
	if abs(_i - i) == 1 or abs(_j - j) == 1:
		return True
	
	# color similarity test
	if abs(img[i, j, 0] - img[_i, _j, 0]) >= tao1:
		return False
	if abs(img[i, j, 1] - img[_i, _j, 1]) >= tao1:
		return False
	if abs(img[i, j, 2] - img[_i, _j, 2]) >= tao1:
		return False
	
	# max arm length
	if abs(i - _i) >= L1 or abs(j - _j) >= L1:
		return False
	
	# strict color test for far pixel
	if abs(i - _i) >= L2 or abs(j - _j) >= L2:
		if abs(img[i, j, 0] - img[_i, _j, 0]) >= tao2:
			return False
		if abs(img[i, j, 1] - img[_i, _j, 1]) >= tao2:
			return False
		if abs(img[i, j, 2] - img[_i, _j, 2]) >= tao2:
			return False
	
	return True
	

def findCross(img):
	h, w, ch = img.shape
	cross = np.zeros((h, w, 4), dtype=int)

	for i in range(h):
		for j in range(w):
			# Up - Down - Left - Right
			cross[i, j, 0] = i
			cross[i, j, 1] = i
			cross[i, j, 2] = j
			cross[i, j, 3] = j

			while(arm_check(i, j, cross[i, j, 0]-1, j, img)):
				cross[i, j, 0] -= 1
			while(arm_check(i, j, cross[i, j, 1]+1, j, img)):
				cross[i, j, 1] += 1
			while(arm_check(i, j, i, cross[i, j, 2]-1, img)):
				cross[i, j, 2] -= 1
			while(arm_check(i, j, i, cross[i, j, 3]+1, img)):
				cross[i, j, 3] += 1
	
	return cross

def cross_based_cost_aggregation(cross_l, cross_r, cost, max_disp, reverse):

	new_cost = np.zeros_like(cost)
	h, w = cross_l.shape[:2]
	for i in range(h):
		for j in range(w):
			for disp in range(max_disp):
				
				# left image
				if not reverse:
					if j - disp < 0:
						new_cost[i, j, disp] = cost[i, j, disp]
					else:
						p_cost, p_cnt = 0, 0
					
						up_edge = max(cross_l[i, j, 0], cross_r[i, j-disp, 0])
						down_edge = min(cross_l[i, j, 1], cross_r[i, j-disp, 1])
						for vertical in range(up_edge, down_edge+1):
							left_edge = max(cross_l[i, j, 2], cross_r[i, j-disp, 2])
							right_edge = min(cross_l[i, j, 3], cross_r[i, j-disp, 3])
							for horizontal in range(left_edge, right_edge+1):
								p_cost += cost[vertical, horizontal, disp]
								p_cnt += 1
					if p_cnt > 0:
						new_cost[i, j, disp] = p_cost / p_cnt
					else:
						new_cost[i ,j, disp] = cost[i, j, disp]
				
				# right image
				else:
					if j + disp >= w:
						new_cost[i, j, disp] = cost[i, j, disp]
					else:
						p_cost, p_cnt = 0, 0
					
						up_edge = max(cross_l[i, j+disp, 0], cross_r[i, j, 0])
						down_edge = min(cross_l[i, j+disp, 1], cross_r[i, j, 1])
						for vertical in range(up_edge, down_edge+1):
							left_edge = max(cross_l[i, j+disp, 2], cross_r[i, j, 2])
							right_edge = min(cross_l[i, j+disp, 3], cross_r[i, j, 3])
							for horizontal in range(left_edge, right_edge+1):
								p_cost += cost[vertical, horizontal, disp]
								p_cnt += 1
					if p_cnt > 0:
						new_cost[i, j, disp] = p_cost / p_cnt
					else:
						new_cost[i ,j, disp] = cost[i, j, disp]

	return new_cost
'''
def computeDisp(Il, Ir, max_disp):
	h, w, ch = Il.shape
	labels_l = np.zeros((h, w), dtype=np.float32)
	labels_r = np.zeros((h, w), dtype=np.float32)
	Il = Il.astype(np.float32)
	Ir = Ir.astype(np.float32)
	# >>> Cost computation
	# TODO: Compute matching cost from Il and Ir
	
	window_size = 5
	r = int(window_size/2)
	Il_gray = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)
	Ir_gray = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)
	
	# border padding
	padded_Il = cv2.copyMakeBorder(Il_gray, top=r, bottom=r, left=r, right=r, borderType=cv2.BORDER_REFLECT)
	padded_Ir = cv2.copyMakeBorder(Ir_gray, top=r, bottom=r, left=r, right=r, borderType=cv2.BORDER_REFLECT)

	matching_cost_l = np.zeros((h, w, max_disp), dtype=np.float32)
	matching_cost_r = np.zeros((h, w, max_disp), dtype=np.float32)
	census_l = np.zeros((h, w, window_size*window_size), dtype = int)
	census_r = np.zeros((h, w, window_size*window_size), dtype = int)

	# Census transform
	for i in range(h):
		for j in range(w):			
			
			padded_i, padded_j = i+r, j+r			
			idx = 0
			for m in range(-r, r+1):
				for n in range(-r, r+1):
					if padded_Il[padded_i+m, padded_j+n] > padded_Il[padded_i, padded_j]:
						census_l[i, j, idx] = 1

					if padded_Ir[padded_i+m, padded_j+n] > padded_Ir[padded_i, padded_j]:
						census_r[i, j, idx] = 1

					idx += 1

	# Census cost
	for disp in range(max_disp):
		for i in range(h):
			for j in range(w):

				# left image
				if j - disp < 0:
					matching_cost_l[i, j, disp] = 1000
				else:
					matching_cost_l[i, j, disp] = np.sum(census_l[i, j]!=census_r[i, j-disp])

				# right-image
				if j + disp >= w:
					matching_cost_r[i, j, disp] = 1000
				else:
					matching_cost_r[i, j, disp] = np.sum(census_l[i, j+disp]!=census_r[i, j])

	'''
	# Block SSD
	for i in range(h):
		for j in range(w):
			for disp in range(max_disp):
				
				if j - disp < 0:
					matching_cost[i, j, disp] = np.inf
				else:
					
					padded_i, padded_j = i+r, j+r
					for m in range(-r, r+1):
						for n in range(-r, r+1):
							tmp_sum = padded_Il[padded_i+m, padded_j+n] - padded_Ir[padded_i+m, padded_j+n-disp]
							matching_cost[i, j, disp] += tmp_sum * tmp_sum  	

	'''
	# >>> Cost aggregation
	# TODO: Refine cost by aggregate nearby costs
	'''
	cross_l = findCross(Il)
	cross_r = findCross(Ir)
	matching_cost_l = cross_based_cost_aggregation(cross_l, cross_r, matching_cost_l, max_disp, False)
	matching_cost_r = cross_based_cost_aggregation(cross_l, cross_r, matching_cost_r, max_disp, True)
	'''
	#agg_cost_l=np.zeros_like(matching_cost_l)
	#agg_cost_r=np.zeros_like(matching_cost_r)
	for i in range(max_disp):
		matching_cost_l[:,:,i] = cv2.bilateralFilter(matching_cost_l[:,:,i], 15, 9, 30)
		matching_cost_r[:,:,i] = cv2.bilateralFilter(matching_cost_r[:,:,i], 15, 9, 30)
	
	
	# >>> Disparity optimization
	# TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
	for i in range(h):
		for j in range(w):
			labels_l[i, j] = np.argmin(matching_cost_l[i, j,:])
			labels_r[i, j] = np.argmin(matching_cost_r[i, j,:])

	# >>> Disparity refinement
	# TODO: Do whatever to enhance the disparity map
	# ex: Left-right consistency check + hole filling + weighted median filtering

	# left-right consistency check
	consistency = np.zeros((h, w), dtype=int)
	for i in range(h):
		for j in range(w):
			if j - labels_l[i, j] < 0:
				consistency[i, j] = 0
			elif labels_r[i, j-int(labels_l[i,j])] != labels_l[i, j]:
				consistency[i, j] = 0
			else:
				consistency[i, j] = 1
	# hole filling
	for i in range(h):
		for j in range(w):
			if consistency[i, j] == 1:
				continue

			left_neighbor, right_neighbor = max_disp+1, max_disp+1
			for left in range(j, -1, -1):
				if consistency[i, left] != 0:
					left_neighbor = labels_l[i, left]
					break
			for right in range(j, w):
				if consistency[i, right] != 0:
					right_neighbor = labels_l[i, right]
					break
			
			labels_l[i, j] = min(left_neighbor, right_neighbor)		
	
	labels_l = weighted_median(labels_l)
	labels_l = labels_l.astype(np.uint8)
	#kernel = np.ones((5,5),np.uint8)
	#labels_blur = cv2.morphologyEx(labels, cv2.MORPH_OPEN, kernel)
	#labels = cv2.ximgproc.guidedFilter(guide=labels_blur, src=labels, radius=16, eps=1000, dDepth=-1)

	return labels_l.astype(np.uint8)
