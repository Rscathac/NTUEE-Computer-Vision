import numpy as np
import cv2
import time
import math


# u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
# this function should return a 3-by-3 homography matrix
def solve_homography(u, v):
	N = u.shape[0]
	if v.shape[0] is not N:
		print('u and v should have the same size')
		return None
	if N < 4:
		print('At least 4 points should be given')
	
	# if you take solution 2:
	A = np.zeros((2*N, 9))
	b = np.zeros((2*N, 1))
	H = np.zeros((3, 3))

	for i in range(N):
		ux, uy = u[i]
		vx, vy = v[i]
		A[2*i] =   [ux, uy, 1, 0, 0, 0, -ux*vx, -uy*vx, -vx]
		A[2*i+1] = [0, 0, 0, ux, uy, 1, -ux*vy, -uy*vy, -vy]
	
	# TODO: compute H from A and b
	U, S, V_T = np.linalg.svd(A)

	return V_T[-1] / V_T[-1, -1]


# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners):
	h, w, ch = img.shape
	height, width = canvas.shape[:2]
	img_corners = np.array([[0, 0], [w, 0], [0, h], [w, h]])
	H = solve_homography(img_corners, corners)
	#HH = cv2.findHomography(img_corners, corners)[0];
	H = H.reshape((3,3))
	#print(HH, H)
	#print(np.dot(H, np.array([w, h, 1])))
	
	for i in range(h):
		for j in range(w):
			proj = np.dot(H, np.array([j, i, 1]))
			new_j, new_i, c = proj / proj[-1]
			new_j, new_i = int(round(new_j)), int(round(new_i))
			#print(new_j, new_i)
			if new_j < 0 or new_i < 0 or new_j >= width or new_i >= height:
				continue
			canvas[new_i, new_j] = img[i, j]
	
	return canvas

def find_neighbors(x, y):

	x_ceil = math.ceil(x)
	x_flr = math.floor(x)
	y_ceil = math.ceil(y)
	y_flr = math.floor(y)
	return [[x_flr, y_flr], [x_ceil, y_flr], [x_flr, y_ceil], [x_ceil, y_ceil]]

def bilinear_interporlation(x, y, points):

	#points = sorted(points)
	(x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points
	#print(points)
	if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
		return q11
	if not x1 <= x <= x2 or not y1 <= y <= y2:
		return q12	
	return (q11 * (x2 - x) * (y2 - y) + 
			q21 * (x - x1) * (y2 - y) + 
			q12 * (x2 - x) * (y - y1) + 
			q22 * (x - x1) * (y - y1)
			) / ((x2 - x1) * (y2 - y1) + 0.0)

def backward_warping(img, canvas, img_corners, canvas_corners):
	
	h, w, ch = canvas.shape
	img_h, img_w, ch = img.shape
	H = solve_homography(canvas_corners, img_corners)
	H = H.reshape((3,3))
	#print(H)

	for i in range(h):
		for j in range(w):
			proj = np.dot(H, np.array([j, i, 1]))
			#if proj[-1] == 0:
			#	proj[-1] = 1e-10
			new_j, new_i, c = proj / proj[-1]
			#print(new_j, new_i)
			neighbors = find_neighbors(new_j, new_i)
			values = []
			points = []
			for x, y in neighbors:
				if x < 0: x = 0
				if y < 0: y = 0
				if x >= img_w: x = img_w-1
				if y >= img_h: y = img_h-1
				points.append([x, y, img[y, x]])
			canvas[i][j] = bilinear_interporlation(new_j, new_i, points)
	return canvas


def main():
	
	# Part 1
	ts = time.time()
	canvas = cv2.imread('./input/Akihabara.jpg')
	img1 = cv2.imread('./input/lu.jpeg')
	img2 = cv2.imread('./input/kuo.jpg')
	img3 = cv2.imread('./input/haung.jpg')
	img4 = cv2.imread('./input/tsai.jpg')
	img5 = cv2.imread('./input/han.jpg')
	img_set = [img1, img2, img3, img4, img5]

	canvas_corners1 = np.array([[779,312],[1014,176],[739,747],[978,639]])
	canvas_corners2 = np.array([[1194,496],[1537,458],[1168,961],[1523,932]])
	canvas_corners3 = np.array([[2693,250],[2886,390],[2754,1344],[2955,1403]])
	canvas_corners4 = np.array([[3563,475],[3882,803],[3614,921],[3921,1158]])
	canvas_corners5 = np.array([[2006,887],[2622,900],[2008,1349],[2640,1357]])
	corners_set = [canvas_corners1, canvas_corners2, canvas_corners3, canvas_corners4, canvas_corners5]

	# TODO: some magic
	for i in range(len(img_set)):
		canvas = transform(img_set[i], canvas, corners_set[i])
		
	cv2.imwrite('part1.png', canvas)
	te = time.time()
	print('Elapse time: {}...'.format(te-ts))
	
	# Part 2
	ts = time.time()
	img = cv2.imread('./input/QR_code.jpg')
	canvas = np.zeros((100, 100, 3), dtype = int)
	canvas[:, :, :] = 255
	QR_corners = np.array([[1980, 1240], [2043, 1213], [2025, 1397], [2085, 1366]])
	output_corners = np.array([[0, 0], [100, 0], [0, 100], [100, 100]])
	
	# TODO: some magic
	canvas = backward_warping(img, canvas, QR_corners, output_corners)
	
	cv2.imwrite('part2.png', canvas)
	te = time.time()
	print('Elapse time: {}...'.format(te-ts))
	
	
	# Part 3
	ts = time.time()
	img_front = cv2.imread('./input/crosswalk_front.jpg')
	front_corners = np.array([[137, 165], [587, 160], [63, 240], [622, 233]])
	img_top = np.zeros((500, 600, 3), dtype = int)
	top_corners = np.array([[75, 175], [525, 175], [75, 325], [525, 325]])
	# TODO: some magic
	img_top = backward_warping(img_front, img_top, front_corners, top_corners)
	
	cv2.imwrite('part3.png', img_top)
	te = time.time()
	print('Elapse time: {}...'.format(te-ts))
	

if __name__ == '__main__':
	main()
