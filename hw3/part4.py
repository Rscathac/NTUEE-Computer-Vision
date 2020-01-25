import numpy as np

import cv2
import sys
import time
import math
from matplotlib.path import Path

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

def main(ref_image,template,video):
	
	ref_image = cv2.imread(ref_image)  ## load gray if you need.
	template = cv2.imread(template, 0)	## load gray if you need.
	ref_resized = cv2.resize(ref_image, (template.shape[:2][::-1]), interpolation = cv2.INTER_AREA)
	#gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	video = cv2.VideoCapture(video)
	film_h, film_w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
	film_fps = video.get(cv2.CAP_PROP_FPS)
	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	videowriter = cv2.VideoWriter("ar_video.mp4", fourcc, film_fps, (film_w, film_h))
	cnt = 0
	
	ts = time.time()
	# Flann base matcher
	MIN_MATCHES = 15
	
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)
			
	flann = cv2.FlannBasedMatcher(index_params, search_params)

	while(video.isOpened()):
		ret, frame = video.read()

		print('Processing frame {}'.format(cnt))
		cnt += 1
		if ret:  ## check whethere the frame is legal, i.e., there still exists a frame
			## TODO: homography transform, feature detection, ransanc, etc.
			frame = cv2.resize(frame, (1080, 720), interpolation = cv2.INTER_AREA)
			height, width = frame.shape[:2]
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# Initialize SIFT detector
			sift = cv2.xfeatures2d.SIFT_create()

			# Find keypoints and descriptors with SIFT
			kp1, des1 = sift.detectAndCompute(template, None)
			kp2, des2 = sift.detectAndCompute(gray, None)
			
			matches = flann.knnMatch(des1, des2, k=2)

			# store all the good matches as per Lowe's ratio test.
			good = []
			for m, n in matches:
				if m.distance < 0.7*n.distance:
					good.append(m)

			if len(good)>MIN_MATCHES:
				src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
				dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

				M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
				H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
				matchesMask = mask.ravel().tolist()

				h, w = template.shape
				pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)				
				dst = cv2.perspectiveTransform(pts, M)
				dst = dst.reshape(4, 2).tolist()

				x, y = np.meshgrid(np.arange(1080), np.arange(720))
				x, y = x.flatten(), y.flatten()
				points = np.vstack((x,y)).T

				p = Path(dst)
				grid = p.contains_points(points)
				inside_pts = points[grid]
				
				# Backward warping
				for j, i in inside_pts:
					proj = np.dot(H, np.array([j, i, 1]))
					new_j, new_i, c = proj / proj[-1]
					neighbors = find_neighbors(new_j, new_i)
					values = []
					points = []
					for x, y in neighbors:
						if x < 0: x = 0
						if y < 0: y = 0
						if x >= w: x = w-1
						if y >= h: y = h-1
						points.append([x, y, ref_resized[y, x]])
					frame[i, j] = bilinear_interporlation(new_j, new_i, points)
				
				frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)
				#cv2.imwrite('frame.png', frame)
				#return
				#cv2.imshow('aaa', frame)
				#if cv2.waitKey(0):
				#	return
				#gray = cv2.polylines(gray,[np.int32(dst)],True,255,3, cv2.LINE_AA)

			else:
				print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCHES))
				matchesMask = None
			
			
			
			#draw_params = dict(matchColor = (0,255,0), # draw matches in green color
			#singlePointColor = None,
			#matchesMask = matchesMask, flags = 2)
			#img3 = cv2.drawMatches(template, kp1, gray, kp2, good, None, **draw_params)
			#cv2.imwrite('matching.png', img3)
			#cv2.imshow('haaa', img3)
			#if cv2.waitKey(0):
			#	return
			
			videowriter.write(frame)

		else:
			break
			
	video.release()
	videowriter.release()
	cv2.destroyAllWindows()
	te = time.time()
	print('Elapse time: {}...'.format(te-ts))


if __name__ == '__main__':
	## you should not change this part
	ref_path = './input/sychien.jpg'
	template_path = './input/marker.png'
	video_path = sys.argv[1]  ## path to ar_marker.mp4
	main(ref_path,template_path,video_path)
