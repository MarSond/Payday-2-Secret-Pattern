import numpy as np
import cv2
import logging
import cipher as cipher
import random


def fix_plate_rotation(plate, overview=None):
	plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
	lines = cv2.HoughLinesP(plate, 1, np.pi / 180, 100, minLineLength=30, maxLineGap=20)
	# Compute the angle of rotation
	angle = 0
	if lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line[0]
			if overview is not None:
				overview = cv2.line(overview, (x1, y1), (x2, y2), (0, 255, 0), 1)
			angle += np.arctan2(y2 - y1, x2 - x1)
		# Average angle
		angle /= len(lines)
	# Convert angle from radians to degrees
	angle = angle * 180.0 / np.pi
	# Rotate the image to straighten the symbols
	corrected = rotate_image(plate, -angle)
	return corrected, overview

def process_cropped_plate(plate):
	overview = plate.copy()
	plate = cv2.threshold(plate, 200, 255, cv2.THRESH_BINARY)[1];show_image(plate, "Plate threshold", logging.DEBUG)
	plate, overview = fix_plate_rotation(plate, overview);show_image(plate, "Plate Rotation fix", logging.DEBUG)
	plate = cv2.flip(plate, 1);show_image(plate, "Plate flipped", logging.DEBUG)
	return plate, overview

def rotate_image(image, angle):
	# Get the dimensions of the image
	h, w = image.shape[:2]
	# Compute the center of the image
	cx, cy = w // 2, h // 2
	# Get the rotation matrix
	M = cv2.getRotationMatrix2D((cx, cy),-angle, 1.0)
	# Perform the rotation
	rotated = cv2.warpAffine(image, M, (w, h))
	return rotated

def rotate_point(origin, point, angle):
	"""
	Rotate a point counterclockwise by a given angle around a given origin.

	The angle should be given in radians.
	"""
	ox, oy = origin
	px, py = point
	qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
	qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
	return qx, qy

def show_image(image, title="", logLevel=logging.DEBUG, scale=1.0):
	if title=="":
		title = random.randint(0, 1000).__str__()
	title = title + "  scale: " + scale.__str__()
	logger = logging.getLogger("main")
	
	if logger.getEffectiveLevel() <= logLevel:
		if scale != 1.0:
			image = cv2.resize(image, (0,0), fx=scale, fy=scale)
		cv2.imshow(title, image)
