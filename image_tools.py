import numpy as np
import cv2
import logging
import cipher as cipher
import random
from config import *

def draw_found_rect(target, loc, scale, angle, size=(CIPHER_WIDTH, CIPHER_HEIGHT)):
	# Define points of the rectangle around the match
	width = int(size[0] * scale)
	height = int(size[1] * scale)

	
	# top-left corner
	top_left = loc
	# top-right corner
	top_right = (top_left[0] + width, top_left[1])
	# bottom-right corner
	bottom_right = (top_left[0] + width, top_left[1] + height)
	# bottom-left corner
	bottom_left = (top_left[0], top_left[1] + height)
	
	# Rotate points
	center = (loc[0] + width / 2, loc[1] + height / 2)
	top_left_rot = rotate_point(center, top_left, angle)
	top_right_rot = rotate_point(center, top_right, angle)
	bottom_right_rot = rotate_point(center, bottom_right, angle)
	bottom_left_rot = rotate_point(center, bottom_left, angle)
	
	# Convert to integer coordinates
	rotated_rect_points = np.intp([top_left_rot, top_right_rot, bottom_right_rot, bottom_left_rot])
	
	# Draw the rotated rectangle on the original image
	cv2.polylines(target, [rotated_rect_points], True, (0, 255, 0), 1)
	return target

def fix_plate_rotation(plate, overview=None):
	lines = cv2.HoughLinesP(plate, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=15)
	# Compute the angle of rotation
	show_image(plate, "", logging.DEBUG)
	angle = 0
	if lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line[0]
			if overview is not None:
				overview = cv2.line(overview, (x1, y1), (x2, y2), (0, 255, 120), 1)
			angle += np.arctan2(y2 - y1, x2 - x1)
		# Average angle
		angle /= len(lines)
	# Convert angle from radians to degrees
	angle = angle * 180.0 / np.pi
	# Rotate the image to straighten the symbols
	corrected = rotate_image(plate, -angle)
	return corrected, overview

def fix_plate_perspective(plate, overview):
	# Dilation to connect nearby contours
	
	# Crop the region with all the symbols
	#cropped_symbols = plate[y:y+h, x:x+w]
	#show_image(cropped_symbols, "Cropped symbols", logging.DEBUG)
	moments = cv2.moments(plate)
	# Central moments
	mu20 = moments['mu20']
	mu02 = moments['mu02']
	mu11 = moments['mu11']
	# Calculating angle using central image moments
	angle = 0.5 * np.arctan2(2 * mu11, (mu20 - mu02))
	
	# Convert angle from radians to degrees
	angle = angle * 180.0 / np.pi
	print("Angle: ", angle)
	# Rotate the image to align the symbols
	corrected_symbols = rotate_image(plate, -angle)
	kernel = np.ones((6, 6), np.uint8)
	dilated = cv2.dilate(plate, kernel, iterations=3)
	show_image(dilated, "Dilated", logging.DEBUG)
	# Find contours
	contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# If no contours are detected, return the original image
	print("Found contours: ", len(contours))
	if not contours:
		return plate
	# Calculate the bounding rectangle that includes all contours
	bound_x, bound_y, bound_w, bound_h = cv2.boundingRect(np.vstack(contours))
	padding = 0
	x = max(0, bound_x - padding)
	y = max(0, bound_y - padding)
	w = min(plate.shape[1] - x, bound_w + 2 * padding)
	h = min(plate.shape[0] - y, bound_h + 2 * padding)
	# Crop the region with all the symbols
	cropped_symbols = corrected_symbols[y:y+h, x:x+w]
	show_image(cropped_symbols, "Cropped symbols", logging.DEBUG)
	if overview is not None:
		overview = draw_found_rect(overview, (x, y), 1, angle, (h,w))
	return	cropped_symbols


def process_cropped_plate(plate):
	overview = plate.copy()
	plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
	plate = cv2.threshold(plate, PLATE_THRESH_LOW, PLATE_THRESH_HIGH, cv2.THRESH_BINARY)[1];show_image(plate, "Plate threshold", logging.DEBUG)
	#plate, overview = fix_plate_rotation(plate, overview);show_image(plate, "Plate Rotation fix", logging.INFO)
	plate = fix_plate_perspective(plate, overview);show_image(plate, "Plate perspective", logging.WARNING)
	plate, overview = fix_plate_rotation(plate, overview);show_image(plate, "Plate Rotation fix 2", logging.INFO)
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
