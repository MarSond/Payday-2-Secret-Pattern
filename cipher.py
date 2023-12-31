import cv2
import numpy as np
import image_tools as tools
from config import *

scales = [0.30]#np.linspace(0.15, 0.6, 60)
rotations = [0]#np.arange(-15, 15, 3)

cipher_image = cv2.imread("cipher-text.jpg", cv2.IMREAD_GRAYSCALE)
plate_image = cv2.imread("plate.png")	
# start position of top left corner of search window in the image

mapping = {
	"A": (0,0),
	"B": (74,0),
	"C": (144,0),
	"D": (206,0),
	"E": (287,0),
	"F": (361,0),
	"G": (428,0),
	"H": (490,0),
	"I": (564,0),
	"J": (639,0),
	"K": (708,0),
	"L": (777,0),
	"M": (844,5),
	"N": (910,0),
	"O": (980,0),
	"P": (1060,0),
	"Q": (1126,0),
	"R": (1200,0),
	"S": (1273,0),
	"T": (1343,0),
	"U": (1424,0),
	"V": (1498,0),
	"W": (1570,0),
	"X": (1650,0),
	"Y": (1710,0),
	"Z": (1775,0),
	"0": (69,184),
	"1": (142,184),
	"2": (207,184),
	"3": (280,184),
	"4": (350,184),
	"5": (420,184),
	"6": (490,184),
	"7": (566,184),
	"8": (640,184),
	"9": (715,184),
	":": (836,184),
	"°": (907,184),
	"\"": (969,184),
	"\'": (1024,184),
}

test_mapping = {
	"A": (0,0),
	"I": (564,0),
}

def generate_cipher_overview():
	# Size of overview image (estimate a size that will fit all characters)
	overview_width = 10 * (CIPHER_WIDTH + 50) # 10 characters per row, plus some spacing for text
	overview_height = int(np.ceil(len(mapping) / 10.0)) * (CIPHER_HEIGHT + 50)
	overview_image = np.zeros((overview_height, overview_width), dtype=np.uint8)


	# Loop through each character
	for i, (key, im_coords) in enumerate(mapping.items()):
		search_window = get_cipher_image(im_coords)
		

		# Ensure that search_window has the expected shape before assigning
		if search_window.shape == (CIPHER_HEIGHT, CIPHER_WIDTH):
			# Position to paste the search_window in the overview_image
			row = i // 10
			col = i % 10
			pos_x = col * (CIPHER_WIDTH + 50)
			pos_y = row * (CIPHER_HEIGHT + 50)
			# Paste the search window in the overview_image
			overview_image[pos_y:pos_y+CIPHER_HEIGHT, pos_x:pos_x+CIPHER_WIDTH] = search_window
		else:
			print(f"Search window for key {key} has incorrect shape: {search_window.shape}")
		# Put the corresponding key (text) below the search window
		text_position = (pos_x, pos_y + CIPHER_HEIGHT + 20)
		cv2.putText(overview_image, key, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA)
	return overview_image


def get_cipher_image(im_coords):
	x, y = im_coords
	output = cipher_image[y:y+CIPHER_HEIGHT, x:x+CIPHER_WIDTH]
	_, output = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	return output

def _match(input, pattern, threshold):
	result = cv2.matchTemplate(input, pattern, cv2.TM_CCOEFF_NORMED)
	locations = np.where(result >= threshold)
	return zip(*locations)

def non_max_suppression(matches, overlapThresh):
	if len(matches) == 0:
		return []

	# If the bounding boxes are integers, convert them to floats -- this is important since we'll be doing a bunch of divisions
	if matches[0]['location'].dtype.kind == "i":
		matches = [{'location': np.array([float(p) for p in match['location']]), 'scale': match['scale']} for match in matches]

	# initialize the list of picked matches
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = np.array([match['location'][0] for match in matches])
	y1 = np.array([match['location'][1] for match in matches])
	x2 = np.array([match['location'][0] + 15 for match in matches])  # assuming width is 15
	y2 = np.array([match['location'][1] + 15 for match in matches])  # assuming height is 15

	# compute the area of the bounding boxes
	area = (x2 - x1 + 1) * (y2 - y1 + 1)

	# sort the bounding boxes by the bottom-right y-coordinate of the bounding box
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have overlap greater than the provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked
	return [matches[i] for i in pick]

def get_all_matches_above_threshold(input, key, threshold=0.8):
	search_image_raw = get_cipher_image(mapping[key])
	search_image = search_image_raw.copy()
	all_matches = []  # This will store all the matches above threshold

	for scale in scales:
		# Resize the template
		resized_template = cv2.resize(search_image, (int(search_image.shape[1]*scale), int(search_image.shape[0]*scale)))
		
		locations = _match(input, resized_template, threshold)
			
		# If current match value is above the threshold, add it to all_matches list
		for loc in locations:

			match_info = {'location': np.array(loc), 'scale': scale}
			all_matches.append(match_info)

	# Apply non-maxima suppression
	return non_max_suppression(all_matches, 0.3)

def get_best_match(input, key):
	search_image_raw = get_cipher_image(mapping[key])
	search_image = search_image_raw.copy()
	#cv2.imshow(f"raw search_image for {key}", search_image)
	betst_match = None # (scale, rotation, max_loc)
	best_match_val = -np.inf
	best_template = search_image

	for scale in scales:
		# Resize the template
		resized_template = cv2.resize(search_image, (int(search_image.shape[1]*scale), int(search_image.shape[0]*scale)))
		#cv2.imshow(f"resized_template for {key} scale {scale}", resized_template)
		# Loop over the rotations
		for angle in rotations:
			# Rotate the template
			rotation_matrix = cv2.getRotationMatrix2D((resized_template.shape[1]//2, resized_template.shape[0]//2), angle, 1)
			rotated_template = cv2.warpAffine(resized_template, rotation_matrix, (resized_template.shape[1], resized_template.shape[0]))
			#cv2.imshow(f"rotated_template for {key} and angle {angle}", rotated_template)
			curr_val, curr_loc = _match(input, rotated_template)
			# Record if this is the best match so far
			if curr_val > best_match_val:
				best_match_val = curr_val
				best_match = (scale, angle, curr_loc)
				best_template = rotated_template

	best_scale, best_rotation, best_loc = best_match
	print(f"Key: {key}, max_val: {best_match_val} at {best_loc} - rotation: {best_rotation} scale: {best_scale}")
	return best_loc, best_match_val, best_rotation, best_scale


def extract_plate(inputImage, crop=False):
	overview = inputImage.copy()
	match = cv2.matchTemplate(inputImage, plate_image, cv2.TM_CCOEFF_NORMED)
	_, _, _, max_loc = cv2.minMaxLoc(match)
	top_left = max_loc
	h, w = plate_image.shape[:2]
	bottom_right = (top_left[0] + w, top_left[1] + h)
	
	# Draw rectangle on the overview
	cv2.rectangle(overview, top_left, bottom_right, (0, 0, 255), 2)
	
	# Cropping
	cropped = inputImage[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
	
	cropped, crop_overview = tools.process_cropped_plate(cropped)
	
	# paste the cropped image overview back into the overview
	overview[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = crop_overview


	# Return the highlighted overview and the perspective-corrected crop
	return overview, cropped