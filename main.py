import cv2
import numpy as np
import cipher as cipher

# Test of systems
#cv2.imshow('Chipher Overview', cipher.generate_cipher_overview())
#cv2.imshow("test_search", cipher.get_cipher_image(cipher.mapping["X"]))


#save overview image+
#cv2.imwrite("overview_image.jpg", overview_image)

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

test1 = cv2.imread("test1.jpg")
test2 = cv2.imread("test2.jpg")

plate_overview, cropped_plate_raw = cipher.extract_plate(test2)
cv2.imshow("plate_overview", plate_overview)
cv2.imshow("cropped_plate raw", cropped_plate_raw)
# zoom in
cropped_plate = cropped_plate_raw[0:cropped_plate_raw.shape[0] - 70, 0:cropped_plate_raw.shape[1] - 70]
# flip
cropped_plate = cv2.flip(cropped_plate, 1)
#scale up
cropped_plate = cv2.resize(cropped_plate, (0,0), fx=2, fy=2)

#cropped_plate = cv2.Canny(cropped_plate, 100, 200)


cv2.imshow(f"cropped_plate canny", cropped_plate)

target = cropped_plate.copy()
target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
for key, values in cipher.test_mapping.items():
	best_loc, best_match_val, best_rotation, best_scale = cipher.get_match(plate=cropped_plate, key=key)
	
	cv2.putText(target, key, best_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
	windowTopLeft = best_loc
	windowBottomRight = (int(best_loc[0] + cipher.cipher_window_size[1] * best_scale), int(best_loc[1] + cipher.cipher_window_size[0] * best_scale))
	print(f"windowTopLeft: {windowTopLeft}, windowBottomRight: {windowBottomRight}")
	
	# Define points of the rectangle around the match
	width = int(cipher.cipher_window_size[1] * best_scale)
	height = int(cipher.cipher_window_size[0] * best_scale)
	angle = best_rotation
	
	# top-left corner
	top_left = best_loc
	# top-right corner
	top_right = (top_left[0] + width, top_left[1])
	# bottom-right corner
	bottom_right = (top_left[0] + width, top_left[1] + height)
	# bottom-left corner
	bottom_left = (top_left[0], top_left[1] + height)
	
	# Rotate points
	center = (best_loc[0] + width / 2, best_loc[1] + height / 2)
	top_left_rot = rotate_point(center, top_left, angle)
	top_right_rot = rotate_point(center, top_right, angle)
	bottom_right_rot = rotate_point(center, bottom_right, angle)
	bottom_left_rot = rotate_point(center, bottom_left, angle)
	
	# Convert to integer coordinates
	rotated_rect_points = np.intp([top_left_rot, top_right_rot, bottom_right_rot, bottom_left_rot])
	
	# Draw the rotated rectangle on the original image
	cv2.polylines(target, [rotated_rect_points], True, (0, 255, 0), 1)
	

# show scaled up image

cv2.imshow("target", cv2.resize(target, (0,0), fx=1.4, fy=1.4))
# Convert to grayscale



cv2.waitKey(0)
cv2.destroyAllWindows()