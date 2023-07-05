import cv2
import numpy as np
import cipher as cipher
import image_tools as tools
import logging

logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)


# Test of systems
tools.show_image(cipher.generate_cipher_overview(), "Cipher suite overview", logging.INFO)
tools.show_image(cipher.get_cipher_image(cipher.mapping["X"]), "Example search pattern", logging.DEBUG)



test1 = cv2.imread("test1.jpg")
test2 = cv2.imread("test2.jpg")

plate_overview, cropped_plate = cipher.extract_plate(test2)

tools.show_image(plate_overview, "Plate extraction overview", logging.WARNING, 0.8)
tools.show_image(cropped_plate, "Cropped Plate", logging.INFO)

cv2.imshow("cropped_plate raw", cropped_plate)
# zoom in
#cropped_plate = cropped_plate_raw[0:cropped_plate_raw.shape[0] - 70, 0:cropped_plate_raw.shape[1] - 70]
# flip


#cropped_plate = cv2.Canny(cropped_plate, 100, 200)


cv2.imshow(f"cropped_plate canny", cropped_plate)

target = cropped_plate.copy()
target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
for key, values in cipher.test_mapping.items():
	best_loc, best_match_val, best_rotation, best_scale = cipher.get_match(plate=cropped_plate, key=key)
	
	cv2.putText(target, key, best_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
	windowTopLeft = best_loc
	windowBottomRight = (int(best_loc[0] + cipher.cipher_window_size[1] * best_scale), int(best_loc[1] + cipher.cipher_window_size[0] * best_scale))

	
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
	top_left_rot = tools.rotate_point(center, top_left, angle)
	top_right_rot = tools.rotate_point(center, top_right, angle)
	bottom_right_rot = tools.rotate_point(center, bottom_right, angle)
	bottom_left_rot = tools.rotate_point(center, bottom_left, angle)
	
	# Convert to integer coordinates
	rotated_rect_points = np.intp([top_left_rot, top_right_rot, bottom_right_rot, bottom_left_rot])
	
	# Draw the rotated rectangle on the original image
	cv2.polylines(target, [rotated_rect_points], True, (0, 255, 0), 1)
	

# show scaled up image

cv2.imshow("target", cv2.resize(target, (0,0), fx=1.4, fy=1.4))
# Convert to grayscale



cv2.waitKey(0)
cv2.destroyAllWindows()