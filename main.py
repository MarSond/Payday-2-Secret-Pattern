import cv2
import numpy as np
import cipher as cipher
import image_tools as tools
import logging

def setup():
	logger = logging.getLogger("main")
	logger.setLevel(logging.DEBUG)
	# Test of systems
	tools.show_image(cipher.generate_cipher_overview(), "Cipher suite overview", logging.INFO)
	tools.show_image(cipher.get_cipher_image(cipher.mapping["X"]), "Example search pattern", logging.DEBUG)

setup()

test1 = cv2.imread("test1.jpg")
test2 = cv2.imread("test2.jpg")

plate_overview, cropped_plate = cipher.extract_plate(test2)

tools.show_image(plate_overview, "Plate extraction overview", logging.WARNING, 0.7)
tools.show_image(cropped_plate, "Cropped Plate", logging.WARNING)


target = cropped_plate.copy()
target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
############ Match symbols

for key, values in cipher.test_mapping.items():
	best_loc, best_match_val, best_rotation, best_scale = cipher.get_best_match(plate=cropped_plate, key=key)
	target = tools.draw_found_rect(target, best_loc, best_scale, best_rotation)
	cv2.putText(target, key, best_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
	pass

###################
tools.show_image(target, "Matched symbols", logging.WARNING, 2)

cv2.waitKey(0)
cv2.destroyAllWindows()