import python.darknet as dn
import cv2
import subprocess


# initializes the yolonet
def init_net():
	net = dn.load_net("yolonet/cfg/yolov2-tiny.cfg", "yolonet/yolov2-tiny.weights", 0)
	meta = dn.load_meta("yolonet/cfg/coco.data")
	return net, meta


# detects cars from frame and crops them out
# inputs:
# frame: opencv frame(image)
# frame_nr: number of frame (to distinguish between frames)
# threshold: used by yolonet - lower means more noise, higher might not detect everything
# net: see init_net
# meta: see init_net
# outputs:
# list of tuples that contains cropped car and its coordinates from initial image (cropped car, x1, y1, x2, y2)
def detect_cars_from_frame(frame, frame_nr, threshold, net, meta, park_spot_coordinates, doShow=False):
	frames_loc = 'videoframes'
	cropped_cars_loc = 'cropped_cars'
	detected_plate = None
	font = cv2.FONT_HERSHEY_SIMPLEX
	frame_copy = frame.copy()

	# contains tuples with cropped cars and its opencv coordinates
	out = []

	# save the frame so darknet could detect it
	name = frames_loc + '/frame' + str(frame_nr) + '.jpg'
	cv2.imwrite(name, frame)

	# detect objects from the frame
	r = dn.detect(net, meta, name, thresh=threshold)

	# only check for cars and trucks
	cars = [x for x in r if x[0] in ['car', 'truck']]

	# Find the cars and crop them out
	if len(cars) > 0:
		for idx, car in enumerate(cars):
			x, y, w, h = car[2]
			x1, y1 = int(x + (w / 2)), int(y + (h / 2))
			x2, y2 = int(x - (w / 2)), int(y - (h / 2))

			# clip negative values
			x1 = 1 if x1 < 1 else x1
			x2 = 1 if x2 < 1 else x2
			y1 = 1 if y1 < 1 else y1
			y2 = 1 if y2 < 1 else y2

			# check if center of the car is inside parking spot
			if cv2.pointPolygonTest(park_spot_coordinates, (x, y), False) == 1.0:
				cropped_frame = frame_copy[y2:y1, x2:x1]

				frame_name = cropped_cars_loc + '/frame' + str(frame_nr) + 'car' + str(idx) + '.jpg'

				cv2.imwrite(frame_name, cropped_frame)

				out.append((frame_name, x1, y1, x2, y2))

				try:
					output = subprocess.check_output(["alpr", "-c", "eu", frame_name])
					if "confidence" in output:
						line1 = output.split('\n')[1]
						detected_plate = line1.split('\t')[0].strip(" \t-")
						print "Detected number plate: " + detected_plate
						if detected_plate is not None:
							cv2.putText(frame, detected_plate, (x2, y2), font, 2, (255, 0, 0), 2, cv2.LINE_AA)
				except:
					pass

				# draw rectangle around a car
				cv2.rectangle(frame, (x2, y2), (x1, y1), (0, 255, 0), 3)
			else:
				# draw rectangle around a car
				cv2.rectangle(frame, (x2, y2), (x1, y1), (0, 0, 255), 3)


	# draw the parking spot
	cv2.polylines(frame, [park_spot_coordinates.reshape((-1, 1, 2))], True, (0, 255, 255), 2)

	if True:
		cv2.imshow('Frame', frame)
		cv2.waitKey(1) & 0xFF

	return out, frame
