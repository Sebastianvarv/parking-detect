import python.darknet as dn
import cv2


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
def detect_cars_from_frame(frame, frame_nr, threshold, net, meta, doShow=False):
	frames_loc = 'videoframes'
	cropped_cars_loc = 'cropped_cars'

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

			cropped_frame = frame[y2:y1, x2:x1]

			frame_name = cropped_cars_loc + '/frame' + str(frame_nr) + 'car' + str(idx) + '.jpg'

			cv2.imwrite(frame_name, cropped_frame)

			out.append((frame_name, x1, y1, x2, y2))

			if doShow:
				cv2.imshow('car', cropped_frame)
				cv2.waitKey(0) & 0xFF

	return out
