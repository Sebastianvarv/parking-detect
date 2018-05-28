import argparse
import os
import cv2
import shutil
from yolonet import find_cars


# read a video file and detect all cars from it
def detect_cars_from_video(video_loc, frames_to_skip, threshold):

	# Start yolonet
	net, meta = find_cars.init_net()

	# what frame are we currently erading
	nr = 0
	skip = frames_to_skip
	frames_loc = 'videoframes'

	if not os.path.exists(frames_loc):
		os.makedirs(frames_loc)

	# Read input video using cv2
	cap = cv2.VideoCapture(video_loc)

	while cap.isOpened():
		# check if open axxually, happens when the skip rate is too high
		if not cap.isOpened():
			break

		ret, frame = cap.read()

		# Stop if the video is over
		if not ret:
			break

		nr += 1

		if nr % skip != 0:
			continue

		detected_cars = find_cars.detect_cars_from_frame(frame, nr, threshold, net, meta)

		print "Found {} car(s) from frame {}".format(str(len(detected_cars)), str(nr))

	cap.release()
	cv2.destroyAllWindows()

	shutil.rmtree(frames_loc, ignore_errors=False)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("-v", "--video", help="path to input video", required=True)

	parser.add_argument("-s", "--skip", help="number of frames to skip", default=15, type=int)

	parser.add_argument("-t", "--threshold", help="threshold for darknet", default=0.5, type=float)

	args = parser.parse_args()
	print 'args'
	print args

	detect_cars_from_video(args.video, args.skip, args.threshold)

