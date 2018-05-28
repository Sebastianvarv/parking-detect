import argparse
import os
import cv2
import shutil
from yolonet import find_cars
import numpy as np
from flask import Flask, request, send_file
from flask_cors import CORS

app = Flask("asi")
CORS(app)


@app.route("/parking", methods=['POST'])
def parking():
    content = request.get_json()
	coordinates = np.array(content["coordinates"])
    video_name = content["videoName"]
    frames_to_skip = int(content["framesToSkip"])
    threshold = float(content["threshold"])
    #detect_cars_from_video(args.video, args.skip, args.threshold, coordinates)
    detect_cars_from_video(video_name, frames_to_skip, threshold, coordinates)
    return "OK"


# read a video file and detect all cars from it
def detect_cars_from_video(video_loc, frames_to_skip, threshold, park_spot_coordinates):

	# Start yolonet
	net, meta = find_cars.init_net()

	# what frame are we currently erading
	nr = 0
	skip = frames_to_skip
	frames_loc = 'videoframes'
	cropped_cars_loc = 'cropped_cars'

	if not os.path.exists(frames_loc):
		os.makedirs(frames_loc)

	if not os.path.exists(cropped_cars_loc):
		os.makedirs(cropped_cars_loc)

	# Read input video using cv2
	cap = cv2.VideoCapture(video_loc)

	# output video dir
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('output.avi', fourcc, cap.get(cv2.CAP_PROP_FPS),
						  (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

	while cap.isOpened():
		# check if open axxually, happens when the skip rate is too high
		if not cap.isOpened():
			break

		ret, frame = cap.read()

		if nr == 0:
			cv2.imwrite("frame0.jpg", frame)

		# Stop if the video is over
		if not ret:
			break

		nr += 1

		if nr % skip != 0:
			continue

		detected_cars, frame_with_cars = find_cars.detect_cars_from_frame(frame, nr, threshold, net, meta, park_spot_coordinates)

		print "Found {} car(s) from frame {}".format(str(len(detected_cars)), str(nr))
		out.write(frame_with_cars)

	cap.release()
	out.release()
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
	app.run(port=5000)
