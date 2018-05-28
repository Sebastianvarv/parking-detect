import argparse
import os
import python.darknet as dn
import cv2
from openalpr import Alpr
import shutil
import sys


def init():
    net = dn.load_net("cfg/tiny-yolo.cfg", "tiny-yolo.weights", 0)
    meta = dn.load_meta("cfg/coco.data")
    alpr = Alpr("eu", "/etc/openalpr/openalpr.conf", "/usr/share/openalpr/runtime_data")
    if not alpr.is_loaded():
        print("Error loading OpenALPR")
        sys.exit(1)

    alpr.set_top_n(20)
    alpr.set_default_region("md")
    return net, meta, alpr


def detect_video(video_loc, frames_to_skip, out_dir, threshold):
    # Start yolonet
    net, meta, alpr = init()

    nr = 0
    skip = frames_to_skip
    frames_loc = 'videoframes'
    cropped_cars_loc = 'cropped-cars'
    other_object_loc = 'notcar'
    files_with_cars = []

    if not os.path.exists(frames_loc):
        os.makedirs(frames_loc)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(cropped_cars_loc):
        os.makedirs(cropped_cars_loc)

    if not os.path.exists(other_object_loc):
        os.makedirs(other_object_loc)

    # Read input video using cv2
    cap = cv2.VideoCapture(video_loc)

    # output video dir
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():

        ret, frame = cap.read()

        # Stop if the video is over
        if not ret:
            break

        nr += 1

        if nr % skip != 0:
            continue

        # save the frame so darknet could detect it
        # could be skipped and feed image straight to network
        name = frames_loc + '/frame' + str(nr) + '.jpg'
        cv2.imwrite(name, frame)

        r = dn.detect(net, meta, name, thresh=threshold)

        # only save image if there is a car in frame
        cars = [x for x in r if x[0] in ['car', 'truck']]

        # crop out cars
        if len(cars) > 0:
            for idx, car in enumerate(cars):
                x, y, w, h = car[2]
                x1, y1 = int(x + (w / 2)), int(y + (h / 2))
                x2, y2 = int(x - (w / 2)), int(y - (h / 2))

                cv2.imwrite(cropped_cars_loc + '/frame' + str(nr) + 'car' + str(idx) + '.jpg', frame[y2:y1, x2:x1])
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                frame = cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1)

                results = alpr.recognize_file(cropped_cars_loc + '/frame' + str(nr) + 'car' + str(idx) + '.jpg')

                print results

        outname = out_dir + '/frame' + str(nr) + '.jpg'
        files_with_cars.append(outname)
        cv2.imwrite(outname, frame)

    out.write(frame)

    print "Found {} car(s) from frame {}".format(str(len(cars)), str(nr))

    for _, conf, coords in cars:
        print "\tConfidence {}".format(conf)


    print "Files with cars: {}".format(", ".join(files_with_cars))
    cap.release()
    out.release()
    alpr.unload()
    cv2.destroyAllWindows()

    shutil.rmtree(frames_loc, ignore_errors=False)

parser = argparse.ArgumentParser()

parser.add_argument("-v", "--video", help="path to input video", required=True)
parser.add_argument("-o", "--output", help="output path for detected cars",
                    default="detected_cars")
parser.add_argument("-s", "--skip", help="number of frames to skip",
                    default=15, type=int)
parser.add_argument("-t", "--threshold", help="threshold for darknet",
                    default=0.5, type=float)

args = parser.parse_args()
print args

detect_video(args.video, args.skip, args.output, args.threshold)
