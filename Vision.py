import argparse

import cv2
import numpy as np
from time import time
import tflite_runtime.interpreter as tflite
from cscore import CameraServer, VideoSource, UsbCamera, MjpegServer
from networktables import NetworkTablesInstance
import cv2
import collections
import json
import sys
import math


class ConfigParser:
    def __init__(self, config_path):
        self.team = -1

        # parse file
        try:
            with open(config_path, "rt", encoding="utf-8") as f:
                j = json.load(f)
        except OSError as err:
            print("could not open '{}': {}".format(config_path, err), file=sys.stderr)

        # top level must be an object
        if not isinstance(j, dict):
            self.parseError("must be JSON object", config_path)

        # team number
        try:
            self.team = j["team"]
        except KeyError:
            self.parseError("could not read team number", config_path)

        # cameras
        try:
            self.cameras = j["cameras"]
        except KeyError:
            self.parseError("could not read cameras", config_path)
    
    def parseError(self, str, config_file):
        """Report parse error."""
        print("config error in '" + config_file + "': " + str, file=sys.stderr)


class PBTXTParser:
    def __init__(self, path):
        self.path = path
        self.file = None


    def parse(self):
        with open(self.path, 'r') as f:
            self.file = ''.join([i.replace('item', '') for i in f.readlines()])
            blocks = []
            obj = ""
            for i in self.file:
                if i == '}':
                    obj += i
                    blocks.append(obj)
                    obj = ""
                else:
                    obj += i
            self.file = blocks
            label_map = []
            for obj in self.file:
                obj = [i for i in obj.split('\n') if i]
                name = obj[2].split()[1][1:-1]
                label_map.append(name)
            self.file = label_map

    def get_labels(self):
        return self.file


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

    def scale(self, sx, sy):
        """Returns scaled bounding box."""
        return BBox(xmin=sx * self.xmin,
                    ymin=sy * self.ymin,
                    xmax=sx * self.xmax,
                    ymax=sy * self.ymax)


class Tester:
    def __init__(self, config_parser):
        print("Initializing TFLite runtime interpreter")
        try:
            model_path = "model.tflite"
            self.interpreter = tflite.Interpreter(model_path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
            self.hardware_type = "Coral Edge TPU"
        except:
            print("Failed to create Interpreter with Coral, switching to unoptimized")
            model_path = "unoptimized.tflite"
            self.interpreter = tflite.Interpreter(model_path)
            self.hardware_type = "Unoptimized"

        self.interpreter.allocate_tensors()

        print("Getting labels")
        parser = PBTXTParser("map.pbtxt")
        parser.parse()
        # self.labels = parser.get_labels()
        self.labels = ['red', 'blue', 'invalid']

        print("Connecting to Network Tables")
        ntinst = NetworkTablesInstance.getDefault()
        ntinst.startClientTeam(config_parser.team)
        ntinst.startDSClient()
        self.entry = ntinst.getTable("ML").getEntry("detections")

        self.coral_entry = ntinst.getTable("ML").getEntry("coral")
        self.fps_entry = ntinst.getTable("ML").getEntry("fps")
        self.resolution_entry = ntinst.getTable("ML").getEntry("resolution")
        self.lowest_distance_blue = ntinst.getTable("ML").getEntry("distance_blue")
        self.lowest_distance_red = ntinst.getTable("ML").getEntry("distance_red")
        self.angle_blue = ntinst.getTable("ML").getEntry("angle_blue")
        self.angle_red = ntinst.getTable("ML").getEntry("angle_red")
        self.temp_entry = []
        self.distanceb = -1
        self.distancer = -1
        self.angleb = 0
        self.angler = 0
        

        print("Starting camera server")
        cs = CameraServer.getInstance()
        camera = cs.startAutomaticCapture()
        camera_config = config_parser.cameras[0]
        WIDTH, HEIGHT = camera_config["width"], camera_config["height"]
        camera.setResolution(WIDTH, HEIGHT)
        self.cvSink = cs.getVideo()
        self.img = np.zeros(shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.output = cs.putVideo("Axon", WIDTH, HEIGHT)
        self.frames = 0

        

        self.coral_entry.setString(self.hardware_type)
        self.resolution_entry.setString(str(WIDTH) + ", " + str(HEIGHT))

        self.size = [320, 240]
        self.center = (
            self.size[0] // 2,
            self.size[1] // 2,
        )
        
        self.dist_matrix = np.zeros((4, 1))

        

    def run(self):
        print("Starting mainloop")
        while True:
            start = time()
            # Acquire frame and resize to expected shape [1xHxWx3]
            ret, frame = self.cvSink.grabFrame(self.img)
            if not ret:
                print("Image failed")
                continue

            # input
            scale = self.set_input(frame)

            # run inference
            self.interpreter.invoke()

            # output
            boxes, class_ids, scores, x_scale, y_scale = self.get_output(scale)
            for i in range(len(boxes)):
                if scores[i] > .5:

                    class_id = class_ids[i]
                    if np.isnan(class_id):
                        continue

                    class_id = int(class_id)
                    if class_id not in range(len(self.labels)):
                        continue

                    frame = self.label_frame(frame, self.labels[class_id], boxes[i], scores[i], x_scale,
                                                 y_scale)
            self.output.putFrame(frame)
            self.entry.setString(json.dumps(self.temp_entry))
            self.lowest_distance_blue.setNumber(self.distanceb)
            self.lowest_distance_red.setNumber(self.distancer)
            self.angle_blue.setNumber(self.angleb)
            self.angle_red.setNumber(self.angler)
            self.temp_entry = []
            self.temp_entry = []
            self.distanceb = -1
            self.distancer = -1
            self.angleb = 0
            self.angler = 0
            if self.frames % 100 == 0:
                print("Completed", self.frames, "frames. FPS:", (1 / (time() - start)))
            if self.frames % 10 == 0:
                self.fps_entry.setNumber((1 / (time() - start)))
            self.frames += 1
            

    def label_frame(self, frame, object_name, box, score, x_scale, y_scale):
        ymin, xmin, ymax, xmax = box
        score = float(score)
        bbox = BBox(xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax).scale(x_scale, y_scale)

        ymin, xmin, ymax, xmax = int(bbox.ymin), int(bbox.xmin), int(bbox.ymax), int(bbox.xmax)

        zoomAmount = 0.5
        width = xmax - xmin
        height = ymax - ymin
        xmin = int(xmin + width * (zoomAmount * 0.5))
        ymin = int(ymin + height * (zoomAmount * 0.5))
        xmax = int(xmax - width * (zoomAmount * 0.5))
        ymax = int(ymax - height * (zoomAmount * 0.5))

        height, width, channels = frame.shape

        if not 0 <= ymin < ymax <= height or not 0 <= xmin < xmax <= width:
            print('invalid')
            print(xmin, xmax, ymin, ymax)
            return frame

        red = [70,  70, 176]
        redtolerance = [80, 80, 80]
        blue = [180, 130,  60]
        bluetolerance = [80, 80, 80]

        cropped = frame[ymin:ymax, xmin: xmax]
        averages = np.average(cropped, axis=(0, 1))
        if self.isWithinTolerance(red, averages, redtolerance):
            object_name = 'red'
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), averages, 2)
        elif self.isWithinTolerance(blue, averages, bluetolerance):
            object_name = 'blue'
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), averages, 2)
        else:
            object_name = 'invalid'
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), averages, 2)

        x, y, z, horizontal_angle, vertical_angle = self.simple_solve(
            np.array([
        [-4.75, 4.75, 0],
        [4.75, 4.75, 0],
        [4.75, -4.75, 0],
        [-4.75, -4.75, 0],
        ], dtype='float'),
        np.array([
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
        ], dtype='float'),
        )
        horizontal_angle = math.degrees(horizontal_angle)
        vertical_angle = math.degrees(vertical_angle)
        distance = math.sqrt(x**2 + y**2 + z**2)
        if(object_name != 'invalid'):
            if(object_name == 'red'):
                if(distance<self.distancer or self.distancer == -1):
                    self.distancer = distance
                    self.angler = horizontal_angle
            if(object_name == 'blue'):
                if(distance<self.distanceb or self.distanceb == -1):
                    self.distanceb = distance
                    self.angleb = horizontal_angle
            self.temp_entry.append({"label": object_name, "box": {"ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax}, "confidence": score, "horizontal_angle": horizontal_angle, "vertical_angle": vertical_angle, "position": {"x": x, "y":y, "z":z}, "distance": distance})

        #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)

        # Draw label
        # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, score * 100)  # Example: 'person: 72%'
        label_size, base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
        label_ymin = max(ymin, label_size[1] + 10)  # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10), (xmin + label_size[0], label_ymin + base - 10),
                      (255, 255, 255), cv2.FILLED)
        # Draw label text
        cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return frame

    def isWithinTolerance(self, arr1, arr2, tolerance):
        for i in range(len(arr1)):
            if abs(arr1[i] - arr2[i]) > tolerance[i]:
                return False
        return True

    def input_size(self):
        """Returns input image size as (width, height) tuple."""
        _, height, width, _ = self.interpreter.get_input_details()[0]['shape']
        return width, height

    def set_input(self, frame):
        """Copies a resized and properly zero-padded image to the input tensor.
        Args:
          frame: image
        Returns:
          Actual resize ratio, which should be passed to `get_output` function.
        """
        width, height = self.input_size()
        h, w, _ = frame.shape
        new_img = np.reshape(cv2.resize(frame, (300, 300)), (1, 300, 300, 3))
        self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], np.copy(new_img))
        return width / w, height / h

    def output_tensor(self, i):
        """Returns output tensor view."""
        tensor = self.interpreter.get_tensor(self.interpreter.get_output_details()[i]['index'])
        return np.squeeze(tensor)

    def get_output(self, scale):
        boxes = self.output_tensor(0)
        class_ids = self.output_tensor(1)
        scores = self.output_tensor(2)

        width, height = self.input_size()
        image_scale_x, image_scale_y = scale
        x_scale, y_scale = width / image_scale_x, height / image_scale_y
        return boxes, class_ids, scores, x_scale, y_scale

    def simple_solve(self, object_points, image_points):
        """Use solvePnP to calculate the x, y, z distances as well as the angles
        to the target.
        The format of object_points and image_points is very important, they
        must be np arrays with the 'float' datatype; you cannot just pass the
        contour. Additionally, the two sets of points must be in the same order.
        """

        _, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_matrix,
        )
        
        x_camera_offset = 0
        y_camera_offset = 0
        z_camera_offset = 0
        horizontal_camera_offset = 0
        vertical_camera_offset = 0

        x = tvec[0][0] + x_camera_offset
        y = -(tvec[1][0] + y_camera_offset)
        z = tvec[2][0] + z_camera_offset

        horizontal_angle = math.atan2(x, z) + horizontal_camera_offset
        vertical_angle = math.atan2(y, z) + vertical_camera_offset

        return x, y, z, horizontal_angle, vertical_angle


if __name__ == '__main__':
    config_file = "/boot/frc.json"
    config_parser = ConfigParser(config_file)
    tester = Tester(config_parser)
    tester.run()
