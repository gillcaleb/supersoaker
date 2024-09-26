#!/usr/bin/env python
import cv2
import os
import sys, getopt
import signal
import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from edge_impulse_linux.image import ImageImpulseRunner

runner = None
res_width = 96                          # Resolution of camera (width)
res_height = 96                         # Resolution of camera (height)
cam_format = "RGB888"                   # Color format
# if you don't want to see a camera preview, set this to False
show_camera = False
if (sys.platform == 'linux' and not os.environ.get('DISPLAY')):
    show_camera = False

def now():
    return round(time.time() * 1000)
    
def sigint_handler(sig, frame):
    print('Interrupted')
    if (runner):
        runner.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

def help():
    print('python classify.py <path_to_model.eim> <Camera port ID, only required when more than 1 camera is present>')

def open_valve(pin):
    GPIO.output(pin, GPIO.HIGH)

def close_valve(pin):
    GPIO.output(pin, GPIO.LOW)

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["--help"])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
            sys.exit()

    if len(args) == 0:
        help()
        sys.exit(2)
    
    #Setup GPIO
    led_pin = 26
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(led_pin, GPIO.OUT)
    GPIO.output(led_pin, GPIO.LOW)
  
    model = args[0]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    print('MODEL: ' + modelfile)

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
            labels = model_info['model_parameters']['labels']
            
            fps = 0
            with Picamera2() as camera:
                # Configure camera settings
                config = camera.create_video_configuration(
                    main={"size": (res_width, res_height), "format": cam_format})
                camera.configure(config)
            
                # Start camera
                camera.start()
                
                # Continuously capture frames
                while True:
                                                        
                    # Get timestamp for calculating actual framerate
                    timestamp = cv2.getTickCount()
                    
                    # Get array that represents the image (in RGB format)
                    img = camera.capture_array()
             
                    # Extract features (e.g. grayscale image as a 2D array)
                    features, cropped = runner.get_features_from_image(img)
                    
                    # Perform inference
                    res = None
                    try:
                        res = runner.classify(features)
                    except Exception as e:
                        print("ERROR: Could not perform inference")
                        print("Exception:", e)
                        
                    if "classification" in res["result"].keys():
                        print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
                        for label in labels:
                            score = res['result']['classification'][label]
                            print('%s: %.2f\t' % (label, score), end='')
                        print('', flush=True)
    
                    elif "bounding_boxes" in res["result"].keys():
                        print('Found %d bounding boxes (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
                        for bb in res["result"]["bounding_boxes"]:
                            print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))
                            img = cv2.rectangle(img, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (255, 0, 0), 1)
                            if  bb['value'] > .6:
                              #open_valve(led_pin)
                              print("Activated!!!")
                    
                    frame_time = (cv2.getTickCount() - timestamp) / cv2.getTickFrequency()
                    print(frame_time)
                    fps = 1 / frame_time
                    print(fps)
        finally:
            if (runner):
                runner.stop()
                GPIO.cleanup()

if __name__ == "__main__":
   main(sys.argv[1:])
