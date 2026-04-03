# utils/iot_content.py

IOT_HARDWARE = [
    {"component": "Raspberry Pi 4 (4GB)", "role": "Main compute unit — runs TFLite model", "cost": "~₹5,500", "why": "Low power Linux SBC with GPIO for servo control"},
    {"component": "Pi Camera Module v2",  "role": "Captures waste image on motion trigger",  "cost": "~₹1,800", "why": "8MP, direct CSI connection, no USB lag"},
    {"component": "PIR Sensor HC-SR501",  "role": "Detects object placed in bin opening",     "cost": "~₹80",    "why": "Passive IR, adjustable sensitivity, 3.3V logic"},
    {"component": "Servo Motors SG90 ×6", "role": "Opens/closes compartment per category",   "cost": "~₹120 ea","why": "PWM control via GPIO, precise 180° rotation"},
    {"component": "16×2 LCD (I2C)",       "role": "Shows predicted class + confidence",       "cost": "~₹150",   "why": "Simple user feedback, 4-wire I2C connection"},
    {"component": "WiFi / 4G",            "role": "Sends data to cloud dashboard",            "cost": "Built-in", "why": "Real-time fill-level alerts to operators"},
]

IOT_WORKFLOW = [
    ("Waste Placed",    "Person drops item into the bin opening"),
    ("Motion Detected", "PIR sensor triggers camera capture"),
    ("Image Captured",  "Camera takes 224×224 photo of the item"),
    ("AI Classifies",   "TFLite model predicts category in ~200ms"),
    ("Servo Opens",     "Correct compartment flap rotates open"),
    ("Item Sorted",     "Waste falls into right section, flap closes"),
    ("Data Logged",     "MQTT publishes: class, confidence, timestamp"),
    ("Dashboard Live",  "Cloud dashboard updates fill levels in real time"),
]

IOT_SOFTWARE = [
    {"layer": "Inference",      "tech": "TensorFlow Lite",        "detail": "Convert .keras → .tflite (quantized, ~5MB, ~200ms/image on Pi 4)"},
    {"layer": "Image Capture",  "tech": "picamera2 (Python)",     "detail": "Triggered by PIR, saves JPEG to /tmp, passes to model"},
    {"layer": "HW Control",     "tech": "RPi.GPIO",               "detail": "PWM signals to SG90 servos on GPIO pins 17,18,27,22,23,24"},
    {"layer": "Messaging",      "tech": "MQTT (Mosquitto)",       "detail": "Publishes events to broker; subscribable by any dashboard"},
    {"layer": "Cloud",          "tech": "Node-RED + Grafana",     "detail": "Receives MQTT, stores in InfluxDB, visualises fill levels"},
    {"layer": "Alerts",         "tech": "Telegram Bot API",       "detail": "Pushes notification when bin reaches 80% capacity"},
]

TFLITE_CODE = '''\
import tensorflow as tf

model = tf.keras.models.load_model('waste_model_FINAL.keras')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('waste_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"TFLite size: {len(tflite_model)/1024/1024:.1f} MB")
# Expected: ~4–6 MB  |  Inference: ~180–250ms on Pi 4
'''

PI_CODE = '''\
import tflite_runtime.interpreter as tflite
import numpy as np, cv2, time
from picamera2 import Picamera2
import RPi.GPIO as GPIO

CLASS_NAMES  = ['cardboard','glass','metal','paper','plastic','trash']
SERVO_PINS   = {'cardboard':17,'glass':18,'metal':27,
                'paper':22,'plastic':23,'trash':24}
PIR_PIN      = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN)

interpreter = tflite.Interpreter('waste_model.tflite')
interpreter.allocate_tensors()
inp  = interpreter.get_input_details()
outp = interpreter.get_output_details()

def classify(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).astype('float32')
    img = (img / 127.5) - 1.0          # MobileNetV2 scale
    img = np.expand_dims(img, 0)
    interpreter.set_tensor(inp[0]['index'], img)
    interpreter.invoke()
    out  = interpreter.get_tensor(outp[0]['index'])[0]
    idx  = int(np.argmax(out))
    return CLASS_NAMES[idx], float(out[idx]) * 100

def open_compartment(cls):
    pin = SERVO_PINS[cls]
    GPIO.setup(pin, GPIO.OUT)
    pwm = GPIO.PWM(pin, 50)
    pwm.start(0)
    pwm.ChangeDutyCycle(7.5)   # open
    time.sleep(1.5)
    pwm.ChangeDutyCycle(2.5)   # close
    time.sleep(0.5)
    pwm.stop()

cam = Picamera2()
cam.start()
print("Smart bin ready...")

while True:
    if GPIO.input(PIR_PIN):
        path = '/tmp/capture.jpg'
        cam.capture_file(path)
        cls, conf = classify(path)
        print(f"{cls}  {conf:.1f}%")
        open_compartment(cls)
    time.sleep(0.1)
'''
