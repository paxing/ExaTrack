import board
import busio
import RPi.GPIO as GPIO
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from functions_EMG import *

import turtle
import numpy as np
import threading
import csv

# Paul Xing Winter 2023 — modified for EMG Pong control

# ─────────────────────────────────────────────
# 1.  HARDWARE SETUP  (unchanged from template)
# ─────────────────────────────────────────────
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c, data_rate=860, mode=256)
chan1 = AnalogIn(ads, ADS.P0)
chan2 = AnalogIn(ads, ADS.P1)

GPIO.setup(12, GPIO.OUT)
GPIO.output(12, 0)
buzzer = GPIO.PWM(12, 10)

# ─────────────────────────────────────────────
# 2.  TRAINING  (unchanged from template)
# ─────────────────────────────────────────────
training_label = np.asarray([0, 1, 0, 0, -1, 0, 0, 1, 0, 0, -1, 0, 0, 1, 0,
                              0, 1, 0, 0, -1, 0, 0, -1, 0, 0, 1, 0, 0, -1, 0])
ntile       = 20
window_size = 50
number_window_training = len(training_label) * ntile

training_label = training_label[:, np.newaxis]
training_label = np.tile(training_label, (1, ntile)).flatten()

training_file_name = create_new_sampling_file("data/", "training_acquisition")
acquire_training_dataset(chan1, chan2, window_size,
                         number_window_training,
                         training_file_name, training_label)

visualize_sampling(training_file_name, window_size, training_label)

classifier = train_classifier(training_file_name, window_size,
                              training_label, max_iter=1000, eta=1, mu=0)

# ─────────────────────────────────────────────
# 3.  SHARED STATE FOR CROSS-THREAD COMMUNICATION
# ─────────────────────────────────────────────
current_label = 0          # 1 = flexion (up), -1 = extension (down), 0 = rest
label_lock    = threading.Lock()
game_running  = True       # set to False when the game ends to stop the EMG thread

# ─────────────────────────────────────────────
# 4.  EMG CLASSIFICATION THREAD
#     Runs in the background, continuously updates current_label.
#     The pong main loop reads current_label every frame.
# ─────────────────────────────────────────────
def emg_loop():
    global current_label, game_running

    testing_file_name = create_new_sampling_file("data/", "testing_acquisition")
    file = open(testing_file_name, 'a', newline="")
    test_writer = csv.writer(file)

    while game_running:
        # acquire one window of EMG
        data1, data2 = acquire_window(chan1, chan2, window_size, test_writer)
        data1 = np.asarray(data1).reshape((1, -1))
        data2 = np.asarray(data2).reshape((1, -1))

        # classify
        label_window = classifier.test(data1, data2)

        # write result to shared variable (thread-safe)
        with label_lock:
            current_label = label_window

        print(f"EMG label: {label_window}")

    file.close()

emg_thread = threading.Thread(target=emg_loop, daemon=True)
emg_thread.start()

# ─────────────────────────────────────────────
# 5.  PONG GAME  (main thread — required by turtle)
# ─────────────────────────────────────────────
PADDLE_SPEED = 40      # pixels per frame the paddle moves when a gesture is active
PADDLE_LIMIT = 250     # y-coordinate boundary so paddle stays on screen

wn = turtle.Screen()
wn.title("EMG Pong  |  Flex = UP   Extend = DOWN")
wn.bgcolor("black")
wn.setup(width=800, height=600)
wn.tracer(0)           # disable auto-refresh; we call wn.update() manually each frame

# --- Left paddle: controlled by EMG ---
paddle_a = turtle.Turtle()
paddle_a.speed(0)
paddle_a.shape("square")
paddle_a.color("white")
paddle_a.shapesize(stretch_wid=5, stretch_len=1)
paddle_a.penup()
paddle_a.goto(-350, 0)

# --- Right paddle: bot ---
paddle_b = turtle.Turtle()
paddle_b.speed(0)
paddle_b.shape("square")
paddle_b.color("white")
paddle_b.shapesize(stretch_wid=5, stretch_len=1)
paddle_b.penup()
paddle_b.goto(350, 0)

# --- Ball ---
ball = turtle.Turtle()
ball.speed(0)
ball.shape("circle")
ball.color("white")
ball.penup()
ball.goto(0, 0)
ball.dx = 8
ball.dy = 8

# --- Score ---
score_a = 0
score_b = 0

pen = turtle.Turtle()
pen.speed(0)
pen.color("white")
pen.penup()
pen.hideturtle()
pen.goto(0, 260)
pen.write("Human : 0   Bot : 0", align="center",
          font=("courier", 24, "normal"))

# --- Bot AI (unchanged) ---
def bot():
    if (ball.xcor() > 0
            and ball.dx == 8
            and np.abs(ball.ycor() - paddle_b.ycor()) > 50):
        paddle_b.goto(350,
                      np.round((0.8 + 0.2 * np.random.rand()) * ball.ycor()
                               + np.random.randint(-10, 10)))

# ─────────────────────────────────────────────
# 6.  MAIN GAME LOOP
# ─────────────────────────────────────────────
while True:
    wn.update()

    # --- Move EMG-controlled paddle ---
    with label_lock:
        label = current_label

    y = paddle_a.ycor()
    if label == 1 and y < PADDLE_LIMIT:        # flexion  → move up
        paddle_a.sety(y + PADDLE_SPEED)
    elif label == -1 and y > -PADDLE_LIMIT:    # extension → move down
        paddle_a.sety(y - PADDLE_SPEED)
    # label == 0  → paddle stays still

    # --- Move ball ---
    ball.setx(ball.xcor() + ball.dx)
    ball.sety(ball.ycor() + ball.dy)

    # --- Bot moves ---
    bot()

    # --- Top / bottom walls ---
    if ball.ycor() > 290:
        ball.sety(290)
        ball.dy *= -1
    if ball.ycor() < -290:
        ball.sety(-290)
        ball.dy *= -1

    # --- Ball out right (human scores) ---
    if ball.xcor() > 390:
        ball.goto(0, 0)
        ball.dx *= -1
        score_a += 1
        pen.clear()
        pen.write(f"Human : {score_a}   Bot : {score_b}",
                  align="center", font=("courier", 24, "normal"))

    # --- Ball out left (bot scores) ---
    if ball.xcor() < -390:
        ball.goto(0, 0)
        ball.dy *= -1
        score_b += 1
        pen.clear()
        pen.write(f"Human : {score_a}   Bot : {score_b}",
                  align="center", font=("courier", 24, "normal"))

    # --- Ball ↔ bot paddle collision ---
    if (340 < ball.xcor() < 350
            and paddle_b.ycor() - 40 < ball.ycor() + 40 < paddle_b.ycor() + 50):
        ball.setx(340)
        ball.dx *= -1

    # --- Ball ↔ EMG paddle collision ---
    if (-350 < ball.xcor() < -340
            and paddle_a.ycor() - 40 < ball.ycor() + 40 < paddle_a.ycor() + 50):
        ball.setx(-340)
        ball.dx *= -1

    # --- End condition ---
    if score_a >= 15 or score_b >= 15:
        game_running = False   # signals the EMG thread to stop
        break

buzzer.stop()
turtle.done()
