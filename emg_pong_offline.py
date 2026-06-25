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
import time

# ─────────────────────────────────────────────
# CONFIGURATION — edit these before running
# ─────────────────────────────────────────────

# Path to the CSV file recorded during a previous acquisition session.
# Example: "data/training_acquisition_24-01-15_10-30-00.csv"
TRAINING_CSV = "data/YOUR_TRAINING_FILE.csv"

# Must match the label sequence used when that CSV was recorded.
training_label = np.asarray([0, 1, 0, 0, -1, 0, 0, 1, 0, 0, -1, 0, 0, 1, 0,
                              0, 1, 0, 0, -1, 0, 0, -1, 0, 0, 1, 0, 0, -1, 0])
ntile       = 20
window_size = 50

training_label = training_label[:, np.newaxis]
training_label = np.tile(training_label, (1, ntile)).flatten()

# Game feel — tune these without touching anything else
TARGET_FPS   = 30    # game render rate (does NOT affect paddle sensitivity)
BALL_SPEED   = 5     # pixels the ball moves per frame (was 8, lower = slower)
PADDLE_STEP  = 20    # pixels the paddle moves per new EMG classification
PADDLE_LIMIT = 250   # max y the paddle center can reach
WIN_SCORE    = 15    # first to this score wins

# ─────────────────────────────────────────────
# 1.  HARDWARE SETUP
# ─────────────────────────────────────────────
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c, data_rate=860, mode=256)
chan1 = AnalogIn(ads, ADS.P0)
chan2 = AnalogIn(ads, ADS.P1)

GPIO.setup(12, GPIO.OUT)
GPIO.output(12, 0)
buzzer = GPIO.PWM(12, 10)

# ─────────────────────────────────────────────
# 2.  OFFLINE TRAINING FROM EXISTING CSV
# ─────────────────────────────────────────────
print(f"\nLoading training data from: {TRAINING_CSV}")
classifier = train_classifier(TRAINING_CSV, window_size,
                              training_label, max_iter=1000, eta=1, mu=0)
print("Classifier ready — launching game.\n")

# ─────────────────────────────────────────────
# 3.  SHARED STATE
#     current_label  : latest EMG prediction
#     label_updated  : flag set by the EMG thread each time a new
#                      classification arrives; cleared by the game loop
#                      after it has acted on it.
#     This decouples paddle movement from the render frame rate:
#     the paddle moves exactly once per EMG window, not once per frame.
# ─────────────────────────────────────────────
current_label  = 0
label_updated  = False     # True only for one game frame after each new EMG result
label_lock     = threading.Lock()
game_running   = True

# ─────────────────────────────────────────────
# 4.  EMG CLASSIFICATION THREAD
# ─────────────────────────────────────────────
def emg_loop():
    global current_label, label_updated, game_running

    testing_file_name = create_new_sampling_file("data/", "testing_acquisition")
    file = open(testing_file_name, 'a', newline="")
    test_writer = csv.writer(file)

    while game_running:
        data1, data2 = acquire_window(chan1, chan2, window_size, test_writer)
        data1 = np.asarray(data1).reshape((1, -1))
        data2 = np.asarray(data2).reshape((1, -1))

        label_window = classifier.test(data1, data2)

        with label_lock:
            current_label = label_window
            label_updated = True      # signal the game loop that fresh data arrived

        print(f"EMG: {label_window:+d}")

    file.close()

emg_thread = threading.Thread(target=emg_loop, daemon=True)
emg_thread.start()

# ─────────────────────────────────────────────
# 5.  PONG GAME
# ─────────────────────────────────────────────
wn = turtle.Screen()
wn.title("EMG Pong  |  Flex = UP   Extend = DOWN")
wn.bgcolor("black")
wn.setup(width=800, height=600)
wn.tracer(0)

# Left paddle — EMG controlled
paddle_a = turtle.Turtle()
paddle_a.speed(0)
paddle_a.shape("square")
paddle_a.color("white")
paddle_a.shapesize(stretch_wid=5, stretch_len=1)
paddle_a.penup()
paddle_a.goto(-350, 0)

# Right paddle — bot
paddle_b = turtle.Turtle()
paddle_b.speed(0)
paddle_b.shape("square")
paddle_b.color("white")
paddle_b.shapesize(stretch_wid=5, stretch_len=1)
paddle_b.penup()
paddle_b.goto(350, 0)

# Ball
ball = turtle.Turtle()
ball.speed(0)
ball.shape("circle")
ball.color("white")
ball.penup()
ball.goto(0, 0)
ball.dx = BALL_SPEED
ball.dy = BALL_SPEED

# Score display
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

def bot():
    if (ball.xcor() > 0
            and ball.dx > 0
            and np.abs(ball.ycor() - paddle_b.ycor()) > 50):
        paddle_b.goto(350,
                      np.round((0.8 + 0.2 * np.random.rand()) * ball.ycor()
                               + np.random.randint(-10, 10)))

# ─────────────────────────────────────────────
# 6.  MAIN GAME LOOP
# ─────────────────────────────────────────────
while True:
    time.sleep(1 / TARGET_FPS)
    wn.update()

    # --- Paddle: move only when a fresh EMG classification has arrived ---
    with label_lock:
        label   = current_label
        updated = label_updated
        label_updated = False          # consume the flag

    if updated:                        # act exactly once per EMG window
        y = paddle_a.ycor()
        if label == 1 and y < PADDLE_LIMIT:
            paddle_a.sety(y + PADDLE_STEP)
        elif label == -1 and y > -PADDLE_LIMIT:
            paddle_a.sety(y - PADDLE_STEP)
        # label == 0 → paddle stays still

    # --- Ball movement ---
    ball.setx(ball.xcor() + ball.dx)
    ball.sety(ball.ycor() + ball.dy)

    bot()

    # --- Walls ---
    if ball.ycor() > 290:
        ball.sety(290)
        ball.dy *= -1
    if ball.ycor() < -290:
        ball.sety(-290)
        ball.dy *= -1

    # --- Ball out right (human scores) ---
    if ball.xcor() > 390:
        ball.goto(0, 0)
        ball.dx = -abs(ball.dx)
        score_a += 1
        pen.clear()
        pen.write(f"Human : {score_a}   Bot : {score_b}",
                  align="center", font=("courier", 24, "normal"))

    # --- Ball out left (bot scores) ---
    if ball.xcor() < -390:
        ball.goto(0, 0)
        ball.dx = abs(ball.dx)
        score_b += 1
        pen.clear()
        pen.write(f"Human : {score_a}   Bot : {score_b}",
                  align="center", font=("courier", 24, "normal"))

    # --- Collisions ---
    # Check: has the ball reached or crossed the paddle line while moving toward it?
    # Using a crossed-line test instead of a narrow band prevents the ball from
    # tunnelling through the paddle when it moves faster than the band is wide.
    # Paddle half-height = stretch_wid(5) * 20px = 100px total → ±50 from centre.
    # Ball radius ≈ 10px, so we add that to the overlap tolerance.
    PADDLE_HALF = 60   # 50px paddle half-height + 10px ball radius

    if ball.dx > 0 and ball.xcor() >= 340:   # moving right, reached bot paddle
        if abs(ball.ycor() - paddle_b.ycor()) < PADDLE_HALF:
            ball.setx(339)
            ball.dx *= -1
        # if missed, let the out-of-bounds check handle scoring

    if ball.dx < 0 and ball.xcor() <= -340:  # moving left, reached player paddle
        if abs(ball.ycor() - paddle_a.ycor()) < PADDLE_HALF:
            ball.setx(-339)
            ball.dx *= -1
        # if missed, let the out-of-bounds check handle scoring

    # --- End condition ---
    if score_a >= WIN_SCORE or score_b >= WIN_SCORE:
        game_running = False
        break

buzzer.stop()
turtle.done()
