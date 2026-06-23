"""
Author = Aku Sarma
github = https://github.com
Modified to simulate EMG signal inputs with original ball speed

Recording fixes applied:
- Captures only the turtle window region (not the full screen) -> faster, smaller files
- Saves the GIF no matter how the loop ends (normal finish OR Ctrl+C), via try/finally
- Tracks real elapsed time per frame so GIF playback speed matches what actually happened
- Fixed a gameplay bug where a miss on the left side flipped dy instead of dx
"""

import turtle
import numpy as np
import time
import pyautogui
import imageio


wn = turtle.Screen()
wn.title("Pong - EMG Signal Simulator")
wn.bgcolor("black")
wn.setup(width=800, height=600)
wn.tracer(0)  # Keep screen tracking manual for performance

# --- Figure out exactly where the turtle window lives on screen ---
# This lets pyautogui grab only the game window instead of your whole desktop.
canvas = wn.getcanvas()
root = canvas.winfo_toplevel()
root.update_idletasks()  # make sure the window has actually been drawn/positioned
CAPTURE_REGION = (
    root.winfo_rootx(),
    root.winfo_rooty(),
    root.winfo_width(),
    root.winfo_height(),
)

# Global EMG State Tracker
#  1 = Flexion (Up), -1 = Extension (Down), 0 = Rest (No movement)
emg_signal = 0

# paddle A (EMG Controlled)
paddle_a = turtle.Turtle()
paddle_a.speed(0)
paddle_a.shape("square")
paddle_a.color("cyan")
paddle_a.shapesize(stretch_wid=5, stretch_len=1)
paddle_a.penup()
paddle_a.goto(-350, 0)

# bot paddle
paddle_b = turtle.Turtle()
paddle_b.speed(0)
paddle_b.shape("square")
paddle_b.color("white")
paddle_b.shapesize(stretch_wid=5, stretch_len=1)
paddle_b.penup()
paddle_b.goto(350, 0)

# ball
ball = turtle.Turtle()
ball.speed(0)
ball.shape("circle")
ball.color("white")
ball.penup()
ball.goto(0, 0)
ball.dx = 8
ball.dy = 8

# score
score_a = 0
score_b = 0

# UI Scoring and EMG telemetry data display
pen = turtle.Turtle()
pen.speed(0)
pen.color("white")
pen.penup()
pen.hideturtle()
pen.goto(0, 230)


def update_display():
    """Renders the standard scores and live simulated EMG data streams."""
    pen.clear()

    if emg_signal == 1:
        status_text = "FLEXION (1) -> Moving UP"
    elif emg_signal == -1:
        status_text = "EXTENSION (-1) -> Moving DOWN"
    else:
        status_text = "REST (0) -> Stationary"

    pen.write(
        f"Human (EMG) : {score_a}   Bot : {score_b}\n"
        f"Live EMG State: {status_text}",
        align="center", font=("courier", 16, "bold")
    )


def simulate_emg():
    """Simulates real-world muscle signals by changing state every 400ms."""
    global emg_signal
    emg_signal = np.random.choice([0, 1, -1], p=[0.4, 0.4, 0.2])
    update_display()
    wn.ontimer(simulate_emg, 400)


def process_emg_movement():
    """Executes paddle movement depending on the active global EMG state variable."""
    y = paddle_a.ycor()

    if emg_signal == 1 and y < 250:
        y += 40
        paddle_a.sety(y)
    elif emg_signal == -1 and y > -250:
        y -= 40
        paddle_a.sety(y)


# bot logic
def bot():
    if ball.xcor() > 0 and ball.dx == 8 and np.abs(ball.ycor() - paddle_b.ycor()) > 50:
        paddle_b.goto(350, np.round((0.8 + 0.2 * np.random.rand()) * ball.ycor() + np.random.randint(-10, 10)))


# Start the EMG simulation timer before main execution begins
simulate_emg()

frames = []
frame_times = []  # real-world timestamp for each captured frame, used for accurate GIF timing
print("Recording... Game ends automatically at 15 points, or press Ctrl+C to stop early.")

try:
    while True:
        wn.update()
        time.sleep(0.03)

        # Apply simulated physical control laws
        process_emg_movement()

        # Move the ball
        ball.setx(ball.xcor() + ball.dx)
        ball.sety(ball.ycor() + ball.dy)

        bot()

        # border collision
        if ball.ycor() > 290:
            ball.sety(290)
            ball.dy *= -1

        if ball.ycor() < -290:
            ball.sety(-290)
            ball.dy *= -1

        if ball.xcor() > 390:
            ball.goto(0, 0)
            ball.dx *= -1
            score_a += 1
            update_display()

        if ball.xcor() < -390:
            ball.goto(0, 0)
            ball.dx *= -1  # fixed: was flipping dy here, leaving the ball moving the same x-direction after a miss
            score_b += 1
            update_display()

        # paddle and ball collision
        if (ball.xcor() > 340) and (ball.xcor() < 350) and (ball.ycor() < paddle_b.ycor() + 50 and ball.ycor() + 40 > paddle_b.ycor() - 40):
            ball.setx(340)
            ball.dx *= -1

        if (ball.xcor() < -340) and (ball.xcor() > -350) and (ball.ycor() < paddle_a.ycor() + 50 and ball.ycor() + 40 > paddle_a.ycor() - 40):
            ball.setx(-340)
            ball.dx *= -1

        # End game criteria
        if score_a >= 15 or score_b >= 15:
            break

        # Capture only the game window, not the full screen
        img = pyautogui.screenshot(region=CAPTURE_REGION)
        frames.append(img)
        frame_times.append(time.time())

except KeyboardInterrupt:
    print("\nStopped early by user.")

finally:
    # This now runs whether the game ended normally (score hit 15) or was
    # interrupted with Ctrl+C, so the recording is saved either way.
    if len(frames) > 1:
        print(f"Saving {len(frames)} frames as GIF...")
        # Use the real measured time between captured frames instead of a fixed
        # guess, so playback speed matches what you actually saw on screen.
        durations = [frame_times[i] - frame_times[i - 1] for i in range(1, len(frame_times))]
        durations.insert(0, durations[0] if durations else 0.1)
        imageio.mimsave('screen_recording.gif', frames, duration=durations)
        print("Done! Saved as screen_recording.gif")
    else:
        print("Not enough frames captured to save a recording.")

turtle.done()