import gymnasium
import ale_py
import cv2
import numpy as np

def heuristic_agent(obs):
    ball_y = get_ball_y_position(obs)
    paddle_y = get_paddle_y_position(obs)

    if paddle_y < ball_y:
        return 2  # move paddle down
    else:
        return 3  # move paddle up

def get_ball_y_position(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_position = None

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 1 < w < 10 and 2 < h < 10:
            ball_position = (x + w // 2, y + h // 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    if ball_position is None:
        ball_position = [80, 105]
    
    return ball_position[1]

def get_paddle_y_position(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, 128, 255, cv2.THRESH_BINARY_INV)
    
    right_half = binary_frame[:, binary_frame.shape[1]//2:]
    contours, _ = cv2.findContours(right_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paddle_y = None

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 10 and 2 < w < 10:
            paddle_y = y + h // 2
            cv2.rectangle(frame, (x + frame.shape[1]//2, y), (x + frame.shape[1]//2 + w, y + h), (0, 255, 0), 2)
    
    if paddle_y is None:
        paddle_y = right_half.shape[0] // 2
    
    return paddle_y

def draw_frame(frame):
    cv2.imshow('Pong', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

if __name__ == "__main__":
    env = gymnasium.make("ALE/Pong-v5")
    obs, info = env.reset()

    for _ in range(1000):
        action = heuristic_agent(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        draw_frame(obs)  # Visualize the processed frame with contours

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    cv2.destroyAllWindows()  # Close all OpenCV windows