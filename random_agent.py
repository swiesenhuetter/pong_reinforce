import gymnasium
import ale_py
import cv2

def heuristic_agent(obs):
    ball_y = get_ball_y_position(obs)  # Define this helper function to extract the ball's y position
    paddle_y = get_paddle_y_position(obs)  # Define this helper function to extract the paddle's y position

    # Move the paddle up or down to follow the ball
    if paddle_y < ball_y:
        return 2  # move paddle down
    else:
        return 3  # move paddle up

def get_ball_y_position(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # ball is a small white square
    # Use simple binary thresholding or adaptive thresholding
    _, thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)

    ball_position = None

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ball_position = None
    # Iterate through contours to find the ball
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Check if the object is ball-like; adjust the size range based on the game
        if 1 < w < 10 and 2 < h < 10:
            ball_position = (x + w // 2, y + h // 2)  # Get center position of the ball

            # Optional: Visualize detected ball
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if ball_position is None:
        ball_position = [80, 105]  # Default position if ball not found

    return ball_position[1]


def get_paddle_y_position(frame):
    # Convert the frame to grayscale to simplify processing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Threshold the frame to get binary image (simplification, adjust threshold if necessary)
    _, binary_frame = cv2.threshold(gray_frame, 128, 255, cv2.THRESH_BINARY_INV)

    # Assume paddle is on the right; search for it
    right_half = binary_frame[:, binary_frame.shape[1]//2:]  # Extract the right half of the frame
    contours, _ = cv2.findContours(right_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    paddle_y = None
    # Loop through contours to find the paddle
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Check for paddle-like structure; tweaking w/h criteria might be necessary
        if h > 10 and 2 < w < 10:  # Simple heuristics: adjust as necessary
            paddle_y = y + h // 2  # Skipped some simplifications

    if paddle_y is None:
        paddle_y = right_half.shape[0] // 2  # Default value if paddle not found

    return paddle_y

def draw_paddle_position(frame, paddle_y):
    color = (0, 255, 0)  # Green
    thickness = 2
    center_x = frame.shape[1] - 10  # Near the right edge
    start_point = (center_x, paddle_y - 10)
    end_point = (center_x, paddle_y + 10)
    cv2.line(frame, start_point, end_point, color, thickness)


if __name__ == "__main__":
    # Create the environment
    env = gymnasium.make("ALE/Pong-v5", render_mode="human")
    obs, info = env.reset()

    for _ in range(1000):
        action = heuristic_agent(obs)

        obs, reward, terminated, truncated, info = env.step(action)


        if terminated or truncated:
            obs, info = env.reset()

    env.close()
