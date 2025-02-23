from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import login
from gymnasium.spaces import Text
from openai import OpenAI 
from google import genai
import gymnasium as gym
import numpy as np
import imageio
import random
import torch
import time
import json 
import csv
import os
import re


# Disable tokenizer parallelism warning.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# Create Gymnasium Environment 
class BrickBreakerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}
    WIN_WIDTH = 80
    WIN_HEIGHT = 24
    INTERIOR_WIDTH = WIN_WIDTH - 2
    INTERIOR_HEIGHT = WIN_HEIGHT - 2
    BRICK_ROWS = 7
    BRICK_COLS = 13
    BRICK_WIDTH = 6
    BRICK_HEIGHT = 1
    GAP_ABOVE_BRICKS = 4
    GAP_BETWEEN_BRICKS_AND_PADDLE = 10

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        max_length = self.WIN_HEIGHT * (self.WIN_WIDTH + 1)
        self.observation_space = Text(max_length=max_length)
        self.action_space = gym.spaces.Discrete(3)
        self.paddle_speed = 2
        self.paddle_width = 7
        self.brick_start_y = 1 + self.GAP_ABOVE_BRICKS
        self.paddle_y = (self.brick_start_y + self.BRICK_ROWS) + self.GAP_BETWEEN_BRICKS_AND_PADDLE

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.score = 0
        self.lives = 3
        self.bricks = np.ones((self.BRICK_ROWS, self.BRICK_COLS), dtype=np.int32)
        self.ball_x = self.WIN_WIDTH // 2
        self.ball_y = self.paddle_y - 1
        self.ball_dx = random.choice([-1, 1])
        self.ball_dy = -1
        self.paddle_x = 1 + (self.INTERIOR_WIDTH - self.paddle_width) // 2
        return self._get_ascii(), {}

    def _get_ascii(self):
        grid = [[" " for _ in range(self.WIN_WIDTH)] for _ in range(self.WIN_HEIGHT)]
        for x in range(self.WIN_WIDTH):
            grid[0][x] = "#"
            grid[self.WIN_HEIGHT - 1][x] = "#"
        for y in range(self.WIN_HEIGHT):
            grid[y][0] = "#"
            grid[y][self.WIN_WIDTH - 1] = "#"
        for i in range(self.BRICK_ROWS):
            for j in range(self.BRICK_COLS):
                if self.bricks[i, j] == 1:
                    brick_x = 1 + j * self.BRICK_WIDTH
                    brick_y = self.brick_start_y + i
                    for bx in range(self.BRICK_WIDTH):
                        ch = "|" if bx == 0 or bx == self.BRICK_WIDTH - 1 else "_"
                        if brick_x + bx < self.WIN_WIDTH - 1:
                            grid[brick_y][brick_x + bx] = ch
        for i in range(self.paddle_width):
            if 0 <= self.paddle_x + i < self.WIN_WIDTH - 1:
                grid[self.paddle_y][self.paddle_x + i] = "="
        bx = int(round(self.ball_x))
        by = int(round(self.ball_y))
        if 0 <= by < self.WIN_HEIGHT and 0 <= bx < self.WIN_WIDTH:
            grid[by][bx] = "O"
        return "\n".join("".join(row) for row in grid)

    def step(self, action):
        if action == 1:
            self.paddle_x = max(1, self.paddle_x - self.paddle_speed)
        elif action == 2:
            self.paddle_x = min(self.WIN_WIDTH - 1 - self.paddle_width, self.paddle_x + self.paddle_speed)
        reward = 0
        new_ball_x = self.ball_x + self.ball_dx
        new_ball_y = self.ball_y + self.ball_dy
        if new_ball_x <= 0:
            new_ball_x = 0
            self.ball_dx = -self.ball_dx
        elif new_ball_x >= self.WIN_WIDTH - 1:
            new_ball_x = self.WIN_WIDTH - 1
            self.ball_dx = -self.ball_dx
        if new_ball_y <= 0:
            new_ball_y = 0
            self.ball_dy = -self.ball_dy
        elif new_ball_y >= self.WIN_HEIGHT - 1:
            self.lives -= 1
            if self.lives <= 0:
                return self._get_ascii(), 0, True, False, {"score": self.score}
            new_ball_x = self.WIN_WIDTH // 2
            new_ball_y = self.paddle_y - 1
            self.ball_dx = random.choice([-1, 1])
            self.ball_dy = -1
            self.paddle_x = 1 + (self.INTERIOR_WIDTH - self.paddle_width) // 2
        if int(new_ball_y) == self.paddle_y and self.paddle_x <= int(new_ball_x) < self.paddle_x + self.paddle_width:
            new_ball_y = self.paddle_y - 1
            self.ball_dy = -abs(self.ball_dy)
            hit_offset = (new_ball_x - self.paddle_x) - (self.paddle_width / 2)
            self.ball_dx = 1 if hit_offset >= 0 else -1
        brick_row = int(new_ball_y) - self.brick_start_y
        if 0 <= brick_row < self.BRICK_ROWS:
            for j in range(self.BRICK_COLS):
                brick_x = 1 + j * self.BRICK_WIDTH
                brick_y = self.brick_start_y + brick_row
                if (self.bricks[brick_row, j] == 1 and 
                    brick_y == int(new_ball_y) and 
                    brick_x <= int(new_ball_x) < brick_x + self.BRICK_WIDTH):
                    self.bricks[brick_row, j] = 0
                    self.ball_dy = -self.ball_dy
                    reward += 10
                    self.score += 10
                    break
        self.ball_x = new_ball_x
        self.ball_y = new_ball_y
        done = (np.sum(self.bricks) == 0)
        if done:
            reward += 50
        return self._get_ascii(), reward, done, False, {"score": self.score}

    def render(self, mode="human"):
        ascii_obs = self._get_ascii()
        if mode == "human":
            print(ascii_obs)
            print(f"Score: {self.score}  Lives: {self.lives}")
        return ascii_obs

    def close(self):
        pass

def extract_action(output_decode: str) -> int:
    pattern = r"<action>\s*([012])\s*</action>"
    matches = re.findall(pattern, output_decode, flags=re.IGNORECASE)
    if not matches:
        raise ValueError("No valid <action> tag found in the output.")
    return int(matches[-1])

def ascii_to_image(ascii_text: str, num_cols=80, num_rows=24) -> Image.Image:
    font = ImageFont.load_default()
    bbox = font.getbbox("A")
    char_width = bbox[2] - bbox[0]
    char_height = bbox[3] - bbox[1]
    img_width, img_height = num_cols * char_width, num_rows * char_height
    image = Image.new("RGB", (img_width, img_height), "black")
    draw = ImageDraw.Draw(image)
    lines = ascii_text.splitlines()
    for row, line in enumerate(lines[:num_rows]):
        draw.text((0, row * char_height), line.ljust(num_cols), font=font, fill="white")
    return image

if __name__ == "__main__":
    api_value = input("Do you want to run OpenAI GPT-4o (1), Gemini Flash 2.0 (2), Gemini Pro 2.0 (3) or Llama 3.2 3B (4)? ")

    if api_value == "1":
        print("\nSetting up GPT-4o...\n")
        with open("OPENAI_API_KEY.txt", "r") as f:
            openai_api_key = f.read().strip()
        client = OpenAI(api_key=openai_api_key)

    elif api_value == "2":
        print("\nSetting up Gemini Flash 2.0...\n")
        with open("GOOGLE_API_KEY.txt", "r") as f:
            google_api_key = f.read().strip()
        # Initialize the Google GenAI client using the new SDK.
        client = genai.Client(api_key=google_api_key)
        # Create a chat session with Gemini 2.0 Flash.
        chat = client.chats.create(model="gemini-2.0-flash")

    elif api_value == "3":
        print("\nSetting up Gemini Pro 2.0...\n")
        with open("GOOGLE_API_KEY.txt", "r") as f:
            google_api_key = f.read().strip()
        # Initialize the Google GenAI client using the new SDK.
        client = genai.Client(api_key=google_api_key)
        # Create a chat session with Gemini 2.0 Flash.
        chat = client.chats.create(model="gemini-2.0-pro-exp-02-05")

    elif api_value == "4":
        print("\nSetting up Llama 3.2 3B...\n")
        with open("HG_API_KEY.txt", "r") as f:
            hugging_face_api_key = f.read().strip()
        login(token=hugging_face_api_key)
        set_seed(42)
        device = get_device()
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError("Invalid selection. Please enter 1, 2, or 3.")

    print("\nModel set up complete...\n")
    
    messages = [{"role": "system", "content": "You are a professional Atari 2600 game playing assistant and will be provided an ASCII representation of Breakout. "
                    "Breakout uses a paddle (=======) to hit a ball (O) with the aim of breaking all bricks (|____|). Your goal is to provide me with "
                    "the best action I could take to break all the bricks while hitting the ball with the paddle, I only have control of the paddle and "
                    "can only move it left or right. I want you to take as long as you need to completely understand the board, and location of the paddle and ball, "
                    "show me the exact game state we are in. Once you have done this take that information and think step by step what the best action you could "
                    "take to keep the paddle in line with the ball. The potential actions I can take are <action> 0 </action> for NOOP, <action> 1 </action> for LEFT and "
                    "<action> 2 </action> for RIGHT. Provide output as a where your reasoning is between <reasoning> meta tags, this is a scratchpad for you to think, take as long as you need. "
                    "In addition, please add the numerical value for the action inbetween <action> </action> meta tags."
    }]
    
    env = BrickBreakerEnv(render_mode="human")
    obs, _ = env.reset(seed=42)
    frames = [ascii_to_image(obs, num_cols=80, num_rows=24)]
    total_rewards = 0
    cumulative_rewards = []
    action_list = []
    max_message_len = 9  # Limit conversation length

    # Game loop: 1000 steps.
    for i in range(1000):
        action = None
        # Provide up to 3 chances for a correct action.
        for _ in range(3):
            messages.append({
                    "role": "user",
                    "content": 
                            f"This is the current game state:\n{obs}\n\n"
                            "Based on the provided game state, please determine the best action to take. Remember:\n"
                            "- Provide your detailed reasoning between <reasoning> and </reasoning> meta tags.\n"
                            "- Then, immediately provide your chosen action between <action> and </action> meta tags, with no extra text or formatting."
                            "Remember your action output should only be one of these 3 options <action> 0 </action> for NOOP, <action> 1 </action> for LEFT and <action> 2 </action> for RIGHT\n"
                            "Show me what the current game state is:\n"
                        })
            
            if api_value == "1":
                response = client.chat.completions.create(
                    model="gpt-4o-2024-11-20",
                    messages=messages,
                    temperature=1,
                )
                output_decode = response.choices[0].message.content
            elif api_value == "2" or api_value == "3":
                # For Gemini Flash, use the chat session.
                response = chat.send_message(message=messages[-1]["content"])
                output_decode = response.text
            elif api_value == "4":
                input_ids = tokenizer.apply_chat_template(
                    messages, return_tensors="pt", return_dict=True
                ).to(get_device())
                outputs = model.generate(**input_ids, max_new_tokens=4096)
                output_decode = tokenizer.decode(outputs[0])
            
            with open('./all_responses.txt', "a") as file:
                file.write(str(output_decode) + '\n\n')

            try:
                action = extract_action(output_decode)
                print(f"Extracted action: {action}")
                break
            except ValueError as e:
                print("Extraction error:", e)
                messages.append({
                    "role": "user",
                    "content": "You did not put the action between the <action> and </action> meta tags. PROVIDE THE ENDING ACTION ONLY BETWEEN THESE META TAGS."
                })

        messages.append({"role": "assistant", "content": output_decode})
        action_list.append(action)
        obs, reward, done, truncated, info = env.step(action)
        total_rewards += reward
        cumulative_rewards.append(total_rewards)
        frames.append(ascii_to_image(obs, num_cols=80, num_rows=24))
        if len(messages) >= max_message_len:
            messages.pop(1)
            messages.pop(1)
        if done:
            obs, _ = env.reset(seed=42)
            frames.append(ascii_to_image(obs))
            done = False
        print("Step ", i)
        time.sleep(0.1)

    print("\n\n Total Reward: ", total_rewards)
    print("Saving video of performance...")
    video_filename = "breakout.mp4"
    images = [np.array(frame) for frame in frames if frame is not None]
    imageio.mimwrite(video_filename, images, fps=env.metadata["render_fps"])
    print(f"Saved video as {video_filename}")

    print("Saving actions and cumulative rewards...")
    header = ["actions", "cumulative_rewards"]
    with open('./actions_rewards.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for action, cum_reward in zip(action_list, cumulative_rewards):
            writer.writerow([action, cum_reward])
    
    print("\nTest complete, Thank you!\n")