from gymnasium.spaces import Text
import gymnasium as gym
import numpy as np
import random

class breakout(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}
    WIN_WIDTH = 80
    WIN_HEIGHT = 34
    INTERIOR_WIDTH = WIN_WIDTH - 2
    INTERIOR_HEIGHT = WIN_HEIGHT - 2
    BRICK_ROWS = 7
    BRICK_COLS = 13
    BRICK_WIDTH = 6
    BRICK_HEIGHT = 1
    GAP_ABOVE_BRICKS = 4
    GAP_BETWEEN_BRICKS_AND_PADDLE = 20

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
        self.LENGTH = self.WIN_HEIGHT
        self.WIDTH = self.WIN_WIDTH

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.score = 0
        self.lives = 5
        self.bricks = np.ones((self.BRICK_ROWS, self.BRICK_COLS), dtype=np.int32)
        self.ball_x = self.WIN_WIDTH // 2
        self.ball_y = self.paddle_y - 1
        self.ball_dx = random.choice([-1, 1])
        self.ball_dy = -1
        self.paddle_x = 1 + (self.INTERIOR_WIDTH - self.paddle_width) // 2

        obs = self._get_ascii()
        return obs, {}
    
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
                obs = self._get_ascii()
                return obs, 0, True, False, {"score": self.score}
            new_ball_x = self.WIN_WIDTH // 2
            new_ball_y = self.paddle_y - 1
            self.ball_dx = random.choice([-1, 1])
            self.ball_dy = -1
            self.paddle_x = 1 + (self.INTERIOR_WIDTH - self.paddle_width) // 2
        if int(round(new_ball_y)) == self.paddle_y and self.paddle_x <= new_ball_x < self.paddle_x + self.paddle_width:
            new_ball_y = self.paddle_y - 1
            self.ball_dy = -abs(self.ball_dy)
            hit_offset = (new_ball_x - self.paddle_x) - (self.paddle_width / 2)
            self.ball_dx = 1 if hit_offset >= 0 else -1
        ball_cell_y = int(round(new_ball_y))
        if self.brick_start_y <= ball_cell_y < self.brick_start_y + self.BRICK_ROWS:
            brick_row = ball_cell_y - self.brick_start_y
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

        final_obs = self._get_ascii()
        return final_obs, reward, done, False, {"score": self.score}

    def render(self, mode="human"):
        ascii_obs = self._get_ascii()
        if mode == "human":
            print(ascii_obs)
            print(f"Score: {self.score}  Lives: {self.lives}")
        return ascii_obs

    def close(self):
        pass
