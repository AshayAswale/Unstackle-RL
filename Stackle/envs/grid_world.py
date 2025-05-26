from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3
    color = 4


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, size=3):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, 1, shape=(size*size,), dtype=int),
                "colored": spaces.Box(0, 1, shape=(size*size,), dtype=int), # this can be a boolean vector. if 1, color it
            }
        )

        # We have 5 actions, corresponding to "right", "up", "left", "down", "color"
        self.action_space = spaces.Discrete(5)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.right.value: self.size,
            Actions.up.value: -1,
            Actions.left.value: -self.size,
            Actions.down.value: 1,
            Actions.color.value: 0,
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "colored":self._colored_cells}

    def _get_info(self):
        return {
            "distance": sum(self._colored_cells)
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.iter = 0
        self.max_iter = 500

        # Choose the agent's location uniformly at random
        self._agent_location = np.zeros((self.size*self.size), dtype=int)
        self._agent_location[np.random.randint(0,self.size*self.size)] = 1

        self._colored_cells = np.zeros((self.size*self.size), dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def getIndexOfPosition(self, position)->int:
        x,y = position
        return self.size*x+y
    
    def getPositionOfIndex(self, index)->np.array:
        return np.array((index//self.size, index%self.size))
    
    def manhattan_distance_to_nearest_zero(self, grid_flat, pos):
      """
      grid_flat: list or 1D array of 0s and 1s representing a 2D grid flattened row-wise
      pos: [x, y] current position
      grid_size: (width, height) of the grid
      """
      min_dist = float('inf')

      for i in range(len(grid_flat)):
          target_pos_mx = min(pos+i, len(grid_flat)-1)
          target_pos_mn = min(pos-i, 0)
          if (not grid_flat[target_pos_mx]) or (not grid_flat[target_pos_mn]):
              return i
      return 0
      
    def step(self, action):
        self.iter+=1
        reward = 0
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        if action != Actions.color.value:
            direction = self._action_to_direction[action]
            old_pos = np.argmax(self._agent_location)
            new_pos = old_pos+direction
            reward -= 0.05
            if 0<=new_pos<self.size*self.size:
              # We use `np.clip` to make sure we don't leave the grid
              self._agent_location[old_pos] = 0
              self._agent_location[new_pos] = 1
        else:
            colored_box_index = np.argmax(self._agent_location)
            if not self._colored_cells[colored_box_index]:
              self._colored_cells[colored_box_index]=1
              give_reward = True
              reward += 1
            else:
              reward -= 0.1

        # An episode is done iff the agent has reached the target
        terminated = sum(self._colored_cells)==self.size*self.size
        # reward = -self.manhattan_distance_to_nearest_zero(self._colored_cells, np.argmax(self._agent_location))
        if terminated:
            reward+=100
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, self.iter>self.max_iter, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        for colored_cell, val in enumerate(self._colored_cells):
          if val:
            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(
                    pix_square_size * self.getPositionOfIndex(colored_cell),
                    (pix_square_size, pix_square_size),
                ),
            )

        # Now we draw the agent
        agent_position = self.getPositionOfIndex(np.argmax(self._agent_location))
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (agent_position + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
