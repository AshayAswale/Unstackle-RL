from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from collections import defaultdict


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3
    color = 4


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, 1, shape=(size,size), dtype=int),
                "colored": spaces.Box(0, 1, shape=(size,size), dtype=int), # this can be a boolean vector. if 1, color it
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
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, -1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, 1]),
            Actions.color.value: np.array([0, 0])
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
        self._agent_location = np.zeros((self.size,self.size), dtype=int)
        self._agent_location[np.random.randint(0,self.size),
                             np.random.randint(0,self.size)] = 1

        self._colored_cells = np.zeros((self.size,self.size), dtype=int)

        observation = self._get_obs()
        info = self._get_info()
        self.remaining_locations=set([(i,j) for i in range(self.size) for j in range(self.size)])
        self.attendance = defaultdict(int)

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def getDistToNearestVacant(self, agent_location):
        ret =float('inf')
        for location in self.remaining_locations:
            dist = abs(location[0]-agent_location[0])
            dist += abs(location[1]-agent_location[1])
            ret = min(dist, ret)
        return ret

    def step(self, action):
        self.iter+=1
        reward = -0.1
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        if action != Actions.color.value:
            direction = self._action_to_direction[action]
            reward -= 0.005
            # We use `np.clip` to make sure we don't leave the grid
            old_pos = np.argwhere(self._agent_location==1)[0]
            new_pos = np.clip(
                old_pos+direction, 0, self.size - 1
            )
            if not self._colored_cells[new_pos[0],new_pos[1]]:
                reward+=0.01
            self._agent_location[old_pos[0],old_pos[1]] = 0
            self._agent_location[new_pos[0],new_pos[1]] = 1
            dist = self.getDistToNearestVacant(new_pos)
            reward -= dist*0.01
            reward -= self.attendance[tuple(new_pos)]*0.01
            self.attendance[tuple(new_pos)]+=1
        else:
            colored_box_index = np.argwhere(self._agent_location==1)[0]
            loc_tuple = tuple(colored_box_index)
            reward -= self.attendance[loc_tuple]*0.01
            self.attendance[loc_tuple]+=1
            if not self._colored_cells[colored_box_index[0], colored_box_index[1]]:
              self._colored_cells[colored_box_index[0], colored_box_index[1]]=1
              reward += 1
              self.remaining_locations.remove(loc_tuple)
            else:
              reward -= 0.2
        # An episode is done iff the agent has reached the target
        terminated = np.sum(self._colored_cells)==self.size*self.size
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

        for val in (np.argwhere(self._colored_cells==1)):
          pygame.draw.rect(
              canvas,
              (0, 255, 0),
              pygame.Rect(
                  pix_square_size * val,
                  (pix_square_size, pix_square_size),
              ),
          )

        # Now we draw the agent
        agent_position = np.argwhere(self._agent_location==1)[0]
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
