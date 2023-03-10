from __future__ import annotations

from copy import deepcopy

from minigrid.core.constants import COLOR_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Key
from minigrid.minigrid_env import MiniGridEnv


class MultiKeyEnv(MiniGridEnv):

    """
    ## Description

    This environment has multiple keys that the agent must pick up in order
    and then get to the green goal square. This environment is difficult,
    because of the sparse reward, to solve using classical RL algorithms. It is
    useful to experiment with curiosity or curriculum learning. Can pick up
    multiple objects in this environment.

    ## Mission Space

    "pick up all the keys in order and then get to the goal"

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Pick up an object         |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal with all keys collected in order.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-MultiKey-16x16-1`
    - `MiniGrid-MultiKey-16x16-2`
    - `MiniGrid-MultiKey-16x16-3`
    - `MiniGrid-MultiKey-16x16-4`

    """

    def __init__(self, size=8, num_keys=1, max_steps: int | None = None, ordered=True, **kwargs):
        if max_steps is None:
            max_steps = 10 * size**2
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space, grid_size=size, max_steps=max_steps, can_carry_multiple_items=True, **kwargs
        )
        assert num_keys >= 1 and num_keys <= 5
        self.num_keys = num_keys
        self.required_key_colors = tuple(k for k in COLOR_TO_IDX if COLOR_TO_IDX[k] < num_keys)
        self.ordered = ordered

    @staticmethod
    def _gen_mission():
        return "pick up all the keys in order and then get to the goal"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent at a random position and orientation
        self.place_agent(size=(width, height))

        # Place all the key on the left side
        for key_color in self.required_key_colors:
            self.place_obj(obj=Key(key_color), top=(0, 0), size=(width, height))

        self.mission = "pick up all the keys in order and then get to the goal"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated and len(self.carrying) != len(self.required_key_colors):
            terminated = True
            reward = -1.0
        if self.ordered:
            colors = [t.color for t in self.carrying]
            took_keys_in_order = all(c == cref for c, cref in zip(colors, self.required_key_colors))
            if not took_keys_in_order:
                terminated = True
                reward = -1.0
        return obs, reward, terminated, truncated, info
