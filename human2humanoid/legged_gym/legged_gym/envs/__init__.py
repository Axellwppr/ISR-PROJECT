from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.h1.h1_teleop_config import H1TeleopCfg, H1TeleopCfgPPO
from legged_gym.envs.xbot.xbot_teleop_config import XbotTeleopCfg, XbotTeleopCfgPPO

from .base.legged_robot_old import LeggedRobot

from legged_gym.utils.task_registry import task_registry

# task_registry.register( "h1:teleop", LeggedRobot, H1TeleopCfg(), H1TeleopCfgPPO())

task_registry.register("xbot:teleop", LeggedRobot, XbotTeleopCfg(), XbotTeleopCfgPPO())
