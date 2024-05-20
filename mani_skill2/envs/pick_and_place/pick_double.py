from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat, qmult

from mani_skill2 import ASSET_DIR, format_path
from mani_skill2.utils.common import random_choice
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import set_actor_visibility, vectorize_pose

from .pick_single import PickSingleYCBEnv, build_actor_ycb


@register_env("PickDoubleYCB-v0", max_episode_steps=200)
class PickDoubleYCBEnv(PickSingleYCBEnv):

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)
        self.all_objs = self._load_model()
        self.obj = self.all_objs[0] # target object
        self.goal_site = self._build_sphere_site(self.goal_thresh)


    def _load_model(self):
        all_objs = []
        for i in range(len(self.model_ids)):
            model_id = self.model_ids[i]
            density = self.model_db[model_id].get("density", 1000)
            obj = build_actor_ycb(
                model_id,
                self._scene,
                scale=self.model_scale,
                density=density,
                root_dir=self.asset_root,
            )
            obj.name = model_id
            obj.set_damping(0.1, 0.1)
            all_objs.append(obj)
        return all_objs


    def _initialize_single_actor(self, obj, factor, use_seed=True):
        # The object will fall from a certain height
        if use_seed:
            rng = self._episode_rng
        else:
            rng = np.random

        xy = rng.uniform(-0.1*factor, 0.1*factor, [2])
        z = self._get_init_z()
        p = np.hstack([xy, z])
        q = [1, 0, 0, 0]

        # Rotate along z-axis
        if self.obj_init_rot_z:
            ori = rng.uniform(0, 2 * np.pi)
            q = euler2quat(0, 0, ori)

        # Rotate along a random axis by a small angle
        if self.obj_init_rot > 0:
            axis = rng.uniform(-1, 1, 3)
            axis = axis / max(np.linalg.norm(axis), 1e-6)
            ori = rng.uniform(0, self.obj_init_rot)
            q = qmult(q, axangle2quat(axis, ori, True))
        obj.set_pose(Pose(p, q))

        # Move the robot far away to avoid collision
        # The robot should be initialized later
        self.agent.robot.set_pose(Pose([-10, 0, 0]))

        # Lock rotation around x and y
        obj.lock_motion(0, 0, 0, 1, 1, 0)
        self._settle(0.5)

        # Unlock motion
        obj.lock_motion(0, 0, 0, 0, 0, 0)
        # Explicit set pose to ensure the actor does not sleep
        obj.set_pose(obj.pose)
        obj.set_velocity(np.zeros(3))
        obj.set_angular_velocity(np.zeros(3))
        self._settle(0.5)

        # Some objects need longer time to settle
        lin_vel = np.linalg.norm(obj.velocity)
        ang_vel = np.linalg.norm(obj.angular_velocity)
        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(0.5)

    def _initialize_actors(self):
        target_obj = self.all_objs[0]
        self._initialize_single_actor(target_obj, factor=1, use_seed=True)
        scene_obj = self.all_objs[1]
        self._initialize_single_actor(scene_obj, factor=3, use_seed=False)
