import pytorch_kinematics as pk
import torch

urdf = "/home/axell/桌面/humanoid/humanoid-benchmark-main/isaacLab/manipulation/assets/urdf/rel2/urdf/rel2.urdf"


class XBot_fk:
    def __init__(self, urdf):
        self.chain = pk.build_chain_from_urdf(open(urdf, mode="rb").read())
        self.joint_names = [j.name for j in self.chain.joints]
        self.link_names = [l.name for l in self.chain.links]
        self.joint_num = len(self.joint_names)
        self.link_num = len(self.link_names)

    def forward_kinematics(self, joint_angles: torch.Tensor):
        joint_angles = joint_angles.reshape(-1)
        if len(joint_angles) != self.joint_num:
            raise ValueError(
                f"joint_angles should have {self.joint_num} elements, but got {len(joint_angles)}"
            )
        joint_angles_dict = {
            name: angle for name, angle in zip(self.joint_names, joint_angles)
        }
        return self.chain.forward_kinematics(joint_angles_dict)

    def forward_kinematics_batch(self, joint_angles: torch.Tensor):
        joint_angles = joint_angles.reshape(-1, self.joint_num)
        joint_angles_dict = [
            {name: angle for name, angle in zip(self.joint_names, angles)}
            for angles in joint_angles
        ]
        return self.chain.forward_kinematics_batch(joint_angles_dict)

    def get_link_names(self):
        return self.link_names

    def get_joint_names(self):
        return self.joint_names


# unit test
def main():
    fk = XBot_fk(urdf)
    joint_angles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fk.forward_kinematics(joint_angles)
    fk.forward_kinematics_batch([joint_angles, joint_angles])
