
import argparse
import os

import cv2
import gym
import mujoco_py
import numpy as np

def humanoid():
    mj_path = mujoco_py.utils.discover_mujoco()
    xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
    model = mujoco_py.load_model_from_path(xml_path)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    viewer.render()

    print(sim.data.qpos)
    # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

    sim.step()
    viewer.render()
    print(sim.data.qpos)
    # [-2.09531783e-19  2.72130735e-05  6.14480786e-22 -3.45474715e-06
    #   7.42993721e-06 -1.40711141e-04 -3.04253586e-04 -2.07559344e-04
    #   8.50646247e-05 -3.45474715e-06  7.42993721e-06 -1.40711141e-04
    #  -3.04253586e-04 -2.07559344e-04 -8.50646247e-05  1.11317030e-04
    #  -7.03465386e-05 -2.22862221e-05 -1.11317030e-04  7.03465386e-05
    #  -2.22862221e-05]

    while True:
        sim.step()
        viewer.render()

class HammerEnvV0():
    def __init__(self, show_viewer, render) -> None:
        self.show_viewer = show_viewer
        self.render = render
        self.setup_env()

    def setup_env(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mj_path = curr_dir + "/one_car.xml"
        model = mujoco_py.load_model_from_path(mj_path)
        self.model = model
        sim = mujoco_py.MjSim(model)
        self.sim = sim


    def get_actuator_names(self):
        actuator_addresses = self.model.name_actuatoradr
        return self.get_names_from_addresses(actuator_addresses)

    def get_names_from_addresses(self, addresses):
        names = self.model.names[
            addresses[0]:addresses[-1]
        ].tolist()
        for i in range(addresses[-1], addresses[-1] + 100):
            val = self.model.names[i]
            if val == b'':
                break
            else:
                names.append(val)

        names = [c.decode() for c in names]

        final_names = []
        cur_name = ""
        for a in names:
            if a == '':
                final_names.append(cur_name)
                cur_name = ""
            else:
                cur_name += a

        return final_names

    def get_body_names(self):
        body_addresses = self.model.name_bodyadr
        return self.get_names_from_addresses(body_addresses)

    def apply_two_axisangles(self, first_rot, second_rot):
        beta = first_rot[3]
        m = first_rot[:3]
        alpha = second_rot[3]
        l = second_rot[:3]

        yeta = np.arccos(np.cos(alpha/2)*np.cos(beta/2) - np.sin(alpha/2)*np.sin(beta/2) * np.dot(l, m)) * 2
        n = (np.sin(alpha/2)*np.cos(beta/2)*l + np.cos(alpha/2)*np.sin(beta/2)*m + np.sin(alpha/2) * np.sin(beta/2) * np.cross(l, m)) / np.sin(yeta/2)
        return np.append(n, yeta)

    def axisangle_to_quat(self, axisangle):
        q0 = np.cos(axisangle[3] / 2)
        q1 = axisangle[0] * np.sin(axisangle[3] / 2)
        q2 = axisangle[1] * np.sin(axisangle[3] / 2)
        q3 = axisangle[2] * np.sin(axisangle[3] / 2)
        return np.array([q0, q1, q2, q3])

    def quat_to_axisangle(self, quat):
        if quat[0] == 0:
            return np.array([1., 0., 0., 0.])
        theta = 2 * np.arccos(quat[0])
        x = quat[1] / np.sin(theta/2)
        y = quat[2] / np.sin(theta/2)
        z = quat[3] / np.sin(theta/2)
        return np.array([x, y, z, theta])

    def run_hand_model(self):
        sim = self.sim
        if self.show_viewer:
            viewer = mujoco_py.MjViewer(sim)
            viewer.render()

        print(self.model.actuator_ctrllimited)
        print(self.model.actuator_ctrlrange)
        actuator_names = self.get_actuator_names()
        print("Actuators:", actuator_names)

        body_names = self.get_body_names()
        print("Bodies:", body_names, len(body_names))
        print(self.model.body_pos.shape)

        steering_id = self.model.actuator_name2id("buddy_steering_pos")
        throttle_id = self.model.actuator_name2id("buddy_throttle_velocity")
        fpv_camera_id = self.model.camera_name2id("buddy_first_person")
        print(
            "Steering id:", steering_id,
            "throttle_id:", throttle_id,
            "fpv_camera_id:", fpv_camera_id)
        start_fpv_camera_quat = self.model.cam_quat[fpv_camera_id]
        start_fpv_camera_axisangle = self.quat_to_axisangle(start_fpv_camera_quat)
        np.testing.assert_allclose(self.axisangle_to_quat(start_fpv_camera_axisangle), start_fpv_camera_quat)

        # twist, down/up, left/right
        angle = np.pi/2
        rot_fpv_camera_axisangle = np.array([0, 0, 1, angle])
        final_fpv_camera_axisangle = self.apply_two_axisangles(start_fpv_camera_axisangle, rot_fpv_camera_axisangle)
        final_fpv_camera_quat = self.axisangle_to_quat(final_fpv_camera_axisangle)
        self.model.cam_quat[fpv_camera_id] = final_fpv_camera_quat

        fpv_camera_xpos = self.sim.data.get_camera_xpos("buddy_first_person")
        print("fpv_camera_xpos:", fpv_camera_xpos)

        while True:

            self.sim.data.ctrl[steering_id] = 0.3
            self.sim.data.ctrl[throttle_id] = 1
            #self.model.cam_fovy[fpv_camera_id] +=1


            sim.step()

            # Rotate left/right
            if self.show_viewer:
                #viewer.cam.azimuth += 0.1
                viewer.render()
            else:
                ret, depth_img = sim.render(width=600, height=400, camera_name="buddy_first_person", depth=True)
                if self.render:
                    rgb_im = cv2.cvtColor(ret, cv2.COLOR_BGR2RGB)
                    rgb_im = cv2.flip(rgb_im, 0)
                    cv2.imshow("Forearm camera", rgb_im)
                    print(depth_img)
                    cv2.imshow("Depth", depth_img)
                    cv2.waitKey()
                    return

    def take_action(self, action):
        for i in range(self.model.nu):
            self.sim.data.ctrl[i] = action[i]

def run_hand_model(show_viewer, render):
    hammer_env = HammerEnvV0(show_viewer, render)
    hammer_env.run_hand_model()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", dest="viewer", nargs="?", default=False, const=True)
    parser.add_argument("--render", dest="render", nargs="?", default=False, const=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    show_viewer = args.viewer
    render = not show_viewer and args.render
    print("viewer:", show_viewer, "render:", render)

    run_hand_model(show_viewer, render)