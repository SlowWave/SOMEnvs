import numpy as np
from Basilisk.architecture import sysModel, messaging
from Basilisk.utilities import RigidBodyKinematics, macros


class ExternalDisturbancesInterface(sysModel.SysModel):
    def __init__(self):
        super(ExternalDisturbancesInterface, self).__init__()

        # define input - output message
        self.sc_state_in_msg = messaging.SCStatesMsgReader()
        self.cmd_torque_out_msg = messaging.CmdTorqueBodyMsg()

        self.use_disturbances = False
        self.disturbance_dict = None
        self.disturbance_torques = np.array([0, 0, 0])

    def Reset(self, current_sim_nanos):
        return

    def UpdateState(self, current_sim_nanos):

        # read input message
        sc_state_msg_buffer = self.sc_state_in_msg()

        if self.use_disturbances:
            current_dcm = RigidBodyKinematics.MRP2C(sc_state_msg_buffer.sigma_BN)
            current_secs = macros.NANO2SEC * current_sim_nanos
            self._compute_torques(current_dcm, current_secs)

        # create output message buffer
        torque_out_msg_buffer = messaging.CmdTorqueBodyMsgPayload()

        # write output message
        torque_out_msg_buffer.torqueRequestBody = (self.disturbance_torques).tolist()
        self.cmd_torque_out_msg.write(torque_out_msg_buffer, current_sim_nanos, self.moduleID)

    def _compute_torques(self, current_dcm, current_secs):

        torques = np.array([0.0, 0.0, 0.0])

        # compute costant torques
        for idx, time_window in enumerate(self.disturbance_dict["constant_torques_time_windows"]):
            if time_window[0] <= current_secs and time_window[1] >= current_secs:
                if self.disturbance_dict["constant_torques_frames"][idx] == "fixed":
                    torques += np.array(self.disturbance_dict["constant_torques_arrays"][idx])
                elif self.disturbance_dict["constant_torques_frames"][idx] == "inertial":
                    torques += np.dot(
                        np.array(self.disturbance_dict["constant_torques_arrays"][idx]),
                        current_dcm
                    )

        # compute random torques
        for idx, time_window in enumerate(self.disturbance_dict["random_torques_time_windows"]):
            if time_window[0] <= current_secs and time_window[1] >= current_secs:
                random_vector = np.random.standard_normal(3)
                random_unit_vector = random_vector / np.linalg.norm(random_vector)
                torques += random_unit_vector * self.disturbance_dict["random_torques_arrays"][idx]

        # compute sinusoidal torques
        for idx, time_window in enumerate(self.disturbance_dict["sinusoidal_torques_time_windows"]):
            if time_window[0] <= current_secs and time_window[1] >= current_secs:
                if self.disturbance_dict["sinusoidal_torques_frames"][idx] == "fixed":
                    torques += np.array(
                        [
                            self.disturbance_dict["sinusoidal_torques_arrays"][idx][0](current_secs),
                            self.disturbance_dict["sinusoidal_torques_arrays"][idx][1](current_secs),
                            self.disturbance_dict["sinusoidal_torques_arrays"][idx][2](current_secs),
                        ]
                    )
                elif self.disturbance_dict["sinusoidal_torques_frames"][idx] == "inertial":
                    torques += np.dot(
                        np.array(
                            [
                                self.disturbance_dict["sinusoidal_torques_arrays"][idx][0](current_secs),
                                self.disturbance_dict["sinusoidal_torques_arrays"][idx][1](current_secs),
                                self.disturbance_dict["sinusoidal_torques_arrays"][idx][2](current_secs),
                            ]
                        ),
                        current_dcm
                    )

        # update disturbances torques
        self.disturbance_torques = torques

    def set_torques_dict(self, disturbance_dict):

        self.disturbance_dict = disturbance_dict
        self.use_disturbances = True

