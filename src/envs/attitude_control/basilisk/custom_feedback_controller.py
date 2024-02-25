import numpy as np
from Basilisk.architecture import sysModel, messaging


class CustomFeedbackController(sysModel.SysModel):
    def __init__(self):
        super(CustomFeedbackController, self).__init__()

        # define input - output messages
        self.guidance_in_msg = messaging.AttGuidMsgReader()
        self.cmd_torque_out_msg = messaging.CmdTorqueBodyMsg()
        
        # controller parameters
        self.k_p = None
        self.k_d = None

    def Reset(self, current_sim_nanos):
        return

    def UpdateState(self, current_sim_nanos):

        # read input message
        guidance_msg_buffer = self.guidance_in_msg()

        # create output message buffer
        torque_out_msg_buffer = messaging.CmdTorqueBodyMsgPayload()

        # compute control signal
        control_signal = - np.array(guidance_msg_buffer.sigma_BR) * self.k_p - np.array(guidance_msg_buffer.omega_BR_B) * self.k_d
        torque_out_msg_buffer.torqueRequestBody = (control_signal).tolist()

        # write output message
        self.cmd_torque_out_msg.write(torque_out_msg_buffer, current_sim_nanos, self.moduleID)

    def set_gains(self, k_p, k_d):

        self.k_p = float(k_p)
        self.k_d = float(k_d)
