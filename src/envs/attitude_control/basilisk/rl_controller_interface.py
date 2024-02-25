import numpy as np
from Basilisk.architecture import sysModel, messaging


class RLControllerInterface(sysModel.SysModel):
    def __init__(self):
        super(RLControllerInterface, self).__init__()

        # define input - output messages
        self.cmd_torque_in_msg = messaging.CmdTorqueBodyMsgReader()
        self.cmd_torque_out_msg = messaging.CmdTorqueBodyMsg()

        # define null reinforcement learning agent action
        self.rl_agent_action = np.array([0, 0, 0])

    def Reset(self, current_sim_nanos):
        return

    def UpdateState(self, current_sim_nanos):

        # read input message
        cmd_torque_msg_buffer = self.cmd_torque_in_msg()

        # create output message buffer
        output_buffer = messaging.CmdTorqueBodyMsgPayload()

        # compute mixed control signal
        mixed_control_signal = cmd_torque_msg_buffer.torqueRequestBody + self.rl_agent_action
        output_buffer.torqueRequestBody = (mixed_control_signal).tolist()

        # write output message
        self.cmd_torque_out_msg.write(output_buffer, current_sim_nanos, self.moduleID)

    def set_rl_agent_action(self, action):

        # set reinforcement learning agent action
        self.rl_agent_action = action
