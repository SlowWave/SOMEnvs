import os
import tomli
import numpy as np
import plotly.graph_objects as go


# get config data
with open(
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
            "configs",
            "config.toml"
        ),
        "rb"
) as config_file:
    CFG = tomli.load(config_file)


class Storage():
    def __init__(self):

        self.dyn_timestamp = None
        self.fsw_timestamp = None
        self.true_sc_mrp = None
        self.true_sc_omega = None
        self.true_sc_omega_dot = None
        self.measured_sc_mrp = None
        self.measured_sc_omega = None
        self.mrp_tracking_error = None
        self.omega_tracking_error = None
        self.angular_attitude_error = None
        self.feedback_control_torque_signal = None
        self.mixed_control_torque_signal = None
        self.rl_agent_actions = None
        self.rw_mapping_control_torque_signal = None
        self.rw_control_voltage_reference_signal = None
        self.rw_control_torque_signal = None
        self.rw_speeds = None
        self.rw_torques = None
        self.current_sim_secs = None
        self.is_last_step = None

        self.obs_model = str(CFG["gymnasium"]["observation_space"]["model_id"])
        self.obs_model_map = {
            "1": self._get_obs_model_1,
            "2": self._get_obs_model_2,
        }

    def reset(self, bsk_states):

        self.dyn_timestamp = bsk_states["dyn_timestamp"]
        self.fsw_timestamp = bsk_states["fsw_timestamp"]
        self.true_sc_mrp = bsk_states["true_sc_mrp"]
        self.true_sc_omega = bsk_states["true_sc_omega"]
        self.true_sc_omega_dot = bsk_states["true_sc_omega_dot"]
        self.measured_sc_mrp = bsk_states["measured_sc_mrp"]
        self.measured_sc_omega = bsk_states["measured_sc_omega"]
        self.mrp_tracking_error = bsk_states["mrp_tracking_error"]
        self.omega_tracking_error = bsk_states["omega_tracking_error"]
        self.angular_attitude_error = bsk_states["angular_attitude_error"]
        self.feedback_control_torque_signal = bsk_states["feedback_control_torque_signal"]
        self.mixed_control_torque_signal = bsk_states["mixed_control_torque_signal"]
        self.rl_agent_actions = np.array([0., 0., 0.])
        self.rw_mapping_control_torque_signal = bsk_states["rw_mapping_control_torque_signal"]
        self.rw_control_voltage_reference_signal = bsk_states["rw_control_voltage_reference_signal"]
        self.rw_control_torque_signal = bsk_states["rw_control_torque_signal"]
        self.rw_speeds = bsk_states["rw_speeds"]
        self.rw_torques = bsk_states["rw_torques"]
        self.current_sim_secs = self.dyn_timestamp[-1]
        self.is_last_step = False

    def update_records(self, bsk_states, action, is_last_step):

        self.dyn_timestamp = np.concatenate((self.dyn_timestamp, bsk_states["dyn_timestamp"]))
        self.fsw_timestamp = np.concatenate((self.fsw_timestamp, bsk_states["fsw_timestamp"]))
        self.true_sc_mrp = np.vstack((self.true_sc_mrp, bsk_states["true_sc_mrp"]))
        self.true_sc_omega = np.vstack((self.true_sc_omega, bsk_states["true_sc_omega"]))
        self.true_sc_omega_dot = np.vstack((self.true_sc_omega_dot, bsk_states["true_sc_omega_dot"]))
        self.measured_sc_mrp = np.vstack((self.measured_sc_mrp, bsk_states["measured_sc_mrp"]))
        self.measured_sc_omega = np.vstack((self.measured_sc_omega, bsk_states["measured_sc_omega"]))
        self.mrp_tracking_error = np.vstack((self.mrp_tracking_error, bsk_states["mrp_tracking_error"]))
        self.omega_tracking_error = np.vstack((self.omega_tracking_error, bsk_states["omega_tracking_error"]))
        self.angular_attitude_error = np.concatenate((self.angular_attitude_error, bsk_states["angular_attitude_error"]))
        self.feedback_control_torque_signal = np.vstack((self.feedback_control_torque_signal, bsk_states["feedback_control_torque_signal"]))
        self.mixed_control_torque_signal = np.vstack((self.mixed_control_torque_signal, bsk_states["mixed_control_torque_signal"]))
        self.rl_agent_actions = np.vstack((self.rl_agent_actions, action))
        self.rw_mapping_control_torque_signal = np.vstack((self.rw_mapping_control_torque_signal, bsk_states["rw_mapping_control_torque_signal"]))
        self.rw_control_voltage_reference_signal = np.vstack((self.rw_control_voltage_reference_signal, bsk_states["rw_control_voltage_reference_signal"]))
        self.rw_control_torque_signal = np.vstack((self.rw_control_torque_signal, bsk_states["rw_control_torque_signal"]))
        self.rw_speeds = np.vstack((self.rw_speeds, bsk_states["rw_speeds"]))
        self.rw_torques = np.vstack((self.rw_torques, bsk_states["rw_torques"]))
        self.current_sim_secs = self.dyn_timestamp[-1]
        self.is_last_step = is_last_step

    def plot_sc_states(self):
        
        # spacecraft attitude
        fig_1 = go.Figure()
        for idx in range(3):
            fig_1.add_trace(
                go.Scatter(
                    x=self.dyn_timestamp,
                    y=self.true_sc_mrp[:, idx],
                    mode='lines',
                    line=dict(dash='dash'),
                    name='true_MRP_{}'.format(idx+1)
                )
            )
            fig_1.add_trace(
                go.Scatter(
                    x=self.dyn_timestamp,
                    y=self.measured_sc_mrp[:, idx],
                    mode='lines',
                    name='meas_MRP_{}'.format(idx+1)
                )
            )
        fig_1.update_layout(
            title='Spacecraft Body Attitude',
            xaxis_title='Time [s]',
            yaxis_title='Attitude [MRP]',
            legend=dict(title='Legend'),
        )
        fig_1.show()

        # spaecraft angular velocity
        fig_2 = go.Figure()
        for idx in range(3):
            fig_2.add_trace(
                go.Scatter(
                    x=self.dyn_timestamp,
                    y=self.true_sc_omega[:, idx],
                    mode='lines',
                    line=dict(dash='dash'),
                    name='true_omega_{}'.format(idx+1)
                )
            )
            fig_2.add_trace(
                go.Scatter(
                    x=self.dyn_timestamp,
                    y=self.measured_sc_omega[:, idx],
                    mode='lines',
                    name='meas_omega_{}'.format(idx+1)
                )
            )
        fig_2.update_layout(
            title='Spacecraft Body Rates',
            xaxis_title='Time [s]',
            yaxis_title='Rates [rad/s]',
            legend=dict(title='Legend'),
        )
        fig_2.show()

    def plot_tracking_errors(self):

        # mrp tracking error
        fig_1 = go.Figure()
        for idx in range(3):
            fig_1.add_trace(
                go.Scatter(
                    x=self.fsw_timestamp,
                    y=self.mrp_tracking_error[:, idx],
                    mode='lines',
                    name='MRP_err_{}'.format(idx+1)
                )
            )
        fig_1.update_layout(
            title='Attitude Tracking Error',
            xaxis_title='Time [s]',
            yaxis_title='Attitude Error [MRP]',
            legend=dict(title='Legend'),
        )
        fig_1.show()

        # omega tracking error
        fig_2 = go.Figure()
        for idx in range(3):
            fig_2.add_trace(
                go.Scatter(
                    x=self.fsw_timestamp,
                    y=self.omega_tracking_error[:, idx],
                    mode='lines',
                    name='omega_err_{}'.format(idx+1)
                )
            )
        fig_2.update_layout(
            title='Omega Tracking Error',
            xaxis_title='Time [s]',
            yaxis_title='Omega Error [rad/s]',
            legend=dict(title='Legend'),
        )
        fig_2.show()

        # angular atittude error
        fig_3 = go.Figure()
        fig_3.add_trace(
            go.Scatter(
                x=self.fsw_timestamp,
                y=self.angular_attitude_error,
                mode='lines',
                name='angular_err_{}'.format(idx+1)
            )
        )
        fig_3.update_layout(
            title='Angular Erorr',
            xaxis_title='Time [min]',
            yaxis_title='Angular Error [deg]',
            legend=dict(title='Legend'),
        )
        fig_3.show()

    def plot_control_signals(self):

        # control signals
        fig = go.Figure()
        for idx in range(3):
            fig.add_trace(
                go.Scatter(
                    x=self.fsw_timestamp,
                    y=self.feedback_control_torque_signal[:, idx],
                    mode='lines',
                    line=dict(dash='dash'),
                    name='feedback_ctrl_signal_{}'.format(idx+1)
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.fsw_timestamp,
                    y=self.mixed_control_torque_signal[:, idx],
                    mode='lines',
                    name='mixed_ctrl_signal_{}'.format(idx+1)
                )
            )
        fig.update_layout(
            title='Control Torque Signals',
            xaxis_title='Time [s]',
            yaxis_title='Control Torque Signal [Nm]',
            legend=dict(title='Legend'),
        )
        fig.show()

    def plot_rw_torques(self):
        
        # reaction wheels torques
        fig = go.Figure()
        for idx in range(len(self.rw_torques[0])):
            fig.add_trace(
                go.Scatter(
                    x=self.fsw_timestamp,
                    y=self.rw_mapping_control_torque_signal[:, idx],
                    mode='lines',
                    line=dict(dash='dash'),
                    name='ctrl_signal_{}'.format(idx+1)
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.dyn_timestamp,
                    y=self.rw_torques[:, idx],
                    mode='lines',
                    name='rw_torque_{}'.format(idx+1)
                )
            )
        fig.update_layout(
            title='Reaction Wheels Torques',
            xaxis_title='Time [s]',
            yaxis_title='Torques [Nm]',
            legend=dict(title='Legend'),
        )
        fig.show()

    def plot_rw_speeds(self):

        # reaction wheels speeds
        fig = go.Figure()
        for idx in range(len(self.rw_speeds[0])):
            fig.add_trace(
                go.Scatter(
                    x=self.dyn_timestamp,
                    y=self.rw_speeds[:, idx],
                    mode='lines',
                    name='rw_speed_{}'.format(idx+1)
                )
            )
        fig.update_layout(
            title='Reaction Wheels Speeds',
            xaxis_title='Time [s]',
            yaxis_title='Speed [rad/s]',
            legend=dict(title='Legend'),
        )
        fig.show()

    def get_observation(self):

        return self.obs_model_map[self.obs_model]()

    def _get_obs_model_1(self):

        observation = np.concatenate(
            (
                self.mrp_tracking_error[-1],
                self.omega_tracking_error[-1],
                self.feedback_control_torque_signal[-1]
            ),
            axis=None
        )

        return observation

    def _get_obs_model_2(self):

        observation = np.concatenate(
            (
                self.mrp_tracking_error[-1],
                self.omega_tracking_error[-1],
                self.feedback_control_torque_signal[-1],
                self.rl_agent_actions[-1],
            ),
            axis=None
        )

        return observation


