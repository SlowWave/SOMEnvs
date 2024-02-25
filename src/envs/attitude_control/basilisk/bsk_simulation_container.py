import os
import tomli
import math
import numpy as np
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import attTrackingError, inertial3D, mrpFeedback, rwMotorTorque, rwMotorVoltage
from Basilisk.simulation import spacecraft, extForceTorque, simpleNav, reactionWheelStateEffector, motorVoltageInterface
from Basilisk.utilities import SimulationBaseClass, macros, unitTestSupport, simIncludeRW
from .custom_feedback_controller import CustomFeedbackController
from .rl_controller_interface import RLControllerInterface
from .external_disturbances_interface import ExternalDisturbancesInterface


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


class BSKSimulationContainer(SimulationBaseClass.SimBaseClass):
    def __init__(self):
        super(BSKSimulationContainer, self).__init__()

        # store processes and tasks ID strings
        self.dyn_process_str = CFG["basilisk"]["dyn_process_str"]
        self.fsw_process_str = CFG["basilisk"]["fsw_process_str"]
        self.dyn_task_str = CFG["basilisk"]["dyn_task_str"]
        self.fsw_task_str = CFG["basilisk"]["fsw_task_str"]

        # define tasks and processes attributes
        self.dyn_process = None
        self.fsw_process = None
        self.dyn_task = None
        self.fsw_task = None

        # define simulation objects attributes
        self.spacecraft_body = None
        self.external_disturbances = None
        self.external_disturbances_interface = None
        self.rw_factory = None
        self.rw_state_effector = None
        self.rw_voltage_interface = None
        self.rw_torque_control_map = None
        self.rw_torque_voltage_map = None
        self.navigation_sensor = None
        self.inertial_reference = None
        self.attitude_tracking_err = None
        self.feedback_controller = None
        self.rl_controller_interface = None

        # define simulation logs attributes
        self.spacecraft_body_log = None
        self.rw_logs = None
        self.rw_state_effector_log = None
        self.rw_voltage_interface_log = None
        self.rw_torque_control_map_log = None
        self.rw_torque_voltage_map_log = None
        self.navigation_sensor_log = None
        self.attitude_tracking_err_log = None
        self.feedback_controller_log = None
        self.rl_controller_log = None

        # define simulaiton states dictionary
        self.simulation_data = dict()

    def _create_processes(self):

        # instantiate dynamic and flight software processes
        self.dyn_process = self.CreateNewProcess(self.dyn_process_str)
        self.fsw_process = self.CreateNewProcess(self.fsw_process_str)

    def _create_tasks(self):

        # get tasks step frequency
        dyn_step_freq = CFG["basilisk"]["dyn_step_frequency"]
        fsw_step_freq = CFG["basilisk"]["fsw_step_frequency"]

        # define tasks time step
        dyn_time_step = macros.sec2nano(1 / dyn_step_freq)
        fsw_time_step = macros.sec2nano(1 / fsw_step_freq)

        # create dynamic and flight software tasks
        self.dyn_task = self.CreateNewTask(self.dyn_task_str, dyn_time_step)
        self.fsw_task = self.CreateNewTask(self.fsw_task_str, fsw_time_step)

        # add tasks to processes
        self.dyn_process.addTask(self.dyn_task)
        self.fsw_process.addTask(self.fsw_task)

    def _add_spacecraft(self):
        
        # instantiate spacecraft object
        self.spacecraft_body = spacecraft.Spacecraft()
        self.spacecraft_body.ModelTag = CFG["basilisk"]["spacecraft_body"]["model_tag_str"]

        # get spacecraft mass
        if CFG["basilisk"]["spacecraft_body"]["use_random_mass"]:
            
            sc_mass = np.random.uniform(
                CFG["basilisk"]["spacecraft_body"]["random_mass_min"],
                CFG["basilisk"]["spacecraft_body"]["random_mass_max"],
            )
        else:
            sc_mass = CFG["basilisk"]["spacecraft_body"]["mass"]

        # get spacecraft moments of inertia
        if CFG["basilisk"]["spacecraft_body"]["use_random_moi"]:

            sc_moi = list()
            
            for idx in range(3):
                sc_moi.append(np.random.uniform(
                    CFG["basilisk"]["spacecraft_body"]["random_moi_min"][idx],
                    CFG["basilisk"]["spacecraft_body"]["random_moi_max"][idx],
                ))

        else:
            sc_moi = CFG["basilisk"]["spacecraft_body"]["moi"]

        # get spacecraft products of inertia
        if CFG["basilisk"]["spacecraft_body"]["use_random_poi"]:      

            sc_poi = list()
            
            for idx in range(3):
                sc_poi.append(np.random.uniform(
                    CFG["basilisk"]["spacecraft_body"]["random_poi_min"][idx],
                    CFG["basilisk"]["spacecraft_body"]["random_poi_max"][idx],
                ))

        else:
            sc_poi = CFG["basilisk"]["spacecraft_body"]["poi"]

        # define tensor of inertia
        sc_inertia = [sc_moi[0], sc_poi[0], sc_poi[1], sc_poi[0], sc_moi[1], sc_poi[2], sc_poi[1], sc_poi[2], sc_moi[2]]

        # get spacecraft initial mrp attitude
        # sc_mrp_init = CFG["basilisk"]["spacecraft_body"]["initial_mrp"]
        # set initial spacecraft mrp atittude to [0, 0, 0]
        sc_mrp_init = [0., 0., 0.]

        # get spacecraft initial angular velocity
        if CFG["basilisk"]["spacecraft_body"]["use_random_angular_velocity"]:      

            sc_omega_init = list()
            
            for idx in range(3):
                sc_omega_init.append(np.random.uniform(
                    CFG["basilisk"]["spacecraft_body"]["random_angular_velocity_max"][idx],
                    CFG["basilisk"]["spacecraft_body"]["random_angular_velocity_min"][idx],
                ))

        else:
            sc_omega_init = CFG["basilisk"]["spacecraft_body"]["initial_angular_velocity"]

        # set spacecraft parameters
        self.spacecraft_body.hub.mHub = sc_mass
        self.spacecraft_body.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(sc_inertia)
        self.spacecraft_body.hub.sigma_BNInit = [sc_mrp_init[0], sc_mrp_init[1], sc_mrp_init[2]]
        self.spacecraft_body.hub.omega_BN_BInit = [sc_omega_init[0], sc_omega_init[1], sc_omega_init[2]]

        # add spacecraft body to dynamic task
        self.AddModelToTask(self.dyn_task_str, self.spacecraft_body, 5)

    def _add_external_disturbances(self):

        # instantiate external disturbances object
        self.external_disturbances = extForceTorque.ExtForceTorque()
        self.external_disturbances.ModelTag = CFG["basilisk"]["external_disturbances"]["model_tag_str"]

        # link external disturbances to spacecraft body
        self.spacecraft_body.addDynamicEffector(self.external_disturbances)

        # add external disturbances object to dynamic task
        self.AddModelToTask(self.dyn_task_str, self.external_disturbances)

    def _add_external_disturbances_interface(self):

        # instantiate external disturbances interface object
        self.external_disturbances_interface = ExternalDisturbancesInterface()
        self.external_disturbances.ModelTag = CFG["basilisk"]["external_disturbances_interface"]["model_tag_str"]

        # add external disturbances interface object to dynamic task
        self.AddModelToTask(self.dyn_task_str, self.external_disturbances_interface)

    def _add_actuators(self):

        # instantiate reaction wheels factory object
        self.rw_factory = simIncludeRW.rwFactory()

        # define reaction wheels distribution matrix
        if CFG["basilisk"]["reaction_wheels"]["configuration"] == "orthogonal":
            distribution_matrix = [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]

        elif CFG["basilisk"]["reaction_wheels"]["configuration"] == "pyramidal":
            distribution_matrix = [
                [math.sqrt(3) / 3, - math.sqrt(3) / 3, - math.sqrt(3) / 3, math.sqrt(3) / 3],
                [math.sqrt(3) / 3, math.sqrt(3) / 3, - math.sqrt(3) / 3, - math.sqrt(3) / 3],
                [math.sqrt(3) / 3, math.sqrt(3) / 3, math.sqrt(3) / 3, math.sqrt(3) / 3]
            ]

        elif CFG["basilisk"]["reaction_wheels"]["configuration"] == "tetrahedral":
            alpha = math.radians(19.47)
            beta = math.radians(30)
            gamma = math.radians(60)

            distribution_matrix = [
                [math.cos(alpha), - math.cos(alpha) * math.cos(gamma), - math.cos(alpha) * math.cos(gamma), 0],
                [0, math.cos(alpha) * math.cos(beta), - math.cos(alpha) * math.cos(beta), 0],
                [- math.sin(alpha), - math.sin(alpha), - math.sin(alpha), 1],
            ]

        # get reaction wheels initial angular velocities
        rw_number = len(distribution_matrix[0])
        if CFG["basilisk"]["reaction_wheels"]["use_random_angular_velocity"]:      

            rw_omega_init = list()
            
            for idx in range(rw_number):
                rw_omega_init.append(np.random.uniform(
                    CFG["basilisk"]["reaction_wheels"]["random_angular_velocity_max"][idx],
                    CFG["basilisk"]["reaction_wheels"]["random_angular_velocity_min"][idx],
                ))

        else:
            rw_omega_init = CFG["basilisk"]["reaction_wheels"]["initial_angular_velocity"]

        # create reaction wheels objects
        for idx in range(rw_number):
            distribution_vector = [distribution_matrix[0][idx], distribution_matrix[1][idx], distribution_matrix[2][idx]]
            self.rw_factory.create(
                rwType=CFG["basilisk"]["reaction_wheels"]["model"],
                gsHat_B=distribution_vector,
                maxMomentum=float(CFG["basilisk"]["reaction_wheels"]["angular_momentum_max"]),
                u_max=float(CFG["basilisk"]["reaction_wheels"]["torque_max"]),
                Omega=float(rw_omega_init[idx]) * 30 / np.pi,
                RWModel=messaging.BalancedWheels,
            )

        # instantiate reaction wheels state effector object
        self.rw_state_effector = reactionWheelStateEffector.ReactionWheelStateEffector()
        self.rw_state_effector.ModelTag = CFG["basilisk"]["reaction_wheels"]["state_effector_model_tag_str"]

        # link reaction wheels cluster to spacecraft body
        self.rw_factory.addToSpacecraft(self.spacecraft_body.ModelTag, self.rw_state_effector, self.spacecraft_body)

        # add reaction wheels state effector object to dynamic task
        self.AddModelToTask(self.dyn_task_str, self.rw_state_effector, 6)

        # instantiate reaction wheels voltage interface object
        self.rw_voltage_interface = motorVoltageInterface.MotorVoltageInterface()
        self.rw_voltage_interface.ModelTag = CFG["basilisk"]["reaction_wheels"]["voltage_interface_model_tag_str"]

        # define reaction wheels voltage interface paramenters
        rw_torque_max = float(CFG["basilisk"]["reaction_wheels"]["torque_max"])
        rw_torque_bias = float(CFG["basilisk"]["reaction_wheels"]["torque_bias"])
        rw_voltage_max = float(CFG["basilisk"]["reaction_wheels"]["voltage_max"])
        rw_voltage_error = float(CFG["basilisk"]["reaction_wheels"]["voltage_error"])

        # set reaction wheels voltage interface parameters
        self.rw_voltage_interface.setGains(np.array([rw_torque_max / rw_voltage_max] * rw_number))
        self.rw_voltage_interface.setScaleFactors(np.array([1 + rw_voltage_error] * rw_number))
        self.rw_voltage_interface.setBiases(np.array([rw_torque_bias] * rw_number))

        # add reaction wheels voltage interface object to dynamic task
        self.AddModelToTask(self.dyn_task_str, self.rw_voltage_interface, 7)

        # instantiate reaction wheels torque control mapping object
        self.rw_torque_control_map = rwMotorTorque.rwMotorTorque()
        self.rw_torque_control_map.ModelTag = CFG["basilisk"]["reaction_wheels"]["torque_control_map_model_tag_str"]

        # set reaction wheels torque control mapping parameters
        self.rw_torque_control_map.controlAxes_B = [1, 0, 0, 0, 1, 0, 0, 0, 1]

        # add reaction wheels torque control mapping object to flight software task
        self.AddModelToTask(self.fsw_task_str, self.rw_torque_control_map, 9)

        # instantiate reaction wheels torque to voltage mapping object
        self.rw_torque_voltage_map = rwMotorVoltage.rwMotorVoltage()
        self.rw_torque_voltage_map.ModelTag = CFG["basilisk"]["reaction_wheels"]["torque_voltage_map_model_tag_str"]

        # set reaction wheels torque to voltage mapping parameters
        self.rw_torque_voltage_map.VMin = 0
        self.rw_torque_voltage_map.VMax = rw_voltage_max

        # add reaction wheels torque to voltage mapping object to flight software task
        self.AddModelToTask(self.fsw_task_str, self.rw_torque_voltage_map, 8)

    def _add_navigation_sensor(self):

        # instantiate navigation sensor object
        self.navigation_sensor = simpleNav.SimpleNav()
        self.navigation_sensor.ModelTag = CFG["basilisk"]["navigation_sensor"]["model_tag_str"]

        if CFG["basilisk"]["navigation_sensor"]["use_measurement_errors"]:

            # define navigation sensor parameters
            attitude_meas_std = CFG["basilisk"]["navigation_sensor"]["attitude_meas_std"] / 180
            rate_meas_std = CFG["basilisk"]["navigation_sensor"]["rate_meas_std"]
            attitude_error_max = CFG["basilisk"]["navigation_sensor"]["attitude_error_max"] / 180
            rate_error_max = CFG["basilisk"]["navigation_sensor"]["rate_error_max"]

            p_matrix = np.diag(
                [
                    0., 0., 0.,
                    0., 0., 0.,
                    attitude_meas_std, attitude_meas_std, attitude_meas_std,
                    rate_meas_std, rate_meas_std, rate_meas_std,
                    0., 0., 0.,
                    0., 0., 0.
                ]
            )

            error_bounds = [
                0., 0., 0.,
                0., 0., 0.,
                attitude_error_max, attitude_error_max, attitude_error_max,
                rate_error_max, rate_error_max, rate_error_max,
                0., 0., 0.,
                0., 0., 0.
            ]

            # set navigation sensors parameters
            self.navigation_sensor.walkBounds = error_bounds
            self.navigation_sensor.PMatrix = p_matrix

        # add navigation sensor object to dynamic task
        self.AddModelToTask(self.dyn_task_str, self.navigation_sensor, 4)

    def _add_inertial_reference(self):

        # instantiate inertial reference object
        self.inertial_reference = inertial3D.inertial3D()
        self.inertial_reference.ModelTag = CFG["basilisk"]["inertial_reference"]["model_tag_str"]

        # add inertial reference object to flight software task
        self.AddModelToTask(self.fsw_task_str, self.inertial_reference, 3)

    def _add_attitude_tracking_error(self):

        # instantiate attitude tracking error object
        self.attitude_tracking_err = attTrackingError.attTrackingError()
        self.attitude_tracking_err.ModelTag = CFG["basilisk"]["attitude_tracking_err"]["model_tag_str"]

        # add attitude tracking error object to flight software task
        self.AddModelToTask(self.fsw_task_str, self.attitude_tracking_err, 2)

    def _add_feedback_controller(self):

        # instantiate feedback controller object
        self.feedback_controller = CustomFeedbackController()
        self.feedback_controller.ModelTag = CFG["basilisk"]["feedback_controller"]["model_tag_str"]
        
        # define feedback controller parameters
        k_p = CFG["basilisk"]["feedback_controller"]["k_p"]
        k_d = CFG["basilisk"]["feedback_controller"]["k_d"]

        # set feedback controller parameters
        self.feedback_controller.set_gains(k_p, k_d)

        # add feedback controller object to flight software task
        self.AddModelToTask(self.fsw_task_str, self.feedback_controller, 1)

    def _add_rl_controller_inteface(self):

        # instantiate reinforcement learning controller interface object
        self.rl_controller_interface = RLControllerInterface()
        self.rl_controller_interface.ModelTag = CFG["basilisk"]["rl_controller_interface"]["model_tag_str"]

        # add reinforcement learning controller interface object to flight software
        self.AddModelToTask(self.fsw_task_str, self.rl_controller_interface, 10)

    def _connect_data_msgs(self):

        # create flight software reaction wheels configuration message
        rw_config_data_msg = self.rw_factory.getConfigMessage()

        # connect messages to modules
        self.external_disturbances_interface.sc_state_in_msg.subscribeTo(self.spacecraft_body.scStateOutMsg)
        self.external_disturbances.cmdTorqueInMsg.subscribeTo(self.external_disturbances_interface.cmd_torque_out_msg)
        self.navigation_sensor.scStateInMsg.subscribeTo(self.spacecraft_body.scStateOutMsg)
        self.attitude_tracking_err.attNavInMsg.subscribeTo(self.navigation_sensor.attOutMsg)
        self.attitude_tracking_err.attRefInMsg.subscribeTo(self.inertial_reference.attRefOutMsg)
        self.feedback_controller.guidance_in_msg.subscribeTo(self.attitude_tracking_err.attGuidOutMsg)
        self.rl_controller_interface.cmd_torque_in_msg.subscribeTo(self.feedback_controller.cmd_torque_out_msg)
        self.rw_torque_control_map.vehControlInMsg.subscribeTo(self.rl_controller_interface.cmd_torque_out_msg)
        self.rw_torque_control_map.rwParamsInMsg.subscribeTo(rw_config_data_msg)
        self.rw_torque_voltage_map.torqueInMsg.subscribeTo(self.rw_torque_control_map.rwMotorTorqueOutMsg)
        self.rw_torque_voltage_map.rwParamsInMsg.subscribeTo(rw_config_data_msg)
        self.rw_voltage_interface.motorVoltageInMsg.subscribeTo(self.rw_torque_voltage_map.voltageOutMsg)
        self.rw_state_effector.rwMotorCmdInMsg.subscribeTo(self.rw_voltage_interface.motorTorqueOutMsg)

    def _setup_data_logs(self):

        # setup data logging
        self.spacecraft_body_log = self.spacecraft_body.scStateOutMsg.recorder()
        self.navigation_sensor_log = self.navigation_sensor.attOutMsg.recorder()
        self.attitude_tracking_err_log = self.attitude_tracking_err.attGuidOutMsg.recorder()
        self.feedback_controller_log = self.feedback_controller.cmd_torque_out_msg.recorder()
        self.mixed_controller_log = self.rl_controller_interface.cmd_torque_out_msg.recorder()
        self.rw_torque_control_map_log = self.rw_torque_control_map.rwMotorTorqueOutMsg.recorder()
        self.rw_torque_voltage_map_log = self.rw_torque_voltage_map.voltageOutMsg.recorder()
        self.rw_voltage_interface_log = self.rw_voltage_interface.motorTorqueOutMsg.recorder()
        self.rw_state_effector_log = self.rw_state_effector.rwSpeedOutMsg.recorder()
        self.rw_logs = list()
        for idx in range(self.rw_factory.getNumOfDevices()):
            self.rw_logs.append(self.rw_state_effector.rwOutMsgs[idx].recorder())

        # add data logging to dynamic and fligth software tasks
        self.AddModelToTask(self.dyn_task_str, self.spacecraft_body_log)
        self.AddModelToTask(self.dyn_task_str, self.navigation_sensor_log)
        self.AddModelToTask(self.fsw_task_str, self.attitude_tracking_err_log)
        self.AddModelToTask(self.fsw_task_str, self.feedback_controller_log)
        self.AddModelToTask(self.fsw_task_str, self.mixed_controller_log)
        self.AddModelToTask(self.fsw_task_str, self.rw_torque_control_map_log)
        self.AddModelToTask(self.fsw_task_str, self.rw_torque_voltage_map_log)
        self.AddModelToTask(self.dyn_task_str, self.rw_voltage_interface_log)
        self.AddModelToTask(self.dyn_task_str, self.rw_state_effector_log)
        for idx in range(self.rw_factory.getNumOfDevices()):
            self.AddModelToTask(self.dyn_task_str, self.rw_logs[idx])

    def _clear_data_logs(self):

        # clear data logs so that only new data are recorded
        self.spacecraft_body_log.clear()
        self.rw_state_effector_log.clear()
        self.rw_voltage_interface_log.clear()
        self.rw_torque_control_map_log.clear()
        self.rw_torque_voltage_map_log.clear()
        self.navigation_sensor_log.clear()
        self.attitude_tracking_err_log.clear()
        self.feedback_controller_log.clear()
        self.mixed_controller_log.clear()
        for idx in range(self.rw_factory.getNumOfDevices()):
            self.rw_logs[idx].clear()

    def _init_simulation_data_dict(self):

        # initialize simulation data dictionary
        self.simulation_data["dyn_timestamp"] = None
        self.simulation_data["fsw_timestamp"] = None
        self.simulation_data["true_sc_mrp"] = None
        self.simulation_data["true_sc_omega"] = None
        self.simulation_data["true_sc_omega_dot"] = None
        self.simulation_data["measured_sc_mrp"] = None
        self.simulation_data["measured_sc_omega"] = None
        self.simulation_data["mrp_tracking_error"] = None
        self.simulation_data["omega_tracking_error"] = None
        self.simulation_data["angular_attitude_error"] = None
        self.simulation_data["feedback_control_torque_signal"] = None
        self.simulation_data["mixed_control_torque_signal"] = None
        self.simulation_data["rw_mapping_control_torque_signal"] = None
        self.simulation_data["rw_control_voltage_reference_signal"] = None
        self.simulation_data["rw_control_torque_signal"] = None
        self.simulation_data["rw_speeds"] = None
        self.simulation_data["rw_torques"] = None

    def _collect_simulation_data(self):

        rw_number = self.rw_factory.getNumOfDevices()

        self.simulation_data["dyn_timestamp"] = self.spacecraft_body_log.times() * macros.NANO2SEC
        self.simulation_data["fsw_timestamp"] = self.feedback_controller_log.times() * macros.NANO2SEC
        self.simulation_data["true_sc_mrp"] = self.spacecraft_body_log.sigma_BN
        self.simulation_data["true_sc_omega"] = self.spacecraft_body_log.omega_BN_B
        self.simulation_data["true_sc_omega_dot"] = self.spacecraft_body_log.omegaDot_BN_B
        self.simulation_data["measured_sc_mrp"] = self.navigation_sensor_log.sigma_BN
        self.simulation_data["measured_sc_omega"] = self.navigation_sensor_log.omega_BN_B
        self.simulation_data["mrp_tracking_error"] = self.attitude_tracking_err_log.sigma_BR
        self.simulation_data["omega_tracking_error"] = self.attitude_tracking_err_log.omega_BR_B
        self.simulation_data["feedback_control_torque_signal"] = self.feedback_controller_log.torqueRequestBody
        self.simulation_data["mixed_control_torque_signal"] = self.mixed_controller_log.torqueRequestBody
        self.simulation_data["rw_mapping_control_torque_signal"] = self.rw_torque_control_map_log.motorTorque
        self.simulation_data["rw_control_voltage_reference_signal"] = self.rw_torque_voltage_map_log.voltage
        self.simulation_data["rw_control_torque_signal"] = self.rw_voltage_interface_log.motorTorque
        self.simulation_data["rw_speeds"] = self.rw_state_effector_log.wheelSpeeds[:, :rw_number]

        # stack reaction wheels torque data
        rw_torques_list = list()
        for idx in range(rw_number):
            rw_torques_list.append(self.rw_logs[idx].u_current)
        self.simulation_data["rw_torques"] = np.stack((rw_torques_list)).T

        # compute angular attitude error from mrp tracking error
        phi = np.sum(self.simulation_data["mrp_tracking_error"] ** 2, axis=1)
        discriminant = 4 * phi ** 2 - 4 * (1 + phi) * (phi - 1)
        angular_error = np.rad2deg(np.arccos(np.abs((2 * phi - np.sqrt(np.abs(discriminant))) / (2 * (1 + phi)))) * 2)
        self.simulation_data["angular_attitude_error"] = angular_error

    def reset(self):
        
        # create simulation tasks and procesess
        self._create_processes()
        self._create_tasks()

        # add objects to the simulation
        self._add_spacecraft()
        self._add_external_disturbances()
        self._add_external_disturbances_interface()
        self._add_actuators()
        self._add_navigation_sensor()
        self._add_inertial_reference()
        self._add_attitude_tracking_error()
        self._add_feedback_controller()
        self._add_rl_controller_inteface()

        # setup message connections and data logging
        self._connect_data_msgs()
        self._setup_data_logs()

    def initialize_simulation(self):

        # initilize simulation
        self.InitializeSimulation()

    def step(self, time_horizon_secs):

        # perform simulation step
        time_horizon_nanos = macros.sec2nano(time_horizon_secs)
        self._clear_data_logs()
        self.ConfigureStopTime(time_horizon_nanos)
        self.ExecuteSimulation()
        self._collect_simulation_data()

    def set_attitude_reference(self, attitude_reference):

        # set a new attitude reference
        self.inertial_reference.sigma_R0N = attitude_reference
        
    def set_torque_disturbances(self, disturbances_dict):

        # set external torque disturbances
        self.external_disturbances_interface.set_torques_dict(disturbances_dict)

    def apply_rl_agent_action(self, action):

        # set reinforcement learning agent control action
        self.rl_controller_interface.set_rl_agent_action(action)

    def get_simulation_data(self):

        # retrieve simulation data dictionary
        return self.simulation_data


