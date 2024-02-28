# AttitudeControlEnv

## Environment Description

The objective of the [AttitudeControlEnv](../../../src/envs/attitude_control/) environment is to simulate the spacecraft attitude control problem using a combination of traditional feedback control algorithms and reinforcement learning (RL) techniques. The environment is built upon Gymnasium and Basilisk, providing high-fidelity and computationally efficient simulations along with a standardize interface for RL frameworks such as Stable-Baselines3.

The following illustration represent the interaction between the environment components during a simulation step.

![ex_image](../../images/envs_attitude_control_1.svg)

At each simulation environment step:
1. Action $a_t$ generated by `RL agent` is stored inside `Storage` and provided to `Basilisk Simulation`.
2. `Basilisk Simulation` propagates the simulation states from $s_t$ to $s_{t+1}$ which are then stored inside `Storage`.
3. Reward $r_{t+1}$ is computed by `Reward Function` based on the data provided by `Storage`.
4. Both reward $r_{t+1}$ provided by `Reward Function` and observation $o_{t+1}$ provided by `Storage` could be normalized ($\tilde{r}_{t+1}$, $\tilde{o}_{t+1}$) by `Normalizator` and then returned to `RL Agent`.


## Basilisk Simulation

In this environment, Basilisk serves as the underlying simulation engine. The simulation layout is shown in the following illustration.

![ex_image](../../images/envs_attitude_control_2.svg)

`Basilisk Simulation` handles two separate simulation processes:
* `Flight Software Task` simulates the flight software algorithm modules:
    * `Inertial Reference` creates a reference attitude that points in fixed inertial direction.
    * `Attitude Tracking Erorr` generates atittude tracking errors relative to a moving reference frame.
    * `Feedback Controller` generates a body-frame control torque signal $u$ using the following control law: $u = - K_p \sigma_e - K_d \omega_e$ where:
        * $K_p$ and $K_d$ are constant parameters.
        * $\sigma_e$ and $\omega_e$ are the spacecraft angular and velocity errors.
    * `RL Agent Interface` serves as interface for `RL Agent` by combining the RL action $a$ with the control torque signal $u$.
    * `Torque Control Map` maps the mixed control torque signal to the available reaction wheels.

* `Dynamic Task` simulates the spacecraft dynamics evolution:
    * `Torque Voltage Map` computes the reaction wheels motor voltage.
    * `Voltage Interface` converts motor input voltages to reaction wheels torques.
    * `Reaction Wheels` simulates reaction wheels dynamics impacting on the spacecraft body.
    * `Disturbances Interface` computes disturbances torques acting on the spacecraft based on the data provided by `Disturbance Generator`.
    * `External Disturbances` provides a direct external torque on spacecraft body.
    * `Spacecraft Body` provides the spacecraft rigid body translational and rotation motion.
    * `Navigation Sensor` perturbs the truth spacecraft state away using a gauss-markov error model.

Separating the two processes enables them to be executed with different update frequencies.


Key features of the environment include:

* Simulation of spacecraft dynamics and kinematics.
* Implementation of a traditional feedback control system for attitude stabilization.
* Integration of a reinforcement learning agent for adaptive control.
* Support for various configurations and parameters to facilitate experimentation and research.

## Environment Customization

Customization is a fundamental aspect of this environment, empowering users to adapt the simulation setup to their research needs. Several aspects of the environment can be customized by updating specific sections of the source code and the [config.toml](../../../src/envs/attitude_control/configs/config.toml) file.



### Modify src


### Modify config.json

