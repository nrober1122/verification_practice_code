import platform

import numpy as np

if platform.system() == "Darwin":
    import matplotlib

    matplotlib.use("MACOSX")
import os
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import torch
from matplotlib import colormaps

dir_path = os.path.dirname(os.path.realpath(__file__))


class Dynamics:
    def __init__(
        self,
        At,
        bt,
        ct,
        u_limits=None,
        dt=1.0,
        c=None,
        sensor_noise=None,
        process_noise=None,
        x_limits=None,
    ):
        # State dynamics
        self.At = At
        self.bt = bt
        self.ct = ct
        self.num_states, self.num_inputs = bt.shape

        # Store dynamics matrices as jnp arrays too
        self.At_jnp = jnp.asarray(At)
        self.bt_jnp = jnp.asarray(bt)
        self.ct_jnp = jnp.asarray(ct)

        # Store dynamics matrices as torch tensors too
        self.At_torch = torch.Tensor(At).transpose(0, 1)
        self.bt_torch = torch.Tensor(bt).transpose(0, 1)
        self.ct_torch = torch.Tensor(ct)

        # Observation Dynamics and Noise
        if c is None:
            c = np.eye(self.num_states)
        self.c = c
        self.num_outputs = self.c.shape[0]
        self.sensor_noise = sensor_noise
        self.process_noise = process_noise

        # Min/max control inputs
        self.u_limits = u_limits

        self.x_limits = x_limits
        self.dt = dt

        self.name = self.__class__.__name__

    def control_nn(self, x, model):
        if x.ndim == 1:
            batch_x = np.expand_dims(x, axis=0)
        else:
            batch_x = x
        us = model.forward(torch.Tensor(batch_x)).data.numpy()
        return us

    def observe_step(self, xs):
        obs = np.dot(xs, self.c.T)
        if self.sensor_noise is not None:
            noise = np.random.uniform(
                low=self.sensor_noise[:, 0],
                high=self.sensor_noise[:, 1],
                size=xs.shape,
            )
            obs += noise
        return obs

    def dynamics_step(self, xs, us):
        raise NotImplementedError

    def dynamics_step_jnp(self, xs, us):
        raise NotImplementedError

    def dynamics_step_torch(self, xs, us):
        raise NotImplementedError

    def tmax_to_num_timesteps(self, t_max):
        num_timesteps = round(t_max / self.dt)
        return num_timesteps

    def num_timesteps_to_tmax(self, num_timesteps):
        return num_timesteps * self.dt

    def colors(self, t_max):
        return [colormaps[self.cmap_name](i) for i in range(t_max + 1)]


    def get_state_and_next_state_samples(
        self,
        input_constraint,
        t_max=1,
        num_samples=1000,
        controller="mpc",
        output_constraint=None,
    ):
        xs, us = self.collect_data(
            t_max,
            input_constraint,
            num_samples,
            controller=controller,
            merge_cols=False,
        )

        return xs[:, 0, :], xs[:, 1, :]


class ContinuousTimeDynamics(Dynamics):
    def __init__(
        self,
        At,
        bt,
        ct,
        u_limits=None,
        dt=1.0,
        c=None,
        sensor_noise=None,
        process_noise=None,
        x_limits=None,
    ):
        super().__init__(
            At, bt, ct, u_limits, dt, c, sensor_noise, process_noise, x_limits
        )
        self.continuous_time = True

    def dynamics(self, xs, us):
        if isinstance(xs, np.ndarray):  # For tracking MC samples
            xdot = (np.dot(self.At, xs.T) + np.dot(self.bt, us.T)).T + self.ct
            if self.process_noise is not None:
                noise = np.random.uniform(
                    low=self.process_noise[:, 0],
                    high=self.process_noise[:, 1],
                    size=xs.shape,
                )
                xdot += noise
        else:  # For solving LP
            xdot = self.At @ xs + self.bt @ us + self.ct
        return xdot

    def dynamics_step(self, xs, us):
        xs_t1 = xs + self.dt * self.dynamics(xs, us)
        return xs_t1

    def dynamics_jnp(self, xs, us):
        xdot = (
            jnp.dot(self.At_jnp, xs.T).T
            + jnp.dot(self.bt_jnp, us.T).T
            + self.ct_jnp
        )
        return xdot

    def dynamics_step_jnp(self, xs, us):
        xs_t1 = xs + self.dt * self.dynamics_jnp(xs, us)
        return xs_t1


class DiscreteTimeDynamics(Dynamics):
    def __init__(
        self,
        At,
        bt,
        ct,
        u_limits=None,
        dt=1.0,
        c=None,
        sensor_noise=None,
        process_noise=None,
        x_limits=None,
    ):
        super().__init__(
            At, bt, ct, u_limits, dt, c, sensor_noise, process_noise, x_limits
        )
        self.continuous_time = False

    def dynamics_step(self, xs, us):
        if isinstance(xs, np.ndarray):  # For tracking MC samples
            xs_t1 = (np.dot(self.At, xs.T) + np.dot(self.bt, us.T)).T + self.ct
            if self.process_noise is not None:
                noise = np.random.uniform(
                    low=self.process_noise[:, 0],
                    high=self.process_noise[:, 1],
                    size=xs.shape,
                )
                xs_t1 += noise
            # if self.x_limits is not None and isinstance(xs, np.ndarray):
            #     for key in self.x_limits:
            #         xs_t1[:, key] = np.minimum(
            #             xs_t1[:, key], self.x_limits[key][1]
            #         )
            #         xs_t1[:, key] = np.maximum(
            #             xs_t1[:, key], self.x_limits[key][0]
            #         )

        else:  # For solving LP
            xs_t1 = self.At @ xs + self.bt @ us + self.ct
            # if self.x_limits is not None and isinstance(xs, np.ndarray):
            #     for key in self.x_limits:
            #         xs_t1[:, key] = np.minimum(
            #             xs_t1[:, key], self.x_limits[key][1]
            #         )
            #         xs_t1[:, key] = np.maximum(
            #             xs_t1[:, key], self.x_limits[key][0]
            #         )

        return xs_t1

    def dynamics_step_jnp(self, xs, us):
        return (
            jnp.dot(self.At_jnp, xs.T).T
            + jnp.dot(self.bt_jnp, us.T).T
            + self.ct_jnp
        )

    def dynamics_step_torch(self, xs, us):
        return (
            torch.mm(xs, self.At_torch)
            + torch.mm(us, self.bt_torch)
            + self.ct_torch
        )


# if __name__ == "__main__":
#     from nfl_veripy.dynamics.DoubleIntegrator import DoubleIntegrator

#     dynamics = DoubleIntegrator()
#     init_state_range = np.array(
#         [
#             # (num_inputs, 2)
#             [2.5, 3.0],  # x0min, x0max
#             [-0.25, 0.25],  # x1min, x1max
#         ]
#     )
#     xs, us = dynamics.collect_data(
#         t_max=10,
#         input_constraint=constraints.LpConstraint(
#             p=np.inf, range=init_state_range
#         ),
#         num_samples=2420,
#     )
#     print(xs.shape, us.shape)
#     system = "double_integrator"
#     with open(
#         dir_path + "/../../datasets/{}/xs.pkl".format(system), "wb"
#     ) as f:
#         pickle.dump(xs, f)
#     with open(
#         dir_path + "/../../datasets/{}/us.pkl".format(system), "wb"
#     ) as f:
#         pickle.dump(us, f)

    # from nfl_veripy.utils.nn import load_model

    # # dynamics = DoubleIntegrator()
    # # init_state_range = np.array([ # (num_inputs, 2)
    # #                   [2.5, 3.0], # x0min, x0max
    # #                   [-0.25, 0.25], # x1min, x1max
    # # ])
    # # controller = load_model(name='double_integrator_mpc')

    # dynamics = QuadrotorOutputFeedback()
    # init_state_range = np.array([ # (num_inputs, 2)
    #               [4.65,4.65,2.95,0.94,-0.01,-0.01], # x0min, x0max
    #               [4.75,4.75,3.05,0.96,0.01,0.01] # x1min, x1max
    # ]).T
    # goal_state_range = np.array([
    #                       [3.7,2.5,1.2],
    #                       [4.1,3.5,2.6]
    # ]).T
    # controller = load_model(name='quadrotor')
    # t_max = 15*dynamics.dt
    # input_constraint = LpConstraint(range=init_state_range, p=np.inf)
    # dynamics.show_samples(
    #     t_max,
    #     input_constraint,
    #     save_plot=False,
    #     ax=None,
    #     show=True,
    #     controller=controller,
    # )
