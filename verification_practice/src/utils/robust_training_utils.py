import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
from ast import literal_eval
from itertools import product
from copy import deepcopy

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
import cl_systems
from cl_systems import ClosedLoopDynamics
import time

class ReachableSet:
    def __init__(self, t, ranges = None, partition_strategy = 'maintain', thread = 0, device='cpu') -> None:
        self.t = t
        
        if ranges is None:
            ranges = torch.tensor([[0, 0], [0, 0]], device=device)
        self.full_set = ranges
        self.subsets = {}
        self.partition_strategy = partition_strategy
        self.get_partitions()
        self.thread = thread
        self.recalculate = True
        self.device = device
        self.t_parent = -1
        self.symbolic = True
        self.populated = False

    def set_range(self, ranges):
        self.full_set = ranges

    def add_subset(self, ranges, index):
        self.subsets[index] = ReachableSet(self.t, ranges, thread=index)

    def get_thread(self, thread):
        if thread == 0 and self.subsets == {}:
            return self
        else:
            return self.subsets[thread]
        
    
    def set_partition_strategy(self, partition_strategy):
        if partition_strategy in ['maintain', 'consolidate'] or isinstance(partition_strategy, np.ndarray):
            self.partition_strategy = partition_strategy
        else:
            raise NotImplementedError
        
    def calculate_full_set(self):
        num_subsets = len(self.subsets)
        num_states = self.subsets[0].full_set.shape[0]
        subset_tensor = torch.zeros((num_subsets, num_states, 2), device=self.device)

        for i, subset in self.subsets.items():
            subset_tensor[i] = subset.full_set
        
        lb, _ = torch.min(subset_tensor[:, :, 0], dim=0)
        ub, _ = torch.max(subset_tensor[:, :, 1], dim=0)
        self.full_set = torch.vstack((lb, ub)).T.to(self.device)
        
    
    def get_partitions(self):
        num_partitions = self.partition_strategy
        
        if self.partition_strategy == 'maintain' or self.partition_strategy == 'consolidate':
            pass
        else:
            # num_partitions = np.array(literal_eval(self.partition_strategy))
            self.subsets = {}
            prev_set = self.full_set

            input_shape = self.full_set.shape[:-1]

            slope = torch.divide(
                (prev_set[..., 1] - prev_set[..., 0]), torch.from_numpy(num_partitions).type(torch.float32).to(self.device)
            )

            ranges = []
            output_range = None

            for element in product(
                *[range(num) for num in num_partitions.flatten()]
            ):
                element_ = torch.tensor(element).reshape(input_shape).to(self.device)
                input_range_ = torch.empty_like(prev_set)
                input_range_[..., 0] = prev_set[..., 0] + torch.multiply(
                    element_, slope
                )
                input_range_[..., 1] = prev_set[..., 0] + torch.multiply(
                    element_ + 1, slope
                )

                ranges.append(input_range_,)

            for i, partition in enumerate(ranges):
                self.subsets[i] = ReachableSet(self.t, torch.tensor(partition).to(self.device), thread = i, device = self.device)

    def consolidate(self):
        if self.partition_strategy != 'consolidate':
            pass
        else:
            self.calculate_full_set()
            self.subsets = {0: ReachableSet(self.t, self.full_set, thread=self.thread, device=self.device)}
        
    
    def populate_next_reachable_set(self, bounded_cl_system, next_reachable_set, training=False):
        if self.subsets == {} and next_reachable_set.recalculate:

            ######################################## Problem 1 ########################################
            # Hint: for problems 1 and 2 see third_party/auto_LiRPA/examples/simple/mip_lp_solver.py
            # input: state_range = torch.tensor([[x0_min, x0_max],
            #                                    [x1_min, x1_max])
            # output: BoundedTensor; auto_LiRPA object
            def get_bounded_tensor(state_range: torch.tensor) -> BoundedTensor:
                # x = torch.mean(state_range, axis=1).reshape((1, -1))
                # eps = (state_range[:, 1] - state_range[:, 0])/2
                # ptb = PerturbationLpNorm(eps = eps)
                # range_tensor = BoundedTensor(x, ptb)
                # return range_tensor
                raise NotImplementedError

            range_tensor = get_bounded_tensor(self.full_set)
            
            # x = torch.mean(self.full_set, axis=1).reshape((1, -1))
            # eps = (self.full_set[:, 1] - self.full_set[:, 0])/2
            # ptb = PerturbationLpNorm(eps = eps)
            # range_tensor = BoundedTensor(x, ptb)

            if training:
                pass
            else:
                ######################################## Problem 2 ########################################
                # get lb and ub with the auto_LiRPA function compute_bounds with the BoundedModule object bounded_cl_system
                # lb, ub = ...
                raise NotImplementedError
            
            # if next_reachable_set.populated:
            #     lb_ = torch.hstack((lb.T, next_reachable_set.full_set))
            #     ub_ = torch.hstack((ub.T, next_reachable_set.full_set))
            #     lb = torch.max(lb_[:,[0, 1]], axis = 1)[0].reshape((1, -1))
            #     ub = torch.min(ub_[:,[0, 2]], axis = 1)[0].reshape((1, -1))
                

            reach_set_range = torch.hstack((lb.T, ub.T))
            next_reachable_set.add_subset(reach_set_range, self.thread)
        else:
            for i, subset in self.subsets.items():
                subset.populate_next_reachable_set(bounded_cl_system, next_reachable_set, training)

        next_reachable_set.calculate_full_set()
        next_reachable_set.t_parent = self.t
        if next_reachable_set.t - self.t == 1:
            next_reachable_set.symbolic = False
        next_reachable_set.populated = True

    def switch_on_off(self, condition, thread=0):
        self.recalculate = not condition(self.full_set)
        if self.recalculate:
            if self.full_set[1, 0] < -1:
                print("Recalculating")
                print(self.full_set)
    
    def plot_reachable_set(self, ax, plot_partitions = True, edgecolor = None, facecolor = 'none', alpha = 0.1):
        if self.subsets == {}:
            set_range = self.full_set.cpu().detach().numpy()
            xy = set_range[:, 0]
            width, height = set_range[:, 1] - set_range[:, 0]
            if edgecolor is None:
                if self.recalculate:
                    edgecolor = 'orange'
                else:
                    edgecolor = 'b'
            rect = Rectangle(xy, width, height, linewidth=1, edgecolor=edgecolor, facecolor='none')
            ax.add_patch(rect)
            rect = Rectangle(xy, width, height, linewidth=1, edgecolor=edgecolor, facecolor=facecolor, alpha=alpha)
            ax.add_patch(rect)
        else:
            for i, subset in self.subsets.items():
                subset.plot_reachable_set(ax, plot_partitions=plot_partitions)

    def sample_from_reachable_set(self, cl_system, num_steps=1, num_trajectories=100, sample_corners=False):
        np.random.seed(0)
        num_states = cl_system.At.shape[0]
        if sample_corners:
            set_range = self.full_set.cpu().detach().numpy()
            test = np.meshgrid(*[set_range.T[:, i] for i in range(set_range.shape[0])])  
            corners = np.array(np.meshgrid([set_range.T[:, i] for i in range(set_range.shape[0])])).T.reshape(-1, num_states)
            # corners = np.array(np.meshgrid(set_range.T[:,0], set_range.T[:,1])).T.reshape(-1, num_states)
            num_trajectories -= len(corners)
            np.meshgrid()
        

        x0s = np.random.uniform(
            low=self.full_set[:, 0].cpu().detach().numpy(),
            high=self.full_set[:, 1].cpu().detach().numpy(),
            size=(num_trajectories, num_states),
        )
        
        if sample_corners:
            xs = np.vstack((corners, x0s))
        else:
            xs = x0s
        
        xt = xs
        for _ in range(num_steps):
            u_nn = cl_system.dynamics.control_nn(xt, cl_system.controller.cpu())
            xt1 = cl_system.dynamics.dynamics_step(xt, u_nn)
            xt = xt1

            xs = np.vstack((xs, xt1))
        
        return xs
            


class Analyzer:
    def __init__(self, cl_system,  num_steps, initial_range, max_diff=10, device='cpu', save_info=True) -> None:
        self.num_steps = num_steps
        self.cl_system = cl_system
        self.device = device
        self.max_diff = max_diff
        self.save_info = save_info
        self.h = 1

        if cl_system.dynamics.name == 'DoubleIntegrator':
            dummy_input = torch.tensor([[2.75, 0.]], device=device)
        elif cl_system.dynamics.name == 'Unicycle_NL':
            dummy_input = torch.tensor([[-12.5, 3.5, -0.5]], device=device)
        bound_opts = {
            'relu': "CROWN-IBP",
            'sparse_intermediate_bounds': False,
            'sparse_conv_intermediate_bounds': False,
            'sparse_intermediate_bounds_with_ibp': False,
            'sparse_features_alpha': False,
            'sparse_spec_alpha': False,
        }
        self.bounded_cl_system = BoundedModule(cl_system, dummy_input, bound_opts=bound_opts, device=device)
        
        self.reachable_sets = {0: ReachableSet(0, initial_range, partition_strategy = 'maintain', device=device)}
        self.bounded_cl_systems = {0: BoundedModule(cl_system, dummy_input, bound_opts=bound_opts, device=device)}
        for i in range(num_steps):
            self.reachable_sets[i+1] = ReachableSet(i+1, device=device)
            cl_system.set_num_steps(i+2)
            if i < self.max_diff:
                self.bounded_cl_systems[i+1] = BoundedModule(cl_system, dummy_input, bound_opts=bound_opts, device=device)

    def set_partition_strategy(self, t, partition_strategy):
        self.reachable_sets[t].set_partition_strategy(partition_strategy)

    def get_parent_set(self, reachable_set):
        if reachable_set.t == 0:
            return IndexError
        
        return self.reachable_sets[reachable_set.t - 1].get_thread(reachable_set.thread)

    
    def calculate_reachable_sets(self, training = False, autorefine = False, visualize = False, condition = None):
        snapshots = []
        if self.save_info:
            current_snapshot = {}
            current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), reachable_set.symbolic) for _, reachable_set in self.reachable_sets.items()])
            current_snapshot['time'] = 1
            current_snapshot['child_idx'] = 0
            current_snapshot['parent_idx'] = 0
        

        for i in range(self.num_steps):
            current_snapshot = {}

            self.reachable_sets[i].get_partitions()
            tstart = time.time()
            ######################################## Problem 3 ########################################
            # Use the Reachable_Set populate_next_reachable_set function to do a concrete (one-step) reachability calculation
            raise NotImplementedError
            tend = time.time()
            self.reachable_sets[i+1].consolidate()
            if self.save_info:
                current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), reachable_set.symbolic, not condition(reachable_set.full_set)) for _, reachable_set in self.reachable_sets.items()])
                current_snapshot['time'] = tend - tstart
                current_snapshot['child_idx'] = i + 1
                current_snapshot['parent_idx'] = i
                snapshots.append(current_snapshot)

            if autorefine:
                if visualize:
                    self.plot_reachable_sets()
                
                self.refine(self.reachable_sets[i+1], condition, snapshots, i)
                    
                if visualize:
                    self.plot_reachable_sets()
            
        return self.reachable_sets, snapshots
    
    def calculate_N_step_reachable_sets(self, training = False, indices = None, condition = None):
        if indices is None:
            indices = list(range(self.num_steps))
        import time

        snapshots = []

        for i in indices:
            current_snapshot = {}
            print("Calculating set {}".format(i))
            tstart = time.time()
            ######################################## Problem 4 ########################################
            # Use the Reachable_Set function populate_next_reachable_set function to do a symbolic (multi-step) reachability calculation from the initial state set (self.reachable_sets[0])
            # Hint: you'll need to use self.cl_systems (the list of cl_systems representing dynamics with increasing number of time steps)
            raise NotImplementedError
            print(self.reachable_sets[i+1].full_set)
            tend = time.time()
            print('Calculation Time: {}'.format(tend-tstart))
            if self.save_info:
                current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), True, not condition(reachable_set.full_set)) for _, reachable_set in self.reachable_sets.items()])
                current_snapshot['time'] = tend - tstart
                current_snapshot['child_idx'] = i + 1
                current_snapshot['parent_idx'] = 0
                snapshots.append(current_snapshot)

        return self.reachable_sets, snapshots
    
    def calculate_hybrid_symbolic_reachable_sets(self, concretization_rate = 5, training = False):
        idx = 0

        for i in range(self.num_steps):
            print("Calculating set {}".format(i))
            cl_system = ClosedLoopDynamics(self.cl_system.controller, self.cl_system.dynamics, i%concretization_rate+1)
            dummy_input = torch.tensor([[2.75, 0.]], device=self.device)
            bounded_cl_system = BoundedModule(cl_system, dummy_input, device=self.device)

            if i % concretization_rate == 0:
                idx = i
            tstart = time.time()
            self.reachable_sets[idx].populate_next_reachable_set(bounded_cl_system, self.reachable_sets[i+1])
            print(self.reachable_sets[i+1].full_set)
            tend = time.time()
            print('Calculation Time: {}'.format(tend-tstart))

        return self.reachable_sets
    
    def refine(self, reachable_set, condition, snapshots, t, force=False):
        refined = not condition(reachable_set.full_set) or force
        tf = reachable_set.t
        min_diff = 2
        max_diff = self.max_diff

        if force and not reachable_set.symbolic:
            marching_back = False
            while not marching_back:
                next_idx = max(tf - self.max_diff, 0)
                print("marching back from set {} to set {}".format(tf, next_idx))
                if self.reachable_sets[next_idx].symbolic:

                    print("{} is symbolic, marching back".format(next_idx))
                    marching_back = True
                    if self.save_info:
                        current_snapshot = {}
                        current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), reachable_set.symbolic, not condition(reachable_set.full_set)) for _, reachable_set in self.reachable_sets.items()])
                        current_snapshot['child_idx'] = tf
                        current_snapshot['parent_idx'] = next_idx

                    tstart = time.time()
                    self.reachable_sets[next_idx].populate_next_reachable_set(self.bounded_cl_systems[tf - next_idx - 1], reachable_set)
                    reachable_set.symbolic = True
                    tend = time.time()
                    if self.save_info:
                        current_snapshot['time'] = tend - tstart
                        snapshots.append(current_snapshot)

                    if self.save_info:
                        current_snapshot = {}
                        current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), reachable_set.symbolic, not condition(reachable_set.full_set)) for _, reachable_set in self.reachable_sets.items()])
                        current_snapshot['time'] = 0
                        current_snapshot['child_idx'] = next_idx + 1
                        current_snapshot['parent_idx'] = next_idx
                        snapshots.append(current_snapshot)
                else:
                    # current_snapshot = {}
                    # current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), reachable_set.symbolic, not condition(reachable_set.full_set)) for _, reachable_set in self.reachable_sets.items()])
                    # current_snapshot['time'] = 0
                    # current_snapshot['child_idx'] = tf
                    # current_snapshot['parent_idx'] = next_idx
                    # snapshots.append(current_snapshot)  
                    self.refine(self.reachable_sets[next_idx], condition, snapshots, next_idx, force=True)
        else:
            final_idx = max(tf - self.max_diff, 0)
            i = tf - 2
            if not condition(reachable_set.full_set):
                print("Collision detected at t = {}".format(tf))
            while i >= final_idx and not condition(reachable_set.full_set):
                diff = tf - i
                if self.reachable_sets[i].symbolic and (diff >= min_diff or i == 0):
                    print("recalculating set {} from time {}".format(tf, i))
                    if self.save_info:
                        current_snapshot = {}
                        current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), reachable_set.symbolic, not condition(reachable_set.full_set)) for _, reachable_set in self.reachable_sets.items()])
                        current_snapshot['child_idx'] = tf
                        current_snapshot['parent_idx'] = i

                    tstart = time.time()
                    self.reachable_sets[i].populate_next_reachable_set(self.bounded_cl_systems[tf - i - 1], reachable_set)
                    reachable_set.symbolic = True
                    tend = time.time()
                    if self.save_info:
                        current_snapshot['time'] = tend - tstart
                        snapshots.append(current_snapshot)

                        current_snapshot = {}
                        current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), reachable_set.symbolic, not condition(reachable_set.full_set)) for _, reachable_set in self.reachable_sets.items()])
                        current_snapshot['time'] = 0
                        current_snapshot['child_idx'] = i+1
                        current_snapshot['parent_idx'] = i
                        snapshots.append(current_snapshot)
                elif diff == max_diff:
                    print("cannot do full symbolic from tf = {}, starting march".format(tf))
                    # if i == 41:
                    #     import pdb; pdb.set_trace()
                    if i >=  1:
                        # current_snapshot = {}
                        # current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), reachable_set.symbolic, not condition(reachable_set.full_set)) for _, reachable_set in self.reachable_sets.items()])
                        # current_snapshot['time'] = 2
                        # current_snapshot['child_idx'] = tf
                        # current_snapshot['parent_idx'] = i
                        # snapshots.append(current_snapshot)
                        self.refine(self.reachable_sets[i], condition, snapshots, i, force=True)
                        i = i + 1
                
                if self.save_info:
                    current_snapshot = {}
                    current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), reachable_set.symbolic, not condition(reachable_set.full_set)) for _, reachable_set in self.reachable_sets.items()])
                    current_snapshot['time'] = 0
                    current_snapshot['child_idx'] = tf
                    current_snapshot['parent_idx'] = i
                    snapshots.append(current_snapshot)

                i -= 1
        return refined
        

    def hybr(self, visualize = False, condition = None):
        snapshots = []
        current_snapshot = {}
        current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), reachable_set.symbolic) for _, reachable_set in self.reachable_sets.items()])
        current_snapshot['time'] = 1
        current_snapshot['child_idx'] = 0
        current_snapshot['parent_idx'] = 0
        
        for i in range(self.num_steps):
            tf = self.reachable_sets[i+1].t
            parent_idx = i
            if (i + 1) % self.max_diff != 0:
                tstart = time.time()
                self.reachable_sets[i].populate_next_reachable_set(self.bounded_cl_system, self.reachable_sets[i+1])
                tend = time.time()
            else:
                parent_idx = max(tf - self.max_diff, 0)
                tstart = time.time()
                #
                self.reachable_sets[parent_idx].populate_next_reachable_set(self.bounded_cl_systems[tf - parent_idx - 1], self.reachable_sets[i+1])
                tend = time.time()


            current_snapshot = {}
            current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), reachable_set.symbolic, not condition(reachable_set.full_set)) for _, reachable_set in self.reachable_sets.items()])
            current_snapshot['time'] = tend - tstart
            current_snapshot['child_idx'] = i + 1
            current_snapshot['parent_idx'] = parent_idx
            snapshots.append(current_snapshot)            
        
            if visualize:
                self.plot_reachable_sets()
            
        return self.reachable_sets, snapshots


    def get_all_ranges(self):
        all_ranges = []
        for i, reachable_set in self.reachable_sets.items():
            if reachable_set.subsets == {}:
                if reachable_set.full_set is None:
                    pass
                else:
                    all_ranges.append(reachable_set.full_set)
            else:
                for _, reachable_subset in reachable_set.subsets.items():
                    all_ranges.append(reachable_subset.full_set)
                     
        return torch.stack(all_ranges, dim=0)
    

    def get_all_reachable_sets(self):
        all_sets = []
        for i, reachable_set in self.reachable_sets.items():
            if reachable_set.subsets == {}:
                all_sets.append(reachable_set)
            else:
                for _, reachable_subset in reachable_set.subsets.items():
                    all_sets.append(reachable_subset)
                     
        return all_sets
    

    def switch_sets_on_off(self, constraint):

        all_sets = self.get_all_reachable_sets()

        y = lambda y: False
        if y.__code__.co_code == constraint.__code__.co_code:
            for reachable_set in all_sets:
                reachable_set.recalculate = True
        else:
            for reachable_set in all_sets: # find colliding sets
                reachable_set.switch_on_off(constraint)

            
            # determine how far back we should step
            t_violation = self.num_steps # start with no partitioning
            for i, reachable_set in self.reachable_sets.items(): # find reachable sets that violate constraint
                if not constraint(reachable_set.full_set):
                    t_violation = min(t_violation, i)
                    print("Collision at t = {}".format(t_violation))
                    break

            walk_back = True # going backwards from violating set
            t = t_violation
            steps_back = 0
            print("violation: {}".format(t_violation))
            while(walk_back):
                t -= 1
                steps_back += 1
                print("stepping back to {}".format(t))
                xs = self.reachable_sets[t].sample_from_reachable_set(self.cl_system, steps_back, sample_corners=True)
                sample_range = np.vstack((np.min(xs, axis=0), np.max(xs, axis=0))).T
                if constraint(sample_range) or t == 0:
                    walk_back = False
                    print("Need to partition t = {}".format(t))

            for reachable_set in all_sets:
                if reachable_set.recalculate:
                    next_set = reachable_set
                    for i in range(t_violation, t, -1):
                        next_set = self.get_parent_set(next_set)
                        next_set.recalculate = True


    def is_safe(self, constraint_list):
        for constraint in constraint_list:
            for i, reachable_set in self.reachable_sets.items():
                if not constraint(reachable_set.full_set):
                    return False
                
        return True      


    def another_plotter(self, info, frames):
        cl_system = self.cl_system
        time_horizon = self.num_steps
        time_multiplier = 5


        
        fig, ax = plt.subplots(figsize=(10,6))
        
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Helvetica",
            "font.size": 20
        })

        reachable_set_snapshots = []
        info[-1]['time'] = 3
        for j, snapshot in enumerate(info):
            reachable_set_snapshot = []
            for i, reach_set_tuple in enumerate(snapshot['reachable_sets']):
                state_range = reach_set_tuple[0]
                is_symbolic = reach_set_tuple[1]
                collides = reach_set_tuple[2]
                edgecolor = '#2176FF'

                if i == snapshot['child_idx'] and snapshot['child_idx'] != snapshot['parent_idx'] + 1:
                    edgecolor = '#D63230' #F45B69'
                
                if i == 0:
                    edgecolor = 'k'
                elif is_symbolic:
                    edgecolor = '#00CC00' # edgecolor = '#D112E2' # '#D14081' # '#53917E' # '#20A39E' # '#44BBA4'

                if i == snapshot['parent_idx'] and j < len(info) - 1:
                    edgecolor = '#FF00FF' # '#FFAE03'
                if collides:
                    edgecolor = '#FF8000' # '#D63230'
                
                
                reachable_set_snapshot.append((state_range, edgecolor))
            
            reachable_set_snapshots.append(reachable_set_snapshot)
                
        def animate(i):    
            ax.clear()
            xs = self.reachable_sets[0].sample_from_reachable_set(self.cl_system, num_steps=time_horizon, sample_corners=False)
            
            ax.scatter(xs[:, 0], xs[:, 1], s=1, c='k')

            if self.cl_system.dynamics.name == "DoubleIntegrator":
                delta = 0.
                fig.set_size_inches(9.6, 7.2)
                constraint_color = '#262626'
                ax.plot(np.array([-1.5, 3.25]), np.array([-1, -1]), c=constraint_color, linewidth=2)
                rect = Rectangle(np.array([0.0, -1.25]), 3.25, 0.25, linewidth=1, edgecolor=constraint_color, facecolor=constraint_color, alpha=0.2)
                ax.add_patch(rect)

                ax.plot(np.array([0+delta, 0+delta]), np.array([-1.25, 1.]), c=constraint_color, linewidth=2)
                rect = Rectangle(np.array([-1.5, -1.25]), 1.5+delta, 2.25, linewidth=1, edgecolor=constraint_color, facecolor=constraint_color, alpha=0.2)
                ax.add_patch(rect)

                ax.set_xlim([-0.5, 3.25])
                ax.set_ylim([-1.25, 0.5])

                ax.set_xlabel('x1')
                ax.set_ylabel('x2')

                linewidth = 1.5

            for reachable_set_snapshot in reachable_set_snapshots[i]:
                set_range = reachable_set_snapshot[0]
                edgecolor = reachable_set_snapshot[1]
                xy = set_range[[0, 1], 0]
                width, height = set_range[[0, 1], 1] - set_range[[0, 1], 0]
                rect = Rectangle(xy, width, height, linewidth=linewidth, edgecolor=edgecolor, facecolor='none')
                ax.add_patch(rect)
                alpha = 0.1
                if edgecolor == '#FF8000': 
                    alpha = 0.2
                if edgecolor != 'k':
                    rect = Rectangle(xy, width, height, linewidth=linewidth, edgecolor=edgecolor, facecolor=edgecolor, alpha=alpha)
                    ax.add_patch(rect)

        for i in frames:
            animate(i)
            plt.show()