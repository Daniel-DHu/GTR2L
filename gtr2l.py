import numpy as np
import torch

from config import BaseConfig, Configurable
from world_model import BatchedGaussianEnsemble, KLevelWorldModel 
from sampling import SampleBuffer
from fcac import FAC
from torch_util import Module, DummyModuleWrapper, device, random_choice
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GTR2L(Configurable, Module):
    class Config(BaseConfig):
        fac_cfg = FAC.Config()
        model_cfg = BatchedGaussianEnsemble.Config()
        model_steps = 500
        horizon = 5
        max_horizon = 8
        min_horizon = 3
        adaptive_horizon = True  # Enable adaptive length
        adaptive_horizon_start_episode = 300
        safe_horizon = 100
        buffer_min = 5
        buffer_max = 10**6
        rollout_batch_size = 64
        solver_updates_per_step = 8
        real_fraction = 0.1
        use_klevel_model = True  # Whether to use a K-level game world model
        k_levels = 3  

        uncertainty_record_interval = 10 
        uncertainty_sample_rate = 0.1  # Only 5% of the rollouts are sampled for recording

    def __init__(self, config, env, state_dim, action_dim, device=None):
        Configurable.__init__(self, config)
        Module.__init__(self)

        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.real_env = env
        self.state_dim, self.action_dim = state_dim, action_dim

        self.solver = FAC(self.fac_cfg, self.state_dim, self.action_dim)
        
        if self.use_klevel_model:
            k_model_cfg = KLevelWorldModel.Config()
            
            for attr in dir(self.model_cfg):
                if not attr.startswith('_') and hasattr(k_model_cfg, attr):
                    setattr(k_model_cfg, attr, getattr(self.model_cfg, attr))
            
            k_model_cfg.k_levels = self.k_levels
            k_model_cfg.ego_veh_indices = (24, 25)
            k_model_cfg.nearby_veh_indices = [(2, 3), (6, 7), (10, 11), (14, 15), (18, 19), (22, 23)]
            k_model_cfg.interaction_weight = 0.8
            k_model_cfg.uncertainty_boost = 1.5
            
            self.model_ensemble = KLevelWorldModel(k_model_cfg, self.state_dim, self.action_dim)
            print("Using K-Level Game Theory World Model with k =", self.k_levels)
        else:
            self.model_ensemble = BatchedGaussianEnsemble(self.model_cfg, self.state_dim, self.action_dim)
            print("Using Standard World Model")

        self.solver.to(self.device)
        self.model_ensemble.to(self.device)

        self.replay_buffer = self._create_buffer(self.buffer_max)
        self.virt_buffer = self._create_buffer(self.buffer_max)

        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
        self.safe_interaction = 0
        self.safe_horizon = 10
        self.env_model_error = 0.0

        # Add uncertainty tracking
        self.episode_uncertainties = []
        self.register_buffer('uncertainty_sum', torch.zeros(1, device=self.device))
        self.register_buffer('uncertainty_count', torch.zeros(1, device=self.device, dtype=torch.long))
        self.should_record_uncertainty = False
        self.current_episode = 0

        self.current_uncertainty_estimate = 0.1  

    def calculate_adaptive_horizon(self, uncertainty=None, episode_num=None):
        if not self.adaptive_horizon or (episode_num is not None and episode_num < self.adaptive_horizon_start_episode):
            return self.horizon
            
        uncertainty_value = uncertainty if uncertainty is not None else self.current_uncertainty_estimate
        adaptive_horizon = max(self.min_horizon, 
                              min(self.max_horizon, 
                                  round(self.max_horizon - 50 * uncertainty_value + 5)))   
        return adaptive_horizon

    @property
    def actor(self):
        return self.solver.actor
    
    @property
    def barrier(self):
        return self.solver.barrier

    def _create_buffer(self, capacity):
        buffer = SafetySampleBuffer(self.state_dim, self.action_dim, capacity)
        buffer.to(device)
        return DummyModuleWrapper(buffer)

    def interaction(self, max_episode_steps, score, v, v_epi, speed_range, max_a, n_epi, warm_up=10, scale=10.0):

        self.current_episode = n_epi
        self.should_record_uncertainty = (n_epi % self.uncertainty_record_interval == 0)
        self.uncertainty_sum.zero_()
        self.uncertainty_count.zero_()

        episode = self._create_buffer(max_episode_steps)
        state = self.real_env.reset()

        for t in range(max_episode_steps):
            if n_epi > warm_up:
                if t == 0:
                    self.update_world_model(self.model_steps)
                self.rollout_and_update()

            mu, _, pi = self.actor(torch.tensor(state, dtype=torch.float32).to(device))
            action = max_a*pi.item()

            r, next_s, done, r_, c_, info = self.real_env.step(action)
            for buffer in [episode, self.replay_buffer]:
                buffer.append(states=torch.tensor(state).to(device), 
                          actions=pi.detach(), 
                          next_states=torch.tensor(next_s).to(device), 
                          rewards=r/scale, 
                          dones=done, 
                          violations=done)
            state = next_s
            score += r
            v.append(state[24]*speed_range)
            v_epi.append(state[24]*speed_range)
            xa = info[0]
            ya = info[1]

            if done is False:
                self.safe_interaction += 1
            else:
                break
        self.safe_horizon = self.safe_interaction
        self.safe_interaction = 0

        if (n_epi+1) % 100 == 0:
            self.solver.save_model(int(score), self.real_env.spec.id)
            self.save_model(self.env_model_error)
            print("#The models are saved!#", n_epi+1)
            print("======>env_model_error:", self.env_model_error)
        if self.should_record_uncertainty and self.uncertainty_count > 0:
            mean_uncertainty = (self.uncertainty_sum / self.uncertainty_count).item()
            self.episode_uncertainties.append(mean_uncertainty)
            self.current_uncertainty_estimate = mean_uncertainty
            adaptive_horizon = self.calculate_adaptive_horizon(episode_num=n_epi)
            print(f"Episode {n_epi+1} uncertainty: {mean_uncertainty:.6f}, adaptive horizon: {adaptive_horizon}")

        return score, v, v_epi, xa, ya, done
    

    def interaction_long_dis_task(self, max_episode_steps, score, v, v_epi, speed_range, n_epi, warm_up=10, scale=10.0, max_sh=10):

        self.current_episode = n_epi
        self.should_record_uncertainty = (n_epi % self.uncertainty_record_interval == 0)
        self.uncertainty_sum.zero_()
        self.uncertainty_count.zero_()

        episode = self._create_buffer(max_episode_steps)
        state = self.real_env.reset()
        infraction_check_epi = False

        for t in range(max_episode_steps):
            if n_epi > warm_up:
                if t == 0:
                    self.update_world_model(self.model_steps)
                self.rollout_and_update()

            mu, _, pi = self.actor(torch.tensor(state, dtype=torch.float32).to(device))
            r, next_s, collision, cost, infraction_check, infraction, navigation_check, done, info = self.real_env.step(pi)
            
            for buffer in [episode, self.replay_buffer]:
                buffer.append(states=torch.tensor(state).to(device), 
                            actions=pi.detach(), 
                            next_states=torch.tensor(next_s).to(device), 
                            rewards=r/scale, 
                            dones=done, 
                            violations=(collision or infraction_check))

            state = next_s
            score += r
            v.append(state[24]*speed_range)
            v_epi.append(state[24]*speed_range)

            if infraction_check is True:
                infraction_check_epi = True

            if done is False:
                self.safe_interaction += 1
            else:
                break

        self.safe_horizon = min(self.safe_interaction, max_sh)
        self.safe_interaction = 0

        if (n_epi+1) % 100 == 0:
            self.solver.save_model(int(score), self.real_env.spec.id)
            self.save_model(self.env_model_error)
            print("#The models are saved!#", n_epi+1)
            print("======>env_model_error:", self.env_model_error)

        if self.should_record_uncertainty and self.uncertainty_count > 0:
            mean_uncertainty = (self.uncertainty_sum / self.uncertainty_count).item()
            self.episode_uncertainties.append(mean_uncertainty)
            self.current_uncertainty_estimate = mean_uncertainty
            adaptive_horizon = self.calculate_adaptive_horizon(episode_num=n_epi)
            print(f"Episode {n_epi+1} uncertainty: {mean_uncertainty:.6f}, adaptive horizon: {adaptive_horizon}")

        return score, v, v_epi, done, infraction_check_epi, navigation_check, collision

    def update_world_model(self, model_steps):
        losses = self.model_ensemble.fit(self.replay_buffer, steps=model_steps) 
        self.env_model_error = np.mean(losses)

        buffer_rewards = self.replay_buffer.get('rewards')
        r_min = buffer_rewards.min().item()
        r_max = buffer_rewards.max().item()
        self.solver.update_r_bounds(r_min, r_max, self.safe_horizon)

    def rollout(self, policy, bar_policy, initial_states=None, k=0.9, c=0.1):  
        if initial_states is None:
            initial_states = random_choice(self.replay_buffer.get('states'), size=self.rollout_batch_size)
        adaptive_horizon = self.calculate_adaptive_horizon(episode_num = self.current_episode)
        buffer = self._create_buffer(self.rollout_batch_size * adaptive_horizon)
        states = initial_states

        record_this_rollout = (
            self.should_record_uncertainty and 
            torch.rand(1).item() < self.uncertainty_sample_rate)

        for t in range(adaptive_horizon):
            with torch.no_grad():
                _, _, actions = policy(states)
                _, _, bound_bars = bar_policy(states)
                if self.use_klevel_model and hasattr(self.model_ensemble, 'sample_with_interaction'):
                    interaction_level = min(1.0, (t + 1) / adaptive_horizon) 
                    next_states, costs, uncertainty, aleatoric = self.model_ensemble.sample_with_interaction(
                        states, actions, interaction_level=interaction_level)
                else:
                    next_states, costs, uncertainty, aleatoric = self.model_ensemble.sample(states, actions)

            obstacle_indices = [0, 4, 8, 12, 16, 20]
            modified_next_states = next_states.clone()

            bound_bar = torch.clamp((bound_bars.mean()+ 1.0) * 0.5, 0.0, 1.0)
            barrier_factor = bound_bar * 10 
            for idx in obstacle_indices:
                uncertainty_factor = barrier_factor.item() * uncertainty.mean().item() 
                uncertain_noise = abs(np.random.normal(0, uncertainty_factor))
                modified_next_states[:, idx] = torch.clamp(next_states[:, idx] - uncertain_noise, min=0.0)

            next_states = modified_next_states

            if t == 0:
                first_step_uncertainty = uncertainty.mean().item()

            if record_this_rollout:
                self.uncertainty_sum += uncertainty.sum()
                self.uncertainty_count += uncertainty.numel()

            violations = costs > c
            dones = violations
            buffer.extend(states=states, actions=actions, next_states=next_states,
                        rewards=0.0, dones=dones, violations=violations)
            continues = ~(dones | violations)
            if continues.sum() == 0:
                break
            states = next_states[continues]

        self.virt_buffer.extend(**buffer.get(as_dict=True))

        if record_this_rollout and first_step_uncertainty is not None and self.current_episode >= self.adaptive_horizon_start_episode:
            self.current_uncertainty_estimate = 0.9 * self.current_uncertainty_estimate + 0.1 * first_step_uncertainty

        return (k*torch.mean(violations.float()) + (1-k)*torch.clamp(uncertainty, 0.0, 1.0)).requires_grad_(True)
    

    def get_mean_uncertainty(self):
        if not self.should_record_uncertainty or self.uncertainty_count == 0:
            return None
        
        with torch.no_grad():
            mean_uncertainty = (self.uncertainty_sum / self.uncertainty_count).item()
        return mean_uncertainty
    

    def update_solver(self, risk):
        solver = self.solver
        n_real = int(self.real_fraction * solver.batch_size)
        real_samples = self.replay_buffer.sample(n_real)
        virt_samples = self.virt_buffer.sample(solver.batch_size - n_real)
        combined_samples = [
            torch.cat([real, virt]) for real, virt in zip(real_samples, virt_samples)
        ]

        solver.update(self.replay_buffer, risk)  
        solver.update_actor(combined_samples[0]) 

    def rollout_and_update(self):
        risk = self.rollout(self.actor, self.barrier)

        for _ in range(self.solver_updates_per_step):
            self.update_solver(risk)

    def process_inputs(self, s, s_mean, s_std, s_epsilon):
        return (torch.tensor(s).to(device) - s_mean) / (s_std + s_epsilon)

    def save_model(self, model_name):
        name = "./models/" + self.real_env.spec.id + "/env_model%d" % model_name
        
        if self.use_klevel_model and hasattr(self.model_ensemble, 'save_model'):
            self.model_ensemble.save_model(model_name)
        else:
            torch.save([self.model_ensemble.state_dict(), self.model_ensemble.state_normalizer.mean, \
                        self.model_ensemble.state_normalizer.std, self.model_ensemble.state_normalizer.epsilon], "{}.pkl".format(name))
        
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.solver.load_state_dict(checkpoint)
        print("Successfully loaded policy model")

    def load_world_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if self.use_klevel_model and hasattr(self.model_ensemble, 'load_model'):
            self.model_ensemble.load_model(path)
        else:
            if isinstance(checkpoint, list):
                self.model_ensemble.load_state_dict(checkpoint[0])
                self.model_ensemble.state_normalizer.mean = checkpoint[1]
                self.model_ensemble.state_normalizer.std = checkpoint[2]
                self.model_ensemble.state_normalizer.epsilon = checkpoint[3]
                print("Successfully loaded world model")
            elif isinstance(checkpoint, dict) and 'base_model' in checkpoint:
                self.model_ensemble.load_state_dict(checkpoint['base_model'])
                self.model_ensemble.state_normalizer.mean = checkpoint['normalizer_mean']
                self.model_ensemble.state_normalizer.std = checkpoint['normalizer_std']
                self.model_ensemble.state_normalizer.epsilon = checkpoint['normalizer_epsilon']
                print("Successfully loaded world model (K-level format)")
            else:
                raise ValueError("Invalid world model checkpoint format")


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


class SafetySampleBuffer(SampleBuffer):
    COMPONENT_NAMES = (*SampleBuffer.COMPONENT_NAMES, 'violations')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_buffer('violations', torch.bool, [])