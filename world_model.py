from abc import ABC, abstractmethod
import random
from math import ceil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from torch_util import torchify
from config import BaseConfig, Configurable
from torch_util import device, Module, mlp


class BaseModel(ABC):
    @abstractmethod
    def sample(self, states, actions):
        """
        Returns a sample of (s', r) given (s, a)
        """
        pass


class BatchedLinear(nn.Module):
    """For efficient MLP ensembles with batched matrix multiplies"""
    def __init__(self, ensemble_size, in_features, out_features, bias=True):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(ensemble_size, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        has_bias = self.bias is not None
        l = nn.Linear(self.in_features, self.out_features, bias=has_bias)
        for i in range(self.ensemble_size):
            l.reset_parameters()
            self.weight.data[i].copy_(l.weight.data)
            if has_bias:
                self.bias.data[i].copy_(l.bias.data)

    def forward(self, input):
        assert len(input.shape) == 3
        assert input.shape[0] == self.ensemble_size
        return torch.bmm(input, self.weight.transpose(1, 2)) + self.bias.unsqueeze(1)


class BatchedGaussianEnsemble(Configurable, Module, BaseModel):
    class Config(BaseConfig):
        ensemble_size = 5
        hidden_dim = 256
        trunk_layers = 2
        head_hidden_layers = 1
        activation = 'relu'
        init_min_log_var = -10.0
        init_max_log_var = 1.0
        log_var_bound_weight = 0.01
        batch_size = 64
        learning_rate = 1e-3

    def __init__(self, config, state_dim, action_dim,
                 device=device, optimizer_factory=torch.optim.Adam):
        Configurable.__init__(self, config)
        Module.__init__(self)

        self.state_dim = state_dim
        self.action_dim = action_dim
        input_dim = state_dim + action_dim
        output_dim = state_dim + 1

        self.min_log_var = nn.Parameter(torch.full([output_dim], self.init_min_log_var, device=device))
        self.max_log_var = nn.Parameter(torch.full([output_dim], self.init_max_log_var, device=device))
        self.state_normalizer = Normalizer(state_dim)

        layer_factory = lambda n_in, n_out: BatchedLinear(self.ensemble_size, n_in, n_out)
        trunk_dims = [input_dim] + [self.hidden_dim] * self.trunk_layers
        head_dims = [self.hidden_dim] * (self.head_hidden_layers + 1) + [output_dim]
        self.trunk = mlp(trunk_dims, layer_factory=layer_factory, activation=self.activation,
                         output_activation=self.activation)
        self.diff_head = mlp(head_dims, layer_factory=layer_factory, activation=self.activation)
        self.log_var_head = mlp(head_dims, layer_factory=layer_factory, activation=self.activation)
        self.to(device)
        self.optimizer = optimizer_factory([
            *self.trunk.parameters(),
            *self.diff_head.parameters(),
            *self.log_var_head.parameters(),
            self.min_log_var, self.max_log_var
        ], lr=self.learning_rate)


    @property
    def total_batch_size(self):
        return self.ensemble_size * self.batch_size

    def _forward1(self, states, actions, index):
        normalized_states = self.state_normalizer(states)
        inputs = torch.cat([normalized_states, actions], dim=-1)
        shared_hidden = unbatched_forward(self.trunk, inputs, index)
        diffs = unbatched_forward(self.diff_head, shared_hidden, index)
        batch_size = inputs.shape[0]
        means = diffs + torch.cat([states, torch.zeros([batch_size, 1], device=device)], dim=1)
        log_vars = unbatched_forward(self.log_var_head, shared_hidden, index)
        log_vars = self.max_log_var - F.softplus(self.max_log_var - log_vars)
        log_vars = self.min_log_var + F.softplus(log_vars - self.min_log_var)
        return means, log_vars

    def _forward_all(self, states, actions):
        if states.shape[0] != self.ensemble_size:
            states = states.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        if actions.shape[0] != self.ensemble_size:
            actions = actions.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        
        normalized_states = self.state_normalizer(states)
        inputs = torch.cat([normalized_states, actions], dim=-1)
        shared_hidden = self.trunk(inputs)
        diffs = self.diff_head(shared_hidden)
        batch_size = inputs.shape[1]
        means = diffs + torch.cat([states, torch.zeros([self.ensemble_size, batch_size, 1], device=device)], dim=-1)

        log_vars = self.log_var_head(shared_hidden)
        log_vars = self.max_log_var - F.softplus(self.max_log_var - log_vars)
        log_vars = self.min_log_var + F.softplus(log_vars - self.min_log_var)

        return means, log_vars

    def _rebatch(self, x):
        total_batch_size = len(x)
        assert total_batch_size % self.ensemble_size == 0, f'{total_batch_size} not divisible by {self.ensemble_size}'
        batch_size = total_batch_size // self.ensemble_size
        remaining_dims = tuple(x.shape[1:])
        return x.reshape(self.ensemble_size, batch_size, *remaining_dims)

    def compute_loss(self, states, actions, targets):
        inputs = [states, actions, targets]
        total_batch_size = len(targets)
        remainder = total_batch_size % self.ensemble_size
        if remainder != 0:
            nearest = total_batch_size - remainder
            inputs = [x[:nearest] for x in inputs]

        states, actions, targets = [self._rebatch(x) for x in inputs]
        means, log_vars = self._forward_all(states, actions)
        inv_vars = torch.exp(-log_vars)
        squared_errors = torch.sum((targets - means)**2 * inv_vars, dim=-1)
        log_dets = torch.sum(log_vars, dim=-1)
        mle_loss = torch.mean(squared_errors + log_dets)
        return mle_loss + self.log_var_bound_weight * (self.max_log_var.sum() - self.min_log_var.sum())

    def fit(self, buffer, steps=None, epochs=None, progress_bar=False, **kwargs):
        n = len(buffer)
        states, actions, next_states, rewards, dones = buffer.get()[:5]
        self.state_normalizer.fit(states)
        targets = torch.cat([next_states, dones.float().unsqueeze(1)], dim=1)

        if steps is not None:
            assert epochs is None, 'Cannot pass both steps and epochs'
            losses = []
            for _ in (trange if progress_bar else range)(steps):
                indices = torch.randint(n, [self.total_batch_size], device=device)
                loss = self.compute_loss(states[indices], actions[indices], targets[indices])
                losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            return losses
        elif epochs is not None:
            adjusted_epochs = self.ensemble_size * epochs
            return epochal_training(self.compute_loss, self.optimizer, [states, actions, targets],
                                    epochs=adjusted_epochs,
                                    batch_size=self.total_batch_size, **kwargs)
        else:
            raise ValueError('Must pass steps or epochs')

    def sample(self, states, actions):
        index = random.randrange(self.ensemble_size)
        means, log_vars = self._forward1(states, actions, index)
        stds = torch.exp(log_vars).sqrt()
        samples = means + stds * torch.randn_like(means)
        return samples[:,:-1], torch.clamp(samples[:,-1], 0, 1), torch.mean(stds)

    # Get all models' means on the same set of states and actions
    def means(self, states, actions):
        states = states.repeat(self.ensemble_size, 1, 1)
        actions = actions.repeat(self.ensemble_size, 1, 1)
        means, _ = self._forward_all(states, actions)
        return means

    # Get average of models' means
    def mean(self, states, actions):
        next_state_means = self.means(states, actions)
        return next_state_means.mean(dim=0)

    def sample_test(self, states, actions):
        index = random.randrange(self.ensemble_size)
        means, log_vars = self._forward1(states, actions, index)
        stds = torch.exp(log_vars).sqrt()
        samples = means + stds * torch.randn_like(means)

        return samples.tolist()


class KLevelWorldModel(BatchedGaussianEnsemble):
    class Config(BatchedGaussianEnsemble.Config):
        k_levels = 3                  
        ego_veh_indices = (24, 25)    # State index corresponding to the speed and orientation angle of the vehicle
        nearby_veh_indices = [(2, 3), (6, 7), (10, 11), (14, 15), (18, 19), (22, 23)] # The speed and orientation angle of surrounding vehicles
        interaction_weight = 0.8      # Interaction influence weight
        uncertainty_boost = 1.5       
        
    def __init__(self, config, state_dim, action_dim,
                 device=device, optimizer_factory=torch.optim.Adam):
        super().__init__(config, state_dim, action_dim, device, optimizer_factory)
        
        self.k_models = nn.ModuleList()
        for k in range(1, self.k_levels + 1):
            k_input_dim = state_dim + action_dim + (state_dim + 1)
            k_output_dim = state_dim + 1
            
            layer_factory = lambda n_in, n_out: BatchedLinear(self.ensemble_size, n_in, n_out)
            trunk_dims = [k_input_dim] + [self.hidden_dim] * self.trunk_layers
            head_dims = [self.hidden_dim] * (self.head_hidden_layers + 1) + [k_output_dim]
            
            k_model = {
                'trunk': mlp(trunk_dims, layer_factory=layer_factory, 
                            activation=self.activation, output_activation=self.activation),
                'diff_head': mlp(head_dims, layer_factory=layer_factory, activation=self.activation),
                'log_var_head': mlp(head_dims, layer_factory=layer_factory, activation=self.activation)
            }
            self.k_models.append(nn.ModuleDict(k_model))
        
        self.optimizer = optimizer_factory([
            *self.trunk.parameters(),
            *self.diff_head.parameters(),
            *self.log_var_head.parameters(),
            *[p for k_model in self.k_models for p in k_model.parameters()],
            self.min_log_var, self.max_log_var
        ], lr=self.learning_rate)
        
    
    def _forward_k_level(self, states, actions, prev_predictions, k_index):
        k_model = self.k_models[k_index]
        normalized_states = self.state_normalizer(states)
        inputs = torch.cat([normalized_states, actions, prev_predictions], dim=-1)
        shared_hidden = k_model['trunk'](inputs)
        diffs = k_model['diff_head'](shared_hidden)
        batch_size = inputs.shape[1]
        means = diffs + torch.cat([states, torch.zeros([self.ensemble_size, batch_size, 1], device=device)], dim=-1)
        log_vars = k_model['log_var_head'](shared_hidden)
        log_vars = self.max_log_var - F.softplus(self.max_log_var - log_vars)
        log_vars = self.min_log_var + F.softplus(log_vars - self.min_log_var)
        return means, log_vars
    
    
    def _extract_vehicle_state(self, states):
        if len(states.shape) == 2:  # [batch, state_dim]
            ego_state = states[:, list(self.ego_veh_indices)]
            
            nearby_states = []
            for speed_idx, angle_idx in self.nearby_veh_indices:
                is_present = states[:, speed_idx-2] < 100  
                veh_speed = states[:, speed_idx] * is_present.float()
                veh_angle = states[:, angle_idx] * is_present.float()
                nearby_states.extend([veh_speed, veh_angle])
            
            nearby_states = torch.stack(nearby_states, dim=1)
            
        elif len(states.shape) == 3:  # [ensemble, batch, state_dim]
            ego_state = states[:, :, list(self.ego_veh_indices)]
            
            nearby_list = []
            for speed_idx, angle_idx in self.nearby_veh_indices:
                is_present = states[:, :, speed_idx-2] < 100
                veh_speed = states[:, :, speed_idx] * is_present.float()
                veh_angle = states[:, :, angle_idx] * is_present.float()
                nearby_list.extend([veh_speed, veh_angle])
            
            nearby_states = torch.stack(nearby_list, dim=2)
        
        return ego_state, nearby_states
    
    def compute_interaction_loss(self, states, actions, targets):

        base_loss = super().compute_loss(states, actions, targets)
        
        if self.k_levels == 0:
            return base_loss
            
        inputs = [states, actions, targets]
        total_batch_size = len(targets)
        remainder = total_batch_size % self.ensemble_size
        if remainder != 0:
            nearest = total_batch_size - remainder
            inputs = [x[:nearest] for x in inputs]
        
        states, actions, targets = [self._rebatch(x) for x in inputs]
        
        level_0_means, level_0_log_vars = self._forward_all(states, actions)
        
        total_loss = base_loss
        loss_weights = [1.0 / (2**i) for i in range(1, self.k_levels + 1)] 
        
        prev_means = level_0_means
        for k in range(self.k_levels):
            means, log_vars = self._forward_k_level(states, actions, prev_means, k)
            inv_vars = torch.exp(-log_vars)
            squared_errors = torch.sum((targets - means)**2 * inv_vars, dim=-1)
            log_dets = torch.sum(log_vars, dim=-1)
            level_loss = torch.mean(squared_errors + log_dets)
            total_loss += loss_weights[k] * level_loss
            prev_means = means
        return total_loss
    

    def fit(self, buffer, steps=None, epochs=None, progress_bar=False, **kwargs):
        n = len(buffer)
        states, actions, next_states, rewards, dones = buffer.get()[:5]
        self.state_normalizer.fit(states)
        targets = torch.cat([next_states, dones.float().unsqueeze(1)], dim=1)

        if steps is not None:
            assert epochs is None, 'Cannot pass both steps and epochs'
            losses = []
            for _ in (trange if progress_bar else range)(steps):
                indices = torch.randint(n, [self.total_batch_size], device=device)
                loss = self.compute_interaction_loss(states[indices], actions[indices], targets[indices])
                losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            return losses
        elif epochs is not None:
            adjusted_epochs = self.ensemble_size * epochs
            return epochal_training(self.compute_interaction_loss, self.optimizer, 
                                    [states, actions, targets],
                                    epochs=adjusted_epochs,
                                    batch_size=self.total_batch_size, **kwargs)
        else:
            raise ValueError('Must pass steps or epochs')
    

    def sample(self, states, actions, k_level=None):

        if k_level is None:
            k_level = self.k_levels   
        if k_level == 0:
            return super().sample(states, actions)
        
        index = random.randrange(self.ensemble_size)
        
        if len(states.shape) == 3 and states.shape[0] == self.ensemble_size:
            states_expanded = states
        elif len(states.shape) == 2:  # [batch_size, feature_dim]
            states_expanded = states.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        else:
            raise ValueError(f"Unexpected states shape: {states.shape}")
        
        if len(actions.shape) == 3 and actions.shape[0] == self.ensemble_size:
            actions_expanded = actions
        elif len(actions.shape) == 2:
            actions_expanded = actions.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        else:
            raise ValueError(f"Unexpected actions shape: {actions.shape}")
        
        # Level 0
        means, log_vars = self._forward_all(states_expanded, actions_expanded)
        all_level_means = [means.clone()]
        
        # Iterative execution of k-level prediction
        for k in range(k_level):
            means, log_vars = self._forward_k_level(states_expanded, actions_expanded, means, k)
            all_level_means.append(means.clone())
        
        means_single = means[index]
        log_vars_single = log_vars[index]
        
        ego_state, nearby_states = self._extract_vehicle_state(states)
        
        uncertainty_factor = torch.ones(means_single.shape[0], device=device)
        if len(nearby_states.shape) == 2:  # [batch, features]
            nearby_speeds = nearby_states[:, ::2]  
            relative_speed = torch.abs(nearby_speeds - ego_state[:, 0].unsqueeze(1))
            uncertainty_factor = torch.clamp(
                relative_speed.max(dim=1)[0] * self.uncertainty_boost / 10.0, 
                min=1.0, 
                max=self.uncertainty_boost)
        
        aleatoric_stds = torch.exp(log_vars_single).sqrt() * uncertainty_factor.unsqueeze(1)
        

        level_means = torch.stack([m[index] for m in all_level_means], dim=0)
        epistemic_variance = level_means.var(dim=0)
        epistemic_stds = epistemic_variance.sqrt() * uncertainty_factor.unsqueeze(1)
        
        total_variance = aleatoric_stds.pow(2) + epistemic_stds.pow(2)
        total_stds = total_variance.sqrt()
        
        # samples = means_single + total_stds * torch.randn_like(means_single)   
        samples = means_single + aleatoric_stds * torch.randn_like(means_single)   
        
        uncertainty_mean = torch.mean(total_stds)
        aleatoric = torch.mean(aleatoric_stds)
        return samples[:,:-1], torch.clamp(samples[:,-1], 0, 1), uncertainty_mean, aleatoric


    def sample_with_interaction(self, states, actions, interaction_level=0.5):
        k_level = max(1, int(interaction_level * self.k_levels))
        return self.sample(states, actions, k_level=k_level)
        
    def save_model(self, model_name):
        name = f"./models/klevel_world_model_{model_name}.pkl"
        torch.save({
            'base_model': self.state_dict(),
            'normalizer_mean': self.state_normalizer.mean,
            'normalizer_std': self.state_normalizer.std,
            'normalizer_epsilon': self.state_normalizer.epsilon,
            'config': {
                'k_levels': self.k_levels,
                'interaction_weight': self.interaction_weight,
                'uncertainty_boost': self.uncertainty_boost
            }
        }, name)
        
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and 'base_model' in checkpoint:
            self.load_state_dict(checkpoint['base_model'])
            self.state_normalizer.mean = checkpoint['normalizer_mean']
            self.state_normalizer.std = checkpoint['normalizer_std']
            self.state_normalizer.epsilon = checkpoint['normalizer_epsilon']
            if 'config' in checkpoint:
                cfg = checkpoint['config']
                if hasattr(self, 'k_levels'):
                    self.k_levels = cfg.get('k_levels', self.k_levels)
                if hasattr(self, 'interaction_weight'):
                    self.interaction_weight = cfg.get('interaction_weight', self.interaction_weight)
                if hasattr(self, 'uncertainty_boost'):
                    self.uncertainty_boost = cfg.get('uncertainty_boost', self.uncertainty_boost)
            print("Successfully loaded K-level world model")
        else:
            return super().load_model(path)


def unbatched_forward(batched_sequential, input, index):
    for layer in batched_sequential:
        if isinstance(layer, BatchedLinear):
            input = F.linear(input, layer.weight[index], layer.bias[index])
        else:
            input = layer(input)
    return input


def epochal_training(compute_loss, optimizer, data, epochs, batch_size=256, max_grad_norm=None,
                     post_epoch_callback=None, post_step_callback=None,
                     progress_bar=False):
    def one_step(batch):
        loss = compute_loss(*batch)
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            for param_group in optimizer.param_groups:
                nn.utils.clip_grad_norm_(param_group['params'], max_grad_norm)
        optimizer.step()
        return loss.item()

    if torch.is_tensor(data):
        data = (data,)
    n = len(data[0])
    for i, data_i in enumerate(data):
        assert len(data_i) == n

    data = [torchify(data_i) for data_i in data]

    n_batches = ceil(float(n) / batch_size)
    iter_range_fn = trange if progress_bar else range
    losses = []
    for epoch_index in range(epochs):
        indices = torch.randperm(n)
        epoch_losses = []
        for batch_index in iter_range_fn(n_batches):
            batch_start = batch_size * batch_index
            batch_end = min(batch_size * (batch_index + 1), n)
            batch_indices = indices[batch_start:batch_end]
            loss_val = one_step([component[batch_indices] for component in data])
            epoch_losses.append(loss_val)
            if post_step_callback is not None:
                post_step_callback(epoch_index, batch_index, n_batches)
        avg_epoch_loss = float(np.mean(epoch_losses))
        losses.append(avg_epoch_loss)

        if post_epoch_callback is not None:
            post_epoch_callback(epoch_index + 1)

    return losses


class Normalizer(Module):
    def __init__(self, dim, epsilon=1e-6):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('std', torch.zeros(dim))

    def fit(self, X):
        assert torch.is_tensor(X)
        assert X.dim() == 2
        assert X.shape[1] == self.dim
        self.mean.data.copy_(X.mean(dim=0))
        self.std.data.copy_(X.std(dim=0))

    def forward(self, x):
        return (x - self.mean) / (self.std + self.epsilon)

    def unnormalize(self, normal_X):
        return self.mean + (self.std * normal_X)