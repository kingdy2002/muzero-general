import copy
import time
import math

import numpy
import ray
import torch

import models

import torch.nn.functional as F

@ray.remote
class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
        self.model.to(torch.device("cuda" if self.config.train_on_gpu else "cpu"))
        self.model.train()

        self.training_step = initial_checkpoint["training_step"]

        if "cuda" not in str(next(self.model.parameters()).device):
            print("You are not training on GPU.\n")

        # Initialize the optimizer
        if self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.lr_init,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.lr_init,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"{self.config.optimizer} is not implemented. You can change the optimizer manually in trainer.py."
            )

        if initial_checkpoint["optimizer_state"] is not None:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(
                copy.deepcopy(initial_checkpoint["optimizer_state"])
            )

    def continuous_update_weights(self, replay_buffer, shared_storage):
        # Wait for the replay buffer to be filled
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        next_batch = replay_buffer.get_batch.remote()
        next_reused_batch = replay_buffer.get_reused_path_batch.remote()
        # Training loop
        while self.training_step < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            index_batch, batch = ray.get(next_batch)
            reused_index_batch,reused_batch = ray.get(next_reused_batch)
            next_batch = replay_buffer.get_batch.remote()
            next_reused_batch = replay_buffer.get_reused_path_batch.remote()
            self.update_lr()
            (
                priorities,
                total_loss,
                value_loss,
                reward_loss,
                policy_loss,
                pc_value_loss,
                consist_loss,
                reused_reward_loss,
                reused_hidden_loss
            ) = self.update_weights(batch,reused_batch=reused_batch)

            if self.config.PER:
                # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
                replay_buffer.update_priorities.remote(priorities, index_batch)

            # Save to the shared storage
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.get_weights()),
                        "optimizer_state": copy.deepcopy(
                            models.dict_to_cpu(self.optimizer.state_dict())
                        ),
                    }
                )
                if self.config.save_model:
                    shared_storage.save_checkpoint.remote()
            shared_storage.set_info.remote(
                {
                    "training_step": self.training_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "total_loss": total_loss,
                    "value_loss": value_loss,
                    "reward_loss": reward_loss,
                    "policy_loss": policy_loss,
                    "pc_value_loss":pc_value_loss,
                    "consist_loss":consist_loss,
                    'reused_reward_loss' : reused_reward_loss,
                    'reused_hidden_loss' : reused_hidden_loss
                }
            )

            # Managing the self-play / training ratio
            if self.config.training_delay:
                time.sleep(self.config.training_delay)
            if self.config.ratio:
                while (
                    self.training_step
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    > self.config.ratio
                    and self.training_step < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

    def update_weights(self, batch , reused_batch = None):
        """
        Perform one training step.
        """

        (
            observation_batch,
            action_batch,
            target_value,
            target_reward,
            target_policy,
            weight_batch,
            gradient_scale_batch,
            pc_value_batch
        ) = batch
        reused_loss = 0

        # Keep values as scalars for calculating the priorities for the prioritized replay
        target_value_scalar = numpy.array(target_value, dtype="float32")
        priorities = numpy.zeros_like(target_value_scalar)

        device = next(self.model.parameters()).device
        
        if reused_batch :
            reused_reward_loss, reused_hidden_loss = self.cal_reused_loss(reused_batch,device)
        
        if self.config.PER:
            weight_batch = torch.tensor(weight_batch.copy()).float().to(device)
        observation_batch = (
            torch.tensor(numpy.array(observation_batch)).float().to(device)
        )
        action_batch = torch.tensor(action_batch).long().to(device).unsqueeze(-1)
        target_value = torch.tensor(target_value).float().to(device)
        target_reward = torch.tensor(target_reward).float().to(device)
        target_policy = torch.tensor(target_policy).float().to(device)
        gradient_scale_batch = torch.tensor(gradient_scale_batch).float().to(device)
        if self.config.PC_constraint :
            target_pc_value = torch.tensor(pc_value_batch).float().to(device)
        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1, 1 (unsqueeze)
        # target_value: batch, num_unroll_steps+1
        # target_reward: batch, num_unroll_steps+1
        # target_policy: batch, num_unroll_steps+1, len(action_space)
        # gradient_scale_batch: batch, num_unroll_steps+1

        target_value = models.scalar_to_support(target_value, self.config.support_size)
        target_reward = models.scalar_to_support(
            target_reward, self.config.support_size
        )
        
        if self.config.PC_constraint :
            target_pc_value = models.scalar_to_support(target_pc_value, self.config.support_size)
        # target_value: batch, num_unroll_steps+1, 2*support_size+1
        # target_reward: batch, num_unroll_steps+1, 2*support_size+1
        projection_batch = []
        projection_target_batch = []
        value_feature_batch = []
        value_feature_target_batch = []
        ## Generate predictions
        value, reward, policy_logits, hidden_state = self.model.initial_inference(
            observation_batch
        )
        predictions = [(value, reward, policy_logits)]

        #make representation self-supervised batch
        if self.config.representation_consistency :
            projection_batch.append( self.model.project(hidden_state,with_grad = True))
        for i in range(1, action_batch.shape[1]):
            value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
                hidden_state, action_batch[:, i]
            )
            # Scale the gradient at the start of the dynamics function (See paper appendix Training)
            hidden_state.register_hook(lambda grad: grad * 0.5)
            predictions.append((value, reward, policy_logits))
            #make representation self-supervised batch
            if self.config.representation_consistency :
                projection_batch.append( self.model.project(hidden_state,with_grad = True))
                projection_target_batch.append(self.model.project(hidden_state,with_grad = False))
        # predictions: num_unroll_steps+1, 3, batch, 2*support_size+1 | 2*support_size+1 | 9 (according to the 2nd dim)
        if self.config.representation_consistency :
            projection_batch.pop()
        ## Compute losses
        value_loss, pc_value_loss, reward_loss, policy_loss = (0, 0, 0, 0)
        value, reward, policy_logits = predictions[0]
        current_pc_value_loss = 0
        # Ignore reward loss for the first batch step
        if self.config.PC_constraint :
            current_value_loss, current_pc_value_loss, _, current_policy_loss = self.loss_function_for_pc_value(
                value.squeeze(-1),
                reward.squeeze(-1),
                policy_logits,
                target_value[:, 0],
                target_reward[:, 0],
                target_policy[:, 0],
                target_pc_value[:, 0]
            )
        else :
            current_value_loss, _, current_policy_loss = self.loss_function(
                value.squeeze(-1),
                reward.squeeze(-1),
                policy_logits,
                target_value[:, 0],
                target_reward[:, 0],
                target_policy[:, 0],
            )
        
        value_loss += current_value_loss
        policy_loss += current_policy_loss
        pc_value_loss += current_pc_value_loss
        # Compute priorities for the prioritized replay (See paper appendix Training)
        pred_value_scalar = (
            models.support_to_scalar(value, self.config.support_size)
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )
        priorities[:, 0] = (
            numpy.abs(pred_value_scalar - target_value_scalar[:, 0])
            ** self.config.PER_alpha
        )
        consist_loss = 0
        if self.config.representation_consistency :
            projection,projection_target = projection_batch[0],projection_target_batch[0]
            consist_loss += self.consist_loss_func(projection,projection_target)        
            for i in range(1, len(projection_batch)) :
                projection,projection_target = projection_batch[i],projection_target_batch[i]
                current_consist_loss = self.consist_loss_func(projection,projection_target).sum()
                consist_loss += current_consist_loss

        for i in range(1, len(predictions)):
            value, reward, policy_logits = predictions[i]
            
            if self.config.PC_constraint :
                (
                    current_value_loss,
                    current_pc_value_loss,
                    current_reward_loss,
                    current_policy_loss,
                ) = self.loss_function_for_pc_value(
                    value.squeeze(-1),
                    reward.squeeze(-1),
                    policy_logits,
                    target_value[:, i],
                    target_reward[:, i],
                    target_policy[:, i],
                    target_pc_value[:, i]
                )            
            else :
                (
                    current_value_loss,
                    current_reward_loss,
                    current_policy_loss,
                ) = self.loss_function(
                    value.squeeze(-1),
                    reward.squeeze(-1),
                    policy_logits,
                    target_value[:, i],
                    target_reward[:, i],
                    target_policy[:, i],
                )
            # Scale gradient by the number of unroll steps (See paper appendix Training)
            current_value_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )
            current_reward_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )
            current_policy_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )
            
            if self.config.PC_constraint :
                current_pc_value_loss.register_hook(
                    lambda grad: grad / gradient_scale_batch[:, i]
                )

            value_loss += current_value_loss
            pc_value_loss += current_pc_value_loss
            reward_loss += current_reward_loss
            policy_loss += current_policy_loss
            

            # Compute priorities for the prioritized replay (See paper appendix Training)
            pred_value_scalar = (
                models.support_to_scalar(value, self.config.support_size)
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            priorities[:, i] = (
                numpy.abs(pred_value_scalar - target_value_scalar[:, i])
                ** self.config.PER_alpha
            )

        # Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze)
        loss = value_loss * self.config.value_loss_weight + reward_loss + policy_loss + pc_value_loss * self.config.pc_value_loss_weight \
            + consist_loss * self.config.consist_loss_weight
        if self.config.PER:
            # Correct PER bias by using importance-sampling (IS) weights
            loss *= weight_batch
        # Mean over batch dimension (pseudocode do a sum)
        loss = loss.mean() + reused_reward_loss + reused_hidden_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1
        
        if reused_batch :
            reused_reward_loss = reused_reward_loss.item()
            reused_hidden_loss = reused_hidden_loss.item()
        
        if self.config.PC_constraint and self.config.representation_consistency : 
            return (
                priorities,
                # For log purpose
                loss.item(),
                value_loss.mean().item(),
                reward_loss.mean().item(),
                policy_loss.mean().item(),
                pc_value_loss.mean().item(),
                consist_loss.mean().item(),
                reused_reward_loss,
                reused_hidden_loss
            )
        elif self.config.PC_constraint :
            return (
                priorities,
                # For log purpose
                loss.item(),
                value_loss.mean().item(),
                reward_loss.mean().item(),
                policy_loss.mean().item(),
                pc_value_loss.mean().item(),
                0,
                reused_reward_loss,
                reused_hidden_loss
            )
        elif self.config.representation_consistency :
            return (
                priorities,
                # For log purpose
                loss.item(),
                value_loss.mean().item(),
                reward_loss.mean().item(),
                policy_loss.mean().item(),
                0,
                consist_loss.mean().item()
                ,
                reused_reward_loss,
                reused_hidden_loss
            )
        else :
            return (
                priorities,
                # For log purpose
                loss.item(),
                value_loss.mean().item(),
                reward_loss.mean().item(),
                policy_loss.mean().item(),
                0,
                0,
                reused_reward_loss,
                reused_hidden_loss
            )

    def update_lr(self):
        """
        Update learning rate
        """
        lr = self.config.lr_init * self.config.lr_decay_rate ** (
            self.training_step / self.config.lr_decay_steps
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def cal_reused_loss(self,batch,device) :
        (   
            observation_batch,
            action_batch,
            reward_batch,
            target_observation_batch,
            weight_batch
        ) = batch
        if self.config.PER:
            weight_batch = torch.tensor(weight_batch.copy()).float().to(device)
        action_batch = torch.tensor(action_batch).long().to(device).unsqueeze(-1)
        reward_batch = torch.tensor(reward_batch).float().to(device)
        observation_batch = (
            torch.tensor(numpy.array(observation_batch)).float().to(device)
        )
        target_observation_batch = (
            torch.tensor(numpy.array(target_observation_batch)).float().to(device)
        )
        reward_batch = models.scalar_to_support(
            reward_batch, self.config.support_size
        )
        reward_loss, hidden_loss = 0,0
        _, reward, _, hidden_state = self.model.recurrent_inference(
                self.model.initial_inference(observation_batch)[3], action_batch
        )
        with torch.no_grad() :
            _, _, _, hidden_state_target = self.model.initial_inference(
                target_observation_batch
            )
        curr_loss = self.reused_loss_function(reward,hidden_state,reward_batch,hidden_state_target)
        curr_reward_loss, curr_hidden_loss = curr_loss
        curr_reward_loss = curr_reward_loss *  self.config.reused_reward_loss_weight
        curr_hidden_loss = curr_hidden_loss * self.config.hidden_loss_weight
        reward_loss += curr_reward_loss.mean() * weight_batch
        hidden_loss += curr_hidden_loss.mean() * weight_batch
        return reward_loss,hidden_loss
    
    @staticmethod
    def loss_function(
        value,
        reward,
        policy_logits,
        target_value,
        target_reward,
        target_policy,
    ):
        # Cross-entropy seems to have a better convergence than MSE
        value_loss = (-target_value * torch.nn.LogSoftmax(dim=1)(value)).sum(1)
        reward_loss = (-target_reward * torch.nn.LogSoftmax(dim=1)(reward)).sum(1)
        policy_loss = (-target_policy * torch.nn.LogSoftmax(dim=1)(policy_logits)).sum(
            1
        )
        return value_loss, reward_loss, policy_loss

    @staticmethod
    def loss_function_for_pc_value(
        value,
        reward,
        policy_logits,
        target_value,
        target_reward,
        target_policy,
        target_pc_value
    ):
        # Cross-entropy seems to have a better convergence than MSE
        value_loss = (-target_value * torch.nn.LogSoftmax(dim=1)(value)).sum(1)
        pc_value_loss = (-target_pc_value * torch.nn.LogSoftmax(dim=1)(value)).sum(1)
        reward_loss = (-target_reward * torch.nn.LogSoftmax(dim=1)(reward)).sum(1)
        policy_loss = (-target_policy * torch.nn.LogSoftmax(dim=1)(policy_logits)).sum(
            1
        )
        return value_loss,pc_value_loss, reward_loss, policy_loss

    @staticmethod
    def reused_loss_function(
        reward,
        hidden,
        target_reward,
        target_hidden,
        
    ):  
        hidden = hidden.view(target_reward.shape[0],-1)
        hidden_size = math.sqrt(hidden.shape[1])
        target_hidden = target_hidden.view(target_reward.shape[0],-1)
        # Cross-entropy seems to have a better convergence than MSE
        reward_loss = ((-target_reward * torch.nn.LogSoftmax(dim=1)(reward)) ).sum(1)
        hidden_loss = ((-target_hidden * torch.nn.LogSoftmax(dim=1)(hidden)) ).sum(1)
        hidden_loss = hidden_loss/ hidden_size
        
        return reward_loss, hidden_loss

    @staticmethod
    def consist_loss_func(f1, f2):
        """Consistency loss function: similarity loss
        Parameters
        """
        f1 = F.normalize(f1, p=2., dim=-1, eps=1e-5)
        f2 = F.normalize(f2, p=2., dim=-1, eps=1e-5)
        return -(f1 * f2).sum(dim=1)