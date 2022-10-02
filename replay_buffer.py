import copy
import time

import numpy
import ray
import torch

import models

from self_play import MCTS

@ray.remote
class ReplayBuffer:
    """
    Class which run in a dedicated thread to store played games and generate batch.
    """

    def __init__(self, initial_checkpoint, initial_buffer, config, shared_storage):
        self.config = config
        self.shared_storage = shared_storage
        self.buffer = copy.deepcopy(initial_buffer)
        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_checkpoint["num_played_steps"]
        self.total_samples = sum(
            [len(game_history.root_values) for game_history in self.buffer.values()]
        )
        if self.total_samples != 0:
            print(
                f"Replay buffer initialized with {self.total_samples} samples ({self.num_played_games} games).\n"
            )

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.replay_buffer_on_gpu else "cpu"))
        self.model.eval()

    def save_game(self, game_history, shared_storage=None):
        if self.config.PER:
            if game_history.priorities is not None:
                # Avoid read only array when loading replay buffer from disk
                game_history.priorities = numpy.copy(game_history.priorities)
            else:
                # Initial priorities for the prioritized replay (See paper appendix Training)
                priorities = []
                for i, root_value in enumerate(game_history.root_values):
                    priority = (
                        numpy.abs(
                            root_value - self.compute_target_value(game_history, i)
                        )
                        ** self.config.PER_alpha
                    )
                    priorities.append(priority)

                game_history.priorities = numpy.array(priorities, dtype="float32")
                game_history.game_priority = numpy.max(game_history.priorities)

        self.buffer[self.num_played_games] = game_history
        self.num_played_games += 1
        self.num_played_steps += len(game_history.root_values)
        self.total_samples += len(game_history.root_values)

        if self.config.replay_buffer_size < len(self.buffer):
            del_id = self.num_played_games - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].root_values)
            del self.buffer[del_id]

        if shared_storage:
            shared_storage.set_info.remote("num_played_games", self.num_played_games)
            shared_storage.set_info.remote("num_played_steps", self.num_played_steps)

    def get_buffer(self):
        return self.buffer

    def get_batch(self):
        (
            index_batch,
            observation_batch,
            action_batch,
            reward_batch,
            value_batch,
            policy_batch,
            gradient_scale_batch,
            pc_value_batch
        ) = ([], [], [], [], [], [], [], [])
        weight_batch = [] if self.config.PER else None

        game_historys = []
        list_game_pos = []
        
        for game_id, game_history, game_prob in self.sample_n_games(
            self.config.batch_size
        ):
            game_pos, pos_prob = self.sample_position(game_history)

            game_historys.append(game_history)
            list_game_pos.append(game_pos)

            values, rewards, policies, actions = self.make_target(
                game_history, game_pos
            )
            

            index_batch.append([game_id, game_pos])
            observation_batch.append(
                game_history.get_stacked_observations(
                    game_pos,
                    self.config.stacked_observations,
                    len(self.config.action_space),
                )
            )
            
            
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            gradient_scale_batch.append(
                [
                    min(
                        self.config.num_unroll_steps,
                        len(game_history.action_history) - game_pos,
                    )
                ]
                * len(actions)
            )
            if self.config.PER:
                weight_batch.append(1 / (self.total_samples * game_prob * pos_prob))

        if self.config.PER:
            weight_batch = numpy.array(weight_batch, dtype="float32") / max(
                weight_batch
            )

        if self.config.PC_constraint :
            pc_value_batch = self.get_pc_value_batch(game_historys,list_game_pos)
            lis = []
            for pc_value in pc_value_batch :
                lis.append(numpy.pad(pc_value, (0, self.config.num_unroll_steps + 1), 'constant', constant_values=0)[:self.config.num_unroll_steps + 1])
            pc_value_batch = numpy.array(lis)
        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1
        # value_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        # weight_batch: batch
        # gradient_scale_batch: batch, num_unroll_steps+1
        
        #pc_value_batch: avg value for PC
        return (
            index_batch,
            (
                observation_batch,
                action_batch,
                value_batch,
                reward_batch,
                policy_batch,
                weight_batch,
                gradient_scale_batch,
                pc_value_batch
            ),
        )

    def sample_game(self, force_uniform=False):
        """
        Sample game from buffer either uniformly or according to some priority.
        See paper appendix Training.
        """
        game_prob = None
        if self.config.PER and not force_uniform:
            game_probs = numpy.array(
                [game_history.game_priority for game_history in self.buffer.values()],
                dtype="float32",
            )
            game_probs /= numpy.sum(game_probs)
            game_index = numpy.random.choice(len(self.buffer), p=game_probs)
            game_prob = game_probs[game_index]
        else:
            game_index = numpy.random.choice(len(self.buffer))
        game_id = self.num_played_games - len(self.buffer) + game_index

        return game_id, self.buffer[game_id], game_prob

    def sample_n_games(self, n_games, force_uniform=False):
        if self.config.PER and not force_uniform:
            game_id_list = []
            game_probs = []
            for game_id, game_history in self.buffer.items():
                game_id_list.append(game_id)
                game_probs.append(game_history.game_priority)
            game_probs = numpy.array(game_probs, dtype="float32")
            game_probs /= numpy.sum(game_probs)
            game_prob_dict = dict(
                [(game_id, prob) for game_id, prob in zip(game_id_list, game_probs)]
            )
            selected_games = numpy.random.choice(game_id_list, n_games, p=game_probs)
        else:
            selected_games = numpy.random.choice(list(self.buffer.keys()), n_games)
            game_prob_dict = {}
        ret = [
            (game_id, self.buffer[game_id], game_prob_dict.get(game_id))
            for game_id in selected_games
        ]
        return ret

    def sample_position(self, game_history, force_uniform=False):
        """
        Sample position from game either uniformly or according to some priority.
        See paper appendix Training.
        """
        position_prob = None
        if self.config.PER and not force_uniform:
            position_probs = game_history.priorities / sum(game_history.priorities)
            position_index = numpy.random.choice(len(position_probs), p=position_probs)
            position_prob = position_probs[position_index]
        else:
            position_index = numpy.random.choice(len(game_history.root_values))

        return position_index, position_prob

    def update_game_history(self, game_id, game_history):
        # The element could have been removed since its selection and update
        if next(iter(self.buffer)) <= game_id:
            if self.config.PER:
                # Avoid read only array when loading replay buffer from disk
                game_history.priorities = numpy.copy(game_history.priorities)
            self.buffer[game_id] = game_history

    def update_priorities(self, priorities, index_info):
        """
        Update game and position priorities with priorities calculated during the training.
        See Distributed Prioritized Experience Replay https://arxiv.org/abs/1803.00933
        """
        for i in range(len(index_info)):
            game_id, game_pos = index_info[i]

            # The element could have been removed since its selection and training
            if next(iter(self.buffer)) <= game_id:
                # Update position priorities
                priority = priorities[i, :]
                start_index = game_pos
                end_index = min(
                    game_pos + len(priority), len(self.buffer[game_id].priorities)
                )
                self.buffer[game_id].priorities[start_index:end_index] = priority[
                    : end_index - start_index
                ]

                # Update game priorities
                self.buffer[game_id].game_priority = numpy.max(
                    self.buffer[game_id].priorities
                )

    def compute_target_value(self, game_history, index):
        # The value target is the discounted root value of the search tree td_steps into the
        # future, plus the discounted sum of all rewards until then.
        bootstrap_index = index + self.config.td_steps
        if bootstrap_index < len(game_history.root_values):
            root_values = (
                game_history.root_values
                if game_history.reanalysed_predicted_root_values is None
                else game_history.reanalysed_predicted_root_values
            )
            last_step_value = (
                root_values[bootstrap_index]
                if game_history.to_play_history[bootstrap_index]
                == game_history.to_play_history[index]
                else -root_values[bootstrap_index]
            )

            value = last_step_value * self.config.discount**self.config.td_steps
        else:
            value = 0

        for i, reward in enumerate(
            game_history.reward_history[index + 1 : bootstrap_index + 1]
        ):
            # The value is oriented from the perspective of the current player
            value += (
                reward
                if game_history.to_play_history[index]
                == game_history.to_play_history[index + i]
                else -reward
            ) * self.config.discount**i

        return value

    def make_target(self, game_history, state_index):
        """
        Generate targets for every unroll steps.
        """
        target_values, target_rewards, target_policies, actions  = [], [], [], []
        for current_index in range(
            state_index, state_index + self.config.num_unroll_steps + 1
        ):
            value = self.compute_target_value(game_history, current_index)
            if current_index < len(game_history.root_values):
                target_values.append(value)
                target_rewards.append(game_history.reward_history[current_index])
                target_policies.append(game_history.child_visits[current_index])
                actions.append(game_history.action_history[current_index])
            elif current_index == len(game_history.root_values):
                target_values.append(0)
                target_rewards.append(game_history.reward_history[current_index])
                # Uniform policy
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(game_history.action_history[current_index])
            else:
                # States past the end of games are treated as absorbing states
                target_values.append(0)
                target_rewards.append(0)
                # Uniform policy
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(numpy.random.choice(self.config.action_space))

        return target_values, target_rewards, target_policies, actions 

    def get_pc_value_batch(self,game_historys,list_game_pos) :
        observation_batchs , action_batchs , index_batchs = self.get_pre_batch_for_pc_value_batch(game_historys,list_game_pos)
        sum_of_value = [[] for i in range(len(observation_batchs))] 
        batchs = self.get_batch_for_pc_value_batch(action_batchs,self.config.heuristic_len)
        first = 1
        device = torch.device("cuda" if self.config.selfplay_on_gpu else "cpu")
        hidden_state = []
        with torch.no_grad(): 
            for action_batch,observation_indexs in batchs :
                action_batch = torch.tensor(action_batch).long().to(device).unsqueeze(-1)
                observation_batch = (
                    torch.tensor(numpy.array(observation_batchs)).float().to(device)
                )
                if first : 
                    value, reward, policy_logits, hidden_state = self.model.initial_inference(
                        observation_batch
                    )
                    first = 0
                    
                value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
                        hidden_state, action_batch
                )
                value = torch.index_select(value, 0, torch.LongTensor(observation_indexs).to(device))
                
                for i,val in zip(observation_indexs , models.support_to_scalar(value, self.config.support_size).detach().cpu().numpy().squeeze()) :
                    if len(sum_of_value[i]) % 2 == 0 :
                        sum_of_value[i].append(-val)
                    else :
                        sum_of_value[i].append(val)
                
                
            pc_value_batch = self.reconstruct_pc_value(game_historys,sum_of_value,index_batchs)
            
        return pc_value_batch
            
    def get_pre_batch_for_pc_value_batch(self,game_historys, list_game_pos) :
        observation_batchs , action_batchs, index_batchs = [], [] , []
        for state_index,game_history in zip(list_game_pos,game_historys) :
            if (state_index + self.config.num_unroll_steps) >= len(game_history.heuristic_path_action) :
                end = len(game_history.heuristic_path_action)
            else :
                end = state_index + self.config.num_unroll_steps
            end = min(state_index + self.config.num_unroll_steps,len(game_history.heuristic_path_action) - 1)
            indexs = list(range(state_index, end))
            index_batchs.append(indexs)
            observation_batch, action_batch = self.game_history_2_observation_action(game_history, indexs)
            observation_batchs.extend(observation_batch)
            action_batchs.extend(action_batch)
        return observation_batchs , action_batchs , index_batchs

    def game_history_2_observation_action(self,game_history, indexs) :
        observation_batch = []
        action_batch = []
        for game_pos in indexs :
            observation_batch.append(
                game_history.get_stacked_observations(
                    game_pos,
                    self.config.stacked_observations,
                    len(self.config.action_space),
                )
            )
            action_batch.append(game_history.heuristic_path_action[game_pos+1[0]])  

        return observation_batch, action_batch

    def get_batch_for_pc_value_batch(self,action_batchs, length) :
        for index in range(length) :
            action_batch,observation_indexs = [],[]
            for i, actions in enumerate(action_batchs) :
                if len(actions) > index :
                    observation_indexs.append(i)
                    action_batch.append(actions[index])
                else :
                    action_batch.append(0)
            yield action_batch ,observation_indexs
    
    def reconstruct_pc_value(self,game_historys,sum_of_value,index_batchs) :
        heruistical_value = self.cal_heruistical_value(sum_of_value,index_batchs)
        historical_value = self.cal_historical_value(game_historys,index_batchs)
        pc_value_batch = []
        for heruistical_paths,historical_paths in zip(heruistical_value,historical_value) :
            lis = []
            for heruistical_path,historical_path in zip(heruistical_paths,historical_paths) :
                value = (sum(heruistical_path) + sum(historical_path)) / (len(heruistical_path) + len(historical_path))
                lis.append(value)
            pc_value_batch.append(lis)
        return pc_value_batch
    
    def cal_heruistical_value(self,sum_of_value,index_batchs) :
        index = 0
        value_batch = []
        for indexs in index_batchs :
            length = len(indexs)
            lis = []
            for i in range(length) :
                lis.append(sum_of_value[index])
                index += 1
            if length < self.config.num_unroll_steps :
                for i in range(self.config.num_unroll_steps - length) :
                    lis.append([0])
            value_batch.append(lis)
            
            
        return value_batch
    
    def cal_historical_value(self,game_historys,index_batchs):
        historical_path = []
        historical_batch = []
        for game_history,indexs in zip(game_historys,index_batchs) :
            root_values = (
                game_history.root_values
                if game_history.reanalysed_predicted_root_values is None
                else game_history.reanalysed_predicted_root_values
            )
            lis = []
            for game_pos in indexs :
            
                if game_pos < self.config.historic_len :
                    historical_path = [i * (-1)**n for n,i in enumerate(root_values[game_pos: : -1])]
                
                else :
                    historical_path = [i * (-1)**n for n,i in enumerate(root_values[game_pos : game_pos-self.config.historic_len-1 : -1])]
                lis.append(historical_path)
            historical_batch.append(lis)
        return historical_batch
    
    def get_reused_path_batch(self) :
        (
            index_batch,
            observation_batch,
            action_batch,
            reward_batch,
            target_observation_batch

        ) = ([], [], [], [], [])
        weight_batch = [] if self.config.PER else None
        for game_id, game_history, game_prob in self.sample_n_games(
            self.config.reuse_batch_size
        ):
            game_pos, pos_prob = self.sample_position(game_history)
            if game_pos == 0 :
                while game_pos == 0 :
                    game_pos, pos_prob = self.sample_position(game_history)
            
            observations ,actions, rewards, target_observation = self.make_reused_target(
                game_history, game_pos
            )
            

            index_batch.append([game_id, game_pos])
            action_batch.extend(actions)
            observation_batch.extend(observations)
            reward_batch.extend(rewards)
            target_observation_batch.extend(target_observation)
            if self.config.PER:
                for i in range(len(actions)) :
                    weight_batch.append(1 / (self.total_samples * game_prob * pos_prob))

        if self.config.PER:
            weight_batch = numpy.array(weight_batch, dtype="float32") / max(
                weight_batch
            )
        return (
            index_batch,
            (
                observation_batch,
                action_batch,
                reward_batch,
                target_observation_batch,
                weight_batch
            ),
        )    
        
    def make_reused_target(self, game_history, state_index):
        """
        Generate targets for every unroll steps.
        """
        observations , actions , rewards, target_observation  = [], [], [], []
        heuristic_path_actions = game_history.heuristic_path_action[state_index]
        heuristic_real_rollout_paths = game_history.heuristic_real_rollout_path[state_index]
        for i , (heuristic_path_action,heuristic_real_rollout_path) in enumerate(zip(heuristic_path_actions,heuristic_real_rollout_paths)) :
            pre_obs = game_history.get_stacked_observations_heuristic(
                    state_index,
                    self.config.stacked_observations,
                    len(self.config.action_space),
                    i,
                    0
                )
            for pos, obs in enumerate(heuristic_real_rollout_path[1:]) :
                pos_obs = game_history.get_stacked_observations_heuristic(
                    state_index,
                    self.config.stacked_observations,
                    len(self.config.action_space),
                    i,
                    pos+1
                )
                observations.append(pre_obs)
                target_observation.append(pos_obs)
                rewards.append(obs[1])
                actions.append(heuristic_path_action[pos])
                pre_obs = pos_obs
        
        choice  = numpy.random.choice(len(observations), self.config.reused_ratio)
        observations = [observations[i] for i in choice]
        actions = [actions[i] for i in choice]
        rewards = [rewards[i] for i in choice]
        target_observation = [target_observation[i] for i in choice]
        
        
        
        return observations ,actions, rewards, target_observation
"""
    def make_PC_value(self,game_history, indexs) :
        l = self.config.historic_len
        k = self.config.heuristic_len
        device = torch.device("cuda" if self.config.selfplay_on_gpu else "cpu")
        historical_path = []
        heuristical_path = []
        observation_batch = []
        action_batch = []
        for game_pos in indexs :
            observation_batch.append(
                game_history.get_stacked_observations(
                    game_pos,
                    self.config.stacked_observations,
                    len(self.config.action_space),
                )
            )
            action_batch.append(game_history.heuristic_path_action[game_pos])   
        self.model.set_weights(ray.get(self.shared_storage.get_info.remote("weights")))
        with torch.no_grad():         
            root_values = (
                    game_history.root_values
                    if game_history.reanalysed_predicted_root_values is None
                    else game_history.reanalysed_predicted_root_values
                )
            
            
            observation_batch = (
                torch.tensor(numpy.array(observation_batch)).float().to(device)
            )

            
            value, reward, policy_logits, hidden_state = self.model.initial_inference(
                observation_batch
            )
            
            for i in range(0, k):
                value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
                    hidden_state, torch.tensor(action_batch[:,i]).to(device)
                )
                value = models.support_to_scalar(value, self.config.support_size).detach().cpu().numpy().squeeze()
                heuristical_path.append(value)
            
            
            if game_pos < l :
                historical_path = [i * (-1)**(n+1) for n,i in enumerate(root_values[:game_pos : -1])]
            
            else :
                historical_path = [i * (-1)**(n+1) for n,i in enumerate(root_values[game_pos -i :game_pos : -1])]
            
            if len(heuristical_path) < k :
                heuristical_path = [i * (-1)**(n+1) for n,i in enumerate(heuristical_path)]
            else :
                heuristical_path = [i * (-1)**(n+1) for n,i in enumerate(heuristical_path[:k])]
                
            length = len(heuristical_path) + len(historical_path) + 1
            sum_value = sum(heuristical_path) + sum(historical_path) + root_values[game_pos]
            result = sum_value / length
        
        return result
"""
    
@ray.remote
class Reanalyse:
    """
    Class which run in a dedicated thread to update the replay buffer with fresh information.
    See paper appendix Reanalyse.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.reanalyse_on_gpu else "cpu"))
        self.model.eval()

        self.num_reanalysed_games = initial_checkpoint["num_reanalysed_games"]

    def reanalyse(self, replay_buffer, shared_storage):
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

            game_id, game_history, _ = ray.get(
                replay_buffer.sample_game.remote(force_uniform=True)
            )

            # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
            if self.config.use_last_model_value:
                observations = numpy.array(
                    [
                        game_history.get_stacked_observations(
                            i,
                            self.config.stacked_observations,
                            len(self.config.action_space),
                        )
                        for i in range(len(game_history.root_values))
                    ]
                )

                observations = (
                    torch.tensor(observations)
                    .float()
                    .to(next(self.model.parameters()).device)
                )
                values = models.support_to_scalar(
                    self.model.initial_inference(observations)[0],
                    self.config.support_size,
                )
                game_history.reanalysed_predicted_root_values = (
                    torch.squeeze(values).detach().cpu().numpy()
                )

            replay_buffer.update_game_history.remote(game_id, game_history)
            self.num_reanalysed_games += 1
            shared_storage.set_info.remote(
                "num_reanalysed_games", self.num_reanalysed_games
            )