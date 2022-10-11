import datetime
import pathlib
import time
from winreg import REG_OPTION_OPEN_LINK

import numpy
import torch

from .abstract_game import AbstractGame


# This is a Game wrapper for open_spiel games. It allows you to run any game in the open_spiel library.

from stockfish import Stockfish

stockfish = Stockfish(path="/home/dongyoung/stockfish_15_linux_x64_bmi2/stockfish_15_src/src/stockfish")

try:
    import pyspiel

except ImportError:
    import sys

    sys.exit(
        "You need to install open_spiel by running pip install open_spiel. For a full documentation, see: https://github.com/deepmind/open_spiel/blob/master/docs/install.md"
    )

# The game you want to run. See https://github.com/deepmind/open_spiel/blob/master/docs/games.md for a list of games
game = pyspiel.load_game("chess")

def logging_time(original_fn):
    import time
    from functools import wraps

    @wraps(original_fn)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)

        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time - start_time))
        return result
    return wrapper


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.game = game

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = 8  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape =  tuple(self.game.observation_tensor_shape()) # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(self.game.policy_tensor_shape()[0]))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(self.game.num_players()))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "self"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 4  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = True
        self.max_moves = 100  # Maximum number of moves if game is not finished before
        
        self.num_simulations = 200  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = 100  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 18  # Number of blocks in the ResNet
        self.channels = 512  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        self.resnet_fc_reward_layers = [64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 1000000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 512  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 50  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available
        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.001  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        ### Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 5  # Number of game moves to keep for every batch element
        self.td_steps = 100  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = True



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

        #PC constraint
        self.replay_buffer_on_gpu = False
        self.PC_constraint = False
        self.heuristic_len = 5
        self.historic_len = 5
        self.pc_value_loss_weight = 1
        
        #representaion self-supervised constraint
        self.representation_consistency = False
        self.consist_loss_weight = 1 
        
        #Train Dynamic consistency
        self.Train_dynamic = False
        
        #reuse MCT
        self.num_branch = 1
        self.reused_ratio = 1
        self.reused_unroll_step = 5
        self.reused_reward_loss_weight = 1
        self.hidden_loss_weight = 0.1
        self.reused_path_is_real = False
        self.reuse_batch_size = self.batch_size * 3


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Spiel()


    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        print('serial : ',self.env.serialize())
        print('serial 2 : ',self.env.serialize_game_and_state())
        observation, reward, done = self.env.step(action)
        self.env.board = self.env.game.BoardFromFEN(self.env.observation_string())
        print()
        
        return observation, reward * 20, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def legal_actions_human(self):
        return self.env.human_legal_actions()

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        while True:
            try:
                print("Legal Actions: ", self.legal_actions_human())
                choice = input("Enter your move: ")
                if choice in self.legal_actions_human():
                    break
            except:
                pass
            print("Wrong input, try again")

        return self.env.board.string_to_action(choice)

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        row = action_number // 3 + 1
        col = action_number % 3 + 1
        return f"Play row {row}, column {col}"


class Spiel:
    def __init__(self):
        self.game = game
        self.board = self.game.new_initial_state()
        self.player = 1

        
    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = self.game.new_initial_state()
        self.player = 1
        return self.get_observation()

    def step(self, action):
        if isinstance(action, str) :
            self.game.deserialize_state(action)
        
        else :
            self.board = self.board.child(action)

        done = self.board.is_terminal()

        reward = 1 if self.have_winner() else 0

        observation = self.get_observation()

        self.player *= -1

        return observation, reward, done

    def get_observation(self):
        if self.player == 1:
            current_player = 1
        else:
            current_player = 0
        return numpy.array(self.board.observation_tensor(current_player)).reshape(
            self.game.observation_tensor_shape()
        )

    def legal_actions(self):
        return self.board.legal_actions()

    def have_winner(self):
        rewards = self.board.rewards()

        if self.player == 1:

            if rewards[0] == 1.0:
                return True

        elif self.player == -1:
            if rewards[1] == 1.0:
                return True

        return False

    def human_legal_actions(self):
        return [self.board.action_to_string(x) for x in self.board.legal_actions()]

    def render(self):
        print(self.board)

    def action_to_string1(self,action) :
        return self.board.action_to_string(self.to_play(),action)

    def action_to_string2(self,action) :
        return self.game.action_to_string(self.to_play(),action)

    def serialize_game_and_state(self) :
        return pyspiel.serialize_game_and_state(self.game,self.board)

    def deserialize_game_and_state(self,str) :
        self.game,self.board =  pyspiel.serialize_game_and_state(str)
        
    def observation_string(self) :
        return self.board.observation_string()
        
    def serialize(self) :
        return self.board.serialize()
    
    def expert_action(self):
        stockfish.set_fen_position(self.observation_string())
        action = stockfish.get_best_move()
        stockfish.make_moves_from_current_position([action])
        return stockfish.get_fen_position()
    
    def action_to_uci(self,action) :
        kUnderPromotionDirectionToOffset = [[0,1],[1,1],[-1,1]]
        kUnderPromotionIndexToType = ['rook','bishop','knight']
        xy = int(action / 73)
        x = int(xy / 8)
        y = int(xy % 8)
        destination_index = action % 73
        is_under_promotion = destination_index < 9
        if is_under_promotion :
            promotion_index = int(destination_index / 3);
            direction_index = int(destination_index % 3);
            promotion_type = kUnderPromotionIndexToType[promotion_index]
            offset = kUnderPromotionDirectionToOffset[direction_index]

        else :
            destination_index -= 9
            offset = 