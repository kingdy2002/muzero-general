from ast import While
import datetime
import pathlib
from urllib.error import ContentTooShortError

import numpy as np
import torch

from collections import deque
import copy

from .abstract_game import AbstractGame

SIZE = 8
BLACK = 1
WHITE = -1
EMPTY = 0

class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = 8  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (5, SIZE+2, SIZE+2)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(SIZE*SIZE))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "random"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 20  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = True
        self.max_moves = SIZE*SIZE  # Maximum number of moves if game is not finished before
        self.num_simulations = 200  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time
        self.self_play_update_interval = 1000

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
        self.channels = 128  # Number of channels in the ResNet
        self.reduced_channels_reward = 8  # Number of channels in reward head
        self.reduced_channels_value = 8  # Number of channels in value head
        self.reduced_channels_policy = 16  # Number of channels in policy head
        self.resnet_fc_reward_layers = [64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64]  # Define the hidden layers in the policy head of the prediction network
        
        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 1000000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 256  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        ### Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 5  # Number of game moves to keep for every batch element
        self.td_steps = 100  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 1  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = True



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = 0.5  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
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
        self.Train_dynamic = True
        
        #reuse MCT
        self.num_branch = 5
        self.reused_ratio = 1
        self.reused_unroll_step = 5
        self.reused_reward_loss_weight = 1
        self.hidden_loss_weight = 0.1
        self.reused_path_is_real = False
        self.reuse_batch_size = self.batch_size * 1


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Hex()

    def step(self, action):
        """
        Apply action to the game.
        Args:
            action : action of the action_space to take.
        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        #print(self.action_to_string(action))
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        Return the current player.
        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play

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
        self.env.show()

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.
        Returns:
            An integer from the action space.
        """
        choice = input(f"Enter the column to play for the player {self.to_play()}: ")
        while choice not in [str(action) for action in self.legal_actions()]:
            choice = input("Enter another column : ")
        return int(choice)

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training
        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.
        Args:
            action_number: an integer from the action space.
        Returns:
            String representing the action.
        """
        player = 'BLACK' if self.to_play() == BLACK else 'WHITE'
        action_id = self.env.action_parse(action_number)
        x = action_number // SIZE
        y = action_number % SIZE
        return f"{player} act {x}, {y}"

class IllegalMove(Exception):
    pass

class Hex:
    def __init__(self, board=np.zeros(shape=(SIZE, SIZE), dtype=np.int8), winner=EMPTY, to_play=BLACK, step=0):
        self.board = board.copy()
        self.winner = winner
        self.to_play = to_play
        self.step = step

    def reset(self) :
        self.board = np.zeros(shape=(SIZE, SIZE))
        self.winner = EMPTY
        self.to_play = BLACK
        self.step = 0
        return self.to_feature()

    def step(self,action) :
        self.apply_action(action)
        reward = 1 if self.is_game_over() != EMPTY else 0
        return self.to_feature(),reward,self.is_game_over()
    
    def is_move_legal(self, x, y):
        return self.board[x][y] == EMPTY

    def reach(self, x, y, target, color):
        mask = np.zeros(shape=(SIZE, SIZE), dtype=np.int8)
        queue = deque()
        queue.append((x, y))
        mask[x][y] = 1
        while len(queue) != 0:
            e = queue.popleft()
            x, y = e[0], e[1]
            if self.board[x][y] == color:
                if color == BLACK and x == target:
                    return True
                elif color == WHITE and y == target:
                    return True

                dx = [0, 0, -1, -1, 1, 1]
                dy = [-1, 1, 0, 1, -1, 0]
                for i in range(6):
                    nx = x + dx[i]
                    ny = y + dy[i]
                    if 0 <= nx and nx < SIZE and 0 <= ny and ny < SIZE \
                            and mask[nx][ny] == 0 and self.board[nx][ny] == color:
                        queue.append((nx, ny))
                        mask[nx][ny] = 1
                
        return False

    def apply_action(self,action, color=None) :
        x = action // SIZE
        y = action % SIZE
        if color == None:
            color = self.to_play
        if x == SIZE and y == SIZE:
            self.winner = -color
            
        if not self.is_move_legal(x, y):
            raise IllegalMove("{} move at {} is illegal: \n{}".format(
                "Black" if self.to_play == BLACK else "White", (x, y), self))
        self.board[x][y] = color
        if self.reach(x, y, 0, color) and self.reach(x, y, SIZE - 1, color):
            self.winner = color
        self.to_play = -self.to_play
        self.step = self.step + 1
    
    def move(self, x, y, color=None):
        pos = copy.deepcopy(self)
        if color == None:
            color = self.to_play
        if x == SIZE and y == SIZE:
            pos.winner = -color
            return pos
        if not pos.is_move_legal(x, y):
            raise IllegalMove("{} move at {} is illegal: \n{}".format(
                "Black" if self.to_play == BLACK else "White", (x, y), self))
        pos.board[x][y] = color
        if pos.reach(x, y, 0, color) and pos.reach(x, y, SIZE - 1, color):
            pos.winner = color
        pos.to_play = -pos.to_play
        pos.step = self.step + 1
        return pos

    def is_game_over(self):
        over = (self.step == SIZE * SIZE or self.winner != EMPTY)
        return over

    def result(self):
        return self.winner

    def __deepcopy__(self, memodict={}):
        new_board = np.copy(self.board)
        return Hex(new_board, self.winner, self.to_play, self.step)

    def legal_actions(self):
        ret = []
        for i in range(SIZE):
            for j in range(SIZE):
                if self.board[i][j] == EMPTY:
                    ret.append(i*SIZE + j)
        return ret

    def to_feature(self):
        ret = np.zeros(shape=(5, SIZE+2, SIZE+2))
        for i in range(SIZE + 2):
            for j in range(SIZE + 2):
                if i == 0 or i == SIZE + 1:
                    if 1 <= j and j <= SIZE:
                        ret[0][i][j] = 1
                elif j == 0 or j == SIZE + 1:
                    if 1 <= i and i <= SIZE:
                        ret[1][i][j] = 1
                else:
                    if self.board[i-1][j-1] == BLACK:
                        ret[0][i][j] = 1
                    elif self.board[i-1][j-1] == WHITE:
                        ret[1][i][j] = 1
                    else:
                        ret[2][i][j] = 1
        if self.to_play == BLACK:
            ret[3, :, :] = 1
        else:
            ret[4, :, :] = 1
        return ret

    def __repr__(self):
        retstr = '\n' + ' '
        for i in range(SIZE):
            retstr += chr(ord('a') + i) + ' '
        retstr += '\n'
        for i in range(0, SIZE):
            if i <= 8:
                retstr += ' ' * i + str(i + 1) + ' '
            else:
                retstr += ' ' * (i - 1) + str(i + 1) + ' '
            for j in range(0, SIZE):
                if self.board[i, j] == BLACK:
                    retstr += 'X'
                elif self.board[i, j] == WHITE:
                    retstr += 'O'
                else:
                    retstr += '.'
                retstr += ' '
            retstr += '\n'
        return retstr

    def show(self):
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i][j] == 1:
                    print(' B ', end="")
                elif self.board[i][j] == -1:
                    print(' W ', end="" )
                else:
                    print(' O ', end="")
            print('\n')