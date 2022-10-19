import datetime
import pathlib
from urllib.error import ContentTooShortError

import numpy
import torch

from collections import deque

from .abstract_game import AbstractGame

class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = 8  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (13, 6, 6)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(32*36))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "random"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 12  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = True
        self.max_moves = 50  # Maximum number of moves if game is not finished before
        self.num_simulations = 200  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

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
        self.batch_size = 64  # Number of parts of games to train on at each training step
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
        self.num_unroll_steps = 30  # Number of game moves to keep for every batch element
        self.td_steps = 50  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

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
        self.reused_unroll_step = 30
        self.reused_reward_loss_weight = 1
        self.hidden_loss_weight = 0.1
        self.reused_path_is_real = False
        self.reuse_batch_size = self.batch_size * 1

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.
        Returns:
            Positive float.
        """
        return 1


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Minichess()

    def step(self, action):
        """
        Apply action to the game.
        Args:
            action : action of the action_space to take.
        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward*10, done

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
        return f"Play column {action_number + 1}"


class Minichess:
    def __init__(self):
        self.reset()
        self.peice = ChessPeice()

    def to_play(self):
        return 0 if self.player == 1 else 1
        #white is 0
        #black is 1
    def reset(self):
        self.player = 1
        self.black_repetitions = 0
        self.white_repetitions = 0
        self.white_action_history = deque(maxlen = 3)
        self.black_action_history = deque(maxlen = 3)
        self.last_action = -1
        self.board = numpy.zeros((10, 6, 6), dtype="int32")
        self.peice_loc = numpy.zeros((6, 6), dtype="int32")
        self.life_peice = {
            'white' : ['p1','p2','p3','p4','p5','p6','r1','r2','n1','n2','q','k'],
            'black' : ['p1','p2','p3','p4','p5','p6','r1','r2','n1','n2','q','k']
        }
        self.white_peice_loc = {
            'p1':(0,1),
            'p2':(1,1),
            'p3':(2,1),
            'p4':(3,1),
            'p5':(4,1),
            'p6':(5,1),
            'r1':(0,0),
            'r2':(5,0),
            'n1':(1,0),
            'n2':(4,0),
            'q':(3,0),
            'k':(2,0)
        }
        self.black_peice_loc = {
            'p1':(0,4),
            'p2':(1,4),
            'p3':(2,4),
            'p4':(3,4),
            'p5':(4,4),
            'p6':(5,4),
            'r1':(0,5),
            'r2':(5,5),
            'n1':(1,5),
            'n2':(4,5),
            'q':(3,5),
            'k':(2,5)
        }
        #pawn
        for i in range(6) :
            self.board[0][i][1] = 1
            self.board[1][i][4] = 1
            self.peice_loc[i][1] = 1
            self.peice_loc[i][4] = -1
        #rook
        self.board[2][0][0] = 1
        self.board[2][5][0] = 1
        self.board[3][0][5] = 1
        self.board[3][5][5] = 1
        self.peice_loc[0][0] = 2
        self.peice_loc[5][0] = 2
        self.peice_loc[0][5] = -2
        self.peice_loc[5][5] = -2
        #knight
        self.board[4][1][0] = 1
        self.board[4][4][0] = 1
        self.board[5][1][5] = 1
        self.board[5][4][5] = 1
        self.peice_loc[1][0] = 3
        self.peice_loc[4][0] = 3
        self.peice_loc[1][5] = -3
        self.peice_loc[4][5] = -3
        #queen
        self.board[6][3][0] = 1
        self.board[7][3][5] = 1
        self.peice_loc[3][0] = 4
        self.peice_loc[3][5] = -4
        #king
        self.board[8][2][0] = 1
        self.board[9][2][5] = 1
        self.peice_loc[2][0] = 5
        self.peice_loc[2][5] = -5
        return self.get_observation()

    def step(self, action):
        if self.player == 1 :
            self.white_action_history.append(action)
            if len(self.white_action_history) > 2 :
                if self.white_action_history[-3] == action:
                    self.white_repetitions += 1
        else :
            self.black_action_history.append(action)
            if len(self.black_action_history) > 2 :
                if self.black_action_history[-3] == action:
                    self.black_repetitions += 1
        self.last_action = action
        move = self.action2move(action)
        xy = action % 36
        x = xy % 6
        y = xy // 6
        destination = (x+move[0], y+move[1])
        life_peice, white_peice_loc, black_peice_loc = self.apply_action(action)
        self.life_peice = life_peice
        self.white_peice_loc = white_peice_loc
        self.black_peice_loc = black_peice_loc
        self.peice_loc[destination[0]][destination[1]] = self.peice_loc[x][y]
        self.peice_loc[x][y] = 0
        pei = 0
        for i,board in enumerate(self.board) :
            if board[x][y] != 0 :
                des = i
                pei = board[x][y]
                board[x][y] = 0
            if board[destination[0]][destination[1]] != 0 :
                board[destination[0]][destination[1]] = 0
        if pei != 0 :
            self.board[des][destination[0]][destination[1]] = pei
        winner = self.who_winner()
        reward = 0
        if winner == 'No' :
            done = False
            reward = 0
        else :
            done = True
            if self.player == 1 and winner == 'white' :
                reward = 1
            if self.player == 1 and winner == 'black' :
                reward = -1
            if self.player == -1 and winner == 'white' :
                reward = -1
            if self.player == -1 and winner == 'black' :
                reward = 1
        self.player *= -1
        
        return self.get_observation(), reward, done

    def get_observation(self):

        board_to_play = numpy.full((1, 6, 6), 1 if self.player == 1 else -1, dtype="int32")
        if self.player == 1 :
            if self.white_repetitions == 0 :
                repetition = numpy.zeros((2, 6, 6), dtype="int32")
            elif self.white_repetitions == 1 :
                repetition1 = numpy.full((6, 6), 1, dtype="int32")
                repetition2 = numpy.full((6, 6), 0, dtype="int32")
                repetition = numpy.stack((repetition1,repetition2))
            else :
                repetition1 = numpy.full((6, 6), 0, dtype="int32")
                repetition2 = numpy.full((6, 6), 1, dtype="int32")
                repetition = numpy.stack((repetition1,repetition2))
        else :
            if self.black_repetitions == 0 :
                repetition = numpy.zeros((2, 6, 6), dtype="int32")
            elif self.black_repetitions == 1 :
                repetition1 = numpy.full((6, 6), 1, dtype="int32")
                repetition2 = numpy.full((6, 6), 0, dtype="int32")
                repetition = numpy.stack((repetition1,repetition2))
            else :
                repetition1 = numpy.full((6, 6), 0, dtype="int32")
                repetition2 = numpy.full((6, 6), 1, dtype="int32")
                repetition = numpy.stack((repetition1,repetition2))
        return numpy.concatenate((self.board, repetition,board_to_play),axis=0)

    def legal_actions(self):
        legal = []
        peice_loc = self.white_peice_loc if self.player == 1 else self.black_peice_loc
        legal_actions_per_peice_dict = self.legal_actions_per_peice()
        for (key,value) in legal_actions_per_peice_dict.items() :
            x,y = peice_loc[key]
            from_pos = (x + y*6)
            for action in value :
                legal.append(from_pos+ action*36)
        return legal

    def legal_actions_per_peice(self,player = None) :
        """
        Returns:
            dict[str : List[int]]:
            posible action per each peice name 
        """
        if player == None :
            player = self.player
        legal_actions = {}
        now_player = 'white' if player == 1 else 'black'
        life_peice = self.life_peice[now_player]
        peice_loc = self.white_peice_loc if player == 1 else self.black_peice_loc
        for peice in life_peice :
            loc = peice_loc[peice]
            peice_id = self.peice.chess_name2number(peice)
            can_action = self.peice.peice_can_action(peice_id)
            actions = []
            for action in can_action :
                move = self.peice.action2move(action)
                destination = (loc[0]+move[0], loc[1]+move[1] * player)
                if (not self.in_board(destination)) :
                    continue
                Obstacle = False
                if action in [i for i in range(20)] :
                    if move[0] != 0 :
                        for i in range(0, move[0], 1 if move[0] > 0 else -1) :
                            if i == 0 :
                                continue
                            if self.peice_loc[loc[0] + i][loc[1]] != 0 :
                                Obstacle = True
                                break
                    if move[1] != 0 :
                        for i in range(0, move[1]* player, 1 if move[1]* player > 0 else -1) :
                            if i == 0 :
                                continue
                            if self.peice_loc[loc[0]][loc[1]+i] != 0 :
                                Obstacle = True
                                break
                if Obstacle :
                    continue
                if (self.peice_loc[destination[0]][destination[1]] * player > 0) :
                        continue
                if peice_id == 1 :
                    if action in [28,29] :
                        if self.peice_loc[destination[0]][destination[1]] * player >= 0 :
                            continue
                    if action in [0,1] :
                        if self.peice_loc[destination[0]][destination[1]] != 0 :
                            continue                        
                actions.append(action)
            legal_actions[peice] = actions
        return legal_actions
    
    def apply_action(self, action,player = None) :
        if player == None :
            player = self.player
        move = self.action2move(action,player=player)
        xy = action % 36
        x = xy % 6
        y = xy // 6
        life_peice = self.life_peice
        white_peice_loc = self.white_peice_loc
        black_peice_loc = self.black_peice_loc
        destination = (x+move[0], y+move[1])
        if player == 1 :
            for peice in life_peice['white'] :
                    loc = white_peice_loc[peice]
                    if loc == (x,y) :
                        white_peice_loc[peice] = destination
                        if self.peice_loc[destination[0]][destination[1]] != 0 :
                            for black_peice in life_peice['black'] :
                                black_loc = black_peice_loc[black_peice]
                                if black_loc == destination :
                                    life_peice['black'].remove(black_peice)
        else :
            for peice in life_peice['black'] :
                    loc = black_peice_loc[peice]
                    if loc == (x,y) :
                        black_peice_loc[peice] = destination
                        if self.peice_loc[destination[0]][destination[1]] != 0 :
                            for white_peice in life_peice['white'] :
                                white_loc = white_peice_loc[white_peice]
                                if white_loc == destination :
                                    life_peice['white'].remove(white_peice)
                                    

        return life_peice, white_peice_loc, black_peice_loc

    def in_board(self,loc) :
        """_summary_
        Args:
            tuple(int, int): (x,y) pos
        Returns:
            bool: is in board
        """
        x = loc[0]
        y = loc[1]
        return 0<=x<6 and 0<=y<6
    
    def who_winner(self):
        if 'k' not in self.life_peice['white'] :
            return 'black'
        
        if 'k' not in self.life_peice['black'] :
            return 'white'
        if self.black_repetitions == 3 :
            return 'white'
        
        if self.white_repetitions == 3 :
            return 'black'
        
        return 'No'
    
    def action2move(self,action,player = None) :
        action = action // 36
        move = self.peice.action2move(action)
        if player == None :
            move = (move[0],move[1]*self.player)
        else :
            move = (move[0],move[1]*player)
        return move
    
    def is_mate(self,life_peice = None, white_peice_loc = None, black_peice_loc = None):
        """
            Returns : 
                tuple(int,int) : (white king mate num, black king mate num)
        """
        if life_peice == None :
            white_life_peice = self.life_peice['white']
            black_life_peice = self.life_peice['black']
        else :
            white_life_peice = life_peice['white']
            black_life_peice = life_peice['black']
        if white_peice_loc == None :
            white_peice_loc = self.white_peice_loc 
        if black_peice_loc == None :
            black_peice_loc = self.black_peice_loc
        white_king_loc = white_peice_loc['k']
        black_king_loc = black_peice_loc['k']
        black_mate = 0
        white_mate = 0
        for peice in white_life_peice :
            loc = white_peice_loc[peice]
            peice_id = self.peice.chess_name2number(peice)
            can_action = self.peice.peice_can_action(peice_id)
            for action in can_action :
                move = self.peice.action2move(action)
                destination = (loc[0]+move[0], loc[1]+move[1])
                if destination == black_king_loc :
                    black_mate+=1
        for peice in black_life_peice :
            loc = black_peice_loc[peice]
            peice_id = self.peice.chess_name2number(peice)
            can_action = self.peice.peice_can_action(peice_id)
            for action in can_action :
                move = self.peice.action2move(action)
                destination = (loc[0]+move[0], loc[1]-move[1])
                if destination == white_king_loc :
                    white_mate+=1
                    
        return (white_mate,black_mate)
    def render(self):
        print(self.peice_loc)


class ChessPeice() :
    def __init__(self) :
        self.action_space = 32
        
        self.peice = {
            'p1' : 1,
            'p2' : 1,
            'p3' : 1,
            'p4' : 1,
            'p5' : 1,
            'p6' : 1,
            'r1' : 2,
            'r2' : 2,
            'n1' : 3,
            'n2' : 3,
            'q' : 4,
            'k' : 5 
        }

    def chess_name2number(self, name) :
        return self.peice[name]
    
    def action2move(self, action) :
        if action < 20 :
            dir = action // 5
            dis = action % 5
            dis += 1
            if dir == 0 :
                return (0,dis)
            elif dir == 1 :
                return (dis,0)
            elif dir == 2 :
                return (0,-dis)
            else :
                return (-dis,0)
            
        action -= 20
        
        if action < 8 :
            if action == 0 :
                return (-2,1)
            elif action == 1 :
                return (-1,2)
            elif action == 2 :
                return (1,2)
            elif action == 3 :
                return (2,1)
            elif action == 4 :
                return (2,-1)
            elif action == 5 :
                return (1,-2)
            elif action == 6 :
                return (-1,-2)
            else :
                return (-2,-1)
            
        action -= 8
        
        if action == 0 :
            return(-1,1)
        elif action == 1 :
            return (1,1)
        elif action == 2 :
            return (1,-1)
        else :
            return (-1,-1)
        
    def peice_can_action(self,peice_id):
        
        if peice_id == 1 :
            return [0,1,28,29]
        elif peice_id == 2 :
            return [i for i in range(20)]
        elif peice_id == 3 :
            return [i for i in range(20,28)]
        elif peice_id == 4 :
            return [i for i in range(20)] + [i for i in range(28,32)]
        else :
            return [i for i in range(28,32)]