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
        self.observation_shape = (12, 6, 6)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(96))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "random"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 20  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = True
        self.max_moves = 100  # Maximum number of moves if game is not finished before
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
        self.num_unroll_steps = 10  # Number of game moves to keep for every batch element
        self.td_steps = 100  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = True



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = 0.1  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
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
        #print(self.action_to_string(action))
        observation, reward, done = self.env.step(action)
        return observation, reward, done

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
        moved_peice,action_id = self.env.action_parse(action_number)
        return f"{moved_peice} act {action_id}"


class Minichess:
    def __init__(self):
        self.reset()
        self.peice = ChessPeice()
        self.pawn_action_count = 4
        self.rook_action_count = 16
        self.knight_action_count = 8
        self.bishop_action_count = 16
        self.queen_action_count = 32
        self.king_action_count = 8
        self.action_count = 152
    def to_play(self):
        return 0 if self.player == 1 else 1
        #white is 0
        #black is 1
    def reset(self):
        self.player = 1
        self.play_turn = 0
        self.black_repetitions = 0
        self.white_repetitions = 0
        self.white_action_history = deque(maxlen = 3)
        self.black_action_history = deque(maxlen = 3)
        self.board = numpy.zeros((10, 6, 6), dtype="int32")
        self.peice_loc = numpy.zeros((6, 6), dtype="int32")
        self.move_count_board = numpy.zeros((6, 6), dtype="int32")
        self.life_peice = {
            'white' : ['p1','p2','p3','p4','p5','p6','r1','r2','n1','n2','q1','k'],
            'black' : ['p1','p2','p3','p4','p5','p6','r1','r2','n1','n2','q1','k']
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
            'q1':(2,0),
            'k':(3,0)
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
            'q1':(2,5),
            'k':(3,5)
        }
        self.white_peice = {
            'p1':pawn(1,(0,1),'white'),
            'p2':pawn(2,(1,1),'white'),
            'p3':pawn(3,(2,1),'white'),
            'p4':pawn(4,(3,1),'white'),
            'p5':pawn(5,(4,1),'white'),
            'p6':pawn(6,(5,1),'white'),
            'r1':rook(1,(0,0),'white'),
            'r2':rook(2,(5,0),'white'),
            'n1':knight(1,(1,0),'white'),
            'n2':knight(2,(4,0),'white'),
            'q1':queen(1,(2,0),'white'),
            'k':king(1,(3,0),'white')
        }
        self.black_peice = {
            'p1':pawn(1,(0,4),'black'),
            'p2':pawn(2,(1,4),'black'),
            'p3':pawn(3,(2,4),'black'),
            'p4':pawn(4,(3,4),'black'),
            'p5':pawn(5,(4,4),'black'),
            'p6':pawn(6,(5,4),'black'),
            'r1':rook(1,(0,5),'black'),
            'r2':rook(2,(5,5),'black'),
            'n1':knight(1,(1,5),'black'),
            'n2':knight(2,(4,5),'black'),
            'q1':queen(1,(2,5),'black'),
            'k':king(1,(3,5),'black')
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
        self.board[6][2][0] = 1
        self.board[7][2][5] = 1
        self.peice_loc[2][0] = 4
        self.peice_loc[2][5] = -4
        #king
        self.board[8][3][0] = 1
        self.board[9][3][5] = 1
        self.peice_loc[3][0] = 5
        self.peice_loc[3][5] = -5
        return self.get_observation()

    def step(self,action):
        self.apply_action(action)
        observation = self.get_observation()
        reward = self.have_winner()
        done = self.is_terminal()
        self.player *= -1
        return observation, reward, done

    def get_observation(self,) :
        board_to_play = numpy.full((1, 6, 6), 1 if self.player == 1 else -1, dtype="int32")
        board = numpy.flip(self.board,axis=1) if self.player == -1 else self.board    
        move_count = numpy.flip(self.move_count_board,axis=1) if self.player == -1 else self.move_count_board
        move_count = numpy.expand_dims(move_count,axis= 0)
        return numpy.concatenate((board,board_to_play,move_count),axis=0)
    
    
    def legal_actions(self,player = None):
        if player == None :
            player = self.player
        legal_actions = []
        legal_actions_dict = self.legal_actions_per_peice(player)
        for peice, legal_action in legal_actions_dict.items() :
            for i in legal_action :
                helper = self.peice.peice_2_action_helper[peice]
                legal_actions.append(helper + i)
        return legal_actions
        
    def legal_actions_per_peice(self,player = None) :
        legal_actions_dict = {}
        if player == None :
            player = self.player
        if player == 1 :
            peice_class = self.white_peice
        else :
            peice_class = self.black_peice
        for peice, peice_inform in peice_class.items() :
            if isinstance(peice_inform,pawn) :
                legal_actions_dict[peice] = self.legal_action_pawn(peice_inform.loc,player,peice_inform.can_two_move)
            elif isinstance(peice_inform,rook) :
                legal_actions_dict[peice] = self.legal_action_straight(peice_inform.loc,player)
            elif isinstance(peice_inform,knight) :
                legal_actions_dict[peice] = self.legal_action_knight(peice_inform.loc,player)
            elif isinstance(peice_inform,bishop) :
                legal_actions_dict[peice] = self.legal_action_diagnoal(peice_inform.loc,player)
            elif isinstance(peice_inform,queen) :
                legal_actions_dict[peice] = self.legal_action_queen(peice_inform.loc,player)
            elif isinstance(peice_inform,king) :
                legal_actions_dict[peice] = self.legal_action_king(peice_inform.loc,player)
        return legal_actions_dict
    
    def apply_action(self, action,player = None) :
        if player == None :
            player = self.player
        self.apply_action_white(action) if player == 1 else self.apply_action_black(action)
        self.play_turn += 1
        
    def apply_action_white(self, action) :
        peice, action_id = self.action_parse(action)
        peice_inform = self.white_peice[peice]
        past_loc = peice_inform.loc
        peice_inform.move(action_id)
        new_loc = peice_inform.loc
        board_index = (self.peice.peice_index[peice] - 1) * 2 
        self.board[board_index][past_loc[0]][past_loc[1]] = 0
        self.board[board_index][new_loc[0]][new_loc[1]] = 1
        self.peice_loc[past_loc[0]][past_loc[1]] = 0
        self.peice_loc[new_loc[0]][new_loc[1]] = self.peice.peice_index[peice]
        self.white_peice_loc[peice] =  new_loc
        self.move_count_board[new_loc[0]][new_loc[1]] += 1 
        del_black_peice = None
        for black_peice, black_loc in self.black_peice_loc.items() :
            if new_loc == black_loc :
                del_black_peice = black_peice
                del self.black_peice[black_peice]
                self.life_peice['black'].remove(black_peice)
                board_index = (self.peice.peice_index[black_peice] - 1) * 2 + 1
                self.board[board_index][new_loc[0]][new_loc[1]] = 0
        if del_black_peice != None :
            del self.black_peice_loc[del_black_peice]
    def apply_action_black(self,action) : 
        peice, action_id = self.action_parse(action)
        peice_inform = self.black_peice[peice]
        past_loc = peice_inform.loc
        peice_inform.move(action_id)
        new_loc = peice_inform.loc
        board_index = (self.peice.peice_index[peice] - 1) * 2 + 1
        self.board[board_index][past_loc[0]][past_loc[1]] = 0
        self.board[board_index][new_loc[0]][new_loc[1]] = 1
        self.peice_loc[past_loc[0]][past_loc[1]] = 0
        self.peice_loc[new_loc[0]][new_loc[1]] = self.peice.peice_index[peice] * -1
        self.black_peice_loc[peice] =  new_loc
        self.move_count_board[new_loc[0]][new_loc[1]] += 1 
        del_white_peice = None
        for white_peice, white_loc in self.white_peice_loc.items() :
            if new_loc == white_loc :
                del_white_peice = white_peice
                del self.white_peice[white_peice]
                self.life_peice['white'].remove(white_peice)
                board_index = (self.peice.peice_index[white_peice] - 1) * 2
                self.board[board_index][new_loc[0]][new_loc[1]] = 0
        if del_white_peice != None :
            del self.white_peice_loc[del_white_peice]
    def legal_action_straight(self,loc,player =None):
        
        legal_list = []
        x,y = loc
        if player == None :
            player = self.player
        temp_x = x - 1
        while temp_x >= 0 : 
            if not self.in_board((temp_x,y)) : 
                break
            if self.peice_loc[temp_x][y] == 0 :
                legal_list.append(temp_x)
                temp_x -= 1
            elif self.peice_loc[temp_x][y] * player > 0 : 
                break
            else :
                legal_list.append(temp_x)
                break 
            
        temp_x = x + 1
        while temp_x < 6 : 
            if not self.in_board((temp_x,y)) : 
                break                    
            if self.peice_loc[temp_x][y] == 0 :
                legal_list.append(temp_x)
                temp_x += 1
            
            elif self.peice_loc[temp_x][y] * player > 0 : 
                break
            else :
                legal_list.append(temp_x)
                break
            
        temp_y = y - 1
        while temp_y >= 0 : 
            if not self.in_board((x,temp_y)) : 
                break
            if self.peice_loc[x][temp_y] == 0 :
                if player == 1 :
                    legal_list.append(temp_y+6)
                else :
                    legal_list.append(5-temp_y+6)
                temp_y -= 1
            
            elif self.peice_loc[x][temp_y] * player > 0 : 
                break
            else :
                if player == 1 :
                    legal_list.append(temp_y+6)
                else :
                    legal_list.append(5-temp_y+6)
                break
            
        temp_y = y + 1
        while temp_y < 6 : 
            if not self.in_board((x,temp_y)) : 
                break                    
            if self.peice_loc[x][temp_y] == 0 :
                if player == 1 :
                    legal_list.append(temp_y+6)
                else :
                    legal_list.append(5-temp_y+6)
                temp_y += 1
            
            elif self.peice_loc[x][temp_y] * player > 0 : 
                break
            else :
                if player == 1 :
                    legal_list.append(temp_y+6)
                else :
                    legal_list.append(5-temp_y+6)
                break
        return legal_list       
    
    def legal_action_diagnoal(self,loc,player=None):
        legal_list = []
        x,y = loc
        if player == None :
            player = self.player
        temp_x = x - 1
        temp_y = y - player
        while temp_x >= 0 : 
            if not self.in_board((temp_x,temp_y)) : 
                break
            if self.peice_loc[temp_x][temp_y] == 0 :
                legal_list.append(temp_x)
                temp_x -= 1
                temp_y -= player
            elif self.peice_loc[temp_x][temp_y] * player > 0 : 
                break
            else :
                legal_list.append(temp_x)
                break
        temp_x = x + 1
        temp_y = y + player
        while temp_x < 6 : 
            if not self.in_board((temp_x,temp_y)) : 
                break
            if self.peice_loc[temp_x][temp_y] == 0 :
                legal_list.append(temp_x)
                temp_x += 1
                temp_y += player
            elif self.peice_loc[temp_x][temp_y] * player > 0 : 
                break
            else :
                legal_list.append(temp_x)
                break
        temp_x = x - 1
        temp_y = y + player
        while temp_x >= 0 : 
            if not self.in_board((temp_x,temp_y)) : 
                break
            if self.peice_loc[temp_x][temp_y] == 0 :
                legal_list.append(temp_x+6)
                temp_x -= 1
                temp_y += player
            elif self.peice_loc[temp_x][temp_y] * player > 0 : 
                break
            else :
                legal_list.append(temp_x+6)
                break
        temp_x = x + 1
        temp_y = y - player
        while temp_x < 6 : 
            if not self.in_board((temp_x,temp_y)) : 
                break
            if self.peice_loc[temp_x][temp_y] == 0 :
                legal_list.append(temp_x+6)
                temp_x += 1
                temp_y -= player
            elif self.peice_loc[temp_x][temp_y] * player > 0 : 
                break
            else :
                legal_list.append(temp_x+6)
                break
        return legal_list
    
    def legal_action_knight(self,loc,player = None)  :
        legal_list = []
        x,y = loc
        if player == None :
            player = self.player
        for action_index in range(8) :
            if action_index == 0 :
                new_pos = (-2,1)
            elif action_index == 1 :
                new_pos = (-1,2)
            elif action_index == 2 :
                new_pos = (1,2)
            elif action_index == 3 :
                new_pos = (2,1)
            elif action_index == 4 :
                new_pos = (2,-1)
            elif action_index == 5 :
                new_pos = (1,-2)
            elif action_index == 6 :
                new_pos = (-1,-2)
            else :
                new_pos = (-2,-1)
            if not self.in_board((x+new_pos[0],y+new_pos[1]*player)):
                continue
            if self.peice_loc[x+new_pos[0]][y+new_pos[1]*player] * player > 0 :
                continue
            legal_list.append(action_index)
        return legal_list


    def legal_action_pawn(self,loc,player = None, first_move = True)  :
        legal_list = []
        x,y = loc
        if player == None :
            player = self.player
        for action_index in range(4) :
            if action_index == 0 :
                new_pos = (0,1)
                if not self.in_board((x+new_pos[0],y+new_pos[1]*player)):
                    continue
                if self.peice_loc[x+new_pos[0]][y+new_pos[1]*player] != 0 :
                    continue

            elif action_index == 1 :
                new_pos = (0,2)
                if not first_move :
                    continue
                if not self.in_board((x+new_pos[0],y+new_pos[1]*player)):
                    continue
                if self.peice_loc[x+new_pos[0]][y+new_pos[1]*player] != 0 :
                    continue
            elif action_index == 2 :
                new_pos = (-1,1)
                if not self.in_board((x+new_pos[0],y+new_pos[1]*player)):
                    continue
                if self.peice_loc[x+new_pos[0]][y+new_pos[1]*player] * player >= 0 :
                    continue
            else :
                new_pos = (1,1)
                if not self.in_board((x+new_pos[0],y+new_pos[1]*player)):
                    continue
                if self.peice_loc[x+new_pos[0]][y+new_pos[1]*player] * player >= 0 :
                    continue
            legal_list.append(action_index)
        return legal_list

    def legal_action_king(self,loc,player = None) :
        legal_list = []
        x,y = loc
        if player == None :
            player = self.player
        for action_index in range(8) :
            if action_index == 0 :
                new_pos = (0,1)
            elif action_index == 1 :
                new_pos = (1,1)
            elif action_index == 2 :
                new_pos = (1,0)
            elif action_index == 3 :
                new_pos = (1,-1)
            elif action_index == 4 :
                new_pos = (0,-1)
            elif action_index == 5 :
                new_pos = (-1,-1)
            elif action_index == 6 :
                new_pos = (-1,0)
            else :
                new_pos = (-1,1)
            if not self.in_board((x+new_pos[0],y+new_pos[1]*player)):
                continue
            if self.peice_loc[x+new_pos[0]][y+new_pos[1]*player] * player > 0  :
                continue
            legal_list.append(action_index)
        return legal_list
    
    def legal_action_queen(self,loc,player = None) :
        if player == None :
            player = self.player
        legal_list = self.legal_action_straight(loc,player)
        diagnoal = self.legal_action_diagnoal(loc,player)
        for i in diagnoal : 
            legal_list.append(i+12)
        return legal_list
        
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
    
    def have_winner(self):
        if 'k' not in self.life_peice['white'] or  'k' not in self.life_peice['black']:
            return 1
        return 0
    
    def action_parse(self,action) :
        if 0<= action < 24 :
            n_pawn = action // 4 + 1
            action_id = action % 4
            moved_peice = 'p' + str(n_pawn)
            return (moved_peice,action_id)
        action -= 24
        if 0<= action < 12*2 :
            n_rook = action // 12 + 1
            action_id = action % 12
            moved_peice = 'r' + str(n_rook)
            return (moved_peice,action_id)
        action -= 24
        if 0<= action < 8*2 :
            n_knight = action // 8 + 1
            action_id = action % 8
            moved_peice = 'n' + str(n_knight)
            return (moved_peice,action_id)
        action -= 16
        if 0<= action < 24 :
            action_id = action % 24
            moved_peice = 'q1'
            return (moved_peice,action_id)
        action -= 24
        moved_peice = 'k'
        return (moved_peice,action)
        
        
    def render(self):
        print(self.peice_loc)
    
    def check_stalemate(self,):
        king_move = list(range(88,96))
        for i in self.legal_actions(1) :
            if i not in king_move :
                return False
        for i in self.legal_actions(-1) :
            if i not in king_move :
                return False
        return True
    
    def is_terminal(self) :
        done = True if self.have_winner() != 0 else False
        if self.check_stalemate() :
            done = True
        if not done :
            if self.play_turn == 100 :
                done = True
        return done
    
    def legal_actions_map(self,peice,legal_actions) : 
        legal_actions
class ChessPeice() :
    def __init__(self) :
        self.action_space = 120
        
        self.peice_index = {
            'p1' : 1,
            'p2' : 1,
            'p3' : 1,
            'p4' : 1,
            'p5' : 1,
            'p6' : 1,
            'p7' : 1,
            'p8' : 1,
            'r1' : 2,
            'r2' : 2,
            'n1' : 3,
            'n2' : 3,
            'q1' : 4,
            'k' : 5 
        }

        self.peice_identical_index = {
            'p1' : 1,
            'p2' : 2,
            'p3' : 3,
            'p4' : 4,
            'p5' : 5,
            'p6' : 6,
            'p7' : 7,
            'p8' : 8,
            'r1' : 9,
            'r2' : 10,
            'n1' : 11,
            'n2' : 12,
            'b1' : 13,
            'b2' : 14,
            'q1' : 15,
            'k' : 16 
        }

        self.peice_2_action_helper = {
            'p1' : 0,
            'p2' : 4,
            'p3' : 8,
            'p4' : 12,
            'p5' : 16,
            'p6' : 20,
            'r1' : 24,
            'r2' : 36,
            'n1' : 48,
            'n2' : 56,
            'q1' : 64,
            'k' : 88 
        }


    def chess_name2index(self, name) :
        return self.peice[name]

    def chess_name2identical_index(self, name) :
        return self.peice_identical_index[name]
        
class pawn :
    #index mean nth pawn
    #loc mean start position
    #taem is black or white
    def __init__(self,index , loc , team) -> None:
        self.index = index
        self.loc = loc
        self.team = team
        self.player = 1 if team == 'white' else -1
        self.can_two_move = True
        self.action_count = 4
    def move(self,action_index) :
        #action index range 0~1 move foward, 2 : eat left foward, 3 : eat right foward
        #4~7 same action and promotion to queen
        #8~11 same action and promotion to knight
        #in this function not implement promotion
        init_action_index = action_index
        x,y = self.loc
        if action_index == 0 :
            new_pos = (x,y + self.player)
        elif action_index == 1 :
            assert(self.can_two_move)
            new_pos = (x,y + self.player * 2)
        elif action_index == 2 :
            new_pos = (x-1,y + self.player)
        else :
            new_pos = (x+1,y + self.player)
        self.can_two_move = False
        self.loc = new_pos
        if not (0<=self.loc[0] < 6 and  0<=self.loc[1] < 6) :
            print('pawn action_index : ',init_action_index,'now loc',self.loc[0],self.loc[1],'player is ',self.player)

class rook : 
    #index mean nth night
    #loc mean start position
    #taem is black or white
    def __init__(self,index , loc , team) -> None:
        self.index = index
        self.loc = loc
        self.team = team
        self.player = 1 if team == 'white' else -1
        self.moved = False #For caslting
        self.action_count = 12
    def move(self,action_index) :
        #action range 0~7 mean move to x is action
        #action range 8~15 mean move to y is action
        init_action_index = action_index
        x,y = self.loc
        if 0<= action_index < 6 :
            self.loc = (action_index,y)
        else :
            action_index-=6
            if self.player == 1 :
                self.loc = (x,action_index)
            else :
                self.loc = (x,5-action_index)
        self.moved = True
        if not (0<=self.loc[0] < 6 and  0<=self.loc[1] < 6) :
            print('rook action_index : ',init_action_index,'now loc',self.loc[0],self.loc[1],'player is ',self.player)

class knight :
    #index mean nth night
    #loc mean start position
    #taem is black or white
    def __init__(self,index , loc , team) -> None:
        self.index = index
        self.loc = loc
        self.team = team
        self.player = 1 if team == 'white' else -1
        self.action_count = 8
    def move(self,action_index) :
        #action index rnage 0~7
        x,y = self.loc
        init_action_index = action_index
        if action_index == 0 :
            new_pos = (-2,1)
        elif action_index == 1 :
            new_pos = (-1,2)
        elif action_index == 2 :
            new_pos = (1,2)
        elif action_index == 3 :
            new_pos = (2,1)
        elif action_index == 4 :
            new_pos = (2,-1)
        elif action_index == 5 :
            new_pos = (1,-2)
        elif action_index == 6 :
            new_pos = (-1,-2)
        else :
            new_pos = (-2,-1)
            
        self.loc = (x + new_pos[0], y + new_pos[1]*self.player)
        if not (0<=self.loc[0] < 6 and  0<=self.loc[1] < 6) :
            print('knight action_index : ',init_action_index,'now loc',self.loc[0],self.loc[1],'player is ',self.player)

class bishop :
    #index mean nth night
    #loc mean start position
    #taem is black or white
    def __init__(self,index , loc , team) -> None:
        self.index = index
        self.loc = loc
        self.team = team
        self.player = 1 if team == 'white' else -1
        self.action_count = 12
    def move(self,action_index) :
        #action index range 0~7 right up
        #action index range 8~15 left up
        x,y = self.loc
        init_action_index = action_index
        if 0<=action_index<6 :
            new_y = y-x*self.player
            new_x = 0
            self.loc = (new_x+action_index,new_y+action_index * self.player)
        else :
            action_index -= 6
            new_y = y+x*self.player
            new_x = 0
            self.loc = (new_x+action_index,new_y-action_index * self.player)
        
        if not (0<=self.loc[0] < 6 and  0<=self.loc[1] < 6) :
            print('bishop action_index : ',init_action_index,'now loc',self.loc[0],self.loc[1],'player is ',self.player)
            
class queen :
    def __init__(self,index , loc , team) -> None:
        self.index = index
        self.loc = loc
        self.team = team
        self.player = 1 if team == 'white' else -1
        self.action_count = 24
    def move(self,action_index) :
        #action index range 0~7 right up
        #action index range 8~15 left up
        x,y = self.loc
        init_action_index = action_index
        if 0<= action_index < 6 :
            self.loc = (action_index,y)
        elif 6<= action_index < 12 :
            action_index-=6
            if self.player == 1 :
                self.loc = (x,action_index)
            else :
                self.loc = (x,5-action_index)       
        elif 12<=action_index<18 :
            action_index -= 12
            new_y = y-x*self.player
            new_x = 0
            self.loc = (new_x+action_index,new_y+action_index * self.player)
        else :
            action_index -= 18
            new_y = y+x*self.player
            new_x = 0
            self.loc = (new_x+action_index,new_y-action_index * self.player)
        if not (0<=self.loc[0] < 6 and  0<=self.loc[1] <6) :
            print('queen action_index : ',init_action_index,'now loc',self.loc[0],self.loc[1],'player is ',self.player)

class king :
    def __init__(self,index , loc , team) -> None:
        self.index = index
        self.loc = loc
        self.team = team
        self.player = 1 if team == 'white' else -1
        self.action_count = 8
    def move(self,action_index) :
        x,y = self.loc
        init_action_index = action_index
        if action_index == 0 :
            new_pos = (0,1)
        elif action_index == 1 :
            new_pos = (1,1)
        elif action_index == 2 :
            new_pos = (1,0)
        elif action_index == 3 :
            new_pos = (1,-1)
        elif action_index == 4 :
            new_pos = (0,-1)
        elif action_index == 5 :
            new_pos = (-1,-1)
        elif action_index == 6 :
            new_pos = (-1,0)
        else :
            new_pos = (-1,1)
        self.loc = (x + new_pos[0], y + new_pos[1]*self.player)
        if not (0<=self.loc[0] < 6 and  0<=self.loc[1] < 6) :
            print('king action_index : ',init_action_index,'now loc',self.loc[0],self.loc[1],'player is ',self.player)
