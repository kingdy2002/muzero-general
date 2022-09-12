import datetime
import pathlib

import numpy as np
import torch

from .abstract_game import AbstractGame
from .JanggiConstants import *
from collections import defaultdict

import random

class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = 2  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (CONFIG_M*CONFIG_T+CONFIG_L, 9, 10)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(CONFIG_X*CONFIG_Y*CONFIG_A + 1))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = CONFIG_T  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "expert"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 5  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = True
        self.max_moves = 42  # Maximum number of moves if game is not finished before
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
        self.blocks = 6  # Number of blocks in the ResNet
        self.channels = 128  # Number of channels in the ResNet
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
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 100000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.005  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        ### Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 42  # Number of game moves to keep for every batch element
        self.td_steps = 42  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = True



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        

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

    def __init__(self, rule=None , seed=None):
        if rule == None :
            c1, c2, mode = (rule)
            self.env = Janggi(0,c1, c2, mode = mode)
        self.env = Janggi(0,0)

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
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
        input("Press enter to take a step ")

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
        pass

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return f"Play column {action_number + 1}"


class Connect4:
    def __init__(self):
        self.board = numpy.zeros((6, 7), dtype="int32")
        self.player = 1

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = numpy.zeros((6, 7), dtype="int32")
        self.player = 1
        return self.get_observation()

    def step(self, action):
        for i in range(6):
            if self.board[i][action] == 0:
                self.board[i][action] = self.player
                break

        done = self.have_winner() or len(self.legal_actions()) == 0

        reward = 1 if self.have_winner() else 0

        self.player *= -1

        return self.get_observation(), reward, done

    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1.0, 0.0)
        board_player2 = numpy.where(self.board == -1, 1.0, 0.0)
        board_to_play = numpy.full((6, 7), self.player, dtype="int32")
        return numpy.array([board_player1, board_player2, board_to_play])

    def legal_actions(self):
        legal = []
        for i in range(7):
            if self.board[5][i] == 0:
                legal.append(i)
        return legal

    def have_winner(self):
        # Horizontal check
        for i in range(4):
            for j in range(6):
                if (
                    self.board[j][i] == self.player
                    and self.board[j][i + 1] == self.player
                    and self.board[j][i + 2] == self.player
                    and self.board[j][i + 3] == self.player
                ):
                    return True

        # Vertical check
        for i in range(7):
            for j in range(3):
                if (
                    self.board[j][i] == self.player
                    and self.board[j + 1][i] == self.player
                    and self.board[j + 2][i] == self.player
                    and self.board[j + 3][i] == self.player
                ):
                    return True

        # Positive diagonal check
        for i in range(4):
            for j in range(3):
                if (
                    self.board[j][i] == self.player
                    and self.board[j + 1][i + 1] == self.player
                    and self.board[j + 2][i + 2] == self.player
                    and self.board[j + 3][i + 3] == self.player
                ):
                    return True

        # Negative diagonal check
        for i in range(4):
            for j in range(3, 6):
                if (
                    self.board[j][i] == self.player
                    and self.board[j - 1][i + 1] == self.player
                    and self.board[j - 2][i + 2] == self.player
                    and self.board[j - 3][i + 3] == self.player
                ):
                    return True

        return False

    def expert_action(self):
        board = self.board
        action = numpy.random.choice(self.legal_actions())
        for k in range(3):
            for l in range(4):
                sub_board = board[k : k + 4, l : l + 4]
                # Horizontal and vertical checks
                for i in range(4):
                    if abs(sum(sub_board[i, :])) == 3:
                        ind = numpy.where(sub_board[i, :] == 0)[0][0]
                        if numpy.count_nonzero(board[:, ind + l]) == i + k:
                            action = ind + l
                            if self.player * sum(sub_board[i, :]) > 0:
                                return action

                    if abs(sum(sub_board[:, i])) == 3:
                        action = i + l
                        if self.player * sum(sub_board[:, i]) > 0:
                            return action
                # Diagonal checks
                diag = sub_board.diagonal()
                anti_diag = numpy.fliplr(sub_board).diagonal()
                if abs(sum(diag)) == 3:
                    ind = numpy.where(diag == 0)[0][0]
                    if numpy.count_nonzero(board[:, ind + l]) == ind + k:
                        action = ind + l
                        if self.player * sum(diag) > 0:
                            return action

                if abs(sum(anti_diag)) == 3:
                    ind = numpy.where(anti_diag == 0)[0][0]
                    if numpy.count_nonzero(board[:, 3 - ind + l]) == ind + k:
                        action = 3 - ind + l
                        if self.player * sum(anti_diag) > 0:
                            return action

        return action

    def render(self):
        print(self.board[::-1])

class Janggi():
    def __init__(self, c1, c2, empty = False, mode = 0):
        # Done
        "Set up initial board configuration."
        
        """
        board = (board, (han_pcs, cho_pcs, move_cnt, curr_player, han_score, cho_score), rep_dict)
        """
        self.mode = mode
        # Create empty board (for optimization)
        if empty:
            self.board = None
            self.rep_dict = None
            self.b_params = None
            return

        self.reset()

    def to_play(self):
        return self.b_params[N_CUR_PLAYER]

    def reset(self):
        # Create the empty board state.
        mode = self.mode
        self.board = [None]*CONFIG_T
        for i in range(CONFIG_T):
            self.board[i] = [0]*CONFIG_X
            for j in range(CONFIG_X):
                self.board[i][j] = [0]*CONFIG_Y

        # Mode
        # 0: BMXPC
        # 1: B
        # 2: BM
        # 3: BMX
        # 4: BMXP
        # 5: BMXPC
        # 6: BC
        EC = (mode == 0) or (mode > 4)
        EP = (mode == 0) or (mode > 3 and mode < 6)
        EM = (mode == 0) or (mode > 2 and mode < 6)
        EX = (mode == 0) or (mode > 1 and mode < 6)

        # Set up first player's board.
        self.board[0][4][1] = NK   # K
        self.board[0][0][0] = NC * EC   # C
        self.board[0][8][0] = NC * EC   # C
        self.board[0][1][2] = NP * EP   # P
        self.board[0][7][2] = NP * EP   # P

        self.board[0][1][0] += NM * int(c1==1 or c1==2) * EM    # M
        self.board[0][2][0] += NM * int(c1==0 or c1==3) * EM    # M
        self.board[0][6][0] += NM * int(c1==1 or c1==3) * EM    # M
        self.board[0][7][0] += NM * int(c1==0 or c1==2) * EM    # M

        self.board[0][1][0] += NX * int(c1==0 or c1==3) * EX    # X
        self.board[0][2][0] += NX * int(c1==1 or c1==2) * EX    # X
        self.board[0][6][0] += NX * int(c1==0 or c1==2) * EX    # X
        self.board[0][7][0] += NX * int(c1==1 or c1==3) * EX    # X

        self.board[0][3][0] = NS    # S
        self.board[0][5][0] = NS    # S
        self.board[0][0][3] = NB    # B
        self.board[0][2][3] = NB    # B
        self.board[0][4][3] = NB    # B
        self.board[0][6][3] = NB    # B
        self.board[0][8][3] = NB    # B

        # Set up opponent's board.
        self.board[0][4][8] = -NK    # K
        self.board[0][0][9] = -NC * EC    # C
        self.board[0][8][9] = -NC * EC    # C
        self.board[0][1][7] = -NP * EP    # P
        self.board[0][7][7] = -NP * EP    # P

        self.board[0][1][9] += -NM * int(c2==0 or c2==3) * EM    # M
        self.board[0][2][9] += -NM * int(c2==1 or c2==2) * EM    # M
        self.board[0][6][9] += -NM * int(c2==0 or c2==2) * EM    # M
        self.board[0][7][9] += -NM * int(c2==1 or c2==3) * EM    # M

        self.board[0][1][9] += -NX * int(c2==1 or c2==2) * EX    # X
        self.board[0][2][9] += -NX * int(c2==0 or c2==3) * EX    # X
        self.board[0][6][9] += -NX * int(c2==1 or c2==3) * EX    # X
        self.board[0][7][9] += -NX * int(c2==0 or c2==2) * EX    # X

        self.board[0][3][9] = -NS    # S
        self.board[0][5][9] = -NS    # S
        self.board[0][0][6] = -NB    # B
        self.board[0][2][6] = -NB    # B
        self.board[0][4][6] = -NB    # B
        self.board[0][6][6] = -NB    # B
        self.board[0][8][6] = -NB    # B

        # Convert to numpy array
        self.board = np.array(self.board)

        score = 1 * self._piece_score(NK) + 2 * self._piece_score(NC) * EC + 2 * self._piece_score(NP) * EP + 2 * self._piece_score(NM) * EM + 2 * self._piece_score(NX) * EX + 2 * self._piece_score(NS) + 5 * self._piece_score(NB)

        # Set up params: han_pcs/cho_pcs (bitmap indicating the live board), move_cnt and curr_player
        han_pcs = 34133    # 10000/10/10/10/10/10/1
        cho_pcs = 34133    # 10000/10/10/10/10/10/1
        move_cnt = 0
        cur_player = 1
        han_score = score + 1.5
        cho_score = score
        captured = False
        is_bic = False
        turnskip_cnt = 0
        self.b_params = np.array([han_pcs, cho_pcs, move_cnt, cur_player, han_score, cho_score, captured, is_bic, turnskip_cnt])

        # Create the empty repetition set
        self.rep_dict = {}
        self.rep_dict = defaultdict(lambda:0, self.rep_dict)
        return self.get_observation()

    def step(self, move):
        """Perform the given move on the board; catch board as necessary.
        color gives the color pf the piece to play
        """
        player = self.b_params[N_CUR_PLAYER]   # 0: Cho, 1: Han
        (a, x, y) = move    # a: action type, (x,y): board position where the action starts

        assert 0 <= a and a <= 58

        ## Duplicate the last board configuration and shift self.board
        self.board = np.delete(self.board, CONFIG_T - 1, 0)
        self.board = np.concatenate(([self.board[0].copy()], self.board), 0)

        assert self.board.shape == (CONFIG_T, CONFIG_X, CONFIG_Y)

        ## Update current player & move count
        self.b_params[N_CUR_PLAYER] = PLAYER_HAN if self.b_params[N_CUR_PLAYER] == PLAYER_CHO else PLAYER_CHO  # change current player
        self.b_params[N_MOVE_CNT] += 1  # increment move count

        ## Rotate board, set captured to false and return if the action is turn skip
        if (a == 58):
            self.board = np.flip(self.board, [1,2])
            self.b_params[N_CAPTURED] = False
            self.b_params[N_TURNSKIP_CNT] += 1
            return
        else:
            self.b_params[N_TURNSKIP_CNT] = 0

        ## Otherwise, add the current board to the rep_dict
        canonical_board = self.board[0]
        if (player == PLAYER_HAN):
            canonical_board = np.flip(canonical_board, [0, 1])
        self.rep_dict[canonical_board.tostring()] += 1

        ## Move the board. First, assert that the moving piece is present.
        assert (self.board[0][x][y] != 0)
        
        ## Move the piece according to the given action
        if (a <= 7):
            newx = x + (a + 1)
            newy = y
        elif (a <= 15):
            newx = x - (a - 7)
            newy = y
        elif (a <= 24):
            newx = x
            newy = y + (a - 15)
        elif (a <= 33):
            newx = x
            newy = y - (a - 24)
        elif (a <= 35):
            newx = x + (a - 33)
            newy = y + (a - 33)
        elif (a <= 37):
            newx = x - (a - 35)
            newy = y + (a - 35)
        elif (a <= 39):
            newx = x - (a - 37)
            newy = y - (a - 37)
        elif (a <= 41):
            newx = x + (a - 39)
            newy = y - (a - 39)
        elif (a <= 43):
            newx = x + 2
            newy = y + 1 if (a == 42) else y - 1
        elif (a <= 45):
            newx = x - 2
            newy = y + 1 if (a == 44) else y - 1
        elif (a <= 47):
            newx = x + 1
            newy = y + 2 if (a == 46) else y - 2
        elif (a <= 49):
            newx = x - 1
            newy = y + 2 if (a == 48) else y - 2
        elif (a <= 51):
            newx = x + 3
            newy = y + 2 if (a == 50) else y - 2
        elif (a <= 53):
            newx = x - 3
            newy = y + 2 if (a == 52) else y - 2
        elif (a <= 55):
            newx = x + 2
            newy = y + 3 if (a == 54) else y - 3
        elif (a <= 57):
            newx = x - 2
            newy = y + 3 if (a == 56) else y - 3
        else:
            # This should not happen. Panic.
            assert False

        assert x >= 0 
        assert x < CONFIG_X
        assert y >= 0
        assert y < CONFIG_Y
        assert newx >= 0
        assert newx < CONFIG_X
        assert newy >= 0
        assert newy < CONFIG_Y
        
        moving_piece = self.board[0][x][y]
        captured_piece = self.board[0][newx][newy]
        self.board[0][x][y] = 0
        self.board[0][newx][newy] = moving_piece

        ## Update han_pcs and cho_pcs
        if (captured_piece != NULL):
            if (player == PLAYER_HAN):  # Han captured Cho
                self.b_params[N_CHO_PCS] = self.remove_piece(self.b_params[N_CHO_PCS], abs(captured_piece))
            else:                       # Cho captured Han
                self.b_params[N_HAN_PCS] = self.remove_piece(self.b_params[N_HAN_PCS], abs(captured_piece))
        
        ## Update han_score and cho_score
        if (captured_piece != NULL):
            if (player == PLAYER_HAN):  # Han captured Cho
                self.b_params[N_CHO_SCORE] -= self._piece_score(abs(captured_piece))
            else:                       # Cho captured Han
                self.b_params[N_HAN_SCORE] -= self._piece_score(abs(captured_piece))

        ## Update captured
        self.b_params[N_CAPTURED] = captured_piece != NULL

        ## Update is_bic
        if (abs(moving_piece) == NK and abs(captured_piece) == NK):
            self.b_params[N_IS_BIC] = True
        
        ## Flip board and return
        self.board = np.flip(self.board, [1,2])
        return self.get_observation(),self.getScore(player),self.game_ended()

    def getObservation(self) :
        encodedBoard = []

        player = self.b_params[N_CUR_PLAYER]
        player_sign = 1 if player == PLAYER_CHO else -1
        move_cnt = self.b_params[N_MOVE_CNT]

        # Set up boards
        for t in range(CONFIG_T):
            pieces_t = self.board[t]

            # Create an encoded board
            enc_t = [0] * CONFIG_M
            for i in range(CONFIG_M):
                enc_t[i] = [0] * CONFIG_X
                for j in range(CONFIG_X):
                    enc_t[i][j] = [0] * CONFIG_Y
            
            # Fill in the pieces
            for i in range(CONFIG_X):
                for j in range(CONFIG_Y):
                    if pieces_t[i][j] == 0:
                        continue
                    if np.sign(pieces_t[i][j]) == player_sign:
                        enc_t[abs(pieces_t[i][j]) - 1][i][j] = 1
                    else:
                        enc_t[7 + abs(pieces_t[i][j]) - 1][i][j] = 1
            
            # Fill in repetition count
            canonical_board = pieces_t
            if (player == PLAYER_HAN):
                canonical_board = np.flip(canonical_board, [0, 1])
            repcnt = self.rep_dict[canonical_board.tostring()]
            if (repcnt >= 1):
                enc_t[14] = np.array(enc_t[14]) + 1
            if (repcnt >= 2):
                enc_t[15] = np.array(enc_t[15]) + 1

            # Append to the encodedBoard
            if t == 0:
                encodedBoard = enc_t
            else:
                encodedBoard = np.concatenate((encodedBoard, enc_t), 0)

        # Set up player
        enc_player = [0] * CONFIG_X
        for i in range(CONFIG_X):
            enc_player[i] = [0] * CONFIG_Y
        enc_player += player
        encodedBoard = np.concatenate((encodedBoard, [enc_player]), 0)

        # Set up move_cnt
        enc_mv = [0] * CONFIG_X
        for i in range(CONFIG_X):
            enc_mv[i] = [0] * CONFIG_Y
        enc_mv += move_cnt
        encodedBoard = np.concatenate((encodedBoard, [enc_mv]), 0)

        return np.array(encodedBoard)
    
    def legal_actions(self):
        """Returns all the legal moves for the current board."""
        moves = []  # stores the legal moves.

        legal_sign = 1 if self.b_params[N_CUR_PLAYER] == PLAYER_CHO else -1

        for y in range(CONFIG_Y):
            for x in range(CONFIG_X):
                if (self.board[0][x][y] == 0):
                    continue
                elif abs(self.board[0][x][y]) == NK and np.sign(self.board[0][x][y]) == legal_sign:
                    moves = moves + self.get_moves_for_K(x,y)
                elif abs(self.board[0][x][y]) == NC and np.sign(self.board[0][x][y]) == legal_sign:
                    moves = moves + self.get_moves_for_C(x,y)
                elif abs(self.board[0][x][y]) == NP and np.sign(self.board[0][x][y]) == legal_sign:
                    moves = moves + self.get_moves_for_P(x,y)
                elif abs(self.board[0][x][y]) == NM and np.sign(self.board[0][x][y]) == legal_sign:
                    moves = moves + self.get_moves_for_M(x,y)
                elif abs(self.board[0][x][y]) == NX and np.sign(self.board[0][x][y]) == legal_sign:
                    moves = moves + self.get_moves_for_X(x,y)
                elif abs(self.board[0][x][y]) == NS and np.sign(self.board[0][x][y]) == legal_sign:
                    moves = moves + self.get_moves_for_S(x,y)
                elif abs(self.board[0][x][y]) == NB and np.sign(self.board[0][x][y]) == legal_sign:
                    moves = moves + self.get_moves_for_B(x,y)
                elif np.sign(self.board[0][x][y]) == -legal_sign:
                    continue
                else:
                    assert False

        # Add turn skip
        moves = moves + [(58, 0, 0)]

        return moves
    
    def getScore(self,player) :
        if player == PLAYER_HAN:
            return self.b_params[N_HAN_SCORE] - self.b_params[N_CHO_SCORE]
        else:
            return self.b_params[N_CHO_SCORE] - self.b_params[N_HAN_SCORE]
    
    def render(self):
        square_content = {
            # 兵/卒: Soldiers (Byung)
            -7: "b",
            +7: "B",
            # 士: Guards (Sa)
            -6: "s",
            +6: "S",
            # 象: Elephants (Xiang)
            -5: "x",
            +5: "X",
            # 馬: Horses (Ma)
            -4: "m",
            +4: "M",
            # 包: Cannons (Po)
            -3: "p",
            +3: "P",
            # 車: Chariots (Cha)
            -2: "c",
            +2: "C",
            # 漢/楚: General (Goong)
            -1: "g",
            +1: "G",
            # Empty space
            +0: "-",
        }
        
        print("   ---------------------")
        for i in range(10):
            y = 9-i
            print(" ", end="")
            print(y, end=" | ")
            for j in range(9):
                x = j
                print(square_content[self.board[0][x][y]], end="")
                if (x == 2 or x == 5) and (y >= 7 or y <= 2):
                    print("|", end="")
                else:
                    print(" ", end="")
            print("|")
        print("   ---------------------")
        print("     0 1 2 3 4 5 6 7 8")
    
    def get_moves_for_K(self, x, y):
        """Returns all the legal moves of K that use the given square as a base."""
        # Assert that the given piece is a K, and it is in a valid place
        assert abs(self.board[0][x][y]) == NK
        assert x >= 3 and x <= 5 and y >= 0 and y <= 2

        my_sign = np.sign(self.board[0][x][y])

        # Ordinary moves are same as S
        moves = self.get_moves_for_S(x, y)
        
        # Draw move
        for i in range(9):
            if y+i+1 >= CONFIG_Y:
                break
            if (abs(self.board[0][x][y+i+1]) == NK):
                moves.append((16+i, x, y))
            elif (abs(self.board[0][x][y+i+1]) != 0):
                break

        # return the generated move list
        return moves
    
    def get_moves_for_C(self, x, y):
        """Returns all the legal moves of C that use the given square as a base."""
        # Assert that the given piece is a C
        assert abs(self.board[0][x][y]) == NC

        my_sign = np.sign(self.board[0][x][y])

        moves = []
        for i in range(8): # 0 ~ 7: (1, 0) ~ (8, 0)
            if (x+i+1 >= CONFIG_X):
                break
            if (self.board[0][x+i+1][y] == 0): # Empty space
                moves.append((i, x, y))
                continue
            elif (np.sign(self.board[0][x+i+1][y]) != my_sign):    # Capture opponent piece
                moves.append((i, x, y))
            break

        for i in range(8): # 8 ~ 15: (-1, 0) ~ (-8, 0)
            if (x-i-1 < 0):
                break
            if (self.board[0][x-i-1][y] == 0): # Empty space
                moves.append((8+i, x, y))
                continue
            elif (np.sign(self.board[0][x-i-1][y]) != my_sign):    # Capture opponent piece
                moves.append((8+i, x, y))
            break

        for i in range(9): # 16 ~ 24: (0, 1) ~ (0, 9)
            if (y+i+1 >= CONFIG_Y):
                break
            if (self.board[0][x][y+i+1] == 0): # Empty space
                moves.append((16+i, x, y))
                continue
            elif (np.sign(self.board[0][x][y+i+1]) != my_sign):    # Capture opponent piece
                moves.append((16+i, x, y))
            break

        for i in range(9): # 25 ~ 33: (0, -1) ~ (0, -9)
            if (y-i-1 < 0):
                break
            if (self.board[0][x][y-i-1] == 0): # Empty space
                moves.append((25+i, x, y))
                continue
            elif (np.sign(self.board[0][x][y-i-1]) != my_sign):    # Capture opponent piece
                moves.append((25+i, x, y))
            break

        if ((x == 3 and (y == 0 or y == 7)) or (x == 4 and (y == 1 or y == 8))): # 34: (1, 1)
            if self.board[0][x+1][y+1] == 0 or np.sign(self.board[0][x+1][y+1]) != my_sign:
                moves.append((34, x, y))

        if (x == 3 and (y == 0 or y == 7)): # 35: (2, 2)
            if self.board[0][x+1][y+1] == 0 and (self.board[0][x+2][y+2] == 0 or np.sign(self.board[0][x+2][y+2]) != my_sign):
                moves.append((35, x, y))

        if ((x == 5 and (y == 0 or y == 7)) or (x == 4 and (y == 1 or y == 8))): # 36: (-1, 1)
            if self.board[0][x-1][y+1] == 0 or np.sign(self.board[0][x-1][y+1]) != my_sign:
                moves.append((36, x, y))

        if (x == 5 and (y == 0 or y == 7)): # 37: (-2, 2)
            if self.board[0][x-1][y+1] == 0 and (self.board[0][x-2][y+2] == 0 or np.sign(self.board[0][x-2][y+2]) != my_sign):
                moves.append((37, x, y))

        if ((x == 4 and (y == 1 or y == 8)) or (x == 5 and (y == 2 or y == 9))): # 38: (-1, -1)
            if self.board[0][x-1][y-1] == 0 or np.sign(self.board[0][x-1][y-1]) != my_sign:
                moves.append((38, x, y))

        if (x == 5 and (y == 2 or y == 9)): # 39: (-2, -2)
            if self.board[0][x-1][y-1] == 0 and (self.board[0][x-2][y-2] == 0 or np.sign(self.board[0][x-2][y-2]) != my_sign):
                moves.append((39, x, y))

        if ((x == 3 and (y == 2 or y == 9)) or (x == 4 and (y == 1 or y == 8))): # 40: (1, -1)
            if self.board[0][x+1][y-1] == 0 or np.sign(self.board[0][x+1][y-1]) != my_sign:
                moves.append((40, x, y))

        if (x == 3 and (y == 2 or y == 9)): # 41: (2, -2)
            if self.board[0][x+1][y-1] == 0 and (self.board[0][x+2][y-2] == 0 or np.sign(self.board[0][x+2][y-2]) != my_sign):
                moves.append((41, x, y))

        # return the generated move list
        return moves

    def get_moves_for_P(self, x, y):
        """Returns all the legal moves of P that use the given square as a base."""
        # Assert that the given piece is a P
        assert abs(self.board[0][x][y]) == NP

        my_sign = np.sign(self.board[0][x][y])

        moves = []
        done = [False, False, False, False]
        jump = [False, False, False, False]
        for i in range(9):
            for j in range(4):
                if (done[j]):
                    continue

                if (j == 0):    # 0 ~ 7: (1, 0) ~ (8, 0)
                    newx = x+i+1
                    newy = y
                    a = i
                elif (j == 1):  # 8 ~ 15: (-1, 0) ~ (-8, 0)
                    newx = x-i-1
                    newy = y
                    a = 8 + i
                elif (j == 2):  # 16 ~ 24: (0, 1) ~ (0, 9)
                    newx = x
                    newy = y+i+1
                    a = 16 + i
                else:           # 25 ~ 33: (0, -1) ~ (0, -9)
                    newx = x
                    newy = y-i-1
                    a = 25 + i
                
                # Invalid destination
                if (newx >= CONFIG_X or newx < 0 or newy >= CONFIG_Y or newy < 0):
                    done[j] = True
                    continue
                
                # Empty destination
                if (self.board[0][newx][newy] == 0): 
                    if (jump[j]):
                        moves.append((a, x, y))
                    continue
                # Nonempty destination
                else:
                    if (not jump[j]):  # The first piece that appear in such direction
                        if (abs(self.board[0][newx][newy]) == NP):   # P cannot jump over another P
                            done[j] = True
                            continue
                        else:
                            jump[j] = True
                            continue
                    else:   # The second piece that appears in such direction
                        if (abs(self.board[0][newx][newy]) != NP and np.sign(self.board[0][newx][newy]) != my_sign):
                            # Capture opponent piece
                            # P cannot capture another P
                            moves.append((a, x, y))
                        done[j] = True
                        continue
        
        if (x == 3 and (y == 0 or y == 7)): # 35: (2, 2)
            if (self.board[0][x+1][y+1] != 0 \
                and abs(self.board[0][x+1][y+1]) != NP \
                and (self.board[0][x+2][y+2] == 0 or np.sign(self.board[0][x+2][y+2]) != my_sign)\
                and abs(self.board[0][x+2][y+2]) != NP):
                moves.append((35, x, y))

        if (x == 5 and (y == 0 or y == 7)): # 37: (-2, 2)
            if (self.board[0][x-1][y+1] != 0 \
                and abs(self.board[0][x-1][y+1]) != NP \
                and (self.board[0][x-2][y+2] == 0 or np.sign(self.board[0][x-2][y+2]) != my_sign)\
                and abs(self.board[0][x-2][y+2]) != NP):
                moves.append((37, x, y))

        if (x == 5 and (y == 2 or y == 9)): # 39: (-2, -2)
            if (self.board[0][x-1][y-1] != 0 \
                and abs(self.board[0][x-1][y-1]) != NP \
                and (self.board[0][x-2][y-2] == 0 or np.sign(self.board[0][x-2][y-2]) != my_sign)\
                and abs(self.board[0][x-2][y-2]) != NP):
                moves.append((39, x, y))

        if (x == 3 and (y == 2 or y == 9)): # 41: (2, -2)
            if (self.board[0][x+1][y-1] != 0 \
                and abs(self.board[0][x+1][y-1]) != NP \
                and (self.board[0][x+2][y-2] == 0 or np.sign(self.board[0][x+2][y-2]) != my_sign)\
                and abs(self.board[0][x+2][y-2]) != NP):
                moves.append((41, x, y))

        return moves

    def get_moves_for_M(self, x, y):
        """Returns all the legal moves of M that use the given square as a base."""
        # Assert that the given piece is a M
        assert abs(self.board[0][x][y]) == NM

        my_sign = np.sign(self.board[0][x][y])

        moves = []
        if self._can_M_move(x, y, 2, 1):    # 42: (2, 1)
            moves.append((42, x, y))

        if self._can_M_move(x, y, 2, -1):   # 43: (2, -1)
            moves.append((43, x, y))
        
        if self._can_M_move(x, y, -2, 1):   # 44: (-2, 1)
            moves.append((44, x, y))
        
        if self._can_M_move(x, y, -2, -1):  # 45: (-2, -1)
            moves.append((45, x, y))
        
        if self._can_M_move(x, y, 1, 2):    # 46: (1, 2)
            moves.append((46, x, y))
        
        if self._can_M_move(x, y, 1, -2):   # 47: (1, -2)
            moves.append((47, x, y))
        
        if self._can_M_move(x, y, -1, 2):   # 48: (-1, 2)
            moves.append((48, x, y))
        
        if self._can_M_move(x, y, -1, -2):  # 49: (-1, -2)
            moves.append((49, x, y))

        return moves

    def _can_M_move(self, x, y, dx, dy):
        midx = int(x + dx/2) if (abs(dx) == 2) else x
        midy = y if (abs(dx) == 2) else int(y + dy/2)
        finx = x + dx
        finy = y + dy

        # Cannot move if the final position is invalid
        if finx < 0 or finx >= CONFIG_X or finy < 0 or finy >= CONFIG_Y:
            return False
        
        # Cannot move if a piece is in the way
        if self.board[0][midx][midy] != 0:
            return False

        # Cannot move if there is my piece in the final position
        if self.board[0][finx][finy] != 0 and np.sign(self.board[0][finx][finy]) == np.sign(self.board[0][x][y]):
            return False
        
        # Else, return true
        return True

    def get_moves_for_X(self, x, y):
        """Returns all the legal moves of X that use the given square as a base."""
        # Assert that the given piece is a X
        assert abs(self.board[0][x][y]) == NX

        moves = []
        if self._can_X_move(x, y, 3, 2):    # 50: (3, 2)
            moves.append((50, x, y))

        if self._can_X_move(x, y, 3, -2):   # 51: (3, -2)
            moves.append((51, x, y))
        
        if self._can_X_move(x, y, -3, 2):   # 52: (-3, 2)
            moves.append((52, x, y))
        
        if self._can_X_move(x, y, -3, -2):  # 53: (-3, -2)
            moves.append((53, x, y))
        
        if self._can_X_move(x, y, 2, 3):    # 54: (2, 3)
            moves.append((54, x, y))
        
        if self._can_X_move(x, y, 2, -3):   # 55: (2, -3)
            moves.append((55, x, y))
        
        if self._can_X_move(x, y, -2, 3):   # 56: (-2, 3)
            moves.append((56, x, y))
        
        if self._can_X_move(x, y, -2, -3):  # 57: (-2, -3)
            moves.append((57, x, y))

        return moves

    def _can_X_move(self, x, y, dx, dy):
        midx1 = int(x + dx/3) if (abs(dx) == 3) else x
        midy1 = y if (abs(dx) == 3) else int(y + dy/3)
        midx2 = int(x + dx/3*2) if (abs(dx) == 3) else int(x + dx/2)
        midy2 = int(y + dy/2) if (abs(dx) == 3) else int(y + dy/3*2)
        finx = x + dx
        finy = y + dy

        # Cannot move if the final position is invalid
        if finx < 0 or finx >= CONFIG_X or finy < 0 or finy >= CONFIG_Y:
            return False
        
        # Cannot move if a piece is in the way
        if self.board[0][midx1][midy1] != 0 or self.board[0][midx2][midy2] != 0:
            return False

        # Cannot move if there is my piece in the final position
        if self.board[0][finx][finy] != 0 and np.sign(self.board[0][finx][finy]) == np.sign(self.board[0][x][y]):
            return False
        
        # Else, return true
        return True

    def get_moves_for_S(self, x, y):
        """Returns all the legal moves of S that use the given square as a base."""
        # Assert that the given piece is a S, and it is in a valid place
        assert abs(self.board[0][x][y]) == NS or abs(self.board[0][x][y]) == NK
        assert x >= 3 and x <= 5 and y >= 0 and y <= 2

        my_sign = np.sign(self.board[0][x][y])

        moves = []
        if (x < 5): # 0: (1, 0)
            if self.board[0][x+1][y] == 0 or np.sign(self.board[0][x+1][y]) != my_sign:
                moves.append((0, x, y))
        if (x > 3): # 8: (-1, 0)
            if self.board[0][x-1][y] == 0 or np.sign(self.board[0][x-1][y]) != my_sign:
                moves.append((8, x, y))
        if (y < 2): # 16: (0, 1)
            if self.board[0][x][y+1] == 0 or np.sign(self.board[0][x][y+1]) != my_sign:
                moves.append((16, x, y))
        if (y > 0): # 25: (0, -1)
            if self.board[0][x][y-1] == 0 or np.sign(self.board[0][x][y-1]) != my_sign:
                moves.append((25, x, y))
        if ((x == 3 and y == 0) or (x == 4 and y == 1)): # 34: (1, 1)
            if self.board[0][x+1][y+1] == 0 or np.sign(self.board[0][x+1][y+1]) != my_sign:
                moves.append((34, x, y))
        if ((x == 5 and y == 0) or (x == 4 and y == 1)): # 36: (-1, 1)
            if self.board[0][x-1][y+1] == 0 or np.sign(self.board[0][x-1][y+1]) != my_sign:
                moves.append((36, x, y))
        if ((x == 4 and y == 1) or (x == 5 and y == 2)): # 38: (-1, -1)
            if self.board[0][x-1][y-1] == 0 or np.sign(self.board[0][x-1][y-1]) != my_sign:
                moves.append((38, x, y))
        if ((x == 3 and y == 2) or (x == 4 and y == 1)): # 40: (1, -1)
            if self.board[0][x+1][y-1] == 0 or np.sign(self.board[0][x+1][y-1]) != my_sign:
                moves.append((40, x, y))

        # return the generated move list
        return moves

    def get_moves_for_B(self, x, y):
        """Returns all the legal moves of B that use the given square as a base."""
        # Assert that the given piece is a B, and it is in a valid place
        assert abs(self.board[0][x][y]) == NB

        my_sign = np.sign(self.board[0][x][y])

        moves = []
        if (x < CONFIG_X - 1): # 0: (1, 0)
            if self.board[0][x+1][y] == 0 or np.sign(self.board[0][x+1][y]) != my_sign:
                moves.append((0, x, y))
        if (x > 0): # 8: (-1, 0)
            if self.board[0][x-1][y] == 0 or np.sign(self.board[0][x-1][y]) != my_sign:
                moves.append((8, x, y))
        if (y < CONFIG_Y - 1): # 16: (0, 1)
            if self.board[0][x][y+1] == 0 or np.sign(self.board[0][x][y+1]) != my_sign:
                moves.append((16, x, y))
        if ((x == 3 and y == 7) or (x == 4 and y == 8)): # 34: (1, 1)
            if self.board[0][x+1][y+1] == 0 or np.sign(self.board[0][x+1][y+1]) != my_sign:
                moves.append((34, x, y))
        if ((x == 5 and y == 7) or (x == 4 and y == 8)): # 36: (-1, 1)
            if self.board[0][x-1][y+1] == 0 or np.sign(self.board[0][x-1][y+1]) != my_sign:
                moves.append((36, x, y))

        # return the generated move list
        return moves

    def remove_piece(self, pcs, cap_piece):
        """ Given han_pcs or cho_pcs, remove one of the captured piece"""
        assert (1 <= cap_piece and cap_piece <= 7)
        pcs = int(pcs)

        if (cap_piece == NK):    # G
            return pcs & ~(1<<0)
        elif (cap_piece == NC):  # C
            if (pcs & (1<<1) != 0):
                return pcs & ~(1<<1)
            elif (pcs & (1<<2) != 0):
                return (pcs & ~(1<<2)) | (1<<1)
            else:
                assert False
        elif (cap_piece == NP):  # P
            if (pcs & (1<<3) != 0):
                return pcs & ~(1<<3)
            elif (pcs & (1<<4) != 0):
                return (pcs & ~(1<<4)) | (1<<3)
            else:
                assert False
        elif (cap_piece == NM):  # M
            if (pcs & (1<<5) != 0):
                return pcs & ~(1<<5)
            elif (pcs & (1<<6) != 0):
                return (pcs & ~(1<<6)) | (1<<5)
            else:
                assert False
        elif (cap_piece == NX):  # X
            if (pcs & (1<<7) != 0):
                return pcs & ~(1<<7)
            elif (pcs & (1<<8) != 0):
                return (pcs & ~(1<<8)) | (1<<7)
            else:
                assert False
        elif (cap_piece == NS):  # S
            if (pcs & (1<<9) != 0):
                return pcs & ~(1<<9)
            elif (pcs & (1<<10) != 0):
                return (pcs & ~(1<<10)) | (1<<9)
            else:
                assert False
        elif (cap_piece == NB):  # B
            if (pcs & (1<<11) != 0):
                return pcs & ~(1<<11)
            elif (pcs & (1<<12) != 0):
                return (pcs & ~(1<<12)) | (1<<11)
            elif (pcs & (1<<13) != 0):
                return (pcs & ~(1<<13)) | (1<<12)
            elif (pcs & (1<<14) != 0):
                return (pcs & ~(1<<14)) | (1<<13)
            elif (pcs & (1<<15) != 0):
                return (pcs & ~(1<<15)) | (1<<14)
            else:
                assert False

    def _get_piece_num(self, pcs, query_piece):
        """ Given han_pcs or cho_pcs, get the number of query_board"""
        assert (1 <= query_piece and query_piece <= 7)

        pcs = int(pcs)
        num = 0

        if (query_piece == NK):    # G
            num = pcs & (1<<0)
        elif (query_piece == NC):  # C
            num = (pcs & (3<<1))>>1
        elif (query_piece == NP):  # P
            num = (pcs & (3<<3))>>3
        elif (query_piece == NM):  # M
            num = (pcs & (3<<5))>>5
        elif (query_piece == NX):  # X
            num = (pcs & (3<<7))>>7
        elif (query_piece == NS):  # S
            num = (pcs & (3<<9))>>9
        elif (query_piece == NB):  # B
            num = (pcs & (31<<11))>>11
        else:
            assert False
        
        if (num == 0):
            return 0
        else:
            return int(np.log2(num) + 1)
    
    def game_ended(self):
        """ Return Cho score if the game is over.
            Return 0 otherwise. """
        # Compare these with han(cho)_pcs & ATTACK_MASK
        cannot_win_yangsa = [
            #         # ㅁㅁㅁ: BBBBB/SS/XX/MM/PP/CC/K
            # # 대삼능
            # 264,    # 포양상: 00000/00/10/00/01/00/0
            # 192,    # 양마상: 00000/00/01/10/00/00/0
            # 288,    # 마양상: 00000/00/10/01/00/00/0

            # # 소삼능
            # 2088,   # 포마졸: 00001/00/00/01/01/00/0
            # 2184,   # 포상졸: 00001/00/01/00/01/00/0
            # 2112,   # 양마졸: 00001/00/00/10/00/00/0
            # 2304,   # 양상졸: 00001/00/10/00/00/00/0
            # 4104,   # 포양졸: 00010/00/00/00/01/00/0
            # 4128,   # 마양졸: 00010/00/00/01/00/00/0
            # 4224,   # 상마졸: 00010/00/01/00/00/00/0
            # 8192,   # ㅁ삼졸: 00100/00/00/00/00/00/0

            # # 차삼능
            # 4106,   # 차이졸: 00010/00/00/00/01/01/0

            # # 차이능/
            # 10,     # ㅁ차포: 00000/00/00/00/01/01/0
            # 34,     # ㅁ차마: 00000/00/00/01/00/01/0
            # 130,    # ㅁ차상: 00000/00/01/00/00/01/0
            # 2050,   # ㅁ차졸: 00001/00/00/00/00/01/0
        ]

        cannot_win_wesa = [
            # 2050,   # ㅁ차졸: 00001/00/00/00/00/01/0
        ]

        # The player that just made a move
        last_player = PLAYER_HAN if self.b_params[N_CUR_PLAYER] == PLAYER_CHO else PLAYER_CHO

        # If turnskip was done 4 times in a row, end the game
        if self.b_params[N_TURNSKIP_CNT] >= 4:
            return 1 if self.b_params[N_CHO_SCORE] > self.b_params[N_HAN_SCORE] else -1

        # If the game just ended with a bic, end the game
        if self.b_params[N_IS_BIC]:
            return 1 if self.b_params[N_CHO_SCORE] > self.b_params[N_HAN_SCORE] else -1

        # Game is over if a K is captured
        if self._get_piece_num(self.b_params[N_HAN_PCS], NK) == 0:
            return 1
        if self._get_piece_num(self.b_params[N_CHO_PCS], NK) == 0:
            return -1

        # Game is over if no player can win
        if ((self._get_piece_num(self.b_params[N_HAN_PCS], NS) == 2 and (int(self.b_params[N_CHO_PCS]) & int(ATTACK_MASK)) in cannot_win_yangsa) \
                or (self._get_piece_num(self.b_params[N_HAN_PCS], NS) == 1 and (int(self.b_params[N_CHO_PCS]) & int(ATTACK_MASK)) in cannot_win_wesa)) \
            and ((self._get_piece_num(self.b_params[N_CHO_PCS], NS) == 2 and (int(self.b_params[N_HAN_PCS]) & int(ATTACK_MASK)) in cannot_win_yangsa) \
                or (self._get_piece_num(self.b_params[N_CHO_PCS], NS) == 1 and (int(self.b_params[N_HAN_PCS]) & int(ATTACK_MASK)) in cannot_win_wesa)):
            return 1 if self.b_params[N_CHO_SCORE] > self.b_params[N_HAN_SCORE] else -1

        # Game is over if repetition happens 3 times
        canonical_board = self.board[0]
        if (self.b_params[N_CUR_PLAYER] == PLAYER_HAN):
            canonical_board = np.flip(canonical_board, [0, 1])
        canon_string = canonical_board.tostring()

        if (self.rep_dict[canon_string] >= 2):
            # If both scores are under 30, check score
            if (self.b_params[N_CHO_SCORE] < 30 and self.b_params[N_HAN_SCORE] < 30):
                return 1 if self.b_params[N_CHO_SCORE] > self.b_params[N_HAN_SCORE] else -1
            # Otherwise, the last player lose
            else:
                return 1 if last_player == PLAYER_HAN else -1

        # For simplicity, the game is over when move_cnt hits 250.
        if self.b_params[N_MOVE_CNT] >= MAX_TURNS:
            return 1 if self.b_params[N_CHO_SCORE] > self.b_params[N_HAN_SCORE] else -1
        
        # Game is over if bic is called when a player has score >= 30.
        # The last player lose.
        if (self.b_params[N_HAN_SCORE] >= 30 or self.b_params[N_CHO_SCORE] >= 30):
            if (self._get_bic_called()):
                return 1 if last_player == PLAYER_HAN else -1
        
        # If the game is not over, return 0.
        return 0
    
    def _get_bic_called(self):
        """ Return true if two K's are facing each other """
        for i in range(3):
            for j in range(3):
                x = i+3
                y = j

                if (abs(self.board[0][x][y]) == NK):
                    for k in range(9):
                        newx = x
                        newy = y+k+1
                        
                        if (newy >= CONFIG_Y):
                            return False
                        
                        if (abs(self.board[0][newx][newy]) == NK):
                            # Two K's are facing each other
                            return True
                        elif (abs(self.board[0][newx][newy]) != 0):
                            # Two K's are not directly facing each other
                            return False
    
    def _piece_score(self, piece):
        if piece == NK:
            return 0
        elif piece == NC:
            return 13
        elif piece == NP:
            return 7
        elif piece == NM:
            return 5
        elif piece == NX:
            return 3
        elif piece == NS:
            return 3
        elif piece == NB:
            return 2
        else:
            assert False
    
    @staticmethod
    def _action_to_dxdy(a):
        if (a <= 7):
            return (a+1, 0)
        elif (a <= 15):
            return (-a+7, 0)
        elif (a <= 24):
            return (0, a-15)
        elif (a <= 33):
            return (0, -a+24)
        elif (a <= 35):
            return (a-33, a-33)
        elif (a <= 37):
            return (-a+35, a-35)
        elif (a <= 39):
            return (-a+37, -a+37)
        elif (a <= 41):
            return (a-39, -a+39)
        elif (a <= 43):
            return (2, 1 if a == 42 else -1)
        elif (a <= 45):
            return (-2, 1 if (a == 44) else -11)
        elif (a <= 47):
            return (1, 2 if (a == 46) else -2)
        elif (a <= 49):
            return (-1, 2 if (a == 48) else -2)
        elif (a <= 51):
            return (3, 2 if (a == 50) else -2)
        elif (a <= 53):
            return (-2, 2 if (a == 52) else -2)
        elif (a <= 55):
            return (2, 3 if (a == 54) else -3)
        elif (a <= 57):
            return (-2, 3 if (a == 56) else -3)
        elif (a == 58):
            return (0, 0)
        else:
            # This should not happen. Panic.
            assert False
    
    @staticmethod
    def _dxdy_to_action(dx, dy):
        for a in range(CONFIG_X*CONFIG_Y*CONFIG_A + 1):
            if (dx, dy) == Board._action_to_dxdy(a):
                return a
        return -1