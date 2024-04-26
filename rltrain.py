
import numpy as np
import pickle
import matplotlib.pyplot as plt

BOARD_ROWS = 3
BOARD_COLS = 3


class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        self.playerSymbol = 1
        self.states_value = {}

    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        return self.boardHash

    def winner(self):

    #row
        for i in range(BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1
            # col
        for i in range(BOARD_COLS):
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1
            # diagonal
        diag_sum1 = sum([self.board[i, i] for i in range(BOARD_COLS)])
        diag_sum2 = sum([self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)])
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == 3:
            self.isEnd = True
            if diag_sum1 == 3 or diag_sum2 == 3:
                return 1
            else:
                return -1
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
            # not end
        self.isEnd = False
        return None
    # Check for winning conditions

    def availablePositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    positions.append((i, j))
            #print(positions)
        return positions
        # Return available positions for a move

    def update_trainState(self, positions):
        self.board[positions] = self.playerSymbol
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    def play(self, rounds, train_mode=False):
        rewards = []
        reward = 0
        if train_mode:
            print("Training")
        for i in range(rounds):
            if i % 1000 == 0:
                print("Rounds {}".format(i))
            exp_rate = max(0.01, 0.3 - 0.001 * i)

            while not self.isEnd:
                positions = self.availablePositions()
                current_board = self.board.copy()  # Store the current board state
                symbol = self.playerSymbol

                if self.playerSymbol == 1:
                    action = self.p1.chooseAction(positions, current_board, symbol)
                    self.update_trainState(action)
                    board_hash = self.getHash()
                    self.p1.addState(board_hash)
                    win = self.winner()
                    if win is not None:
                        reward = self.calculateReward(win)
                        self.giveReward(board_hash, action, reward)
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break
                else:
                    action = self.p2.chooseAction(positions, current_board, symbol)
                    self.update_trainState(action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)
                    win = self.winner()
                    if win is not None:
                        reward = self.calculateReward(win)
                        self.giveReward(board_hash, action, reward)
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break
                rewards.append(reward)

        plt.plot(rewards)
        plt.xlabel('Actions')
        plt.ylabel('Rewards')
        plt.title('Rewards received for each action during training')
        plt.show()

    def calculateReward(self, win):
        if win == 1:
            return 1  # Win reward
        elif win == -1:
            return -1  # Lose reward
        else:
            return 0.1  # Intermediate reward for tie

    def giveReward(self, state, action, reward):
        state_action = f"{state}_{action}"
        if state_action not in self.states_value:
            self.states_value[state_action] = 0.0
        next_state_value = max(self.states_value.values()) if self.states_value else 0.0
        self.states_value[state_action] += self.p1.lr * (reward + self.p1.decay_gamma * next_state_value - self.states_value[state_action])

    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1


class Player:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.states = []  # record all positions taken
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}

    def chooseAction(self, positions, current_board, symbol):
        # Choose action based on epsilon-greedy strategy
        if np.random.uniform(0, 1) <= self.exp_rate:
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -np.inf
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_board_hash = self.getHash(next_board)
                value = self.states_value.get(next_board_hash, 0.0)
                if value >= value_max:
                    value_max = value
                    action = p
        return action

    def addState(self, state):
        self.states.append(state)

    def reset(self):
        self.states = []

    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return boardHash

    def savePolicy(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.states_value, f)

    def loadPolicy(self, file_name):
        with open(file_name, 'rb') as f:
            self.states_value = pickle.load(f)

if __name__ == "__main__":
    p1 = Player("p1")
    p2 = Player("p2")
    st = State(p1, p2)
    print("Training...")
    st.play(50000)
    p1.savePolicy("policy_p1.pkl")

