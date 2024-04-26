from nltk.corpus import wordnet as wn
import numpy as np
import pickle
from gensim.models import Word2Vec, KeyedVectors
import re

import w2v
import random

from w2v import word2vec_model

BOARD_ROWS = 3
BOARD_COLS = 3
# word2vec_model = 'word2vec_wordnet_model.bin'
class State:
    def __init__(self, p1, p2, base_words_p1, base_words_p2):
        # Initialize original board with integer symbols and duplicate board for visualization
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)  # Initialize the board with zeros
        self.duplicate_board = np.empty((BOARD_ROWS, BOARD_COLS), dtype=object)
        self.duplicate_board.fill("")  # Initialize the duplicate board with empty strings
        self.p1 = p1
        self.p2 = p2
        self.base_words_p1 = base_words_p1
        self.base_words_p2 = base_words_p2
        self.isEnd = False
        self.boardHash = None
        self.playerSymbol = 1
        self.name = p2.name
        self.actions_p1 = []
        self.actions_p2 = []
        self.states_value = {}


    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        return self.boardHash

    def winner(self):
        # row
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
                return -1  # Return None if no winner or draw yet
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
        # not end
        self.isEnd = False
        return None

    def is_similar_word(self, word, base_word):
        synsets_base_word = wn.synsets(str(base_word))  # Ensure base_word is converted to string
        synsets_word = wn.synsets(str(word))  # Ensure word is converted to string
        if synsets_base_word and synsets_word:
            for syn_base in synsets_base_word:
                for syn_word in synsets_word:
                    if syn_base.wup_similarity(syn_word) is not None:
                        return True
        return False

    def availablePositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    positions.append((i, j))
        # print(positions)
        return positions

    def updateState(self, positions, playerSymbol, word):
        row, col = positions
        self.duplicate_board[row][col] = word  # Update duplicate board with input word
        self.board[row][col] = self.playerSymbol  # Update original board with player symbol
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1
        if playerSymbol == 1:
            self.actions_p1.append((positions, word))
        elif playerSymbol == -1:
            self.actions_p2.append((positions, word))

    def getActionsPlayer1(self):
        return self.actions_p1

    def getActionsPlayer2(self):
        return self.actions_p2

    def update_trainState(self, positions):
        self.board[positions] = self.playerSymbol  # Update original board with player symbol
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1



    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.duplicate_board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1



    def play(self, rounds, train_mode=False):
        rewards = []
        if train_mode:
            print("Training")
        for i in range(rounds):
            if i % 1000 == 0:
                print("Rounds {}".format(i))
                exp_rate = max(0.01, 0.3 - 0.001 * i)    # Your existing code here

            while not self.isEnd:
                # Player 1
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)

                self.update_trainState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)

                win = self.winner()
                if win is not None:
                    reward = self.calculateReward(win)  # Calculate reward based on game outcome
                    self.giveReward(board_hash, p1_action, reward)  # Update rewards based on state-action pair
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # Player 2
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)

                    self.update_trainState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        reward = self.calculateReward(win)  # Calculate reward based on game outcome                            self.giveReward(board_hash, p2_action, reward)  # Update rewards based on state-action pair
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break
                    rewards.append(self.p1_reward if win == 1 else self.p2_reward)

            # Plot rewards as before

    def calculateReward(self, win):
        if win == 1:
            return 1  # Win reward
        elif win == -1:
            return -1  # Lose reward
        else:
            return 0.1  # Intermediate reward for tie

    def giveReward(self, state, action, reward):
        state_action = f"{state}_{action}"  # Unique identifier for state-action pair
        if self.states_value.get(state_action) is None:
            self.states_value[state_action] = 0
        self.states_value[state_action] += p1.lr * (reward - self.states_value[state_action])

    def play2(self):
        duplicate_board = self.duplicate_board

        self.playerSymbol = - 1
        print("Current board:")
        self.print_display_board(self.duplicate_board)
        print(st.board)

        while not self.isEnd:
            self.playerSymbol = 1
            word_h = input(f"{p2.name}, input your word: ")
            if p2.is_word_valid(word_h, base_words_p2):
                row, col = p2.get_word_position(word_h)
                if duplicate_board[row][col] == "":
                    duplicate_board[row][col] = word_h
                    print("Move accepted!")
                    print("Updated board:")
                    self.updateState((row, col), self.playerSymbol, word_h)
                    self.print_display_board(duplicate_board)

                    print(st.board)
                    # Update game state with agent move
                    win = self.winner()
                    if win is not None:
                        self.display_winner(win)
                        break
                    else:
                        self.playerSymbol = -1
                # Use loaded policy for computer player
                    agent_action = self.p1.chooseAction(self.availablePositions(), self.board,
                                                        self.playerSymbol,
                                                        base_word=self.base_words_p1)
                    # print(position)# Pass the agent's base word
                    if agent_action:
                        row, col = agent_action
                        # print(agent_action)
                        word = self.p1.get_word_similar_to_base(self.base_words_p1)
                        # print(word)
                        # word = self.is_similar_word(c_word, base_words_p1)
                        duplicate_board[row][col] = word  # Display agent's word
                        print(f"{p1.name} placed word '{duplicate_board[row][col]}' in position '({row},{col})'")
                        self.updateState(agent_action, self.playerSymbol, word)
                        print("Updated board:")
                        self.print_display_board(duplicate_board)
                        # print(st.board)

                        win = self.winner()
                        if win is not None:
                            self.display_winner(win)
                            break
                    else:
                        print("Agent action is None. Skipping agent's turn.")
                else:
                    print("position already occupied ({},{}).".format(row, col))
            else:
                print("Word {} is not similar to base {}. Try again.".format(word_h, base_words_p2))



    def print_display_board(self, duplicate_board):
        for row in duplicate_board:
            # Convert None values to empty strings and extract the word part from tuples
            row_str = [cell[0] if isinstance(cell, tuple) and cell != "" else str(cell) for cell in row]
            print(" | ".join(row_str))
            print("-" * (len(row_str) * 4 - 1))

    def display_winner(self, winner):
        if winner == -1:
            print("Computer wins!")
        elif winner == 1:
            print("Human wins!")
        else:
            print("It's a draw!")


class Player:
    def __init__(self, name, exp_rate=0.3):
        self.p2 = HumanPlayer
        self.name = name
        self.states = []
        self.lr = 0.1
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}
        self.word2vec = Word2Vec.load('word2vec_wordnet_model.bin')  # Load Word2Vec model
        self.policy_loaded = False
        self.q_values = {}
        self.chosen_words = []# Load Word2Vec model

    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return boardHash

    def is_player_word_valid(self, word, base_word):
        if word and isinstance(word, str):  # Check if word is not None and is a string
            synsets_base_word = wn.synsets(base_word)
            synsets_word = wn.synsets(word)
            if synsets_base_word and synsets_word:
                for syn_base in synsets_base_word:
                    for syn_word in synsets_word:
                        if syn_base.wup_similarity(syn_word) is not None:
                            return True
        return False

    def chooseAction(self, positions, board, playerSymbol, base_word):
        action = None
        if np.random.uniform(0, 0.5) <= self.exp_rate:
            # take random action
            idx = np.random.choice(len(positions))
            action = positions[idx]
            return action
        else:
            value_max = -999
            for p in positions:
                next_board = board.copy()
                next_board[p] = playerSymbol
                next_boardHash = self.getHash(next_board)
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
                # print("value", value)
                if value >= value_max:
                    value_max = value
                    action = p
        #print("agent takes action {}".format(action))
        return action

    def get_synonyms(self, word):
        synonyms = set()
        synsets = wn.synsets(word)
        for synset in synsets:
            for lemma in synset.lemmas():
                synonyms.add(lemma.name())
        return list(synonyms)

    def get_word_similar_to_base(self, base_word):
        synsets_base_word = self.get_synonyms(base_word)
        random_word = None
        while not random_word or random_word in self.chosen_words:  # Ensure a new word is chosen
            random_word = random.choice(synsets_base_word).split('.')[0]
        self.chosen_words.append(random_word)

        if isinstance(random_word, str):# check if randomword is a string

            print(random_word)
        return random_word


    def addState(self, state):
        self.states.append(state)

    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    def reset(self):
        self.states = []

    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()
        self.policy_loaded = True


class HumanPlayer:
    def __init__(self, name):
        self.name = name
        self.base_word = base_words_p2
        self.base_word = self.base_word.lower()
        self.glove_model_path = 'glove.840B.300d.txt'


    def is_word_valid(self, base_word, word, similarity_threshold=0.5):
        #Check WordNet synsets for semantic similarity
        synsets_base_word = wn.synsets(base_word)
        synsets_word = wn.synsets(word)
        for syn1 in synsets_base_word:
            for syn2 in synsets_word:
                if syn1.wup_similarity(syn2) is not None and syn1.wup_similarity(syn2) >= similarity_threshold:
                    return True
        # Calculate cosine similarity using Word2Vec for additional simi
        # larity measure
        if base_word in word2vec_model.wv and word in word2vec_model.wv:
            cosine_similarity = word2vec_model.wv.similarity(base_word, word)
            if cosine_similarity >= similarity_threshold:
                return True
        return False

    def get_word_position(self, word):
        # Some logic to convert word to position
        # For simplicity, let's assume user will input row and column directly
        row = int(input("Input your action row: "))
        col = int(input("Input your action col: "))
        return row, col


if __name__ == "__main__":
    base_words_p1 = input("Enter base word for Player 1, computer: ").lower()
    base_words_p2 = input("Enter base word for Player 2, human: ").lower()

    p1 = Player("computer", exp_rate=0.2)
    p1.loadPolicy("policy_p1.pkl")
    p2 = HumanPlayer("human")
    st = State(p1, p2, base_words_p1, base_words_p2)
    initial_positions = st.availablePositions()
    if not initial_positions:
        print("Error: No available positions on the board. Adjust board initialization or game rules.")
    else:
        st.play2()
