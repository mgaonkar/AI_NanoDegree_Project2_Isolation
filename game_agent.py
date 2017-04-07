"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import sys


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def move_towards_center(game, player):
    """Calculates the heuristic value as the distance from the center positions
    Reward for maximizing player is for moves involving  center positions 

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player):
        return float("inf")

    if game.is_loser(player):
        return float("-inf")

    height = game.height
    width = game.width

    center_positions = [(width/2, (height / 2) - 1),
               ((width / 2) - 1, height/2),
               ((width / 2) - 1, (height / 2) - 1),
               (width/2,height/2)]

    own_moves = game.get_legal_moves(player)
    own_center = [move for move in own_moves if move in center_positions]

    # opponent
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    opponent_center = [move for move in opponent_moves if move in center_positions]

    # reward based on moves with center positions
    return float(len(own_moves) + (len(own_center))
                 - (len(opponent_moves) + (len(opponent_center))))

def move_to_opponent(game, player):
    """Penalize large distance between the player and the opponent

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player):
        return float("inf")

    if game.is_loser(player):
        return float("-inf")

    #player has not moved.
    own_location = game.get_player_location(player)
    if own_location == None:
        return 0

    #Opponent has not moved.
    opponent_location = game.get_player_location(game.get_opponent(player))
    if opponent_location == None:
        return 0
    #difference in co-ordinates of player and opponent squared
    diff = (own_location[0] - opponent_location[0])**2 + (own_location[1]- opponent_location[1])**2

    return float(-diff)



def custom_score(game, player):
    """Calculates the heuristic value as a difference between the number of 
    legal moves available to the two players.
    Penalty for maximizing player is for moves involving the edges on the board 
    This penalty is heightened as the number of legal positions to move decreases 
    by 75% as corner moves are more probable to end in self isolation on a 
    crowded board
    Also added in is the distance from the center positions
    Reward for maximizing player is for moves involving  center positions
    This is a amalgamation of 'board_usage_with_corner' and the 'move_towards_center' heuristic

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player):
        return float("inf")

    if game.is_loser(player):
        return float("-inf")

    #use board_usage_value as a factor to reward/penalise players, initialized with 1 for no reward /no penalty

    height = game.height
    width = game.width

    board_usage_value = 1
    total_spaces = height * width


    board_edges = [(0,0),
                   (0, (game.height - 1)),
                   ((width -1), 0),
                   ((width -1),(height -1 ))]

    center_positions = [(width / 2, (height / 2) - 1),
                        ((width / 2) - 1, height / 2),
                        ((width / 2) - 1, (height / 2) - 1),
                        (width / 2, height / 2)]
   #Being in a corner will lead to having less legal moves available for player, increasing board usage value accordingly
    if len(game.get_blank_spaces()) < int(total_spaces / 4.):
        board_usage_value= 4

   #player
    own_moves = game.get_legal_moves(player)
    own_corner = [move for move in own_moves if move in board_edges]
    own_center = [move for move in own_moves if move in center_positions]

    #opponent
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    opponent_corner = [move for move in opponent_moves if move in board_edges]
    opponent_center = [move for move in opponent_moves if move in center_positions]

    #penalize and reward based on board usage value
    return float((len(own_moves) - (board_usage_value * len(own_corner)) + len(own_center))
                  - (len(opponent_moves) + (board_usage_value * len(opponent_corner))+ len(opponent_center)))


def board_usage_with_corner(game, player):
    """Calculates the heuristic value as a difference between the number of 
    legal moves available to the two players.
    Penalty for maximizing player is for moves involving the edges on the board 
    This penalty is heightened as the number of legal positions to move decreases 
    by 75% as corner moves are more probable to end in self isolation on a 
    crowded board

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player):
        return float("inf")

    if game.is_loser(player):
        return float("-inf")

    #use board_usage_value as a factor to reward/penalise players, initialized with 1 for no reward /no penalty

    board_usage_value = 1
    total_spaces = game.height * game.width

    board_edges = [(0,0),
                   (0, (game.height - 1)),
                   ((game.width -1), 0),
                   ((game.width -1),(game.height -1 ))]
   #Being in a corner will lead to having less legal moves available for player, increasing board usage value accordingly
    if len(game.get_blank_spaces()) < int(total_spaces / 4.):
        board_usage_value= 4

   #player
    own_moves = game.get_legal_moves(player)
    own_corner = [move for move in own_moves if move in board_edges]

    #opponent
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    opponent_corner = [move for move in opponent_moves if move in board_edges]

    #penalize and reward based on board usage value
    return float(len(own_moves) - (board_usage_value * len(own_corner))
                  - (len(opponent_moves) + (board_usage_value * len(opponent_corner))))


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)  This parameter should be ignored when iterative = True.

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).  When True, search_depth should
        be ignored and no limit to search depth.

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            DEPRECATED -- This argument will be removed in the next release

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        #No legal moves, isolated
        if not legal_moves:
            return (-1, -1)
        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        result = None

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                if self.method == "minimax":
                    for depth in range(sys.maxsize): #The largest positive integer supported by the platform’s Py_ssize_t type
                        _, move = self.minimax(game, depth)
                        result = move
                if self.method == "alphabeta":
                    for depth in range(sys.maxsize):
                        _, move = self.alphabeta(game, depth)
                        result = move
            else:
                if self.method == "minimax":
                    _, result = self.minimax(game, self.search_depth)
                if self.method == "alphabeta":
                    _, result = self.alphabeta(game, self.search_depth)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return result

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        #Get legal moves for the current player
        legal_moves = game.get_legal_moves()

        #There are no more legal moves, game over and (-1, -1) for no legal moves
        if not legal_moves:
            return game.utility(self), (-1,1)

        #Terminal depth reached
        if depth == 0:
            return self.score(game,self), (-1,1)

        best_move = None
        if maximizing_player:
            # Best for maximizing player is highest score
            best_score = float("-inf")
            for move in legal_moves:
                # Forecast_move switches the active player
                next_state = game.forecast_move(move)
                score, _ = self.minimax(next_state, depth - 1, False)
                if score > best_score:
                    best_score, best_move = score, move
        # Else minimizing player
        else:
            # Best for minimizing player is lowest score
            best_score = float("inf")
            for move in legal_moves:
                next_state = game.forecast_move(move)
                score, _ = self.minimax(next_state, depth - 1, True)
                if score < best_score:
                    best_score, best_move = score, move
        return best_score, best_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        legal_moves = game.get_legal_moves()

        # There are no more legal moves, game over and (-1, -1) for no legal moves
        if not legal_moves:
            return game.utility(self), (-1, 1)

        # Terminal depth reached
        if depth == 0:
            return self.score(game, self), (-1, 1)

        best_move = None
        if maximizing_player:
            # Best for maximizing player is highest score
            best_score = float("-inf")
            for move in legal_moves:
                # Forecast_move switches the active player
                next_state = game.forecast_move(move)
                score, _ = self.alphabeta(next_state, depth - 1, alpha, beta, False)
                if score > best_score:
                    best_score, best_move = score, move
                # Prune if possible
                if best_score >= beta:
                    return best_score, best_move
                # Update alpha, if necessary
                alpha = max(alpha, best_score)
        # Else minimizing player
        else:
            # Best for minimizing player is lowest score
            best_score = float("inf")
            for move in legal_moves:
                next_state = game.forecast_move(move)
                score, _ = self.alphabeta(next_state, depth - 1, alpha, beta, True)
                if score < best_score:
                    best_score, best_move = score, move
                # Prune if possible
                if best_score <= alpha:
                    return best_score, best_move
                # Update beta, if necessary
                beta = min(beta, best_score)
        return best_score, best_move

