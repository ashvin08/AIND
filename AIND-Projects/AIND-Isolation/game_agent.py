"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import math


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

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
    #Follow the opponent
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    r1,c1 = game.get_player_location(player)
    r2,c2 = game.get_player_location(game.get_opponent(player))
    #Distance between two points
    return float(math.sqrt((r1-r2)**2+(c1-c2)**2))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

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
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    #Percentage of board filled
    board_filled = float(game.move_count/float(game.width*game.height))

    #Percentage of board not filled
    board_not_filled = 1 - board_filled

    diff_moves = float(len(game.get_legal_moves(player)) - len(game.get_legal_moves(game.get_opponent(player))))
    open_moves = float(len(game.get_legal_moves(player)))
    return float((board_filled * open_moves) + (board_not_filled * diff_moves))

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

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


    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    board_filled = float(game.move_count / float(game.width * game.height))
    board_not_filled = 1 - board_filled
    diff_moves = float(len(game.get_legal_moves(player)) - len(game.get_legal_moves(game.get_opponent(player))))
    open_moves = float(len(game.get_legal_moves(player)))

    #Awarding moves which block the opponent
    if board_filled > 0.5:
        return float((board_filled * open_moves)*2 + (board_not_filled * diff_moves)*0.5)

    #Awarding open moves since the board is over half filled
    else:
        return float((board_filled * open_moves)*0.5 + (board_not_filled * diff_moves)*2)

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score_3, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

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

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            return best_move  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        def min_value(parent_move, current_game, current_depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            #Game ended
            if current_game.utility(self) != 0:
                return current_game.utility(self)
            current_depth += 1
            min_game = current_game.forecast_move(parent_move)

            #Depth condition reached
            if current_depth > depth:
                return self.score(min_game, min_game.inactive_player)
            v = float("inf")
            #Search max positions
            for move in min_game.get_legal_moves():
                v = min(v, max_value(move, min_game, current_depth))
            return v

        def max_value(parent_move, current_game, current_depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            if current_game.utility(self) != 0:
                return current_game.utility(self)
            current_depth += 1
            max_game = current_game.forecast_move(parent_move)
            if current_depth > depth:
                return self.score(max_game, max_game.active_player)
            v = float("-inf")
            #Search min positions
            for move in max_game.get_legal_moves():
                v = max(v, min_value(move, max_game, current_depth))
            return v

        moves = game.get_legal_moves()
        if len(moves) == 0:
            return (-1, -1)
        best_score = float("-inf")
        best_move = moves[0]
        for move in moves:
            score = min_value(move, game, 1)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

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

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            depth = 1
            while True:
                if len(game.get_legal_moves()) == 0:
                    return best_move
                best_move = self.alphabeta(game, depth, float("-inf"), float("inf"))
                depth += 1

        except SearchTimeout:
            return best_move  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

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

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        def min_value(parent_move, current_game, alpha, beta, current_depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            if current_game.utility(self) != 0:
                return current_game.utility(self)
            current_depth += 1
            min_game = current_game.forecast_move(parent_move)
            if current_depth > depth:
                return self.score(min_game, min_game.inactive_player)
            v = float("inf")
            for move in min_game.get_legal_moves():
                v = min(v, max_value(move, min_game, alpha, beta, current_depth))
                #Prune the tree
                if v <= alpha:
                    return v
                #Set beta
                beta = min(v, beta)
            return v

        def max_value(parent_move, current_game, alpha, beta, current_depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            if current_game.utility(self) != 0:
                return current_game.utility(self)
            current_depth += 1
            max_game = current_game.forecast_move(parent_move)
            if current_depth > depth:
                return self.score(max_game, max_game.active_player)
            v = float("-inf")
            for move in max_game.get_legal_moves():
                v = max(v, min_value(move, max_game, alpha, beta, current_depth))
                #Prune the tree
                if v >= beta:
                    return v
                #Set alpha
                alpha = max(v, alpha)
            return v

        moves = game.get_legal_moves()
        if len(moves) == 0:
            return (-1, -1)
        best_score = float("-inf")
        best_move = moves[0]
        for move in moves:
            score = min_value(move, game, best_score, float("inf"), 1)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move
