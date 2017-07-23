"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import isolation
import game_agent

from importlib import reload
from game_agent import MinimaxPlayer
from game_agent import AlphaBetaPlayer

class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = MinimaxPlayer()
        self.player2 = AlphaBetaPlayer(search_depth=2)

    def test1(self):
        self.game = isolation.Board(self.player1, self.player2, width=9, height=9)
        test_board_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                            1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0,
                            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 57]
        self.game.set_board_state(test_board_state)
        move, message = self.game.play_one_move()
        print("Played Move:", move)
        self.game.print_board()
        self.assertTrue(move in [(4, 4)])

    def test2(self):
        self.game = isolation.Board(self.player2, self.player1, width=9, height=9)
        test_board_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 68]
        self.game.set_board_state(test_board_state)
        move, message = self.game.play_one_move()
        print("Played Move:", move)
        self.game.print_board()
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
