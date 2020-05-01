import chess
import numpy as np

rep = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}

#returns (16,8,8) vector
def convert_board(board):
    assert isinstance(board, chess.Board)
    state = np.zeros((16, 8, 8))
    
    #pieces
    for i in range(64):
        piece = board.piece_at(i)
        if piece is not None:
            state[rep[str(piece)]][7 - int(i / 8)][i % 8] = 1.0
    
    #castling
    if board.has_queenside_castling_rights(chess.WHITE):
        state[12][7][0] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE):
        state[12][7][7] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        state[12][0][0] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        state[12][0][7] = 1.0
    
    #repetition
    rep_count = _get_repetition_count(board)
    if rep_count == 1 or rep_count == 2:
        state[12 + rep_count] = 1.0
    
    #color
    state[15] = board.turn * 1.0

    return state

def _get_repetition_count(board):
    for i in range(3):
        if board.is_repetition(i + 1):
            return i + 1
    raise Exception("invalid repition count")
