import chess
import numpy as np

pieces = {"P": 0, "N": 1, "B": 2,"R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}

#fen_list from 1 to 8, with 8 as last move
def bitboard(fen_list):
    """
    P1 piece: 6
    P2 piece: 6
    Repetions: 2
    total: 112, because of an eight step history
    ____________________________________________

    color: 1 w:0,b:1
    total move count: 1 (fullmove count)
    p1 castling: 2
    p2 castling: 2
    no-progress count: 1 (halfmove count)
    total: 7
    ____________________________________________
    
    total: 119
    output: (batch_size,119,8,8)
    """
    assert isinstance(fen_list, list)

    board_arr = np.zeros((119,8,8))

    board = chess.Board(fen_list[len(fen_list) - 1])

    board_arr[:112] = _encode_history(fen_list)
    #turn
    board_arr[112] = 0 if board.turn == chess.WHITE else 1
    #fullmove count
    board_arr[113] = board.fullmove_number
    #castling
    board_arr[114:118] = _get_castling_positions(board)
    #halfmove count
    board_arr[118] = board.halfmove_clock

    return board_arr 

def _encode_board(board):
    assert isinstance(board, chess.Board)
    board_arr = np.zeros((14,8,8))
    #encodes board
    for i in range(8):
        for j in range(8):
            piece_num = i * 8 + (7 - j) if board.turn == chess.BLACK else (7 - i) * 8 + j
            piece = board.piece_at(piece_num)
            if piece is not None:
                board_arr[pieces[str(piece)]][i][i] = 1.0
    #adds repetition count to the last two planes
    rep_count = _get_repetition_count(board)
    if(rep_count == 1 or rep_count == 2):
        board_arr[11 + rep_count] = 1.0
    return board_arr

def _encode_history(fen_list):
    assert isinstance(fen_list, list)
    boards_arr = np.zeros((112,8,8))
    for index, fen in enumerate(fen_list):
        board = chess.Board(fen)
        boards_arr[index * 14 :(index+1) * 14] = _encode_board(board)
    return boards_arr


def _get_repetition_count(board):
    for i in range(4):
        if board.is_repetition(i):
            return i

    raise Exception("invalid repetition count")

def _get_castling_positions(board):

    out = np.zeros((4,8,8))
    if board.has_queenside_castling_rights(chess.WHITE):
        if board.turn == chess.WHITE:
            out[0][7][2] = 1.0
        else:
            out[0][0][5] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        if board.turn == chess.WHITE:
            out[1][7][6] = 1.0
        else:
            out[1][0][1] = 1.0

    if board.has_queenside_castling_rights(chess.BLACK):
        if board.turn == chess.BLACK:
            out[2][7][5] = 1.0
        else:
            out[2][0][2] = 1.0

    if board.has_kingside_castling_rights(chess.BLACK):
        if board.turn == chess.BLACK:
            out[3][7][1] = 1.0
        else:
            out[3][0][6] = 1.0

    return out
