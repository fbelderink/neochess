import chess
import numpy as np
import argparse
import random

import convert

def generate(max_size, preprocess=False ,shuffle=False):
    dir = {"1-0": 1, "1/2-1/2": 0, "0-1": -1}
    f = open("data/raw_data.txt", "r")
    raw_data = f.read()
    f.close()

    raw_data = raw_data.split("\n")[5:]
    
    if shuffle:
        random.shuffle(raw_data)
    
    size = 0
    data = []
    labels = []

    for match in raw_data:
        if size >= max_size:
            break
        result = match.split(" ")[2]
        moves = match.split("###")[1].split(" ")
        moves = list(filter(None, moves))
        board = chess.Board()
        history = []
        for move in moves:
            if size >= max_size:
                break
            size += 1
            
            raw_move = move.split(".")[1]
            try:
                board.push_san(raw_move.strip())
            except:
                raise Exception("invalid data point")

            if len(history) == 8:
                history.pop(0)
            history.append(board.fen())
            
            if preprocess:
                data.append(convert.bitboard(history.copy()))
                labels.append(dir[result])
            else:
                data.append(history.copy())
                labels.append(result)
    
    if shuffle: 
        c = list(zip(data, labels))
        random.shuffle(c)
        data, labels = zip(*c)

    return data, labels

def format_number(num):
    suffixes = ['', 'K', 'M', 'G', 'T', 'P']
    num = float('{:.3g}'.format(num))
    m = 0
    while abs(num) > 1000:
        m += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), suffixes[m])

def write_files(data, labels, path, size, is_testset=False):
    #writes data and labels
    
    name = "%strain_data_%s" if not is_testset else "%stest_data_%s"
    size = 0.25 * size if is_testset else size

    with open(name  % (path, format_number(size)), 'w') as f:
        for list in data:
            for index, fen in enumerate(list):
                if index < len(list) - 1:
                    f.writelines("%s," % fen)
                else:
                    f.writelines("%s\n" % fen)
        f.close()
    
    name = "%strain_labels_%s" if not is_testset else "%stest_labels_%s"
    with open(name % (path, format_number(size)), 'w') as f:
        f.writelines("%s\n" % result for result in labels)
        f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', required=True, type=int)
    parser.add_argument('-p', '--path', required=True)
    parser.add_argument('-gt', '--generate_testset', action='store_true')
    parser.add_argument('-pp', '--preprocess', action='store_true')
    parser.add_argument('-sh', '--shuffle', action='store_true')

    args = parser.parse_args()
    size = args.size
    path = args.path
    generate_testset = args.generate_testset
    preprocess = args.preprocess
    shuffle = args.shuffle

    data, labels = generate(size, preprocess=preprocess, shuffle=shuffle)
    
    if preprocess:
        data, labels = np.array(data), np.array(labels)
        name = "%strainset_%s.npz" % (path, format_number(size))
        np.savez(name, data, labels)
    else:
        write_files(data, labels, path, size)

    if generate_testset:
        data, labels = generate(size * 0.25, preprocess=preprocess, shuffle=shuffle)
        
        if preprocess:
            data, labels = np.array(data), np.array(labels)
            name = "%stestset_%s.npz" % (path, format_number(size))
            np.savez(name, data, labels)
        else:
            write_files(data, labels, path, size, True)
