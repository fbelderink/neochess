import chess
import numpy as np
import argparse
import random

import convert

def generate(raw_data, max_size, preprocess=False ,shuffle=False):
    dir = {"1-0": 1, "1/2-1/2": 0, "0-1": -1}

    raw_data = raw_data.split("\n")[5:]
    
    if shuffle:
        random.shuffle(raw_data)
    
    size = 0
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
<<<<<<< Updated upstream
=======
                bb = convert.bitboard(history)
                l = dir[result]
                yield bb, l
            else:
                yield history, result
            """
            if preprocess:
>>>>>>> Stashed changes
                data.append(convert.bitboard(history))
                labels.append(dir[result])
            else:
                data.append(history.copy())
                labels.append(result)
            """
    """
    if shuffle: 
        c = list(zip(data, labels))
        random.shuffle(c)
        data, labels = zip(*c)

    return data, labels
    """

def format_number(num):
    suffixes = ['', 'K', 'M', 'G', 'T', 'P']
    num = float('{:.3g}'.format(num))
    m = 0
    while abs(num) > 1000:
        m += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), suffixes[m])

def write_files(data_gen, path, size, is_testset=False):
    #writes data and labels
    
    trainfilename = "%strain_data_%s" if not is_testset else "%stest_data_%s"
    labelsfilename = "%strain_labels_%s" if not is_testset else "%stest_labels_%s"
    size = 0.25 * size if is_testset else size
    
    datafile = open(trainfilename  % (path, format_number(size)), 'w')
    labelsfile = open(labelsfilename % (path, format_number(size)), 'w') 
    for data, labels in data_gen: 
        for idx, fen in enumerate(data):
            if idx < len(data) - 1:
                datafile.writelines("%s," % fen)
            else:
                datafile.writelines("%s\n" % fen)
        
        labelsfile.writelines("%s\n" % label for label in labels)
    
    datafile.close()
    labelsfile.close()

def shuffle(paths, preprocessed=False):
    assert isinstance(paths, list)
    assert len(paths) == 2
    assert isinstance(paths[0], tuple) and isinstance(paths[1], tuple)
    #TODO
    if preprocessed:
        pass
    else:
        for path in paths[:1]:
            datafile = open(path[0], "r+")
            labelsfile = open(path[1], "r+")
            
            data = datafile.read()
            labels = labelsfile.read()

            datafile.truncate(0)
            labelsfile.truncate(0)

            data = data.split("\n")
            labels = labels.split("\n")

            c = list(zip(data, labels))
            random.shuffle(c)
            data, labels = zip(*c)
            
            for fen in data:
                datafile.writelines("%s\n" % fen)

            for label in labels:
                labelsfile.writelines("%s\n" % label)

            datafile.close()
            labelsfile.close()


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
    bool_shuffle = args.shuffle
    
    #shuffle([("data/train_data_100K", "data/train_labels_100K"), ()])
    f = open("data/raw_data.txt", "r")
    raw_data = f.read()
    f.close()

    data_gen = generate(raw_data, size, preprocess=preprocess, shuffle=bool_shuffle)
    
    #TODO preprocess with generator 
    if preprocess:
        name = "%strainset_%s.npz" % (path, format_number(size))
        for data, labels in data_gen:
            data, labels = np.array(data), np.array(labels)
            np.savez(name, data, labels)
    else:
        write_files(data_gen, path, size)

    if generate_testset:
        data_gen = generate(raw_data, size * 0.25, preprocess=preprocess, shuffle=shuffle)
        
        #TODO preprocess with generator
        if preprocess:
            name = "%stestset_%s.npz" % (path, format_number(size))
            for data, labels in data_gen:
                data, labels = np.array(data), np.array(labels)
                np.savez(name, data, labels)
        else:
            write_files(data_gen, path, size, is_testset=True)
    #TODO 
    if shuffle:
        if preprocessed:
            shuffle(["","","",""], preprocessed=True)
        else:
            shuffle(["","","",""])
