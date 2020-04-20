import chess
import numpy as np
import argparse
import random
import os

import convert

def generate(raw_data, max_size, shuffle=False):
    dir = {"1-0": 1, "1/2-1/2": 0, "0-1": -1}

    raw_data = raw_data.split("\n")[5:]
    
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
                raise Exception("invalid data point after %d examples" % size)

            if len(history) == 8:
                history.pop(0)
            history.append(board.fen())
            
            yield history, result

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
    size = int(round(0.25 * size)) if is_testset else size
    
    datafile = open(trainfilename  % (path, format_number(size)), 'a')
    labelsfile = open(labelsfilename % (path, format_number(size)), 'a')
    for data, label in data_gen: 
        for idx, fen in enumerate(data):
            if idx < len(data) - 1:
                datafile.writelines("%s," % fen)
            else:
                datafile.writelines("%s\n" % fen)
        
        labelsfile.writelines("%s\n" % label)
    
    datafile.close()
    labelsfile.close()

def empty_file(path):
    open(path, "w").close()

def shuffle(paths):
    assert isinstance(paths, list)
    assert isinstance(paths[0], tuple)
    for path in paths:
        datafile = open(path[0], "r")
        labelsfile = open(path[1], "r")
            
        data = datafile.readlines()
        labels = labelsfile.readlines()
        
        open(path[0], "w").close()
        open(path[1], "w").close()

        c = list(zip(data, labels))
        random.shuffle(c)
        data, labels = zip(*c)
        
        datafile = open(path[0], "w")
        labelsfile = open(path[1], "w")
        
        datafile.writelines("%s" % fen for fen in data)
        labelsfile.writelines("%s" % label for label in labels)

        datafile.close()
        labelsfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', required=True, type=int)
    parser.add_argument('-p', '--path', required=True)
    parser.add_argument('-gt', '--generate_testset', action='store_true')
    parser.add_argument('-sh', '--shuffle', action='store_true')

    args = parser.parse_args()
    size = args.size
    path = args.path
    generate_testset = args.generate_testset
    bool_shuffle = args.shuffle
    
    f = open("data/raw_data.txt", "r")
    raw_data = f.read()
    f.close()
    
    data_gen = generate(raw_data, size, shuffle=bool_shuffle)
    write_files(data_gen, path, size)

    if generate_testset:
        data_gen = generate(raw_data, size * 0.25, shuffle=bool_shuffle)
        write_files(data_gen, path, size, is_testset=True)
    
    if bool_shuffle:
        trainset = ("%strain_data_%s" % (path, format_number(size)), "%strain_labels_%s" % (path, format_number(size)))
        if generate_testset:
            testset = ("%stest_data_%s" % (path, format_number(size * 0.25)), "%stest_labels_%s" % (path, format_number(size * 0.25)))
            shuffle([trainset,testset])
        else:
            shuffle([trainset])
