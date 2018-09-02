# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 11:15:15 2018

@author: jjunh
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 09:34:24 2018

@author: jjunh
@content: Matching trained word2vec to text data (S, R, O)

@Input Data:

    svo_data.csv (only use single (S,R,O))
===============================================================================
col1,   col2,          col3,      col4,     col5,   col6
title,  clean_title,   subject,   verb,   object,   topic code
===============================================================================

@Output Data
numpy array for subject, verb, object
vecS.npy
vecR.npy
vecO.npy

"""
from multiprocessing import Process, Queue
import pandas as pd
import numpy as np
import time
         
def get_data(df, q1, q2):
    print("loading data and numpy index >> putting them into the queue")
    for i in range(len(df)):
        q1.put(df['verb'].iloc[i])
        q2.put(i)
            
def drop_no_full_match(q1, q2, error_list, data_size):
    """ ls: e.g. ['macau', 'gambling', 'revenue'] """
    drop_idx_verb = np.zeros((data_size, ))
    start = time.time()
    for i in range(data_size):
        data = q1.get()
        idx = q2.get()
        if data in error_list:
            drop_idx_verb[idx] = 1
        if (i+1) % 100000 == 0:
            end = time.time()
            print(str(i+1) + " th data / " + str((end-start)/60) + " Minutes")

    np.save(dat_path + '/drop_idx_verb.npy', drop_idx_verb)
    return drop_idx_verb
            

if __name__ == "__main__":
    """ Change the file name!! """
    dat_path = '/home/junhyuki/DLproject/DAT'
    df = pd.read_csv(dat_path + '/2-SVO/SVO_chunker_truncated_fullmatch.csv', encoding='utf-8-sig')
    error_list = list(np.load(dat_path + '/error_list.npy'))
    data_size = len(df)

    # make queue
    q1 = Queue()
    q2 = Queue()
    
    # get processor
    p1 = Process(target=get_data, args=(df, q1, q2))
    p2 = Process(target=drop_no_full_match, args=(q1, q2, error_list, data_size))
    # begin processor
    p1.start()
    p2.start()
    # closing queue
    q1.close()
    
    # complete the process
    p1.join()
    p2.join()
    #__________________________________________________________________
    print("complete")