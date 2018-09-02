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
import re
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Word2Vec
from nltk import word_tokenize

         
def get_avgVec_phrase(phrase, w2v_model, dim):
    """ get averaged subject or verb or object vector
     Note:
      - If there is no matched word, then pass it.
      - If there is no matched word in total, then return zero vectors
      
    e.g) get_avgVec_phrase(svo_set, skipgram, 'subject', dim)
    """
    match_list = word_tokenize(re.sub('[-&()=<>{}:;]', ' ',
                                        re.sub('[!‘’"“”#%\'*+,./?@[\\]^_`|~]', '', phrase)).lower())
    nwords = 0
    avg_vec = np.zeros((1,dim),  dtype='float32')
    for word in match_list:
        if word in w2v_model.wv.vocab:
            nwords += 1
            avg_vec = np.add(avg_vec, w2v_model.wv[word])
    if nwords > 0:
        avg_vec = np.divide(avg_vec, nwords)
        
    return avg_vec

def get_data(df, q1, q2):
    print("loading (S,V,O) data and numpy index >> putting them into the queue")
    for j in range(3):
        for i in range(len(df)):
            if j == 0:
                q1.put(df.loc[i, 'subject'])
                q2.put(i)
            elif j == 1:
                q1.put(df.loc[i, 'verb'])
                q2.put(i)
            else:
                q1.put(df.loc[i, 'object'])
                q2.put(i)
                

def get_vecSVO(q1, q2, w2v_model, dim, data_size, dat_path, npy_name):
    """ matching word2vec to (S,V,O) """
    # Container
    vecS = np.zeros((data_size, dim))
    vecV = np.zeros((data_size, dim))
    vecO = np.zeros((data_size, dim))
    # filling
    start = time.time()
    for j in range(3):
        for i in range(data_size):
            text = q1.get()
            idx = q2.get()  # for keeping order
            processed = get_avgVec_phrase(text, w2v_model, dim)
        # assigning
            if j == 0:
                vecS[idx,:] = processed
            elif j == 1:
                vecV[idx,:] = processed
            else:
                vecO[idx,:] = processed
            
            if (i+1) % 100000 == 0:
                end = time.time()
                print(str(i+1) + " th (S,V,O) matched / " + str((end-start)/60) + " Minutes")
            
    assert vecS.shape == vecV.shape == vecO.shape
    print(vecS.shape, vecV.shape, vecO.shape)
    np.save(dat_path + npy_name[0], vecS)
    np.save(dat_path + npy_name[1], vecV)
    np.save(dat_path + npy_name[2], vecO)
    
    return vecS, vecV, vecO


if __name__ == "__main__":
    """ Change the file name!! """
    dat_path = '/home/junhyuki/DLproject/DAT'
    svo_data = pd.read_csv(dat_path + '/2-SVO/SVO_chunker_final_unique.csv')
    print("svo data shape:" + str(svo_data.shape))
    data_size = len(svo_data)
    
    d, w = 100, 7
    skipgram = Word2Vec.load(dat_path + '/3-Word2Vec/skipgram_d{}_w{}_m5_ng5'.format(d, w))
    print("Load Skipgram Model... 'skipgram_d{}_w{}_m5_ng5'".format(d, w) + '----------------------------------')
    npy_name = ['/4-SVO_vector/vecS_chunker_unique_d{}_w{}.npy'.format(d, w),
                '/4-SVO_vector/vecV_chunker_unique_d{}_w{}.npy'.format(d, w),
                '/4-SVO_vector/vecO_chunker_unique_d{}_w{}.npy'.format(d, w)]
    #__________________________________________________________________
    # make queue
    q1 = Queue()
    q2 = Queue()
            
    # get processor
    p1 = Process(target=get_data, args=(svo_data, q1, q2))
    p2 = Process(target=get_vecSVO, args=(q1, q2, skipgram, d,
                                                  data_size,
                                                  dat_path, npy_name))
    # begin processor
    p1.start()
    p2.start()
    # closing queue
    q1.close()
            
    # complete the process
    p1.join()
    p2.join()
    #__________________________________________________________________
    print("(S,V,O) data size is: " + str(data_size))
    
    
    # dat_path = '/home/junhyuki/DLproject/DAT'
    # svo_data = pd.read_csv(dat_path + '/2-SVO/SVO_chunker_final.csv')
    # print("svo data shape:" + str(svo_data.shape))
    # data_size = len(svo_data)
    # 
    # d, w = 100, 7
    # skipgram = Word2Vec.load(dat_path + '/3-Word2Vec/skipgram_d{}_w{}_m5_ng5'.format(d, w))
    # print("Load Skipgram Model... 'skipgram_d{}_w{}_m5_ng5'".format(d, w) + '----------------------------------')
    # npy_name = ['/4-SVO_vector/vecS_chunker_d{}_w{}.npy'.format(d, w),
    #             '/4-SVO_vector/vecV_chunker_d{}_w{}.npy'.format(d, w),
    #             '/4-SVO_vector/vecO_chunker_d{}_w{}.npy'.format(d, w)]
    # #__________________________________________________________________
    # # make queue
    # q1 = Queue()
    # q2 = Queue()
    #         
    # # get processor
    # p1 = Process(target=get_data, args=(svo_data, q1, q2))
    # p2 = Process(target=get_vecSVO, args=(q1, q2, skipgram, d,
    #                                               data_size,
    #                                               dat_path, npy_name))
    # # begin processor
    # p1.start()
    # p2.start()
    # # closing queue
    # q1.close()
    #         
    # # complete the process
    # p1.join()
    # p2.join()
    # #__________________________________________________________________
    # print("(S,V,O) data size is: " + str(data_size))
    
    
    # d_list = [300, 200, 100]
    # w_list = [7, 6, 5]
    # for i in range(3):
    #     d = d_list[i]
    #     for j in range(3):
    #         w = w_list[j]
    #         skipgram = Word2Vec.load(dat_path + '/3-Word2Vec/skipgram_d{}_w{}_m5_ng5'.format(d, w))
    #         print("Load Skipgram Model... 'skipgram_d{}_w{}_m5_ng5'".format(d, w) + '----------------------------------')
    #         npy_name = ['/4-SVO_vector/vecS_chunker_d{}_w{}.npy'.format(d, w),
    #                     '/4-SVO_vector/vecV_chunker_d{}_w{}.npy'.format(d, w),
    #                     '/4-SVO_vector/vecO_chunker_d{}_w{}.npy'.format(d, w)]
    #         #__________________________________________________________________
    #         # make queue
    #         q1 = Queue()
    #         q2 = Queue()
    #         
    #         # get processor
    #         p1 = Process(target=get_data, args=(svo_data, q1, q2))
    #         p2 = Process(target=get_vecSVO, args=(q1, q2, skipgram, d,
    #                                               data_size,
    #                                               dat_path, npy_name))
    #         # begin processor
    #         p1.start()
    #         p2.start()
    #         # closing queue
    #         q1.close()
    #         
    #         # complete the process
    #         p1.join()
    #         p2.join()
    #         #__________________________________________________________________
    #         print("(S,V,O) data size is: " + str(data_size))

#    svo_file_name = '/2-SVO/SVO.csv'
#    dat_path = '/home/junhyuki/DLproject/DAT'
#    svo_data = pd.read_csv(dat_path + svo_file_name)
#    data_size = len(svo_data)
#    
#    d_list = [300, 200, 100]
#    w_list = [7, 6, 5]
#    for i in range(3):
#        d = d_list[i]
#        for j in range(3):
#            w = w_list[j]
#            skipgram = Word2Vec.load(dat_path + '/3-Word2Vec/skipgram_d{}_w{}_m5_ng5'.format(d, w))
#            print("Load Skipgram Model... 'skipgram_d{}_w{}_m5_ng5'".format(d, w))
#            npy_name = ['/4-SVO_vector/vecS_d{}_w{}.npy'.format(d, w),
#                        '/4-SVO_vector/vecV_d{}_w{}.npy'.format(d, w),
#                        '/4-SVO_vector/vecO_d{}_w{}.npy'.format(d, w)]
#            #__________________________________________________________________
#            # make queue
#            q1 = Queue()
#            q2 = Queue()
#            
#            # get processor
#            p1 = Process(target=get_data, args=(svo_data, q1, q2))
#            p2 = Process(target=get_vecSVO, args=(q1, q2, skipgram, d,
#                                                  data_size,
#                                                  dat_path, npy_name))
#            # begin processor
#            p1.start()
#            p2.start()
#            # closing queue
#            q1.close()
#            
#            # complete the process
#            p1.join()
#            p2.join()
#            #__________________________________________________________________
#            print("(S,V,O) data size is: " + str(data_size))    



#    d, w = 300, 5
#    skipgram = Word2Vec.load(dat_path + '/3-Word2Vec/skipgram_d{}_w{}_m5_ng5'.format(d, w))
#    print("Load Skipgram Model... 'skipgram_d{}_w{}_m5_ng5'".format(d, w))
#    npy_name = ['/4-SVO_vector/vecS_d{}_w{}.npy'.format(d, w),
#                '/4-SVO_vector/vecV_d{}_w{}.npy'.format(d, w),
#                '/4-SVO_vector/vecO_d{}_w{}.npy'.format(d, w)]
#    # make queue
#    q1 = Queue()
#    q2 = Queue()
#    
#    # get processor
#    p1 = Process(target=get_data, args=(svo_data, q1, q2))
#    p2 = Process(target=get_vecSVO, args=(q1, q2, skipgram, d,
#                                          data_size,
#                                          dat_path, npy_name))
#    # begin processor
#    p1.start()
#    p2.start()
#    # closing queue
#    q1.close()
#    
#    # complete the process
#    p1.join()
#    p2.join()
#    #__________________________________________________________________
#    print("(S,V,O) data size is: " + str(data_size))        
