# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 12:56:49 2018

@author: jjunh
"""
from multiprocessing import Process, Queue
import pandas as pd
import numpy as np
import ast
import os
import time
# change your java path and add it as an environment variable
#java_path = "C:/Program Files/Java/jdk-9.0.4/bin/java.exe"
java_path = "usr/bin/java/java.exe"
os.environ['JAVAHOME'] = java_path
# load python wrapper for Stanford CoreNLP
# https://stanfordnlp.github.io/CoreNLP/
from stanfordcorenlp import StanfordCoreNLP
#nlp = StanfordCoreNLP(r'D:\Working_file\DLproject\ExternalLib\stanford-corenlp-full-2018-02-27',
#                      memory='8g')
nlp = StanfordCoreNLP('/home/junhyuki/DLproject/ExternalLib/stanford-corenlp-full-2018-02-27',
                      memory='8g')
# We will use OpenIE (Open Information Extraction)
props={'annotators': 'tokenize, ssplit, pos, lemma, depparse, natlog, openie',
       'pipelineLanguage':'en',
       'outputFormat':'json',                        # one of {json, xml, text}
       'openie.format': 'default',    # One of {reverb, ollie, default, qa_srl}
       'openie.triple.strict': 'true',
       'openie.affinity_probability_cap': '1',
       'openie.max_entailments_per_clause': '1000',   # default = 1000
       }
# =============================================================================
""" functions """ 
# =============================================================================
def deleteNaN(df, which_col):
    """eliminate 'nan'"""
    df_new = df.dropna(how='any', subset=which_col)
    #df[df['clean_firstlines'].notnull()]
    print('original dataframe rows: ' + str(df.shape[0]) +
          ' -> after delete: ' + str(df_new.shape[0]))
    return df_new.reset_index(drop=True)

def ExtractSVO(text):
    """Extract first (S,V,O) from each sentence """
    openie = nlp.annotate(text, properties=props)
    openie = ast.literal_eval(openie)              # convert str to dict
    if openie["sentences"][0]["openie"] != []:
        first_svo = openie["sentences"][0]["openie"][0]
#        tuples = (first_sro['subject'], first_sro['relation'], first_sro['object'])
        s, v, o = first_svo['subject'], first_svo['relation'], first_svo['object']
        return s, v, o
    else:
        s, v, o = np.repeat(np.nan, 3)
        return s, v, o

def put_data_into_queue(df, which_col, q1, q2):
    print("loading data & position index >>> putting them into the queue")
    for i in df.index:
        q1.put(df.loc[i, which_col][0])
        q2.put(i)      # for maintaining the order
    
def get_tuples(q1, q2, df, data_size, result_save_path):
    start = time.time()
    for i in range(data_size):
        text = q1.get()
        idx  = q2.get()  # for keeping order
        s, v, o = ExtractSVO(text)
        # assigning
        df.loc[idx, ('subject', 'verb', 'object')] = s, v, o
        if (i+1) % 10000 == 0:
            end = time.time()
            print(str(i+1) + " th (S,V,O) extracted / " + str((end-start)/60) + " Minutes")
    # delete no (subject, relation, object) rows
    df = deleteNaN(df, ['subject', 'verb', 'object'])
    print(df.head())
    print('final shape : ' + str(df.shape))
    # save file
    df.to_csv(result_save_path, index=False)
    return df


if __name__ == "__main__":
    print("# of Logical Processors : " + str(os.cpu_count()))
    # read data
    #dat_path = 'D:/Working_file/DLproject/DAT'
    dat_path = '/home/junhyuki/DLproject/DAT'
    usecols = ['connected_url', 'keywords', 'timestamp', 'clean_title', 'location']
    
#    for year in np.arange(2012, 2019):
    for year in np.arange(2015, 2019):
        df = pd.read_csv(dat_path + '/1-DailyNews/cleaned_news_{}.csv'.format(year),
                         usecols=usecols)
        print('Processing "' + str(year) + '" news data ...')
        col_name = ['clean_title']
        df = deleteNaN(df, col_name)
        data_size = len(df)
        # make new columns
        df['subject'], df['verb'], df['object'] = np.repeat(np.nan, 3)
        # result save
        result_save_path = dat_path + '/2-SVO/SVO_{}.csv'.format(year)
        # make queue
        q1 = Queue()
        q2 = Queue()

        # get processor
        p1 = Process(target=put_data_into_queue, args=(df, col_name, q1, q2))
        p2 = Process(target=get_tuples, args=(q1, q2, df, data_size, result_save_path))
        # begin processor
        p1.start()
        p2.start()
        # closing queue
        q1.close()
        q2.close()
        # complete the process
        p1.join()
        p2.join()
        print("Completed")


    
#    df = pd.read_csv(dat_path + '/1-DailyNews/cleaned_news_2012.csv')
#    print('Processing "2012" news data ...')
#    col_name = ['clean_titles']
#    df = deleteNaN(df, col_name)
#    df = df[:1000]
#    data_size = len(df)
#    # make new columns
#    df['subject'], df['verb'], df['object'] = np.repeat(np.nan, 3)
#    # result save
#    result_save_path = dat_path + '/2-SVO/SVO_2012.csv'
#    # make queue
#    q1 = Queue()
#    q2 = Queue()
#
#    # get processor
#    p1 = Process(target=put_data_into_queue, args=(df, col_name, q1, q2))
#    p2 = Process(target=get_tuples, args=(q1, q2, df, data_size, result_save_path))
#    # begin processor
#    p1.start()
#    p2.start()
#    # closing queue
#    q1.close()
#    q2.close()
#    # complete the process
#    p1.join()
#    p2.join()
#    print("Completed")

