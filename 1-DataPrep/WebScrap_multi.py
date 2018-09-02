# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 15:04:14 2018

@author: jjunh
"""
from multiprocessing import Pool
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import time


def get_daily_news(date):
    # =========================================================================
    # request url and check
    # =========================================================================
#    date = 20150115
    url = 'https://www.reuters.com/resources/archive/us/{}.html'.format(date)
    result = requests.get(url)
    content = result.content
    content.decode().strip().replace('\t', '').split('\n')

    # =========================================================================
    # get titles and timestamp
    # =========================================================================
    soup = BeautifulSoup(content, 'html.parser')
    headlines = soup.find_all('div', {'class': 'headlineMed'})
    time.sleep(5)
    if headlines == []:
        print(str(date) + 'news: No news!')
    elif 'news_' + str(date) + '.csv' in os.listdir():
        pass
    else:
        start = time.time()
        # containers
        titles = []
        timestamp = []

        for title in headlines:
            titles.append(title.find('a').get_text())  # titles
            timestamp.append(date)  # timestamp

        # =========================================================================
        # get contents and meta data
        # =========================================================================
        # containers
        firstline = []
        contents = []
        keywords = []
        connected_url = []

        for url in headlines:
#            url = headlines[0]
            url_connected = url.find('a').attrs['href']
            connected_url.append(url_connected)

            result = requests.get(url_connected)
            content = result.content
            content.decode().strip().replace('\t', '').split('\n')
            soup_meta = BeautifulSoup(content, 'lxml')

            # error: page is not found (unavailable website)
            if soup_meta.find_all('div', {'class': 'body_1gnLA'}) != []:
                body = soup_meta.find_all('div', {'class': 'body_1gnLA'})
            elif soup_meta.find_all('div', {'class': 'StandardArticleBody_body'}) != []:
                body = soup_meta.find_all('div', {'class': 'StandardArticleBody_body'})
            try:
                firstline.append(body[0].find('p').get_text())
            except:
                firstline.append(np.NaN)
            try:
                contents.append(body[0].get_text())
            except:
                contents.append(np.NaN)
            try:
                # meta data
                tmp = soup_meta.find_all('meta', attrs={'name': 'news_keywords'})
                keywords.append(tmp[0]['content'])  # e.g. Politics, etc.
            except:
                keywords.append(np.NaN)
    
        assert len(titles) == len(firstline) == len(timestamp) == len(keywords) == len(connected_url)
    
        df = pd.DataFrame({'title': titles,
                           'firstline': firstline,
                           'content' : contents,
                           'timestamp': timestamp,
                           'keywords': keywords,
                           'connected_url': connected_url})
    
        # If there are no title, content, url, we safely assume that
        # this article is not important
        # e.g. 'Following is test release'
        # e.g. Video titles
        df_new = df.dropna(how='any', subset=['connected_url', 'title', 'content'])
        print(str(date) + 'news: completed' + ' / ' + str((time.time()-start)/60) + " Muinutes")
        # save by date
        df_new.to_csv('/home/junhyuki/DLproject/DAT/daily_news/news_' + date + '.csv',
                      encoding='utf-8-sig', index = False)
#        df_new.to_csv('D:/Working_file/DLproject/DAT/daily_news/news_' + date + '.csv',
#        encoding='utf-8-sig', index = False)
                
if __name__ == "__main__":
    print("# of Logical Processors : " + str(os.cpu_count()))
    import sys
    sys.setrecursionlimit(1000000)
    start = '7/1/2016'
    end =   '12/31/2016'
    date_range = pd.date_range(start=start, end=end, freq='D').strftime('%Y%m%d')
    
    with Pool(processes=16) as p:
        p.map(get_daily_news, date_range)
    
    