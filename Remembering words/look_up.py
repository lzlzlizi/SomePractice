import requests 
import bs4
import re
import pandas as pd
import urllib.request
import json
import requests
import re
import os
import random
import urllib

import MySQLdb
db = MySQLdb.connect(host = 'localhost',#本地数据库
                             user = 'root', #用户名
                             passwd = 'yourpw', #数据库密码
                             db = 'mydict', #数据库名
                             charset = 'utf8') 

f =open("words.txt")
words = f.read()
f.close()
words = re.split(" |\n",words)
words = set(words)

NN = 1

try:
    with open('data.json', 'r') as f:
        word_base = json.load(f)
except:
    word_base = dict()

def look_up(word):
    url = 'https://cn.bing.com/dict/search?q='+word

    da = requests.request(url=url,method='get')
    soup = bs4.BeautifulSoup(da.text)
    rubbish = [s.extract() for s in soup('script')]
    soup = soup.body
    
    try:
        qdef = soup.findAll('ul')[1]
        quick_def = ''
        for s in qdef.findAll('li'):
            quick_def = quick_def + s.text + '\n'
        quick_def = quick_def.replace('网络','Web.')
    except:
        return None
    
    try:
        p = soup.findAll('div', {"class": "hd_prUS"})[0].text
        p = re.findall(r'\[(.*?)\]', p)[0]
    except:
        p = ''
    
    try:
        s_url = soup.find('div', {"class": "hd_tf"}).a['onclick']
        s_url = s_url.replace("javascript:BilingualDict.Click(this,'",'')
        s_url = s_url.replace("','akicon.png',false,'dictionaryvoiceid')",'')
    except:
        s_url = ''
    
    try:
        more_def = ''
        for pos in soup.findAll('div', {"class": "li_pos"}):
            w_pos = pos.find('div', {"class": "pos"}).text
            for w_def in pos.findAll('div', {"class": "de_co"}):
                more_def = more_def +w_pos+w_def.text+ '\n'
    except:
        return None

    return {'word':word,'sound':p,'chinese':quick_def,'more_def':more_def,'sound_url':s_url}

def get_definition(w):
    global NN
    if(w in word_base):
        da = word_base[w]
    else:   
        da = look_up(w)
        NN = NN+1
        if da != None:
            word_base[w] = da
        else:
            return None
    return da

def insert_res(w):
    sql = 'insert ignore into mydict.dicts values("'+w["word"]+'","'+w["sound"]+'","'+w['more_def']+'","'+w["chinese"]+'")'
    cursor = db.cursor()
    cursor.execute(sql)
    db.commit()   

    if os.path.exists('audio/'+w['word']+'.mp3'):
        return
    else:
        try:
            url = w['sound_url']
            urllib.request.urlretrieve(url,'audio/'+w['word']+'.mp3')
        except:
            return 
        
hhd = 1     
for word in words:
    w = get_definition(word)
    if NN % 10 == 0:
        with open('data.json', 'w') as f:
            json.dump(word_base, f)
    if w != None:
        try:
            insert_res(w)
        except:
            pass
        print('已完成第'+str(hhd)+'个:  '+w['word'])
        hhd = hhd+1  
    
with open('data.json', 'w') as f:
    json.dump(word_base, f)

    
print('ALL DONE!!!\n')
