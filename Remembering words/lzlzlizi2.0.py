#!/usr/bin/env python
import sys
from PyQt5.QtCore import QSize,QCoreApplication,Qt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPalette, QBrush,QCursor
from PyQt5.QtWidgets import *
import MySQLdb
import re
import datetime
import warnings
import markdown
import random
import os
import json
import subprocess
import copy
import re

N =10

def short(s):
    return re.sub("[A-Za-z0-9\[\`\~\!\@\#\$\^\&\*\(\)\=\|\{\}\'\:\;\'\,\[\]\.\<\>\/\?\~\！\@\#\\\&\*\%]", "", s)

def group_pre():
    sql = """SELECT g.word,timestampdiff(minute,time,now()) diff 
            FROM group5_30 g join remembering r on g.word = r.word
            where timestampdiff(minute,time,now()) > 0 and timestampdiff(minute,time,now()) <= 10  and r.status = 0
            order by diff desc ;"""
    cursor = db.cursor()
    cursor.execute(sql)
    db.commit()
    res = cursor.fetchall()
    try:
        group5 = [rr[0] for rr in res]
    except:
        group5 = []

    sql = """SELECT g.word,timestampdiff(minute,time,now()) diff 
            FROM group5_30 g join remembering r on g.word = r.word
            where timestampdiff(minute,time,now()) > 30 and timestampdiff(minute,time,now()) <= 40 and r.status = 1
            order by diff desc;"""
    cursor = db.cursor()
    cursor.execute(sql)
    db.commit()
    res = cursor.fetchall()

    try:
        group30 = [rr[0] for rr in res]
    except:
        group30 = []
    
    f = open('5_30_min_log.txt','a',encoding = 'utf8')
    f.write('$#$5min\n'+str(group5)+'\n$#$30min\n'+str(group30)+'\n###############################\n\n')
    f.close()
    res = group5 + group30
    return set(res)

def fetch(w):
    # from dicts
    sql = "select word,sound,chinese,eng from dicts where dicts.word ='%s'; " %(w)
    cursor = db.cursor()
    cursor.execute(sql)
    db.commit()

    result = cursor.fetchone()
    if result == None: return None,''
    if result[3] != '':
        w_def = result[3]
    else:
        w_def = result[2]
    www  = result[2]
    if w_def == '' or '隐私声明和 Cookie' in w_def: return None,''
    p = result[1]
    
    # from 3000
    w_3000 = ''
    sql = "select word,def from gre_3000 where word ='%s'; " %(w)
    cursor.execute(sql)
    db.commit()
    result = cursor.fetchone()
    if result != None: 
        w_3000 = '<ul><li>'+result[1]+'</li>'
        while 1:
            result = cursor.fetchone()
            if result!=None:
                w_3000 = w_3000 +'<li>'+ result[1]+'</li>'
            else:
                break
        w_3000 = w_3000.replace('\n','')+'</ul>'
        
    short_mean = short(w_3000)
    if short_mean == '':
        short_mean = short(www)
        
    if w_3000 == '' and ('Web' in w_def.split('\n')[0]):
        return None,''
        
    if w_3000 == '':
        w_def = w_def.replace('\n','<BR>') 
    else:
        w_def = w_3000 + '\n\n----------------\n' + w_def.replace('\n','<BR>')
  
    w_info = {'word':w, 'p':p, 'def':w_def}
    info = "**"+w_info['word']+"**  ["+w_info['p']+"]"
    info = info +"<BR>"+ w_info['def']
    info = markdown.markdown(info)
    
    return info,short_mean

def play(word):
    path = 'audio/'+word+'.mp3'
    if os.path.exists(path):
       # print(['ffplay', "-loglevel", "-8", "-nodisp",'-autoexit', path])
        subprocess.Popen(['ffplay', "-loglevel", "-8", "-nodisp",'-autoexit', path])
    

def make_table(word):
    col = ['#FF7256','white','#7FFFD4']
    res = '<table border="1" cellspacing="0" width = "500px" height = "300px">'
    
    for ww in word:
        t = """
        <tr bgcolor = '%s'>
        <td>%s</td>
        <td>%s</td>
        </tr>
        """%(col[status[ww[0]]],ww[0],ww[3].replace('：',''))
        res = res + t
    res = res + '</table>\n </body>'
    return res


class words:
    def __init__(self,my_word):
        #my_words = my_word
        self.words = []
        self.word_inf = dict()
        error = set()
        
        for w in my_word:
            info,s_m = fetch(w)
            if info != None:
                if w in status:
                    if status[w] >=2: continue
                    self.words.append(w)
                    self.word_inf[w] = [w,info,status[w],s_m]
                else:
                    self.words.append(w)
                    self.word_inf[w] = [w,info,1,s_m]
                    status[w] = 1
                    sql = """insert ignore into remembering values('%s',%s)""" % (w, 1)
                    cursor = db.cursor()
                    cursor.execute(sql)
                    db.commit()
            else:
                error.add(w)
        print(len(self.words))
        f = open('error.txt','a',encoding = 'utf8')
        for w in error:
            f.write(w+'\n')
        f.close()
        
        # get current time
        date_time = datetime.datetime.now()
        date_time = date_time.strftime("%Y-%m-%d %H:%M:%S")
        # insert new words
        cursor = db.cursor()
        for w in self.words:
            # appdata
            sql = """insert ignore into appdata values('%s',%s,'%s')""" % (w, 0, date_time)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cursor.execute(sql)
                db.commit()
  
        self.group = []
        self.grouping()
        self.current_group = self.group.pop(0)
        
    def grouping(self):
        self.group = []
        first = []
        for w in list(group_pre()):
            if w in self.words:
                first.append(w)
        
        random.shuffle(first)
        second  = list(set(self.words).difference(first))
        random.shuffle(second)
        waiting = first + second
        t = []
        for w in waiting:
            if len(t) == N:
               self.group.append(t)
               t = []
            t.append(w)
        self.group.append(t)
            

class SystemTrayIcon(QtWidgets.QSystemTrayIcon):
    def __init__(self, icon, parent=None):
        super(SystemTrayIcon, self).__init__(icon, parent)
        menu = QtWidgets.QMenu(parent)
        exitAction = menu.addAction("Exit")
        exitAction.triggered.connect(parent.close)

        clear_remembring = menu.addAction('clear remembering')
        clear_remembring.triggered.connect(self.clear_remembring_fun)

        self.setContextMenu(menu)

    def clear_remembring_fun(self,event):
        cursor = db.cursor()
        sql = 'delete from mydict.remembering'
        cursor.execute(sql)
        db.commit()

class MainWindow(QMainWindow):
    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle('lzlzlizi words 2.0')
        icon1 = QtGui.QIcon('pic/tray.png')
        tray = SystemTrayIcon(icon1,self)
        tray.show()


        self.initUI()
        self.setWindowIcon( QtGui.QIcon('pic/tray.png'))
        oImage = QImage("pic/background.jpg")
        sImage = oImage.scaled(QSize(1280,720))                   # resize Image to widgets size
        palette = QPalette()
        palette.setBrush(10, QBrush(sImage))                     # 10 = Windowrole
        self.setPalette(palette)
        
        
    def initUI(self):
        butfont = QtGui.QFont("Noto Sans CJK SC", 18, QtGui.QFont.Bold)
        butfont.setItalic(True)


        self.setGeometry(100,100,1280,720)
    
        self.but_quit = QPushButton('Exit', self)
        self.but_quit.setFont(butfont)
        self.but_quit.setGeometry(1100,600,80,80)
        self.but_quit.setStyleSheet("QPushButton{border:0px;}")
        self.but_quit.clicked.connect(QCoreApplication.instance().quit)
        self.but_quit.setShortcut('esc')
            
        self.word_info = QtWidgets.QLabel('',self)
        font = QtGui.QFont("Source Han Sans SC", 14, QtGui.QFont.Normal) 
        self.word_info.setFont(font)
        self.word_info.setAlignment(Qt.AlignLeft|Qt.AlignTop)
        self.word_info.setWordWrap(True)
        self.word_info.setStyleSheet("background-color: rgba(255, 255, 255, 200);")
        self.word_info.setGeometry(240,100,800,500)
        self.word_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        self.but_know = QPushButton('I Know', self)
        self.but_know.setFont(butfont)
        self.but_know.setGeometry(1050,260,120,80)
        self.but_know.setStyleSheet("QPushButton{border:0px;}")
        self.but_know.clicked.connect(self.know)
        self.but_know.setShortcut('K')
        self.but_know.hide()
        
        self.but_forget = QPushButton('I Forget', self)
        self.but_forget.setFont(butfont)
        self.but_forget.setGeometry(1050,380,120,80)
        self.but_forget.setStyleSheet("QPushButton{border:0px;}")
        self.but_forget.clicked.connect(self.forget)
        self.but_forget.setShortcut('Return')
        self.but_forget.hide()
        
        self.but_meaning = QPushButton('Meaning', self)
        self.but_meaning.setFont(butfont)
        self.but_meaning.setGeometry(1050,380,120,80)
        self.but_meaning.setStyleSheet("QPushButton{border:0px;}")
        self.but_meaning.clicked.connect(self.show_meaning)
        self.but_meaning.setShortcut("Return")
        
        self.but_next_group = QPushButton('Next\nGroup', self)
        self.but_next_group.setFont(butfont)
        self.but_next_group.setGeometry(1050,320,120,80)
        self.but_next_group.setStyleSheet("QPushButton{border:0px;}")
        self.but_next_group.clicked.connect(self.next_group)
        self.but_next_group.hide()
        self.but_next_group.setShortcut("Return")
        
        self.but_play = QPushButton("P", self)
        butfont.setItalic(False)
        self.but_play.setFont(butfont)
        butfont.setItalic(True)
        self.but_play.setGeometry(750,70,30,30)
        self.but_play.setStyleSheet("QPushButton{border:0px;}")
        self.but_play.clicked.connect(self.p)
        self.but_play.setShortcut('P')
        
        self.info = QtWidgets.QLabel('',self) 
        self.info.setFont(butfont)
        self.info.setGeometry(350,70,300,30)
        
        self.lzlzlizi = QtWidgets.QLabel("lzlzlizi words 2.1",self)
        self.lzlzlizi.setGeometry(5,5,150,20)
        lzlzlizifont = QtGui.QFont("Serif", 13, QtGui.QFont.Normal)
        lzlzlizifont.setItalic(True)
        self.lzlzlizi.setFont(lzlzlizifont)
        
        self.summary = []
        self.ng = 1

        try:
            self.current_word = my_list.word_inf[my_list.current_group[0]]
        except:
            self.current_word = None
        self.show_next_word()
        
        info = "Group: " + str(self.ng) + '   ' + str(len(my_list.words)) +' of ' +str(len(my_list.word_inf))+' left'
        self.info.setText(info)
        

    def mousePressEvent(self, event):
        if event.button()==Qt.LeftButton:
            self.m_drag=True
            self.m_DragPosition=event.globalPos()-self.pos()
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))

    def mouseMoveEvent(self, QMouseEvent):

        if Qt.LeftButton and self.m_drag:
            self.move(QMouseEvent.globalPos()-self.m_DragPosition)
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_drag=False
        self.setCursor(QCursor(Qt.ArrowCursor))

    
        
    def p(self):
        play(self.current_word[0])

    def show_next_word(self):
        self.but_know.hide()
        self.but_forget.hide()
        self.but_meaning.show()
        if self.current_word == None:
            self.word_info.setText('<font size = 15>ALL DONE!</font>')
            self.but_play.hide()
            self.but_meaning.hide()
            return
        
        if len(my_list.current_group) == 0:
            self.but_meaning.hide()
            self.but_next_group.show()
            
        if status[self.current_word[0]] ==2:
            info = "Group: " + str(self.ng) + '   ' + str(len(my_list.words)) +' of ' +str(len(my_list.word_inf))+' left'
            self.info.setText(info)
        
        if len(my_list.current_group) > 0:
            w = my_list.word_inf[my_list.current_group[0]]
            self.current_word = copy.copy(w)
            self.word_info.setText('<font size = 20>'+w[0]+'<font>')
        else:
            if len(my_list.group) != 0:
                if self.ng % 3 == 0: my_list.grouping()
                my_list.current_group = my_list.group.pop(0)
                self.ng = self.ng + 1
            else:
                if len(my_list.words) != 0:
                    my_list.grouping()
                    my_list.current_group = my_list.group.pop(0)
                else:
                    print("ALL DONG!")
                    self.word_info.setText('<font size = 15>ALL DONE!</font>')
                    return
            res = make_table(self.summary)
            self.word_info.setText(res)
            self.summary = []    
    
    def next_group(self):
        self.but_next_group.hide()
        self.but_meaning.show()
        self.show_next_word()
        info = "Group: " + str(self.ng) + '   ' + str(len(my_list.words)) +' of ' +str(len(my_list.word_inf))+' left'
        self.info.setText(info)
            
    def show_meaning(self):
        w = my_list.current_group.pop(0)
        w = my_list.word_inf[w]
        
        self.word_info.setText(w[1])
        self.summary.append(w)
        self.but_meaning.hide()
        self.but_know.show()
        self.but_forget.show()         

    def forget(self):
        w = self.current_word
        # log
        date_time = datetime.datetime.now()
        date_time = date_time.strftime("%Y-%m-%d %H:%M:%S")
        sql = """insert ignore into mylog values('%s','%s',%s,0)"""%(w[0],date_time,w[2])
        cursor = db.cursor()
        cursor.execute(sql)
        db.commit()
        ## deal with status
        status[w[0]] = 0
        sql = """update remembering set status = %s  where word = '%s';"""%(status[w[0]],w[0])
        cursor = db.cursor()
        cursor.execute(sql)
        db.commit()
        ## next
        self.show_next_word()
        
    def know(self):
        w = self.current_word
        # log
        date_time = datetime.datetime.now()
        date_time = date_time.strftime("%Y-%m-%d %H:%M:%S")
        sql = """insert ignore into mylog values('%s','%s',%s,1)"""%(w[0],date_time,status[w[0]])
        cursor = db.cursor()
        cursor.execute(sql)
        db.commit()
        ## deal with status
        status[w[0]] = status[w[0]] + 1
        sql = """update remembering set status = %s  where word = '%s';"""%(status[w[0]],w[0])
        cursor = db.cursor()
        cursor.execute(sql)
        db.commit()
        ## discard remembered
        if int(status[w[0]]) == 2:
            my_list.words.remove(w[0])
            sql= """update appdata set level = level + 1, last_date = '%s' where word = '%s';"""%(date_time,w[0])
            cursor = db.cursor()
            cursor.execute(sql)
            db.commit()
        ## next
        self.show_next_word()

if __name__=="__main__":
    
    #from wx.lib.wordwrap import wordwrap
    db = MySQLdb.connect(host = '127.0.0.1',#本地数据库
                        port = 3306,
                                 user = 'root', #用户名
                                 passwd = 'tw123123', #数据库密码
                                 db = 'mydict', #数据库名
                                 charset = 'utf8') 
   

    
    # Remembering info
    status = {}
    sql = "select word,status from remembering; " 
    cursor = db.cursor()
    cursor.execute(sql)
    db.commit()
    res = cursor.fetchall()
    for r in res:
        status[r[0]] = r[1]

    # get new words
    f =open("words.txt")
    my_words_list = f.read()
    f.close()
    my_words_list = re.split(" |\n",my_words_list)
    my_words_list = set(my_words_list)
    if '' in my_words_list:
        my_words_list.discard('')
        
    my_list = words(my_words_list)
    
    app = QtWidgets.QApplication(sys.argv)
    oMainWindow = MainWindow()
    oMainWindow.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.CustomizeWindowHint)
    oMainWindow.show()
    exit(app.exec_())
