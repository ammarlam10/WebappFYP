from flask import render_template, flash, redirect, url_for,request
from app import app
from app.forms import LoginForm
from backend.search import Basic
import json

#####
import sys
import os
import pysolr
import happybase
import pydoop.hdfs as hdfs
#import json
#from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
#machine learning variable start
import cPickle as pickle
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re,string
from pymongo import MongoClient
import pymongo
import pprint


#end
solr = pysolr.Solr('http://localhost:8983/solr/solr',timeout = 10)
print "solr connected"
# connection = happybase.Connection('localhost')
print "connected to hbase"
path = ""
# App config.
DEBUG = True
#app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'




#/home/hishamsajid/Documents/FlaskFront/backend

@app.route('/')
def main():
    return render_template('index2.html')


@app.route('/detect')
def detect():
        #### MACHINE LEARNING FUNCTIONS

    def tokenize(s):
        return re_tok.sub(r' \1 ', s).split()

    def pr(y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

    def get_mdl(y):
            y = y.values
            r = np.log(pr(1,y) / pr(0,y))
            m = LogisticRegression(C=4, dual=True)
            models = {}
            x_nb = x.multiply(r)
            result = m.fit(x_nb, y)
            return result, r,m

####
    # --------------MONGODB ACCESS---------------



    mongo_db = pymongo.MongoClient().conversations

    cursor = mongo_db.conversations.find({"messages.author":"user4"},{"messages":1, "_id":0})
    members = mongo_db.conversations.distinct('members')
    messages = mongo_db.conversations.distinct('messages')
    # print messages

    #--------------MONGODB ACCESS---------------

    train = pd.read_csv('/home/ammar/study/8th semester/FYP/toxic comments/final/train_800.csv')


    users = {}
    tag = {}
    for member in members:
        users[member['user id']] = []
    for message in messages:
        users[message['author']].append(message['body'])
    print(users)



    for key, value in users.iteritems():
        test = pd.DataFrame(value, columns=['comment_text'])
        test['id'] = key



        label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        train['none'] = 1-train[label_cols].max(axis=1)
        train.describe()

        COMMENT = 'comment_text'
        train[COMMENT].fillna("unknown", inplace=True)
        test[COMMENT].fillna("unknown", inplace=True)


        re_tok = re.compile(re.escape(string.punctuation))


        n = train.shape[0]
        vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
                       min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                       smooth_idf=1, sublinear_tf=1 )
        trn_term_doc = vec.fit_transform(train[COMMENT])
        # with open('/home/ammar/study/8th semester/FYP/toxic comments/final/800/vec','wb') as fp:
        #     pickle.dump(vec,fp)

        test_term_doc = vec.transform(test[COMMENT])

        
        x = trn_term_doc
        test_x = test_term_doc

        
        get_mdl.counter = 0

        preds = np.zeros((len(test), len(label_cols)))
        # models = {}
        # print(test_x)

        print label_cols

        for i, j in enumerate(label_cols):
            # print('fit', j)
            m,r,model = get_mdl(train[j])
            # print r
            print "running model on user " + key
            with open('/home/ammar/study/8th semester/FYP/toxic comments/final/800/model'+str(i),'wb') as fp:
                pickle.dump(m,fp)
            with open('/home/ammar/study/8th semester/FYP/toxic comments/final/800/r'+str(i),'wb') as fp:
                pickle.dump(r,fp)
            preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]

        submid = pd.DataFrame({'id': test["id"]})
        submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
        users = submission['id'].unique()

        for user in users:
            tag[user] = False

        for user in users:
            for index, row in submission.iterrows():
                if row['id'] == user:
                    if row['toxic'] >= 0.6 or row['severe_toxic'] >= 0.6 or row['threat'] >= 0.6:
                        tag[user] = True



    return render_template('detected.html', type=tag)




@app.route('/inputml')
def inputml():

    def tokenize(s):
        return re_tok.sub(r' \1 ', s).split()

    def pr(y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

    def get_mdl(y):
            y = y.values
            r = np.log(pr(1,y) / pr(0,y))
            m = LogisticRegression(C=4, dual=True)
            models = {}
            x_nb = x.multiply(r)
            result = m.fit(x_nb, y)
            return result, r,m

    train = pd.read_csv('/home/ammar/study/8th semester/FYP/toxic comments/final/train_800.csv')

    comment = request.args.get('search')
    cl = []
    cl.append(comment)
    test = pd.DataFrame(cl, columns=['comment_text'])
    test['id'] = 1


    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train['none'] = 1-train[label_cols].max(axis=1)
    train.describe()

    COMMENT = 'comment_text'
    train[COMMENT].fillna("unknown", inplace=True)
    test[COMMENT].fillna("unknown", inplace=True)


    re_tok = re.compile(re.escape(string.punctuation))


    n = train.shape[0]
    vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
                   min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                   smooth_idf=1, sublinear_tf=1 )
    trn_term_doc = vec.fit_transform(train[COMMENT])
    # with open('/home/ammar/study/8th semester/FYP/toxic comments/final/800/vec','wb') as fp:
    #     pickle.dump(vec,fp)

    test_term_doc = vec.transform(test[COMMENT])

    
    x = trn_term_doc
    test_x = test_term_doc

    
    get_mdl.counter = 0

    preds = np.zeros((len(test), len(label_cols)))


    for i, j in enumerate(label_cols):
        # print('fit', j)
        m,r,model = get_mdl(train[j])
        # print r
        print "running model on user "
        with open('/home/ammar/study/8th semester/FYP/toxic comments/final/800/model'+str(i),'wb') as fp:
            pickle.dump(m,fp)
        with open('/home/ammar/study/8th semester/FYP/toxic comments/final/800/r'+str(i),'wb') as fp:
            pickle.dump(r,fp)
        preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]

    submid = pd.DataFrame({'id': test["id"]})
    submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
    return render_template('inputml.html', title='search',type=submission)



@app.route('/index')
def index():
    user = {'username': 'Miguel'}
    posts = [
        {
            'author': {'username': 'John'},
            'body': 'Beautiful day in Portland!'
        },
        {
            'author': {'username': 'Susan'},
            'body': 'The Avengers movie was so cool!'
        }
    ]
    return render_template('index2.html', title='Home', user=user, posts=posts)

@app.route('/search')
def search():
	return render_template('search.html', title='search')

@app.route('/threat')
def threat():
    return render_template('threat.html', title='Threat')


@app.route('/login', methods =['GET','POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():  

        return redirect(url_for('search'))
        
    return render_template('login.html', title = 'Sign In', form=form)    

@app.route("/start", methods=['GET', 'POST'])
def start():
    #form = LoginForm()
    #solr = pysolr.Solr('http://localhost:8983/solr/', timeout=10)
    

    reload(sys)
    #so that we don't get unicode encode error
    sys.setdefaultencoding('utf-8')
    #path = "C:\\Users\\HP\\Desktop\\test_data\\"
    global path
    path = request.args.get('search')
    result = solr.search(path)
    save_path = '/home/ammar/test/'
    type_dict = {'filename':[],'user':[],'location':[]}
    for r in result:
        print r['title'][0]
        type_dict['filename'].append(r['title'][0])
        # table = connection.table('details')
        # row = table.row(r['title'][0])
        # print row['cf:userID']
        # type_dict['user'].append(row['cf:userID'])
        # print row['cf:path']
        # type_dict['location'].append(row['cd:path'])
        # hdfs.get(row['cf:path'],'/home/ammar/test')

    return render_template('form2.html',type=type_dict)

