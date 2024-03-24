import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd 
import numpy as np 
import os
import io
import requests
import csv
import json
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import pprint
import matplotlib.pyplot as plt
import random

from process import *

def classification_report_csv(report):
    report_data = []
    lines = report.split('\n')

    for i, line in enumerate(lines[2:-3]):

        if i ==0 or i==1:
            row = {}

            row_data = line.split(' ') 
            row_data = list(filter(None, row_data))
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])

        if i==3:
            row_data = line.split(' ') 
            row_data = list(filter(None, row_data))
            row['acc'] = float(row_data[1])

        report_data.append(row)
    print("final report", report_data[0:2])
    dataframe = pd.DataFrame.from_dict(report_data[0:2])
    return dataframe

def remove_extras(df):
    extras = df[df['reference'].isnull()]
    extras = extras[extras['label'] == 1]
    df = df.drop(extras.index)
    return df
  
train_alldata = {}
test_alldata = {}


def contain_prefix(word):
    # common emakhuwa
    prefixes = ['ya', 'ni', 'kh', 'va', 'wo', 'sin', 'oh', 'mw', 'yaa', 'ana', 'ah', 'ani', 'ki', 'oni', 'ahi', 'soo', 'mwa', 'oo', 'vo', 'aa', 'waa', 'omu', 'nam', 'woo', 'kha', 'yoo', 'ott', 'oth', 'mut', 'anam', 'ow', 'ohi', 'okh', 'nin', 'kin', 'khi', 'muk', 'sini', 'yah', 'mm', 'nn', 'nama', 'yan', 'ov', 'yaah', 'wii', 'van', 'yaahi', 'ann', 'otth', 'mwaa', 'mun', 'yi', 'yahi', 'oph', 'eh', 'mak', 'eni', 'mur', 'voo', 'noo', 'api', 'onn', 'nik', 'anni', 'wan', 'ih', 'ne', 'nii', 'nt', 'aah', 'sot', 'ett', 'naa', 'mat', 'anama', 'vak', 'akh', 'otha', 'amu', 'mir', 'san', 'oko', 'mit', 'to', 'kini', 'mwi', 'mul', 'apin', 'apina', 'mutt', 'yar', 'onni', 'yat', 'vat', 'nikh']
    for prefix in prefixes:
        if word.startswith(prefix):
            return 1
    return 0

def contain_suffix(word):
    #suffixes = ['wa', 'iwa', 'ela', 'ha', 'aka', 'we', 'rya', 'erya', 'ana', 'liwa', 'eya', 'eliwa', 'iha', 'ani', 'riwa', 'ene', 'ala', 'lo', 'xa', 'iwe', 'herya', 'awe', 'aya', 'ye', 'lela', 'elo', 'hiwa', 'eriwa', 'asa', 'ke', 'pa', 'yo', 'waka']
    suffixes = ['ela', 'aka', 'iwa', 'rya', 'erya', 'awe', 'iha', 'ala', 'aya', 'liwa', 'ene', 'lela', 'ke', 'ye', 'eke', 'eliwa', 'nya', 'hala', 'xa', 'laka', 'herya', 'ele', 'aa', 'hu', 'elo', 'he', 'riwa', 'iwe', 'tta', 'nle', 'aca', 'khala', 'asa', 'ula', 'eene', 'yawe', 'rela', 'va', 'wela', 'iherya', 'hiwa', 'elela', 'yaka', 'ho', 'waka', 'elaka', 'tha', 'ira', 'hela', 'hiya', 'eela', 'aha', 'eriwa', 'uma', 'wo', 'uwa', 'owa', 'aawe', 'uwela', 'lana', 'nyu', 'aru', 'kela', 'ihiwa', 'haka', 'anya', 'neya', 'liha', 'epa', 'eni', 'yaa', 'exa', 'ere', 'iye', 'thi', 'ile', 'ona', 'mela', 'enle', 'aana', 'kha', 'hani', 'iwaka', 'yaya', 'ini', 'aneya', 'tho', 'aaya', 'tte', 'rye', 'thu', 'leya', 'ryo', 'una', 'mwa', 'riwe', 'niwa']
    for suffix in suffixes:
        if word.startswith(suffix):
            return 1
    return 0

long_vowels = ['aa', 'ee', 'ii', 'oo',  'uu']
emakhuwa_only_alpha = ['fy', 'kh', 'kw', 'khw', 'lw', 'ly', 'mw', 'my', 'ny', 'ng', 'ph', 'pw', 'py', 'phy', 'phw', 'rw', 'ry', 'sy', 'th', 'thw', 'tt', 'tth', 'ttw']
emakhuwa_only_accents = ['è', 'ì', 'ò', 'ù']

def has_letter_outside_alphabet(word):
    accents = ['à', 'á', 'è', 'é', 'ì', 'í', 'ò', 'ó', 'ù', 'ú']
    alpha = ['a', 'c', 'e',  'f', 'h', 'i',  'k', 'kh', 'l', 'm', 'n', 'ny', 'ng', 'o',  'p', 'ph', 'r', 's', 't', 'th', 'tt', 'u', 'v', 'w', 'x', 'y'] + long_vowels + emakhuwa_only_alpha + accents

    words = list(word)
    for w in words:
        if w not in alpha:
            return 1
    return 0

def has_letter_from_emakhuwa(word):
    
    alpha =  emakhuwa_only_alpha + emakhuwa_only_accents + long_vowels
    for v in emakhuwa_only_alpha: 
        if word.__contains__(v):
            return 1  
    return 0

def has_vowel_seq(word):
    vowel_seq = ['ae', 'ao', 'au', 'ea', 'ei', 'eo', 'eu', 'ia', 'ie', 'io', 'iu', 'oa', 'oe', 'oi', 'ou', 'ua', 'ue', 'ui', 'uo']
    for v in vowel_seq: 
        if word.__contains__(v):
            return 1  
    return 0

def has_consonat_seq(word):

    # Define the alphabet
    alphabet = list('abcdefghijklmnopqrstuvwxyz')

    # Define a string of vowels
    vowels = list('aeiou')
    exceptions = emakhuwa_only_alpha + emakhuwa_only_accents + long_vowels + vowels

    consonat_seq = []
    # Iterate through the alphabet
    for left_letter in alphabet:
        for right_letter in alphabet:
            # Check if both current and next letters are consonants
            if left_letter not in exceptions and right_letter not in exceptions and left_letter+right_letter not in exceptions:
                consonat_seq.append(left_letter+right_letter)

    for v in consonat_seq: 
        if word.__contains__(v):
            return 1  
    return 0

def save_predition(test_set, y_test, y_pred, classifier, report, repot_txt):

    data = {
        'word': [test_set["word"].values[i] for i in range(len(y_test))],
        'tag': [test_set["tag"].values[i] for i in range(len(y_test))],
        'label': y_test,
        'prediction': y_pred
    }
    
    df = pd.DataFrame(data)
    df.to_csv('../../predictions/prediction-'+classifier+'.csv', index=False)
    class_report = classification_report_csv(report)
    with open('../../predictions/reports/'+f'{classifier}-report.txt', 'w', encoding='utf-8') as file:
        file.write(repot_txt)
        file.close()
    class_report.to_csv(('../../predictions/reports/'+f'{classifier}-report.csv'), index=False)


augment_pt_dic = False
aug_tag = '-aug' if augment_pt_dic else ''
data_train = pd.read_csv('data/ML/train'+aug_tag+'.csv')
data_test = pd.read_csv('data/ML/test'+aug_tag+'.csv')

data_train['match_'] = data_train.apply(lambda x: '' if x['word'] == x['match_'] else x['match_'], axis=1)
data_train['prefix'] = data_train.apply(lambda x: contain_prefix(x['word']), axis=1)
data_train['suffix'] = data_train.apply(lambda x: contain_suffix(x['word']), axis=1)
data_train['vowel_seq'] = data_train.apply(lambda x: vowel_seq(x['word']), axis=1)
data_train['consonat_seq'] = data_train.apply(lambda x: consonat_seq(x['word']), axis=1)
data_train['has_letter_outside_alphabet'] = data_train.apply(lambda x: has_letter_outside_alphabet(x['word']), axis=1)
data_train['has_letter_from_emakhuwa'] = data_train.apply(lambda x: has_letter_from_emakhuwa(x['word']), axis=1)

data_train.to_csv('train'+aug_tag+'.csv')

#missing_values = data_train.isna().sum()
#print(missing_values)

data_test['match_'] = data_test.apply(lambda x: '' if x['word'] == x['match_'] else x['match_'], axis=1)
data_test['prefix'] = data_test.apply(lambda x: contain_prefix(x['word']), axis=1)
data_test['suffix'] = data_test.apply(lambda x: contain_suffix(x['word']), axis=1)
data_test['vowel_seq'] = data_test.apply(lambda x: has_vowel_seq(x['word']), axis=1)
data_test['consonat_seq'] = data_test.apply(lambda x: has_consonat_seq(x['word']), axis=1)
data_test['has_letter_outside_alphabet'] = data_test.apply(lambda x: has_letter_outside_alphabet(x['word']), axis=1)
data_test['has_letter_from_emakhuwa'] = data_test.apply(lambda x: has_letter_from_emakhuwa(x['word']), axis=1)

data_test.to_csv('test'+aug_tag+'.csv')

train_alldata = data_train
test_alldata = data_test
    
display = [
        "word",
        "translation",
        "match_", 
        "tag"  
]

similarity_features = [
        "distance_soundex_match_emakhuwa_", 
        "distance_soundex_translation_emakhuwa_",
]

outside_alpha_features = [
        "has_letter_outside_alphabet",
        "has_letter_from_emakhuwa"        
]

adjacency = [
    "vowel_seq", 
    "consonat_seq", 
] 


affix_features = [
        'prefix',
        'suffix',   
]





language_specific = outside_alpha_features + adjacency + affix_features
split = 'LS-all'+ aug_tag

features =  language_specific 
features =   similarity_features 

features =  language_specific + similarity_features


labels = ['label']


