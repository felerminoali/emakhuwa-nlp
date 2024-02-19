import os
import pandas as pd
import spacy
import panphon
import panphon.distance
import pickle
import numpy as np
from utils.soudexpt import SoundexBRPT
from utils.spellchecker import SpellChecker

import sys

sys.path.append('../loan_translation')

from inference import get_translation_candidate

soundex = SoundexBRPT()
checker = SpellChecker()
spacy_pt = spacy.load("pt_core_news_sm")
dst = panphon.distance.Distance()

with open('utils/natura-dic.pkl', 'rb') as file:
    myvar = pickle.load(file)

with open('utils/pt_word.pkl', 'rb') as file:
    myvar_pt = pickle.load(file)

def find_best_cadidate(word, candidates, normalize=True):
    best_candidate=None
    MAX=10
    for c in candidates:
        distance_panpon = dst.fast_levenshtein_distance(word, c)
                
        if distance_panpon < MAX:
            MAX = distance_panpon
            best_candidate = c

        if MAX == 0:
            return best_candidate, MAX
    
    if normalize:
        maxlen = max(len(word), len(candidates))
        MAX = MAX / maxlen

    return best_candidate, MAX

def match_in_portuguese_not_stopword(match):
    return 1 if match in myvar_pt else 0

def distance_match_emakhuwa(word, match):
    dst = distance_between_two_words(word, match)
    return dst

def distance_between_match_emakhuwa(e_word, match_, normalize=True):
    distance_panpon = dst.fast_levenshtein_distance(e_word, match_)
    if normalize:
        maxlen = max(len(e_word), len(match_))
        distance_panpon = distance_panpon / maxlen
    
    return str(distance_panpon)

def distance_between_two_words(e_word, match_, normalize=True):
    distance_panpon = dst.fast_levenshtein_distance(e_word, match_)
    if normalize:
        maxlen = max(len(e_word), len(match_))
        maxlen = 1 if maxlen == 0 else maxlen
        distance_panpon = distance_panpon / maxlen
    
    return str(distance_panpon)

def match(word_e, word):
    key = soundex.phonetics(word).lower()
    # Open the file in binary mode

    lema = " ".join([t.lemma_ for t in spacy_pt(word)]).lower()

    best_candidate = None
    MAX = 10

    if key in myvar and lema in myvar[key]:
        best_candidate, MAX = find_best_cadidate(word, myvar[key][lema])
        return word_e, word, best_candidate
    else:
        
        if key in myvar:  
            # correct possible lemma mispeling               
            word_corrections = checker.check(word)
            if word_corrections: 
                word_correction = word_corrections[0][0]
                lema = " ".join([t.lemma_ for t in spacy_pt(word_correction)]).lower()
                if lema in myvar[key]:
                    best_candidate, MAX = find_best_cadidate(word, myvar[key][lema])
                    return word_e, word, best_candidate
                return word_e, word, word_e
        else:
            # correct possible key mispeling 
            word_corrections = checker.check(word)

            if word_corrections: 
                word_correction = word_corrections[0][0]
                key = soundex.phonetics(word_correction).lower()
                lema = " ".join([t.lemma_ for t in spacy_pt(word_correction)]).lower()

                if key in myvar and lema in myvar[key]:
                    best_candidate, MAX = find_best_cadidate(word, myvar[key][lema])
                    return word_e, word, best_candidate
                return word_e, word, word_e
            return word_e, word, word_e
    return word_e, word, word_e


def has_letter_outside_alphabet(word):
    accents = ['â', 'ã', 'ê', 'ô', 'õ', 'ü']
    out_alpha = ['b', 'd', 'g', 'j', 'z', 'q', 'ç'] + accents
    words = list(word)
    for w in words:
        if w in out_alpha:
            return 1
    return 0

def has_letter_from_emakhuwa(word):
    from_alpha = ['fy', 'Kh', 'kw', 'khw', 'lw', 'ly', 'mw', 'my', 'ph', 'pw', 'py', 'phy', 'phw', 'rw', 'ry', 'sy', 'th', 'thw', 'tt', 'tth', 'ttw']
    for v in from_alpha: 
        if word.__contains__(v):
            return 1  
    return 0

def start_with_f(word):
    return 1 if word.startswith('f') else 0

def start_with_ch(word):
    return 1 if word.startswith('ch') else 0   

def has_vowel_seq(word):
    vowel_seq = ['ae', 'ao', 'au', 'ea', 'ei', 'eo', 'eu', 'ia', 'ie', 'io', 'iu', 'oa', 'oe', 'oi', 'ou', 'ua', 'ue', 'ui', 'uo', 'uu']
    for v in vowel_seq: 
        if word.__contains__(v):
            return 1  
    return 0


def has_consonat_seq(word):
    consonat_seq = [ 'bl', 'br', 'bs', 'ch', 'cl', 'cr', 'ct', 'dr', 'dv', 'fl', 'fr', 'gl', 'gr', 'gn', 'pl', 'pr', 'ps', 'pt', 'pn', 'tr', 'tm', 'rt', 'kr', 'kl', 'kn', 'lh', 'vr', 'ss', 'sr', 'st', 'sk', 'hr', 'tl' 'ws']
    for v in consonat_seq: 
        if word.__contains__(v):
            return 1  
    return 0

def consonat_vowels(word):
    consonat_vowels = ['ca', 'co', 'cu', 'qu']
    for v in consonat_vowels: 
        if word.__contains__(v):
            return 1  
    return 0 

def vowel_consonat(word):
    vowel_consonat = ['ef', 'eg', 'ad', 'ab']
    for v in vowel_consonat: 
        if word.__contains__(v):
            return 1  
    return 0

def distance_soundex_(word, other_word):
    soundex_match = soundex.phonetics(other_word)
    soundex_word = soundex.phonetics(word)

    dst = distance_between_two_words(soundex_match,soundex_word)
    return dst

def process_word_translation_match(word):
    translation= get_translation_candidate(word)
    word_match=match(word, translation)[2]

    return word, translation, word_match

def process_input(word):
    word, translation, word_match = process_word_translation_match(word) 
    #translation= get_translation_candidate(word)
    #word_match=match(word, translation)[2]
    
    match_in_portuguese_not_stopword_ = match_in_portuguese_not_stopword(word_match)
    distance_match_emakhuwa_ = distance_match_emakhuwa(word, word_match)
    distance_soundex_match_emakhuwa = distance_soundex_(word, word_match)
    distance_soundex_translation_emakhuwa = distance_soundex_(word, translation)
    has_letter_outside_alphabet_ = has_letter_outside_alphabet(word)
    has_letter_from_emakhuwa_ = has_letter_from_emakhuwa(word)
    vowel_seq = has_vowel_seq(word)
    consonat_seq = has_consonat_seq(word)

    data = [
        word,
        translation,
        word_match,
        match_in_portuguese_not_stopword_,
        distance_match_emakhuwa_,
        distance_soundex_match_emakhuwa,
        distance_soundex_translation_emakhuwa,
        has_letter_outside_alphabet_,
        has_letter_from_emakhuwa_,
        vowel_seq,
        consonat_seq
    ]

    display = [
            "word",
            "translation",
            "match",   
    ]

    phonetic_features = [
            "match_in_portuguese_not_stopword", 
            "distance_match_emakhuwa",
            "distance_soundex_match_emakhuwa", 
            "distance_soundex_translation_emakhuwa",
    ]

    outside_alpha_features = [
            "has_letter_outside_alphabet",
            "has_letter_from_emakhuwa"        
    ]

    adjacency = [  
        "vowel_seq", 
        "consonat_seq", 
    ]

    lexical_features = outside_alpha_features + adjacency
    features =  lexical_features + phonetic_features 
    
    inputs = pd.DataFrame(zip(          
          [float(data[3])],
          [float(data[4])],
          [float(data[5])],
          [float(data[6])],
          [float(data[7])],
          [float(data[8])],
          [float(data[9])],
          [float(data[10])],),  columns=features)

    inputs = inputs[features].values
    x_means = np.array([0.22178157, 0.16259915, 0.14378686, 0.14968477, 0.77404922, 0.43709421, 0.33963823, 0.40926518])
    x_stds = np.array([0.41544495, 0.36899954, 0.35087348, 0.35676216, 0.41820692, 0.3187777, 0.34631366, 0.33349559])
    inputs = (inputs - x_means)/x_stds

    return inputs