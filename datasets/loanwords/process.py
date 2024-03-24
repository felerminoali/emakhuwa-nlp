import spacy
import panphon
import panphon.distance
import pickle

from utils.soudexpt import SoundexBRPT
from spellchecker import SpellChecker
from utils.vmwloans import loanword

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

def match_(word_e, word):
    word, translation, m = match(word_e, word)
    return m


def get_data(filename_src, filename_tgt, label):
    
    lines_src = open(filename_src, 'r', encoding='utf-8').readlines()
    lines_src = [line.strip().replace('\n', '') for line in lines_src]

    lines_tgt = open(filename_tgt, 'r', encoding='utf-8').readlines()
    lines_tgt = [line.strip().replace('\n', '') for line in lines_tgt]

    labels = [label for i in range(len(lines_tgt))]

    df = pd.DataFrame(zip(lines_src, lines_tgt, labels), columns=['word', 'translation', 'label'])
    
    return df

def match_equal_to_emakhuwa(word, match):
    return 1 if word == match else 0

def match_in_portuguese_not_stopword(match):
    return 1 if match in myvar_pt else 0

def distance_match_emakhuwa(word, match):
    dst = distance_between_two_words(word, match)
    return dst

def distance_soundex_match_emakhuwa(word, translation, match):
    soundex_match = soundex.phonetics(match)
    soundex_word = soundex.phonetics(word)

    dst = distance_between_two_words(soundex_match,soundex_word)
    return dst

def distance_soundex_translation_emakhuwa(word, translation, match):
    soundex_word = soundex.phonetics(word)
    soundex_translation = soundex.phonetics(translation)

    dst = distance_between_two_words(soundex_translation, soundex_word)
    return dst



def has_letter_outside_alphabet(word):
    accesent  = ['â', 'ã', 'à', 'á', 'é', 'ê', 'í',  'î', 'ô', 'õ', 'ó',  'ú',  'ü']
    out_alpha = ['b', 'd', 'g', 'j', 'z', 'q', 'ç'] + accesent
    words = list(word)
    for w in words:
        if w in out_alpha:
            return 1
    return 0

def has_letter_from_emakhuwa(word):
    from_alpha = ['fy', 'kh', 'kw', 'khw', 'lw', 'ly', 'mw', 'my', 'ph', 'pw', 'py', 'phy', 'phw', 'rw', 'ry', 'sy', 'th', 'thw', 'tt', 'tth', 'ttw', 'è', 'ì', 'ò', 'ù']
    for v in from_alpha: 
        if word.__contains__(v):
            return 1  
    return 0

def start_with_f(word):
    return 1 if word.startswith('f') else 0

def start_with_ch(word):
    return 1 if word.startswith('ch') else 0   

def vowel_seq(word):
    vowel_seq = ['ae', 'ao', 'au', 'ea', 'ei', 'eo', 'eu', 'ia', 'ie', 'io', 'iu', 'oa', 'oe', 'oi', 'ou', 'ua', 'ue', 'ui', 'uo', 'uu']
    for v in vowel_seq: 
        if word.__contains__(v):
            return 1  
    return 0

# adjanct vowels
def consonat_seq(word):
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



def has_letter_outside_alphabet(word):
    accesent  = ['â', 'ã', 'à', 'á', 'é', 'ê', 'í',  'î', 'ô', 'õ', 'ó',  'ú',  'ü']
    out_alpha = ['b', 'd', 'g', 'j', 'z', 'q', 'ç'] + accesent
    words = list(word)
    for w in words:
        if w in out_alpha:
            return 1
    return 0

def has_letter_from_emakhuwa(word):
    from_alpha = ['fy', 'kh', 'kw', 'khw', 'lw', 'ly', 'mw', 'my', 'ph', 'pw', 'py', 'phy', 'phw', 'rw', 'ry', 'sy', 'th', 'thw', 'tt', 'tth', 'ttw', 'è', 'ì', 'ò', 'ù']
    for v in from_alpha: 
        if word.__contains__(v):
            return 1  
    return 0

def start_with_f(word):
    return 1 if word.startswith('f') else 0

def start_with_ch(word):
    return 1 if word.startswith('ch') else 0   

def vowel_seq(word):
    vowel_seq = ['ae', 'ao', 'au', 'ea', 'ei', 'eo', 'eu', 'ia', 'ie', 'io', 'iu', 'oa', 'oe', 'oi', 'ou', 'ua', 'ue', 'ui', 'uo', 'uu']
    for v in vowel_seq: 
        if word.__contains__(v):
            return 1  
    return 0

# adjanct vowels
def consonat_seq(word):
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