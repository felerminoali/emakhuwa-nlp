import re
import string
from collections import Counter
import numpy as np

import panphon.distance

import sys
#sys.path.append('../../')
from utils.soudexpt import SoundexBRPT
soundex = SoundexBRPT()


dst = panphon.distance.Distance()
  
class SpellChecker(object):

  def __init__(self, corpus_file_path="utils/wordsList"):
    with open(corpus_file_path, "r", encoding = "ISO-8859-1") as file:
      lines = file.readlines()
      words = []
      letters = []
      for line in lines:
        words.append(line.strip().replace('\n', '').lower())
        letters += line.strip().replace('\n', '').lower()

    self.vocabs = set(words)
    self.letters_vocabs = list(set(letters))
    self.word_counts = Counter(words)
    total_words = float(sum(self.word_counts.values()))
    self.word_probas = {word: self.word_counts[word] / total_words for word in self.vocabs}

  def _level_one_edits(self, word):
    letters =''.join(list(set([l for l in string.ascii_lowercase] + self.letters_vocabs)))
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [l + r[1:] for l,r in splits if r]
    swaps = [l + r[1] + r[0] + r[2:] for l, r in splits if len(r)>1]
    replaces = [l + c + r[1:] for l, r in splits if r for c in letters]
    inserts = [l + c + r for l, r in splits for c in letters] 

    return set(deletes + swaps + replaces + inserts)

  def _level_two_edits(self, word):
    return set(e2 for e1 in self._level_one_edits(word) for e2 in self._level_one_edits(e1))

  def distance_(self, w1, w2, normalize=True):
      distance_panpon = dst.fast_levenshtein_distance(w1, w2)
      if normalize:
          maxlen = max(len(w1), len(w2))
          maxlen = 1 if maxlen == 0 else maxlen
          distance_panpon = distance_panpon / maxlen
      
      return distance_panpon

  def distance_soundex_(self, word, candidate):
      soundex_candidate = soundex.phonetics(candidate)
      soundex_word = soundex.phonetics(word)
      dst = self.distance_(soundex_candidate,soundex_word)
      return dst

  def check(self, word):
    candidates = self._level_one_edits(word) or self._level_two_edits(word) or [word]
    valid_candidates = [w for w in candidates if w in self.vocabs]
    return sorted([(c, self.word_probas[c], self.distance_soundex_(word,c)) for c in valid_candidates], key=lambda tup: tup[1]*tup[2], reverse=False)

checker = SpellChecker()

#print(checker.check('coner'))