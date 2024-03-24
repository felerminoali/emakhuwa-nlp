import re
import string
from collections import Counter
import numpy as np

import sys
#sys.path.append('../data/utils/')

class SpellChecker(object):

  def __init__(self, corpus_file_path="wordsList"):
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

  def check(self, word):
    candidates = self._level_one_edits(word) or self._level_two_edits(word) or [word]
    valid_candidates = [w for w in candidates if w in self.vocabs]
    return sorted([(c, self.word_probas[c]) for c in valid_candidates], key=lambda tup: tup[1], reverse=True)

checker = SpellChecker()

#print(checker.check('coner'))