
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

import spacy
spacy_pt = spacy.load("pt_core_news_sm")

import sys
sys.path.append('../loan_translation')

from NNClassifier import predict
class LoanwordModel():
    """
    The LoanwordModel defines the interface of interest to clients.
    """

    def __init__(self, TrainStrategy: TrainStrategy) -> None:
        """
        Usually, the LoanwordModel accepts a TrainStrategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._TrainStrategy = TrainStrategy

    @property
    def TrainStrategy(self) -> TrainStrategy:
        """
        The LoanwordModel maintains a reference to one of the TrainStrategy objects. The
        LoanwordModel does not know the concrete class of a TrainStrategy. It should work
        with all strategies via the TrainStrategy interface.
        """

        return self._TrainStrategy

    @TrainStrategy.setter
    def TrainStrategy(self, TrainStrategy: TrainStrategy) -> None:
        """
        Usually, the LoanwordModel allows replacing a TrainStrategy object at runtime.
        """

        self._TrainStrategy = TrainStrategy

    def detect(self, sentence) -> List:
        """
        The LoanwordModel delegates some work to the TrainStrategy object instead of
        implementing multiple versions of the algorithm on its own.
        """

        # ...
        tokens = [token.text.lower()  for token in spacy_pt(sentence) if token.is_alpha]
        result = self._TrainStrategy.find_loanwords(tokens)
        return result

        # ...


class TrainStrategy(ABC):
    """
    The TrainStrategy interface declares operations common to all supported versions
    of some algorithm.

    The LoanwordModel uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    @abstractmethod
    def find_loanwords(self, data: List) -> List:
        pass


"""
Concrete Strategies implement the algorithm while following the base TrainStrategy
interface. The interface makes them interchangeable in the LoanwordModel.
"""


class LexicalBased(TrainStrategy):


    def has_outside_letters(self, word:str) -> List:
        accents = ['â', 'ã', 'ê', 'ô', 'õ', 'ü']
        outside_letters = ['b', 'd', 'g', 'j', 'z', 'q', 'ç'] + accents
        for letter in outside_letters:
            if word.__contains__(letter):
                return True        
        return False 

    def find_loanwords(self, data: List) -> List:
        
        result = []
        for word in data:
            if self.has_outside_letters(word):            
                loanword = (word, word, 100)
                result.append(loanword)
         
        return result



class ClassifierBased(TrainStrategy):

    def has_outside_letters(self, word:str) -> List:
        accents = ['â', 'ã', 'ê', 'ô', 'õ', 'ü']
        outside_letters = ['b', 'd', 'g', 'j', 'z', 'q', 'ç'] + accents
        for letter in outside_letters:
            if word.__contains__(letter):
                return True        
        return False 

    def find_loanwords(self, data: List) -> List:
        
        result = []
        for word in data:
            predition = predict(word)
            if predition:            
                result.append(predition)
        return result
    