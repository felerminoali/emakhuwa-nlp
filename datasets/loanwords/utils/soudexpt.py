from pyphonetics.phonetics import PhoneticAlgorithm
from pyphonetics.utils import translation, squeeze, check_str, check_empty

import re
from unidecode import unidecode


class SoundexBRPT(PhoneticAlgorithm):
    def __init__(self):
        super().__init__()

        self.translations_first = {
            'Y': 'I',
            'BR': 'B',
            'PH': 'F',
            'GR': 'G',
            'MG': 'G',
            'NG': 'G',
            'RG': 'G',
            'GE': 'J',
            'GI': 'J',
            'RJ': 'J',
            'MJ': 'J',
            'NJ': 'J',
            'Q': 'K',
            'CA': 'K',
            'CO': 'K',
            'CU': 'K',
            'C': 'K',
            'LH': 'L',
            'N': 'M',
            'RM': 'M',
            'GM': 'M',
            'MD': 'M',
            'SM': 'M',
            'AO': 'M',
            'NH': 'N',
            'PR': 'P',
            'Ç': 'S',
            'X': 'S',
            'TS': 'S',
            'C': 'S',
            'Z': 'S',
            'RS': 'S',
            'LT': 'T',
            'TR': 'T',
            'CT': 'T',
            'ST': 'T',
            'W': 'V'
        }

        self.end_remove = ['S', 'Z', 'R', 'R', 'M', 'N', 'L']
        
        self.vowels = ['A', 'E', 'I', 'O', 'U', 'H']

    def check_ao_end(self, word):
      for match in re.finditer('AO', word):
        return (match.end() == len(word))
      return False
    
    def remove_termination(self, word):
      # 16.	Eliminar as terminações S, Z, R, R, M, N, AO e L 
        splits = word.split(' ')
        modification = []
        for token in splits:
          if(token[-1] in self.end_remove or self.check_ao_end(token)):
            if(self.check_ao_end(token)):
              token = token.replace('AO', '')
            else:
              token = token[:-1]
          modification.append(token)
        word = " ".join(modification)
        return word

    def replacer(self, s, newstring, index, nofail=False):
      # raise an error if index is outside of the string
      if not nofail and index not in range(len(s)):
          raise ValueError("index outside given string")

      # if not erroring, but the index is still not in the correct range..
      if index < 0:  # add it to the beginning
          return newstring + s
      if index > len(s):  # add it to the end
          return s + newstring

      # insert the new string between "slices" of the original
      return s[:index] + newstring + s[index + 1:]

    def phonetics(self, word):
        check_str(word)
        check_empty(word)


        # posicoes Ç aparece
        special = [i for i, ltr in enumerate(word.upper()) if ltr == 'Ç']
       
        # 1.	Converter todas as letras para Maiúsculo;
        word = unidecode(word).upper()
        for index in special:
          word = self.replacer(word, 'Ç', index)
        
        # Substituicoes 
        for key, value in self.translations_first.items():
          # print(key)
          if(key == 'AO'):
            splits = word.split(' ')
            modification = []
            for token in splits:
              for match in re.finditer(key, token):
                if (match.end() == len(token)):
                  token = token.replace(key, value)
              modification.append(token)
            word = " ".join(modification)
          else:
            word = re.sub(key, value, word)
         
        # print(word)
        
        word = self.remove_termination(word)

        # 17.	Substituir R por L;
        word = re.sub('L', 'R', word)
        # word = re.sub('R', 'L', word)
        # print(word)

        # 18.	Eliminar todas as vogais e o H;
        for vowel in self.vowels:
          word = re.sub(vowel, '', word)
        # print(word)
        
        # 19.	Eliminar todas as letras em duplicidade;
        word = re.sub(r'(.)\1+', r'\1', word)

        # word = self.remove_termination(word)

        return word
        # print(word)