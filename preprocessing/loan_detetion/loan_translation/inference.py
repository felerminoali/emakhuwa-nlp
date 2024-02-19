
import sys
import os

from pathlib import Path


path = '../loan_translation/'
sys.path.append('../loan_translation/')


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import random
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard


from mt_utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torchtext.data import BucketIterator

from sacrebleu.metrics import BLEU, CHRF
from torchtext.data import Field, TabularDataset
from tokenizer import Tokenizer, BPETokenizer, CharTokenizer, SentencePiece, WordCharTokenizer, MosesTokenizer
import pandas as pd

import pickle




### We're ready to define everything we need for training our Seq2Seq model ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        # x: (seq_length, N) where N is batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        encoder_states, (hidden, cell) = self.rnn(embedding)
        # outputs shape: (seq_length, N, hidden_size)

        # Use forward, backward cells and hidden through a linear layer
        # so that it can be input to the decoder which is not bidirectional
        # Also using index slicing ([idx:idx+1]) to keep the dimension
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return encoder_states, hidden, cell


class Decoder(nn.Module):
    def __init__(
        self, input_size, embedding_size, hidden_size, output_size, num_layers, p
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers)

        self.energy = nn.Linear(hidden_size * 3, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x, encoder_states, hidden, cell):
        x = x.unsqueeze(0)
        # x: (1, N) where N is the batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        # h_reshaped: (seq_length, N, hidden_size*2)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        # energy: (seq_length, N, 1)

        attention = self.softmax(energy)
        # attention: (seq_length, N, 1)

        # attention: (seq_length, N, 1), snk
        # encoder_states: (seq_length, N, hidden_size*2), snl
        # we want context_vector: (1, N, hidden_size*2), i.e knl
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)

        rnn_input = torch.cat((context_vector, embedding), dim=2)
        # rnn_input: (1, N, hidden_size*2 + embedding_size)

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs).squeeze(0)
        # predictions: (N, hidden_size)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(portuguese.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        encoder_states, hidden, cell = self.encoder(source)

        # First input will be <SOS> token
        x = target[0]

        for t in range(1, target_len):
            # At every time step use encoder_states and update hidden, cell
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)

            # Store prediction for current time step
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to. This was a long comment.
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

bpe_tokenizer = BPETokenizer('test_bpe.codes', 'test_bpe.vocab', num_symbols=100, use_moses=True)
item_list= [('a', 1652), ('o', 1222), ('e@@', 1119), ('c@@', 1022), ('i@@', 994), ('m@@', 981), ('s@@', 919), ('o@@', 913), ('p@@', 881), ('a@@', 869), ('u', 746), ('t@@', 711), ('r@@', 709), ('e', 707), ('u@@', 694), ('i', 693), ('b@@', 649), ('er@@', 636), ('n@@', 610), ('an@@', 608), ('at@@', 607), ('d@@', 596), ('in@@', 564), ('ar@@', 561), ('ri@@', 520), ('l@@', 519), ('or@@', 516), ('f@@', 502), ('et@@', 501), ('h@@', 494), ('es@@', 491), ('al@@', 461), ('it@@', 452), ('ia', 451), ('g@@', 440), ('on@@', 429), ('k@@', 407), ('is@@', 385), ('ik@@', 383), ('ol@@', 377), ('om@@', 376), ('el@@', 375), ('w@@', 346), ('am@@', 343), ('il@@', 339), ('as@@', 326), ('em@@', 324), ('en@@', 303), ('ist@@', 297), ('á@@', 289), ('ep@@', 281), ('ad@@', 281), ('ek@@', 276), ('ap@@', 275), ('ur@@', 273), ('ent@@', 272), ('st@@', 253), ('os@@', 250), ('z@@', 250), ('y@@', 247), ('ul@@', 241), ('í@@', 241), ('x@@', 224), ('un@@', 220), ('é@@', 220), ('v@@', 216), ('j@@', 216), ('ic@@', 213), ('op@@', 203), ('ab@@', 200), ('im@@', 194), ('ir@@', 189), ('est@@', 185), ('as', 184), ('os', 181), ('ip@@', 180), ('iv@@', 171), ('iya', 170), ('id@@', 164), ('ari@@', 161), ('af@@', 157), ('ó@@', 156), ('to', 155), ('ak@@', 155), ('ef@@', 154), ('al', 154), ('ac@@', 148), ('iy@@', 143), ('ok@@', 143), ('oc@@', 140), ('gr@@', 136), ('ed@@', 132), ('iri@@', 132), ('io', 132), ('ot@@', 129), ('ão', 127), ('ro', 118), ('ya', 118), ('ica', 118), ('art@@', 118), ('ro@@', 117), ('ec@@', 117), ('qu@@', 115), ('amp@@', 113), ('ev@@', 111), ('ut@@', 110), ('s', 109), ('ex@@', 107), ('aat@@', 107), ('aw@@', 107), ('us@@', 106), ('pro@@', 104), ('ont@@', 100), ('ov@@', 99), ('ade', 99), ('ag@@', 98), ('yu', 97), ('r', 96), ('au', 95), ('cia', 94), ('es', 93), ('ob@@', 91), ('moçambique', 91), ('rio', 88), ('aal@@', 88), ('l', 87), ('ar', 87), ('ação', 84), ('av@@', 83), ('ant@@', 82), ('m', 76), ('n', 71), ('ú@@', 59), ('â@@', 49), ('amb@@', 49), ('ç@@', 49), ('é', 47), ('ê@@', 44), ('t', 32), ('ò@@', 26), ('d', 26), ('ã@@', 26), ('ção', 26), ('á', 22), ('-@@', 22), ('ô@@', 22), ('y', 20), ('que', 20), ('ã', 18), ('ique', 16), ('mo@@', 15), ('k', 14), ('g', 14), ('õ@@', 13), ('z', 13), ('à@@', 12), ('’', 11), ('è@@', 10), ('h', 10), ('moçamb@@', 9), ('ú', 7), ('x', 6), ('b', 6), ('c', 6), ('í', 5), ('ó', 5), ('v', 4), ('ì@@', 3), ('f', 3), ('q@@', 2), ('moç@@', 2), ('ö@@', 2), ('p', 2), ('ê', 2), ('j', 1), ('w', 1), ('î@@', 1), ('ẽ', 1)]
bpe_tokenizer.vocab = item_list
bpe_tokenizer.set_bpe(path+bpe_tokenizer.codes_file)
bpe_tokenizer.update_word2idx()

def tokenize_bpe_inputs(text):
    
    tokenized, inputs = bpe_tokenizer.tokenize_custom(text)
    return inputs

def text_detokenized(translated_sentence):
  targets = []
  for t in translated_sentence:
    targets.append(bpe_tokenizer.word2idx(t))
  inputs = torch.LongTensor(targets)
  return bpe_tokenizer.detokenize(inputs)

def train_val_test_split(emakhuwa, portuguese):
    train_file = "train.csv"
    valid_file = "dev.csv"
    test_file = "test.csv"

    # Load your data
    train_data, valid_data, test_data = TabularDataset.splits(
        path="../loan_translation/data/csv/",
        train=train_file,
        validation=valid_file,
        test=test_file,
        format="csv",
        fields=[("emakhuwa", emakhuwa), ("portuguese", portuguese)],
    )

    return train_data, valid_data, test_data

from torchtext.vocab import Vocab
from collections import Counter

def get_translation_candidate(word):

    emakhuwa = Field(lower=True, init_token="<sos>", eos_token="<eos>")
    portuguese = Field(lower=True, init_token="<sos>", eos_token="<eos>")
    
    emakhuwa_dict_vocab = {'<unk>': 0, '<pad>': 1, '<sos>': 2, '<eos>': 3, 'a': 4, 'u': 5, 'i': 6, 'a@@': 7, 'e@@': 8, 'o@@': 9, 'i@@': 10, 'o': 11, 'm@@': 12, 'e': 13, 'u@@': 14, 'er@@': 15, 'k@@': 16, 'at@@': 17, 'ik@@': 18, 'ar@@': 19, 'et@@': 20, 'it@@': 21, 'in@@': 22, 's@@': 23, 'an@@': 24, 'w@@': 25, 'p@@': 26, 'or@@': 27, 'es@@': 28, 'h@@': 29, 't@@': 30, 'ek@@': 31, 'om@@': 32, 'on@@': 33, 'al@@': 34, 'is@@': 35, 'el@@': 36, 'y@@': 37, 'ep@@': 38, 'as@@': 39, 'r@@': 40, 'ol@@': 41, 'os@@': 42, 'ri@@': 43, 'il@@': 44, 'x@@': 45, 'ur@@': 46, 'n@@': 47, 'am@@': 48, 'f@@': 49, 'ia': 50, 'em@@': 51, 'ap@@': 52, 'b@@': 53, 'iya': 54, 'en@@': 55, 'ak@@': 56, 'ip@@': 57, 'op@@': 58, 'ist@@': 59, 'iy@@': 60, 'g@@': 61, 'd@@': 62, 'ul@@': 63, 'ent@@': 64, 'l@@': 65, 'ok@@': 66, 'im@@': 67, 'st@@': 68, 'iri@@': 69, 'un@@': 70, 'ya': 71, 'ari@@': 72, 'aat@@': 73, 'af@@': 74, 'z@@': 75, 'aw@@': 76, 'ot@@': 77, 'yu': 78, 'amp@@': 79, 'ab@@': 80, 'ef@@': 81, 'iv@@': 82, 'aal@@': 83, 'ir@@': 84, 'ex@@': 85, 'c@@': 86, 'v@@': 87, 'au': 88, 'est@@': 89, 'ob@@': 90, 'oc@@': 91, 'us@@': 92, 'ev@@': 93, 'ut@@': 94, 'ad@@': 95, 'art@@': 96, 'j@@': 97, 'ed@@': 98, 'id@@': 99, 'ant@@': 100, 'ro@@': 101, 'ic@@': 102, 'ont@@': 103, 'ov@@': 104, 'av@@': 105, 'ac@@': 106, 'to': 107, 'io': 108, 'amb@@': 109, 'á@@': 110, 'ag@@': 111, 'ec@@': 112, 'n': 113, 'gr@@': 114, 'ò@@': 115, 'é@@': 116, 'ica': 117, 'al': 118, 'í@@': 119, 'ó@@': 120, 'ade': 121, 'd': 122, 'ique': 123, 's': 124, 'as': 125, 'qu@@': 126, 'rio': 127, 'es': 128, 't': 129, 'à@@': 130, 'r': 131, '’': 132, 'y': 133, 'cia': 134, 'g': 135, 'k': 136, 'l': 137, 'm': 138, 'ro': 139, 'ç@@': 140, '-@@': 141, 'è@@': 142, 'ê@@': 143, 'h': 144, 'mo@@': 145, 'os': 146, 'pro@@': 147, 'z': 148, 'â@@': 149, 'ú@@': 150, 'v': 151, 'x': 152, 'á': 153, 'é': 154, 'ar': 155, 'b': 156, 'moç@@': 157, 'moçamb@@': 158, 'ã': 159, 'ção': 160, 'ì@@': 161, 'ô@@': 162}
    emakhuwa_dummy_counter = Counter(emakhuwa_dict_vocab)
    emakhuwa_vocab = Vocab(counter=emakhuwa_dummy_counter)
    emakhuwa_vocab.stoi = emakhuwa_dict_vocab
    emakhuwa_vocab.itos = list(emakhuwa_dict_vocab.keys())
    emakhuwa.vocab = emakhuwa_vocab


    portuguese_dict_vocab = {'<unk>': 0, '<pad>': 1, '<sos>': 2, '<eos>': 3, 'c@@': 4, 'a': 5, 'o': 6, 'p@@': 7, 's@@': 8, 'e@@': 9, 'm@@': 10, 'b@@': 11, 'r@@': 12, 'd@@': 13, 'i@@': 14, 'n@@': 15, 't@@': 16, 'l@@': 17, 'o@@': 18, 'g@@': 19, 'f@@': 20, 'ri@@': 21, 'ia': 22, 'e': 23, 'an@@': 24, 'a@@': 25, 'u@@': 26, 'in@@': 27, 'á@@': 28, 'er@@': 29, 'at@@': 30, 'í@@': 31, 'h@@': 32, 'es@@': 33, 'ad@@': 34, 'or@@': 35, 'ar@@': 36, 'al@@': 37, 'é@@': 38, 'os': 39, 'ol@@': 40, 'as': 41, 'ic@@': 42, 'ent@@': 43, 'z@@': 44, 'et@@': 45, 'j@@': 46, 'on@@': 47, 'ist@@': 48, 'em@@': 49, 'al': 50, 'ó@@': 51, 'am@@': 52, 'is@@': 53, 'v@@': 54, 'en@@': 55, 'st@@': 56, 'il@@': 57, 'it@@': 58, 'id@@': 59, 'ão': 60, 'el@@': 61, 'to': 62, 'gr@@': 63, 'ul@@': 64, 'ac@@': 65, 'ab@@': 66, 'qu@@': 67, 'ir@@': 68, 'om@@': 69, 'ro': 70, 'io': 71, 'pro@@': 72, 'est@@': 73, 'moçambique': 74, 'ica': 75, 'r': 76, 'as@@': 77, 'ec@@': 78, 'un@@': 79, 'es': 80, 's': 81, 'ade': 82, 'ap@@': 83, 'ar': 84, 'ação': 85, 'cia': 86, 'rio': 87, 'ur@@': 88, 'iv@@': 89, 'l': 90, 'm': 91, 'ag@@': 92, 'oc@@': 93, 'im@@': 94, 'af@@': 95, 'ed@@': 96, 'u': 97, 'ú@@': 98, 'ro@@': 99, 'art@@': 100, 'i': 101, 'op@@': 102, 'ari@@': 103, 'ov@@': 104, 'ont@@': 105, 'â@@': 106, 'ev@@': 107, 'é': 108, 'ê@@': 109, 'ef@@': 110, 'ant@@': 111, 'av@@': 112, 'os@@': 113, 'us@@': 114, 'n': 115, 'ep@@': 116, 'ut@@': 117, 'ç@@': 118, 'ot@@': 119, 'x@@': 120, 'ip@@': 121, 'ã@@': 122, 'ção': 123, 'amp@@': 124, 'k@@': 125, 'w@@': 126, 'á': 127, 'ex@@': 128, 't': 129, 'ô@@': 130, '-@@': 131, 'que': 132, 'ã': 133, 'd': 134, 'ob@@': 135, 'amb@@': 136, 'õ@@': 137, 'mo@@': 138, 'y@@': 139, 'g': 140, 'moçamb@@': 141, 'y': 142, 'z': 143, 'c': 144, 'h': 145, 'iri@@': 146, 'ó': 147, 'ú': 148, 'b': 149, 'ik@@': 150, 'k': 151, 'au': 152, 'ok@@': 153, 'x': 154, 'í': 155, 'aw@@': 156, 'f': 157, 'è@@': 158}
    portuguese_dummy_counter = Counter(portuguese_dict_vocab)
    portuguese_vocab = Vocab(counter=portuguese_dummy_counter)
    portuguese_vocab.stoi = portuguese_dict_vocab
    portuguese_vocab.itos = list(portuguese_dict_vocab.keys())
    portuguese.vocab = portuguese_vocab

    # Training hyperparameters
    num_epochs = 1000
    #num_epochs = 2
    learning_rate = 3e-4
    batch_size = 32

    # Model hyperparameters
    input_size_encoder = 163
    input_size_decoder = 159
    output_size = 159
    encoder_embedding_size = 300
    decoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 1
    enc_dropout = 0.0
    dec_dropout = 0.0

    encoder_net = Encoder(
        input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
    ).to(device)

    decoder_net = Decoder(
        input_size_decoder,
        decoder_embedding_size,
        hidden_size,
        output_size,
        num_layers,
        dec_dropout,
    ).to(device)

    model = Seq2Seq(encoder_net, decoder_net).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model_name ='best-seq2seq_attention-1000-32'
    model_path = path+'checkpoints/'+model_name+'.pth.tar'
    load_checkpoint(torch.load(model_path, map_location=torch.device(device)), model, optimizer)


    tokenized, loan_tokenized = bpe_tokenizer.tokenize_custom(word)
    translated_sentence = translate_sentence(
            model, 
            loan_tokenized, 
            emakhuwa, 
            portuguese, 
            device, 
            max_length=50
    )
    
    return ''.join(text_detokenized(translated_sentence[:-1]))