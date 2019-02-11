#!/usr/bin/env python3
import os
import re
import random
from collections import Counter

import torch
import torch.nn as nn
from torchtext.data import Field, TabularDataset
from torchtext.vocab import Vocab

class WordToPhonemeModel:
    UNK_TOKEN = '<unk>'
    PAD_TOKEN = '<pad>'
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'

    def __init__(self, model_dir, **load_kwargs):
        self.load_model(model_dir, **load_kwargs)

    @classmethod
    def tokenize_word(cls, word):
        return list(word)

    @classmethod
    def tokenize_phonemes(cls, text):
        return re.split(r'\s+', text)

    @classmethod
    def load_vocab(cls, freqs_path):
        counter = Counter()
        with open(freqs_path, 'r') as freqs_file:
            for line in freqs_file:
                name, freq = re.split(r'\s+', line.strip(), maxsplit=1)
                counter[name] = int(freq)

        return Vocab(counter, specials=[WordToPhonemeModel.UNK_TOKEN,
                                        WordToPhonemeModel.PAD_TOKEN,
                                        WordToPhonemeModel.SOS_TOKEN,
                                        WordToPhonemeModel.EOS_TOKEN])

    def load_dataset(self, csv_path):
        return TabularDataset(path=csv_path, format='csv',
                              fields=[('src', self.src_field), ('trg', self.trg_field)])

    def load_model(self, model_dir, lower=True, device='cpu'):
        self.src_field = Field(tokenize=WordToPhonemeModel.tokenize_word,
                               init_token=WordToPhonemeModel.SOS_TOKEN,
                               eos_token=WordToPhonemeModel.EOS_TOKEN,
                               lower=lower)

        self.src_field.vocab = WordToPhonemeModel.load_vocab(
            os.path.join(model_dir, 'src-freqs.txt'))

        self.trg_field = Field(tokenize=WordToPhonemeModel.tokenize_phonemes,
                               init_token=WordToPhonemeModel.SOS_TOKEN,
                               eos_token=WordToPhonemeModel.EOS_TOKEN,
                               lower=lower)

        self.trg_field.vocab = WordToPhonemeModel.load_vocab(
            os.path.join(model_dir, 'trg-freqs.txt'))

        input_dim = len(self.src_field.vocab)
        output_dim = len(self.trg_field.vocab)
        enc_emb_dim = 256
        dec_emb_dim = 256
        hid_dim = 512
        enc_dropout = 0.5
        dec_dropout = 0.5

        self.sos_idx = self.trg_field.vocab.stoi[WordToPhonemeModel.SOS_TOKEN]
        self.eos_idx = self.trg_field.vocab.stoi[WordToPhonemeModel.EOS_TOKEN]
        self.pad_idx = self.src_field.vocab.stoi[WordToPhonemeModel.PAD_TOKEN]

        enc = Encoder(input_dim, enc_emb_dim, hid_dim, enc_dropout)
        dec = Decoder(output_dim, dec_emb_dim, hid_dim, dec_dropout)

        self.model = Seq2Seq(enc, dec, self.pad_idx, self.sos_idx, self.eos_idx, device)

        state_path = os.path.join(model_dir, 'g2p-model.pt')
        if os.path.exists(state_path):
            self.model.load_state_dict(torch.load(state_path))

    def word2phonemes(self, word, lower=True, device='cpu'):
        if lower:
            word = word.lower()

        tokenized = WordToPhonemeModel.tokenize_word(word)
        tokenized = [WordToPhonemeModel.SOS_TOKEN] + tokenized + [WordToPhonemeModel.EOS_TOKEN]
        numericalized = [self.src_field.vocab.stoi[t] for t in tokenized]

        src = torch.LongTensor(numericalized).unsqueeze(1).to(device)
        self.model.eval()
        output = self.model(src, None, teacher_forcing_ratio=0)[1:]

        predicted = torch.argmax(output.squeeze(1), 1)
        tokenized = [self.trg_field.vocab.itos[int(i)] for i in predicted
                    if i != self.eos_idx]

        return tokenized

# -----------------------------------------------------------------------------
# Built using the tutorial from http://github.com/bentrevett/pytorch-seq2seq
#
# Specifically, the "Learning Phrase Representations unsing RNN Encoder-Deocer
# for Statistical Machine Translation".
# -----------------------------------------------------------------------------

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, sos_idx, eos_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):

        #src = [sent len, batch size]
        #trg = [sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        if trg is None:
            inference = True
            assert teacher_forcing_ratio == 0, "Must be zero during inference"
            trg = torch.zeros_like(src, dtype=torch.long).fill_(self.sos_idx).to(self.device)
        else:
            inference = False

        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        #last hidden state of the encoder is the context
        context = self.encoder(src)

        #context also used as the initial hidden state of the decoder
        hidden = context

        #first input to the decoder is the <sos> tokens
        input = trg[0,:]

        for t in range(1, max_len):

            output, hidden = self.decoder(input, hidden, context)
            outputs[t] = output
            teacher_force = (random.random() < teacher_forcing_ratio) if teacher_forcing_ratio > 0 else False
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)

        return outputs

# -----------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim) #no dropout as only one layer!

        self.rnn = nn.GRU(emb_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        #src = [sent len, batch size]

        embedded = self.dropout(self.embedding(src))

        #embedded = [sent len, batch size, emb dim]

        outputs, hidden = self.rnn(embedded) #no cell state!

        #outputs = [sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]

        #outputs are always from the top hidden layer

        return hidden

# -----------------------------------------------------------------------------

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)

        self.out = nn.Linear(emb_dim + hid_dim*2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):

        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #context = [n layers * n directions, batch size, hid dim]

        #n layers and n directions in the decoder will both always be 1, therefore:
        #hidden = [1, batch size, hid dim]
        #context = [1, batch size, hid dim]

        input = input.unsqueeze(0)

        #input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        #embedded = [1, batch size, emb dim]

        emb_con = torch.cat((embedded, context), dim=2)

        #emb_con = [1, bsz, emb dim + hid dim]

        output, hidden = self.rnn(emb_con, hidden)

        #output = [sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]

        #sent len, n layers and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [1, batch size, hid dim]

        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim=1)

        #output = [batch size, emb dim + hid dim * 2]

        prediction = self.out(output)

        #prediction = [batch size, output dim]

        return prediction, hidden
