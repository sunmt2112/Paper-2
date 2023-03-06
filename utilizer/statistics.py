from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import re
import numpy as np

def preprocess_sentence(w):
    # w = 'Tüm ofis çalışanlarının neredeyse üçte biri gözlük takar.' # tur
    # w = 'Denne æblejuice er 100 % ren.' # dan
    # w = 'Dahan-dahan magsalita, nga!' # tgl
    # w = 'Tomun heç vaxt sürücülük vəsiqəsi olmayıb.' # aze
    # w = '너 왜 그렇게 미친 사람처럼 행동해?' # kor
    # w = '"Terima kasih." "Sama-sama."' # ind
    # w = 'Tom så ikke trøtt ut, spør du meg.' # nor
    # w = "We didn't ask any questions." # eng

    w = w.lower().strip()
    w = re.sub(r"([?.!,])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[-#$%&'()*+/:;<=>@\[\\\]^_`{|}~]+", " ", w)
    w = w.strip()
    w = '<start> ' + w + ' <end>'

    return w


def create_dataset(path, num_examples):
    lines = open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

    return zip(*word_pairs)


def max_length(tensor):
    return max([len(t) for t in tensor])
def mode(tensor):
    return np.argmax(np.bincount([len(t) for t in tensor]))
def avg(tensor):
    return np.mean([len(t) for t in tensor])
def median(tensor):
    return np.median([len(t) for t in tensor])
def min_length(tensor):
    return min([len(t) for t in tensor])

def tokenize(lang):
    lang = [" ".join(each_lang.split()) for each_lang in lang]
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)

    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer