"""
Extracting data from books
"""

import cPickle
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import ebooklib
from ebooklib import epub
import os

import sys
reload(sys)
sys.setdefaultencoding("utf8")


def sentence_to_wordlist(sentence, remove_stopwords=False):
    """
    Sentence to wordlist
    """

    sentence_text = BeautifulSoup(sentence).get_text()
    sentence_text = re.sub("[^a-zA-Z]", " ", sentence_text)
    words = sentence_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    return(words)


def chapter_to_sentences(chapter, tokenizer, remove_stopwords=False):
    """
    Chapter to list of sentences (list of words)
    """

    raw_sentences = tokenizer.tokenize(chapter.strip())

    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(sentence_to_wordlist(raw_sentence,
                                                 remove_stopwords))
    return sentences


def book_to_sentences(book_path, tokenizer):
    """
    Return sentences from given book
    """

    sentences = []

    try:
        book = epub.read_epub(book_path)
    except:
        return -1

    for x in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        sentences += chapter_to_sentences(BeautifulSoup(x.content).text, tokenizer, remove_stopwords=True)
    return sentences


def get_sentence_vector(book_path, model, tokenizer):
    """
    Return sentence averaged word vectors from the model
    """

    sentences = []
    try:
        book = epub.read_epub(book_path)
    except:
        return -1

    for x in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        sentences += chapter_to_sentences(BeautifulSoup(x.content).text, tokenizer, remove_stopwords=True)

    def _get_sig(sentence):
        vec = []
        for word in sentence:
            try:
                vec.append(model[word])
            except KeyError:
                continue
        return np.array(vec)

    wordvecs = map(_get_sig, sentences)
    sentvec = map(lambda x: x.mean(axis=0), wordvecs)

    remove = []
    for idx in xrange(len(sentvec)):
        try:
            sentvec[idx].shape[0]
        except IndexError:
            remove.append(idx)

    sentvec = [item for idx, item in enumerate(sentvec) if idx not in remove]

    sentvec = np.asarray(sentvec)

    # Clip 5% from beginning and end
    return sentvec[int(0.05 * sentvec.shape[0]):int(-0.05*sentvec.shape[0]), :]
