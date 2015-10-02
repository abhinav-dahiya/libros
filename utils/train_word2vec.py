# Script to train word2vec model from the ebooks in the repository

from gensim.models import word2vec
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import cPickle
import ebooklib
from ebooklib import epub
import pyprind
import os
import sys
reload(sys)
sys.setdefaultencoding("utf8")


def chapter_to_wordlist(chapter, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    chapter_text = BeautifulSoup(chapter).get_text()
    # 2. Remove non-letters
    chapter_text = re.sub("[^a-zA-Z]", " ", chapter_text)
    # 3. Convert words to lower case and split them
    words = chapter_text.lower().split()
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    #
    # 5. Return a list of words
    return(words)


def chapter_to_sentences(chapter, tokenizer, remove_stopwords=False):
    # Function to split a chapter into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(chapter.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(chapter_to_wordlist(raw_sentence,
                                                 remove_stopwords))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


# Find all books
all_books = []
for f in os.listdir("../gutenberg/"):
    if f.endswith(".epub"):
        all_books.append(f)


# Create sentences
tokenizer = nltk.data.load("tokenizers\punkt\english.pickle")

sentences = []

bar = pyprind.ProgBar(len(all_books))
for book_name in all_books:
    book = epub.read_epub(os.path.join("../gutenberg/", book_name))

    for x in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        sentences += chapter_to_sentences(BeautifulSoup(x.content).text, tokenizer, remove_stopwords=True)

    bar.update()

# Dump data
cPickle.dump(sentences, open("sentences_dump.pkl", "w"))


# Make word2vec model
num_features = 100
min_word_count = 10
num_workers = 2
context = 10
downsampling = 1e-3


print("Training word2vec model . . .")
model = word2vec.Word2Vec(sentences,
                          workers=num_workers,
                          size=num_features,
                          min_count=min_word_count,
                          window=context,
                          sample=downsampling)

model.init_sims(replace=True)
model.save("word2vec_model")
