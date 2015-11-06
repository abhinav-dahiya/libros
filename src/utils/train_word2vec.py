"""
Script to train word2vec model from the ebooks
"""

from gensim.models import word2vec
import nltk.data
import cPickle
import pyprind
import os

import book_data

# Learn from files in this directory
root_path = "../../gutenberg/mask/"
# Dump word2vec model here
model_dump = "../data/word2vec_model"

# Find all books
all_books = []
for f in os.listdir(root_path):
    if f.endswith(".epub"):
        all_books.append(os.path.join(root_path, f))


tokenizer = nltk.data.load("tokenizers\punkt\english.pickle")

sentences = []
bar = pyprind.ProgBar(len(all_books))

for book_path in enumerate(all_books):

    ret_sent = book_data.book_to_sentences(book_path, tokenizer)
    if ret_sent != 1:
        sentences += ret_sent
    bar.update()


# Make word2vec model
num_features = 100
min_word_count = 10
num_workers = 3
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
model.save(model_dump)
