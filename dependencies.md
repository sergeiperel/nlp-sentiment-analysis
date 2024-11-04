import pandas as pd
import re
from collections import Counter
import squarify
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from html import unescape
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from wordcloud import WordCloud
import spacy

def count_emails(tweet):
    return len(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', tweet))

def count_nicknames(tweet):
    return len(re.findall(r'@([A-Za-z0-9_]{1,})', tweet))

def count_urls(tweet):
    return len(re.findall(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})', tweet))

def count_hashtags(tweet):
    return len(re.findall(r'#\w+', tweet))

def delete_stopwords(tokens):
    cleaned = []
    for word in tokens:
        if word not in stop_words:
            cleaned.append(word)
    return cleaned

def lemmatize_tokens(tokens):
    lemmatized = []
    for word in tokens:
        lemmatized.append(lemmatizer.lemmatize(word))
    return lemmatized

def spacy_lemmatize_tokens(text):
    c = []
    doc = nlp(text)
    k = [(token.text, token.pos_) for token in doc]
    c.extend(k)
    return c

def replace_contractions(text):
    # Проверяем, содержатся ли целевые фразы в тексте
    if any(key in text for key in contraction_patterns.keys()):
    # Заменяем каждую найденную фразу на нужное значение
        return pattern.sub(lambda x: contraction_patterns[x.group(0).lower()], text)
    return text

def plot_wordcloud(freq_dist, title):
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(freq_dist)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()
