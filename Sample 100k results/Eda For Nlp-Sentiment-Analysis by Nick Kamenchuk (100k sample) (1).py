#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install wordcloud')
get_ipython().system('pip install tqdm')
get_ipython().system('pip install dask swifter')
from tqdm import tqdm
import multiprocessing as mp
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import re
import nltk
import dask.dataframe as dd
import swifter
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[2]:


# Загрузка данных
try:
    df = pd.read_csv('processed_tweets.csv')
except FileNotFoundError:

file_path = 'Sentiment Analysis Dataset.csv'
df = pd.read_csv(file_path, quoting=csv.QUOTE_NONE, on_bad_lines='skip', nrows=100000)

# Загрузка данных (если файл уже обработан, загружаем его)
#try:
#    df = pd.read_csv('processed_tweets.csv')
#except FileNotFoundError:
#file_path = 'Sentiment Analysis Dataset.csv'
#    df = pd.read_csv(file_path, nrows=1000)  # Используем 1000 строк для теста
#    df = pd.read_csv(file_path, nrows=10000)  # Используем 10000 строк для теста
#    df = pd.read_csv(file_path, nrows=100000)  # Используем 100000 строк для теста


# In[3]:


# 1) Загрузка и первичный осмотр данных
# Проверка размера датасета
print(f"Размер датасета: {df.shape}")

# Просмотр первых нескольких записей
print(df.head())

# Проверка типов данных и отсутствующих значений
print(df.info())
print("Проверка на пропуски в данных:")
print(df.isnull().sum())


# In[4]:


# Предположим, что целевая переменная - это 'Sentiment', а текст твитов - 'Text'.
# Если столбцы называются по-другому, их можно переименовать
df = df.rename(columns={'Sentiment': 'Sentiment', 'SentimentText': 'Text'})


# In[5]:


# 2) Анализ целевой переменной
# Подсчет положительных и отрицательных твитов
sentiment_count = df['Sentiment'].value_counts()
print("Распределение по тональностям:")
print(sentiment_count)

# Визуализация распределения тональностей
sns.countplot(x='Sentiment', data=df)
plt.title("Распределение положительных и отрицательных твитов")
plt.show()


# In[6]:


# 3) Анализ текстовых данных
# Длина твитов в символах и словах
df['Text_length_chars'] = df['Text'].apply(len)
df['Text_length_words'] = df['Text'].apply(lambda x: len(x.split()))

# Визуализация распределения длины твитов
plt.figure(figsize=(10,5))
sns.histplot(df['Text_length_chars'], bins=50, kde=True)
plt.title('Распределение длины твитов (в символах)')
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(df['Text_length_words'], bins=50, kde=True)
plt.title('Распределение длины твитов (в словах)')
plt.show()


# In[7]:


# Частота слов: топ-20 самых частых слов для каждой тональности
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Токенизация и лемматизация
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    return words

 # 1. Стандартное использование apply с tqdm для прогресс-бара
tqdm.pandas()  # Активируем прогресс бар для Pandas
df['Processed_Text_Apply'] = df['Text'].progress_apply(preprocess_text)

# Функция для визуализации топ-20 слов
def plot_top_words(processed_col, sentiment, title):
    words = df[df['Sentiment'] == sentiment][processed_col].sum()
    common_words = Counter(words).most_common(20)
    words_df = pd.DataFrame(common_words, columns=['word', 'count'])
    
    plt.figure(figsize=(10,5))
    sns.barplot(x='count', y='word', data=words_df)
    plt.title(title)
    plt.show()

# Построение графиков для каждого метода
plot_top_words('Processed_Text_Apply', 1, 'Топ-20 слов (положительные твиты) - Apply')


# In[8]:


# Частота слов: топ-20 самых частых слов для каждой тональности
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Токенизация и лемматизация
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    return words

 # 1. Стандартное использование apply с tqdm для прогресс-бара
tqdm.pandas()  # Активируем прогресс бар для Pandas
df['Processed_Text_Apply'] = df['Text'].progress_apply(preprocess_text)

# Функция для визуализации топ-20 слов
def plot_top_words(processed_col, sentiment, title):
    words = df[df['Sentiment'] == sentiment][processed_col].sum()
    common_words = Counter(words).most_common(20)
    words_df = pd.DataFrame(common_words, columns=['word', 'count'])
    
    plt.figure(figsize=(10,5))
    sns.barplot(x='count', y='word', data=words_df)
    plt.title(title)
    plt.show()

# Построение графиков для каждого метода
plot_top_words('Processed_Text_Apply', 0, 'Топ-20 слов (негативные твиты) - Apply')


# In[9]:


# Частота слов: топ-20 самых частых слов для каждой тональности
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Токенизация и лемматизация
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    return words


# 4. Использование swifter для автоматического выбора метода с прогресс баром
df['Processed_Text_Swifter'] = df['Text'].swifter.progress_bar(enable=True).apply(preprocess_text)

# Функция для визуализации топ-20 слов
def plot_top_words(processed_col, sentiment, title):
    words = df[df['Sentiment'] == sentiment][processed_col].sum()
    common_words = Counter(words).most_common(20)
    words_df = pd.DataFrame(common_words, columns=['word', 'count'])
    
    plt.figure(figsize=(10,5))
    sns.barplot(x='count', y='word', data=words_df)
    plt.title(title)
    plt.show()

plot_top_words('Processed_Text_Swifter', 1, 'Топ-20 слов (положительные твиты) - Swifter')


# In[10]:


# Частота слов: топ-20 самых частых слов для каждой тональности
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Токенизация и лемматизация
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    return words


# 4. Использование swifter для автоматического выбора метода с прогресс баром
df['Processed_Text_Swifter'] = df['Text'].swifter.progress_bar(enable=True).apply(preprocess_text)

# Функция для визуализации топ-20 слов
def plot_top_words(processed_col, sentiment, title):
    words = df[df['Sentiment'] == sentiment][processed_col].sum()
    common_words = Counter(words).most_common(20)
    words_df = pd.DataFrame(common_words, columns=['word', 'count'])
    
    plt.figure(figsize=(10,5))
    sns.barplot(x='count', y='word', data=words_df)
    plt.title(title)
    plt.show()

plot_top_words('Processed_Text_Swifter', 0, 'Топ-20 слов (негативные твиты) - Swifter')


# In[11]:


# Облака слов для положительных и отрицательных твитов
def plot_wordcloud(sentiment, title):
    words = ' '.join(df[df['Sentiment'] == sentiment]['Text'])
    wordcloud = WordCloud(stopwords=stop_words, background_color='white').generate(words)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

plot_wordcloud(1, "Облако слов для положительных твитов")
plot_wordcloud(0, "Облако слов для отрицательных твитов")


# In[12]:


# 4) Предобработка текста
# Уже выполнена в preprocess_text (токенизация, удаление стоп-слов, лемматизация)

# 5) Анализ n-грамм
def plot_ngrams(sentiment, n, title):
    text = df[df['Sentiment'] == sentiment]['Text'].str.cat(sep=' ')
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')
    ngrams = vectorizer.fit_transform([text])
    sum_ngrams = ngrams.sum(axis=0)
    ngram_counts = [(word, sum_ngrams[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    ngram_counts = sorted(ngram_counts, key=lambda x: x[1], reverse=True)[:20]
    
    ngram_df = pd.DataFrame(ngram_counts, columns=['ngram', 'count'])
    
    plt.figure(figsize=(10,5))
    sns.barplot(x='count', y='ngram', data=ngram_df)
    plt.title(title)
    plt.show()

plot_ngrams(1, 2, 'Топ-20 биграмм для положительных твитов')
plot_ngrams(0, 2, 'Топ-20 биграмм для отрицательных твитов')

plot_ngrams(1, 3, 'Топ-20 триграмм для положительных твитов')
plot_ngrams(0, 3, 'Топ-20 триграмм для отрицательных твитов')


# In[13]:


# 6) Анализ пунктуации и специальных символов
def count_punctuation(text, symbol):
    return text.count(symbol)

df['Exclamations'] = df['Text'].apply(lambda x: count_punctuation(x, '!'))
df['Questions'] = df['Text'].apply(lambda x: count_punctuation(x, '?'))

print(f"Среднее количество восклицательных знаков в положительных твитах: {df[df['Sentiment'] == 1]['Exclamations'].mean()}")
print(f"Среднее количество восклицательных знаков в отрицательных твитах: {df[df['Sentiment'] == 0]['Exclamations'].mean()}")

print(f"Среднее количество вопросительных знаков в положительных твитах: {df[df['Sentiment'] == 1]['Questions'].mean()}")
print(f"Среднее количество вопросительных знаков в отрицательных твитах: {df[df['Sentiment'] == 0]['Questions'].mean()}")


# In[14]:


# 7) Временной анализ (если доступны метки времени)
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day_of_Week'] = df['Date'].dt.day_name()
    
    plt.figure(figsize=(10,5))
    sns.countplot(x='Day_of_Week', data=df, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.title("Распределение твитов по дням недели")
    plt.show()

if 'Date' in df.columns:
    print("Столбец 'Date' найден.")
else:
    print("Столбец 'Date' отсутствует.")

# Сохранение обработанных данных для последующего использования
    df.to_csv('processed_tweets.csv', index=False)
# Загрузка данных (если файл уже обработан, загружаем его)
#try:
#    df = pd.read_csv('processed_tweets.csv')
#except FileNotFoundError:
#file_path = 'Sentiment Analysis Dataset.csv'
#    df = pd.read_csv(file_path, nrows=1000)  # Используем 1000 строк для теста
#    df = pd.read_csv(file_path, nrows=10000)  # Используем 10000 строк для теста
#    df = pd.read_csv(file_path, nrows=100000)  # Используем 100000 строк для теста


# Далее код для выгрузки данных:

# In[15]:


# Сохранение DataFrame в CSV файл
df.to_csv('analysis_results.csv', index=False)


# In[16]:


# Сохранение DataFrame в JSON файл
df.to_json('analysis_results.json', orient='records', lines=True)


# In[17]:


# Сохранение частотных слов для положительной тональности в CSV
positive_words = Counter(df[df['Sentiment'] == 1]['Processed_Text_Apply'].sum()).most_common(20)
pd.DataFrame(positive_words, columns=['word', 'count']).to_csv('positive_words.csv1', index=False)

# Сохранение частотных слов для отрицательной тональности в CSV
negative_words = Counter(df[df['Sentiment'] == 0]['Processed_Text_Apply'].sum()).most_common(20)
pd.DataFrame(negative_words, columns=['word', 'count']).to_csv('negative_words.csv1', index=False)


# In[18]:


# Сохранение DataFrame в Excel файл
with pd.ExcelWriter('analysis_results.xlsx') as writer:
    df.to_excel(writer, sheet_name='Processed Data', index=False)
    
    # Сохранение топ-20 слов для каждой тональности
    pd.DataFrame(positive_words, columns=['word', 'count']).to_excel(writer, sheet_name='Positive Words', index=False)
    pd.DataFrame(negative_words, columns=['word', 'count']).to_excel(writer, sheet_name='Negative Words', index=False)


# In[ ]:




