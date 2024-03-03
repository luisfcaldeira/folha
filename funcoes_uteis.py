import time
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer


def remover_stopwords(text):
    
    portugues_stops = stopwords.words('portuguese')
    personal_stop = []
    text = text.split(' ')
    text = [x for x in text if x not in portugues_stops]
    text = [x for x in text if x not in personal_stop]
    return ' '.join(text)

def gerar_cloud(text, title=None): 
    if isinstance(text, list):
        text.sort()
        text = ' '.join(text)
        
    if(title != None):
        print(title)

    wordcloud = WordCloud(collocations=False).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    
def plot_frequencia_tokens(tokens, title=None):
    fd_words = FreqDist(tokens)
    fd_words.plot(20, title=title)

    
def criar_tf_idf(palavras, min_df=0.0):
    if not isinstance(palavras, list):
        raise Exception("precisa ser informado um array de textos")
    vect = TfidfVectorizer()
    vect.set_params(ngram_range=(1,3))
    # vect.set_params(max_df=0.5)
    vect.set_params(min_df=min_df)
    docs_tdidf = vect.fit_transform(palavras)
    return pd.DataFrame(docs_tdidf.todense(), columns=vect.get_feature_names(), index=["doc"+str(i+1) for i in range(0,len(palavras))])
    
def stem_corpus(corpus: str):
    words_tokens = nltk.word_tokenize(corpus, language='portuguese')
    
    return stem_list(words_tokens)

def stem_list(word_list: list) :
    return [stem_word(word) for word in word_list]

def stem_word(word : str):
    stemmer = nltk.stem.RSLPStemmer()
    return stemmer.stem(word)



def estimar_tempo_execucao_em_minutos(tempo_execucao_ate_agora, executados, total_executar):
    return round(((tempo_execucao_ate_agora * total_executar / executados) - tempo_execucao_ate_agora) / 60, 1)

def create_stemm_from_articles(articles: list, tempo_faltante: bool =True, as_array:bool = True):
    corpus_all_stems = []
    count = 0
    total = len(articles)
    tempo_inicial = time.time()
    for article in articles:
        count += 1
        arr_stemmed = stem_list(article.split(' '))
        if not as_array: 
            arr_stemmed = ' '.join(arr_stemmed)
        corpus_all_stems.append(arr_stemmed)
        tempo_final = time.time()
        tempo_execucao = tempo_final - tempo_inicial
        tempo_estimado = estimar_tempo_execucao_em_minutos(tempo_execucao, count, total)
        if tempo_faltante:
            print(f'{count}/{total}: estimado {tempo_estimado} mins para terminar', end='\r')
    if tempo_faltante:
        print('', end='\r\n')
        print('end...', end='\r\n')
    return corpus_all_stems

def extract_words_from_list_of_articles(articles: list, tempo_faltante=True):
    all_words = []
    count = 0
    total = len(articles)
    tempo_inicial = time.time()
    for article in articles:
        arr_of_words = article
        if isinstance(article, str):
            arr_of_words = article.split(' ')
        for word in arr_of_words:
            count += 1
            all_words.append(word)

        tempo_final = time.time()
        tempo_execucao = tempo_final - tempo_inicial
        tempo_estimado = estimar_tempo_execucao_em_minutos(tempo_execucao, count, total)
        if tempo_faltante:
            print(f'{count}/{total}: estimado {tempo_estimado} mins para terminar', end='\r')
    return all_words