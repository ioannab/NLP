import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora

# ------------------------------------ REMOVE TV ARTICLES -------------------------------- #
df = pd.read_csv('elections_2015/tm_articles_with_text.csv')

df_sent = pd.read_csv('elections_2015/tm_sentences_with_text.csv', encoding='utf-8')

# keep only articles derived from newspapers
df2 = df_sent.loc[(df_sent['media_type'] == 'newspaper')]

# get a list with the mongo_ids of only the newspapers' articles
mongo_ids = df2['mongo_id']

# check which articles come from newspapers in the "per article" dataset
mask = df['mongo_id'].isin(mongo_ids)

# dataframe df2 has only the newspaper articles
df2 = df[mask]

#create list of articles
texts = df2['proc_text'].tolist()
doc_complete = [str(i) for i in texts]

# ---------------------------------- Cleaning and preprocessing ---------------------------- #

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
# Some commonly used words that do not infer anything about politics
common_words = ['s', '-','will', 'said','saying', 'say', 't', 'one', 'mr', 'can', 'also', 'says', 'like', 'just', 'say', 'get', 'make', 'go', 'want', 'last', 'first', 'day', 'may', '000', 'per', 'com', 'year', 'would', 'one']
common_words = set(common_words)

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop and i not in common_words])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]

# ----------------------------------- Create doc-term matrix ----------------------------------- #

# Creating the term dictionary of our corpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using the Gensim library
lda = gensim.models.ldamodel.LdaModel

# Running and training LDA model on the document-term matrix

ldamodel = lda(doc_term_matrix, num_topics = 30, id2word = dictionary, passes = 20)

result_topics = ldamodel.print_topics(num_topics=30, num_words=20)

#result_topics.to_csv('lda_elections_2015_articles.csv',encoding='utf-8')

for i in result_topics:
    print(i)
