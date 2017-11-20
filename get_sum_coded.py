import pandas as pd
from gensim.summarization import summarize

# Input: the coded articles that have >4 sentences
df = pd.read_csv('text_without_small_articles.csv', sep=',', encoding='utf-8')
articles = df['text']

sum = []

# summarize each article in 100 words
for i in articles:
    sum.append(summarize(i,word_count=100))

sum2 = []

# Delete the newline characters from the summaries
for i in sum:
    sum2.append(i.strip())

df['sum']=sum2

# Write an Excel file containing the texts and summaries of the coded articles
# with >4 sentences
df.to_excel('mysums.xlsx', encoding='utf-8', index=False)




