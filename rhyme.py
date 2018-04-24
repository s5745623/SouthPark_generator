import nltk
from nltk import word_tokenize
from nltk.corpus import cmudict
import pandas as pd
from collections import Counter
import json
nltk.download('punkt')

quotes = pd.read_csv('data/All-seasons.csv')
quotes_lines = quotes.Line

kyle_quotes_lower = quotes_lines.apply(str.lower).apply(str.rstrip, '\n')

kyle_tokens = kyle_quotes_lower.apply(nltk.word_tokenize)

#kyle_quotes.head()

kyle_tokens_list =  [ word for inner_list in list(kyle_tokens) for word in inner_list]
tokenCounter = Counter()
tokenCounter.update(kyle_tokens_list)
tokens_list = []
arpabet = cmudict.dict()

file = open('rhyme.csv','w')

token_dict = {}
for k,v in tokenCounter.items():
    # tokens_list.append(k)

    rhymePron = arpabet.get(k,-1)
    print(rhymePron)
    if rhymePron != -1:
        token_dict[k] = rhymePron
with open('tokenRhyme.json','w') as output:
    json.dump(token_dict,output,indent=1)






