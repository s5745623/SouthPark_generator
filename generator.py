import pandas as pd
import numpy as np
import nltk
from math import log
from collections import defaultdict
import random

WHO = ''
WHO = input('Give us a Character: ')


quotes = pd.read_csv('data/All-seasons.csv')

quotes_by_character = quotes.groupby('Character')
quotes_by_character.count()[quotes_by_character.count().Line > 1000]

kyle_quotes = quotes[quotes.Character == WHO].Line

#kyle_quotes.head()

kyle_quotes_lower = kyle_quotes.apply(str.lower).apply(str.rstrip, '\n')

kyle_tokens = kyle_quotes_lower.apply(nltk.word_tokenize)

#kyle_quotes.head()

kyle_tokens_list =  [ word for inner_list in list(kyle_tokens) for word in inner_list]

kyle_lexical_diversity = len(set(kyle_tokens_list)) / len(kyle_tokens_list)
print(kyle_lexical_diversity)

#len(kyle_tokens_list)/len(kyle_tokens)

top_characters = quotes_by_character.count()[quotes_by_character.count().Line > 100].index

#This function will redo all the computation explained above to compute the lexical diversity and the average sentence length
def get_character_params(data, character):
    character_quotes = data[data.Character == character].Line
    character_quotes_lower = character_quotes.apply(str.lower).apply(str.rstrip, '\n')
    character_tokens = character_quotes_lower.apply(nltk.word_tokenize)
    character_tokens_list =  [ word for inner_list in list(character_tokens) for word in inner_list]
    number_of_unique_words = len(set(character_tokens_list))
    character_lexical_diversity = number_of_unique_words / len(character_tokens_list)
    character_avg_sentence_length = len(character_tokens_list)/len(character_tokens)
    
    return [len(character_tokens), len(character_tokens_list),  character_avg_sentence_length, number_of_unique_words, character_lexical_diversity]

top_characters_tokens = []

columns = ['Name', 'Number of Lines','Total Word Count', "Average Sentence Length", 'Unique Words', "Lexical Diversity"]
character_quotes_parameters_df = pd.DataFrame(columns=columns)

for speaker in top_characters:
    temp_entry_dict = {'Name':"", 'Number of Lines':"",'Total Word Count':"", 
                       "Average Sentence Length":"", 'Unique Words':"", "Lexical Diversity":""}
    
    character_params = get_character_params(quotes, speaker)
    
    temp_entry_dict['Name'] = speaker
    temp_entry_dict['Number of Lines'] = character_params[0]
    temp_entry_dict['Total Word Count'] = character_params[1]
    temp_entry_dict['Average Sentence Length'] = character_params[2]
    temp_entry_dict['Unique Words'] = character_params[3]
    temp_entry_dict['Lexical Diversity'] = character_params[4]
    
    character_quotes_parameters_df = character_quotes_parameters_df.append(temp_entry_dict, ignore_index=True)
    
#character_quotes_parameters_df.head()


