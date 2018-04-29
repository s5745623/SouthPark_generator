import pandas as pd
import numpy as np
import nltk
from math import log
from collections import defaultdict
import random
import topic as tp
import re
import math
import string
from nltk.corpus import cmudict
import viz

# nltk.download('punkt')

WHO = ''
WHO = input('Who is the author: ')
stanzas = int(input("How many stanzas? "))
mood = input("Pos or Neg? ")
Rhyme = input('SR: Rhyme by stress; FSR: Rhyme by final syllable.\nGive us a rhyme type(SR/FSR): ')
who = tp.get_topic(WHO)


quotes = pd.read_csv('data/All-seasons.csv')

quotes_by_character = quotes.groupby('Character')
quotes_by_character.count()[quotes_by_character.count().Line > len(quotes)]

kyle_quotes = quotes[quotes.Character == WHO].Line

# kyle_quotes.head()

kyle_quotes_lower = kyle_quotes.apply(str.lower).apply(str.rstrip, '\n')

kyle_tokens = kyle_quotes_lower.apply(nltk.word_tokenize)

# kyle_quotes.head()

kyle_tokens_list = [word for inner_list in list(kyle_tokens) for word in inner_list]

kyle_tokens_list = [re.sub(r'[^A-Za-z0-9\'\-{1}]+$|\'$', 'punc', i) for i in kyle_tokens_list]

kyle_lexical_diversity = len(set(kyle_tokens_list)) / len(kyle_tokens_list)
# print(kyle_lexical_diversity)

# len(kyle_tokens_list)/len(kyle_tokens)

top_characters = quotes_by_character.count()[quotes_by_character.count().Line > 100].index
pro_dict = cmudict.dict()

# This function will redo all the computation explained above to compute the lexical diversity and the average sentence length
def get_character_params(data, character):
    character_quotes = data[data.Character == character].Line
    character_quotes_lower = character_quotes.apply(str.lower).apply(str.rstrip, '\n')
    character_tokens = character_quotes_lower.apply(nltk.word_tokenize)
    character_tokens_list = [word for inner_list in list(character_tokens) for word in inner_list]
    character_tokens_list = [re.sub(r'[^A-Za-z0-9\'\-{1}]+$|\'$', 'punc', i) for i in character_tokens_list]
    number_of_unique_words = len(set(character_tokens_list))
    character_lexical_diversity = number_of_unique_words / len(character_tokens_list)
    character_avg_sentence_length = len(character_tokens_list) / len(character_tokens)

    return [len(character_tokens), len(character_tokens_list), character_avg_sentence_length, number_of_unique_words,
            character_lexical_diversity]


top_characters_tokens = []

columns = ['Name', 'Number of Lines', 'Total Word Count', "Average Sentence Length", 'Unique Words',
           "Lexical Diversity"]
character_quotes_parameters_df = pd.DataFrame(columns=columns)

for speaker in top_characters:
    temp_entry_dict = {'Name': "", 'Number of Lines': "", 'Total Word Count': "",
                       "Average Sentence Length": "", 'Unique Words': "", "Lexical Diversity": ""}

    character_params = get_character_params(quotes, speaker)

    temp_entry_dict['Name'] = speaker
    temp_entry_dict['Number of Lines'] = character_params[0]
    temp_entry_dict['Total Word Count'] = character_params[1]
    temp_entry_dict['Average Sentence Length'] = character_params[2]
    temp_entry_dict['Unique Words'] = character_params[3]
    temp_entry_dict['Lexical Diversity'] = character_params[4]

    character_quotes_parameters_df = character_quotes_parameters_df.append(temp_entry_dict, ignore_index=True)

# character_quotes_parameters_df.head()

kyle_word_freq = nltk.FreqDist(kyle_tokens_list)
kyle_word_log_probability = [{word: -log(float(count) / len(kyle_tokens_list))} for word, count in
                             kyle_word_freq.items()]
kyle_word_log_probability[:10]


# Build n-gram builder
def build_ngram(text, n):
    # sanity check, but generally speaking we want the text to be much much longer than the sentence length to get
    # interesting\funny results
    if len(text) < n:
        # print("Text length is less than n")
        return text
    index = 0
    tokenized_text = nltk.word_tokenize(text)  # Try it with lower case, i.e. text.lower()

    ngram = defaultdict(int)

    # Loop over all text, except the last n words, since they cannot have n words after
    for index in range(len(tokenized_text) - n):
        # Get current word from the corpus
        current_word = tokenized_text[index]

        # Get the next n words, so that we can push them into the current word's entry in the ngram dictionary
        ngram_tail = " ".join(tokenized_text[index + 1: index + n])

        # The general structure of an entry is as follows: the beginning of the ngram is the key. Its contents is a dictionary
        # that contains the total number of grams that are started by this first word, plus another dictionary of all the grams
        # and their counts. To save a little space, only the tail is stored in that last dictionary. That way, we can compute
        # easily the probability since everything needed is already stored inside

        # If this is a new entry, create a new one
        if current_word not in ngram.keys():
            ngram[current_word] = {
                'total_grams_start': 1,
                'grams': {ngram_tail: 1}
            }
        else:
            # increase the total count of grams starting with this word
            ngram[current_word]['total_grams_start'] += 1
            # If this ngram tail is new, create a new sub-entry with this ngram
            if ngram_tail not in ngram[current_word]['grams'].keys():
                ngram[current_word]['grams'][ngram_tail] = 1
            # else, increment the entry count by one
            else:
                ngram[current_word]['grams'][ngram_tail] += 1

    return ngram


def Generate_quote(grammed_input, gram_size, start_word, quote_length):
    output_str = start_word

    # This is like the seed based on which we will pick the next word.
    current_word = start_word.lower()

    next_word = ""
    # i=0
    # iterate length by gram size times + 1. We want to iterate as much as needed to build a sentence of n size
    # while i < quote_length // gram_size + 1:
    for i in range(quote_length // gram_size + 1):
        # We want some randomness in picking the next word, not just pick the highest probable next word. So we are going to
        # set a minimum probability under which the gram is not going to get picked.
        random_num = random.random()

        # cumulative probability
        cum_prob = 0
        for potential_next_word, count in grammed_input[current_word]['grams'].items():
            # The cumulative probability is the count of this gram-tail divided by how many time the see word appeared
            cum_prob += float(count) / grammed_input[current_word]['total_grams_start']
            # print cum_prob, random_num
            # If the cumulative probability has reached the minimum probability threshold, then this is the gram to use
            if cum_prob > random_num:

                # if i != quote_length // gram_size and current_word == 'punc':
                #     break
                # i+=1
                output_str += (" " + potential_next_word)
                current_word = potential_next_word.split()[-1]
                if i == quote_length // gram_size and current_word != 'punc':
                    output_str = Generate_quote(grammed_input, gram_size, start_word, quote_length)
                break
            # else, i.e. this gram's probability is lower than our random threshold, get the next gram
            else:
                # print(i)
                continue

    return output_str


def sentiment(sentences):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
    sid = SIA()
    score = defaultdict(float)
    scores = sid.polarity_scores(sentences)
    # for sentence in nltk.word_tokenize(sentences):
    #     score['pos'] += sid.polarity_scores(sentence)['pos']
    #     score['neg'] += sid.polarity_scores(sentence)['neg']
    #     score['compound'] += sid.polarity_scores(sentence)['compound']
    #     score['neu'] += sid.polarity_scores(sentence)['neu']
    for key in sorted(scores):
        # print('{0}: {1}, '.format(key, scores[key]), end='')
        score[key] = scores[key]

    return score


def generate_poem(stanzas, target):
    poem_list = {}
    for k in range(stanzas):
        i = 0
        poem_list[k] = []
        while len(poem_list[k]) < 4:
            poem = Generate_quote(kyle_bigram, 2, target, 20)
            if len(poem_list[k]) == 0:
                while CheckOOV(poem) is False:
                    poem = Generate_quote(kyle_bigram, 2, target, 20)
                rhyme = poem.split()[-2]
            else:
                while CheckRhyme(rhyme, poem) is False:
                    poem = Generate_quote(kyle_bigram, 2, target, 20)
            poem = 'Oh ' + poem
            poem_tok = nltk.word_tokenize(re.sub(r'\spunc', '!', poem[:-5]))
            poem = "".join(
                [" " + i if not (i.startswith("'") or i.startswith("n")) and i not in string.punctuation else i for i in
                 poem_tok]).strip()
            # if sentiment(poem)['compound'] >= 0.5:
            score = sentiment(poem)
            if score['compound'] <= -0.5 and mood == 'Neg':
                if len(poem_list[k]) < 3:
                    poem_list[k].append(poem + ',')
                    # print(score)
                else:
                    poem_list[k].append(poem + '.')
                    # print(score)
            elif score['compound'] >= 0.5 and mood == 'Pos':
                if len(poem_list[k]) < 3:
                    poem_list[k].append(poem + ',')
                    # print(score)
                else:
                    poem_list[k].append(poem + '.')
                    # print(score)
                    # print(poem + '\tSentiment: ' + str(dict(score)))
                    # print('\n')
            #prev_rhyme = poem_list[k][0].split().[-1]
    for stans in poem_list.values():
        for lines in stans:
            print(lines)
        print('\n')

def CheckOOV(line):
    word_to_check = line.split()[-2]
    if word_to_check in pro_dict.keys():
        return True
    else:
        return False

def FinalSyllable(rhyme_word, line):
#Check if the last syllable matches the rhyme:
    pron_list1 = []
    for pron1 in pro_dict[rhyme_word]:
        pron_list1.append(pron1)

    rhyme_to_check = line.split()[-2]
    pron_list2 = []
    if rhyme_to_check in pro_dict.keys():
        for pron2 in pro_dict[rhyme_to_check]:
            pron_list2.append(pron2)
    else:
        return False

    count_match = 0
    for item1 in pron_list1:
        index1 = len(item1) - 1
        while re.search('[AEIOU]', item1[index1]) is None:
            index1 -= 1
        #item1 = item1[index1:len(item1)] #uncomment this to keep any coda consonant.
        item1 = item1[index1]
        #item1 = ' '.join(item1)    #uncomment this to keep any coda consonant.
        item1 = re.sub(r'\d', '', item1)
        for item2 in pron_list2:
            index2 = len(item2) - 1
            while re.search('[AEIOU]', item2[index2]) is None:
                index2 -= 1
            #item2 = item2[index2:len(item2)]   #uncomment this to keep any coda consonant.
            item2 = item2[index2]
            #item2 = ' '.join(item2)    #uncomment this to keep any coda consonant.
            item2 = re.sub(r'\d', '', item2)
            if item1 == item2:
                count_match += 1

    if count_match == 0:
        return False
    else:
        return True

def Stress(rhyme_word, line):
#Check if the vowel sequence after the stressed syllable matches the rhymed word:
    pron_list1 = []
    for i in pro_dict[rhyme_word]:
        pron_list1.append(" ".join(i))

    rhyme_to_check = line.split()[-2]
    pron_list2 = []
    if rhyme_to_check in pro_dict.keys():
        for j in pro_dict[rhyme_to_check]:
            pron_list2.append(" ".join(j))
    else:
        return False

    # print(pron_list1)
    # print(pron_list2)
    count_match = 0
    for item1 in pron_list1:
        if re.search('1', item1) is not None:
            while re.search('1', item1.split()[0]) is None:
                item1 = item1[1:]
            item1 = ' '.join(item1)
            if re.search('\b[^AEIOU\s][^AEIOU\s]*', item1) is not None:
                item1 = re.sub(r'\d', '', item1)
                item1 = re.sub(r'\b[^AEIOU\s][^AEIOU\s]*', '', item1)
            for item2 in pron_list2:
                if re.search('1', item2) is not None:
                    while re.search('1', item2.split()[0]) is None:
                        item2 = item2[1:]
                    item2 = ' '.join(item2)
                    if re.search('\b[^AEIOU\s][^AEIOU\s]*', item2) is not None:
                        item2 = re.sub(r'\d', '', item2)
                        item2 = re.sub(r'\b[^AEIOU\s][^AEIOU\s]*', '', item2)

                if item1 == item2:
                    count_match += 1

        elif re.search('0', item1) is not None and re.search('1', item1) is None:
            if re.search('\b[^AEIOU\s][^AEIOU\s]*', item1) is not None:
                item1 = re.sub(r'\d', '', item1)
                item1 = re.sub(r'\b[^AEIOU\s][^AEIOU\s]*', '', item1)
            for item2 in pron_list2:
                if re.search('0', item2) is not None:
                    if re.search('\b[^AEIOU\s][^AEIOU\s]*', item2) is not None:
                        item2 = re.sub(r'\d', '', item2)
                        item2 = re.sub(r'\b[^AEIOU\s][^AEIOU\s]*', '', item2)

                if item1 == item2:
                    count_match += 1

    if count_match == 0:
        return False
    else:
        return True

if Rhyme == 'SR':
    CheckRhyme = Stress
elif Rhyme == 'FSR':
    CheckRhyme = FinalSyllable
target = who
# stanzas = 4
kyle_bigram = build_ngram(' '.join(kyle_tokens_list), 2)
#print(kyle_bigram)
generate_poem(stanzas, target)

# viz.viz(character_quotes_parameters_df)








