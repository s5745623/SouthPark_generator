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
import viz
from nltk.corpus import cmudict

nltk.download('punkt')
nltk.download('vader_lexicon')

stop_words_list = ['i','a','an', 'if', 'of','the','their','to','with']

who_list = [
'Cartman',
'Kenny',
'Kyle',
'Stan']

ngram = 2
quote_len = 20
# quote_len/ngram = poem line length
WHO = ''
while WHO not in who_list:
    print('ONLY the character: '+', '.join(who_list))
    WHO = input('Who is the author: ')
    WHO = WHO[0].upper() + WHO[1:].lower() 

stanzas = int(input("\nHow many Stanzas for the poem? "))
mood = input("POS, NEG or WTV? ")
mood  = mood.upper()
Rhyme = input('SR: Rhyme by stress; FSR: Rhyme by final syllable.\nGive us a rhyme type(SR/FSR): ')
Rhyme = Rhyme.upper()
who = tp.get_topic(WHO)
# Rhyme = input('Give us a Rhyme: ')
file = open('results.txt','w')
quotes = pd.read_csv('data/All-seasons.csv')

quotes_by_character = quotes.groupby('Character')
quotes_by_character.count()[quotes_by_character.count().Line > len(quotes)]

kyle_quotes = quotes[quotes.Character == WHO].Line
#kyle_quotes.head()

kyle_quotes_lower = kyle_quotes.apply(str.lower).apply(str.rstrip, '\n')
kyle_tokens = kyle_quotes_lower.apply(nltk.word_tokenize)
#kyle_quotes.head()

kyle_tokens_list =  [ word for inner_list in list(kyle_tokens) for word in inner_list]
kyle_tokens_list = [re.sub(r'[^A-Za-z0-9\'\-{1}]+$|\'$', 'punc', i) for i in kyle_tokens_list]
kyle_lexical_diversity = len(set(kyle_tokens_list)) / len(kyle_tokens_list)
#print(kyle_lexical_diversity)
#len(kyle_tokens_list)/len(kyle_tokens)

top_characters = quotes_by_character.count()[quotes_by_character.count().Line > 100].index
pro_dict = cmudict.dict()

def get_character_params(data, character):
    
    character_quotes = data[data.Character == character].Line
    character_quotes_lower = character_quotes.apply(str.lower).apply(str.rstrip, '\n')
    character_tokens = character_quotes_lower.apply(nltk.word_tokenize)
    character_tokens_list =  [ word for inner_list in list(character_tokens) for word in inner_list]
    character_tokens_list = [re.sub(r'[^A-Za-z0-9\'\-{1}]+$|\'$', 'punc', i) for i in character_tokens_list]
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

file.write(str(character_quotes_parameters_df.head()))
# print(character_quotes_parameters_df.head())

kyle_word_freq = nltk.FreqDist(kyle_tokens_list)
kyle_word_log_probability = [ {word: -log(float(count)/ len(kyle_tokens_list))} for word, count in kyle_word_freq.items() ]
kyle_word_log_probability[:10]
file.write('\n\n')
file.write(str(kyle_word_log_probability[:10]))

# n-gram 
def build_ngram(text, n):


    if len(text) < n:
        print("Text length is less than n")
        return text
    index = 0
    tokenized_text = nltk.word_tokenize(text)  

    ngram = defaultdict(int)

    for index in range(len(tokenized_text) - n):
        current_word = tokenized_text[index]

        ngram_tail = " ".join(tokenized_text[index + 1: index + n])

        if current_word not in ngram.keys():
            ngram[current_word] = {
                'total_grams_start': 1,
                'grams': {ngram_tail: 1}
            }
        else:
            ngram[current_word]['total_grams_start'] += 1
            if ngram_tail not in ngram[current_word]['grams'].keys():
                ngram[current_word]['grams'][ngram_tail] = 1
            else:
                ngram[current_word]['grams'][ngram_tail] += 1

    return ngram


def Generate_quote(grammed_input, gram_size, start_word, quote_length):
    
    output_str = start_word

    current_word = start_word.lower()
    sentence_prob = 1
    perple = 0

    next_word = ""
    # i=0
    # while i < quote_length // gram_size + 1:
    for i in range(quote_length // gram_size + 1):

        random_num = random.random()
        # random_num = 0.5

        cum_prob = 0
        for potential_next_word, count in grammed_input[current_word]['grams'].items():
            cum_prob += float(count) / grammed_input[current_word]['total_grams_start']
            current_prob = (float(count) + 0.1) / (
                grammed_input[current_word]['total_grams_start'] + 0.1 * len(grammed_input[current_word]['grams']))

            # print cum_prob, random_num
            if cum_prob > random_num:

                # if i != quote_length // gram_size and current_word == 'punc':
                #     break
                # i+=1
                output_str += (" " + potential_next_word)
                current_word = potential_next_word.split()[-1]
                # punc last
                # if i == quote_length // gram_size and current_word != 'punc':
                if i == quote_length // gram_size and current_word == 'punc':
                    output_str,perple  = Generate_quote(grammed_input, gram_size, start_word, quote_length)
                if i == quote_length // gram_size and current_word == 'punc':
                    #print(output_str)
                    perple = 1 / (pow(sentence_prob, 1.0 / quote_length))

                if current_prob != 0:
                    sentence_prob *= current_prob
                break
            else:
                #print(i)
                continue


    perple = 1 / (pow(sentence_prob, 1.0 / quote_length))
    return output_str,perple


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
    perplexityFile = open('perplexity.txt','a')
    poem_list = {}
    for k in range(stanzas):
        i = 0
        poem_list[k] = []
        while len(poem_list[k]) < 4:
            perpleList = []
            poem,perple = Generate_quote(kyle_bigram, ngram, target, quote_len)
            perpleList.append(perple)
            if len(poem_list[k]) == 0:
                while CheckOOV(poem,stop_words_list) is False:
                    poem,perple = Generate_quote(kyle_bigram, ngram, target, quote_len)
                    perpleList.append(perple)
                rhyme = poem.split()[-2]
            else:
                while CheckRhyme(rhyme, poem,stop_words_list) is False:
                    poem,perple = Generate_quote(kyle_bigram, ngram, target, quote_len)
                    perpleList.append(perple)
            poem = 'Oh ' + poem
            #punc last
            #poem_tok = nltk.word_tokenize(re.sub(r'\spunc','!',poem[:-5]))
            poem_tok = nltk.word_tokenize(re.sub(r'\spunc','!',poem))[:-1]
            poem = "".join([" "+i if not (i.startswith("'") or i.startswith("n")) and i not in string.punctuation else i for i in poem_tok]).strip()
            #if sentiment(poem)['compound'] >= 0.5:
            score = sentiment(poem)
            # seperate same rhyme word 
            # for line in range(len(poem_list[k])):
            # if len(poem_list[k]) == 3 and nltk.word_tokenize(poem_list[k][1])[-1] == nltk.word_tokenize(poem_list[k][2])[-1]:
            #     del poem_list[k][2]

            if score['compound'] <= -0.5 and mood == 'NEG':       
                if len(poem_list[k]) < 3:
                    poem_list[k].append(poem + ',')
                    print("perplexity:"+ str(perpleList[-1]))
                    perplexityFile.write(str(perpleList[-1])+"\n")
                    # print(score)
                else: 
                    poem_list[k].append(poem + '.')
                    print("perplexity:" + str(perpleList[-1]))
                    perplexityFile.write(str(perpleList[-1]) + "\n")
                    # print(score)
            elif score['compound'] >= 0.5 and mood == 'POS':
                if len(poem_list[k]) < 3:
                    poem_list[k].append(poem + ',')
                    print("perplexity:" + str(perpleList[-1]))
                    perplexityFile.write(str(perpleList[-1]) + "\n")
                    # print(score)
                else: 
                    poem_list[k].append(poem + '.')
                    print("perplexity:" + str(perpleList[-1]))
                    perplexityFile.write(str(perpleList[-1]) + "\n")
                    # print(score)
            elif mood == 'WTV':
                if len(poem_list[k]) < 3:
                    poem_list[k].append(poem + ',')
                    print("perplexity:" + str(perpleList[-1]))
                    perplexityFile.write(str(perpleList[-1]) + "\n")
                    # print(score)
                else: 
                    poem_list[k].append(poem + '.')
                    print("perplexity:" + str(perpleList[-1]))
                    perplexityFile.write(str(perpleList[-1]) + "\n")
                    # print(score)

            # print(poem + '\tSentiment: ' + str(dict(score)))
        #print('\n')

    for stans in poem_list.values():    
        for lines in stans:
            print(lines)
        print('\n')
    perplexityFile.close()
    return poem_list

def CheckOOV(line,stop_words_list):




    word_to_check = line.split()[-2]
    if word_to_check in stop_words_list:
        return False
    if word_to_check in pro_dict.keys():
        return True
    else:
        return False

def FinalSyllable(rhyme_word, line, stop_words_list):
    
#Check if the last syllable matches the rhyme:
    pron_list1 = []
    for pron1 in pro_dict[rhyme_word]:
        pron_list1.append(pron1)

    rhyme_to_check = line.split()[-2]
    if rhyme_to_check in stop_words_list:
        return False
    pron_list2 = []
    if rhyme_to_check in pro_dict.keys():
        for pron2 in pro_dict[rhyme_to_check]:
            pron_list2.append(pron2)
    else:
        return False

    count_match = 0
    for item1 in pron_list1:
        index1 = len(item1) - 1
        while index1>=0 and re.search('[AEIOU]', item1[index1]) is None:
            index1 -= 1
        item1 = item1[index1:len(item1)] #uncomment this to keep any coda consonant.
        # item1 = item1[index1]
        item1 = ' '.join(item1)    #uncomment this to keep any coda consonant.
        item1 = re.sub(r'\d', '', item1)
        for item2 in pron_list2:
            index2 = len(item2) - 1
            while index2>=0 and re.search('[AEIOU]', item2[index2]) is None:
                index2 -= 1
            item2 = item2[index2:len(item2)]   #uncomment this to keep any coda consonant.
            # item2 = item2[index2]
            item2 = ' '.join(item2)    #uncomment this to keep any coda consonant.
            item2 = re.sub(r'\d', '', item2)
            if item1 == item2:
                count_match += 1

    if count_match == 0:
        return False
    else:
        return True

def Stress(rhyme_word, line,stop_words_list):
#Check if the vowel sequence after the stressed syllable matches the rhymed word:
    

    pron_list1 = []
    for i in pro_dict[rhyme_word]:
        pron_list1.append(" ".join(i))

    rhyme_to_check = line.split()[-2]
    if rhyme_to_check in stop_words_list:
        return False
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
            if re.search('\b[^AEIOU\s]*', item1) is not None:
                item1 = re.sub(r'\d', '', item1)
                item1 = re.sub(r'\b[^AEIOU\s]*', '', item1)
            for item2 in pron_list2:
                if re.search('1', item2) is not None:
                    while re.search('1', item2.split()[0]) is None:
                        item2 = item2[1:]
                    item2 = ' '.join(item2)
                    if re.search('\b[^AEIOU\s]*', item2) is not None:
                        item2 = re.sub(r'\d', '', item2)
                        item2 = re.sub(r'\b[^AEIOU\s]*', '', item2)

                if item1 == item2:
                    count_match += 1

        elif re.search('0', item1) is not None and re.search('1', item1) is None:
            if re.search('\b[^AEIOU\s]*', item1) is not None:
                item1 = re.sub(r'\d', '', item1)
                item1 = re.sub(r'\b[^AEIOU\s]*', '', item1)
            for item2 in pron_list2:
                if re.search('0', item2) is not None:
                    if re.search('\b[^AEIOU\s]*', item2) is not None:
                        item2 = re.sub(r'\d', '', item2)
                        item2 = re.sub(r'\b[^AEIOU\s]*', '', item2)

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
kyle_bigram = build_ngram(' '.join(kyle_tokens_list), ngram)
#print(kyle_bigram)
final_poem = generate_poem(stanzas, target)
file.write('\n\n\n\nOh {}!\n\n'.format(target))

for stans in final_poem.values():    
    for lines in stans:
        file.write(lines+'\n')
    file.write('\n')
file.close()

viz.viz(character_quotes_parameters_df)






