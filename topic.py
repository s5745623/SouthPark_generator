import pandas as pd
from collections import defaultdict
import random
import nltk
import re 

def get_topic(chac):
	#chac = input('Characters: ')
	quotes = pd.read_csv('data/All-seasons.csv')

	chac_dic = defaultdict(list) 

	for i in range(len(quotes)):        
		chac_dic[quotes['Character'][i]].append(quotes['Line'][i])

	chacter = []
	for i in chac_dic.keys():                                                                                         
		chacter.append(i)                                                        

	chacter.remove('M')
	chacter.remove('Al')
	Kyle_dic = {}
	top_ten_chac = []
	for k in range(len(chac_dic[chac])):                                          
		for i in chacter:
			#pat = re.compile('\W*'+i+'\W')
			if i in chac_dic[chac][k]:
				if i not in Kyle_dic: 
					Kyle_dic[i]=1
				else:    
					Kyle_dic[i]+=1
				#print(Kyle_dic)

	top10_chac = sorted(((v,k) for k,v in Kyle_dic.items()), reverse=True)[0:5]
	# print(top10_chac)
	result = random.choice(top10_chac)[1]
	print('\nOh {}!\tby {}\n'.format(result, chac))
	
	
	return result