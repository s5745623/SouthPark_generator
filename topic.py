import pandas as pd
from collections import defaultdict
import random

def get_topic(chac):
	#chac = input('Characters: ')
	quotes = pd.read_csv('data/All-seasons.csv')

	chac_dic = defaultdict(list) 

	for i in range(len(quotes)):        
		chac_dic[quotes['Character'][i]].append(quotes['Line'][i])

	chacter = []
	for i in chac_dic.keys():                                                                                         
		chacter.append(i)                                                        

	Kyle_dic = {}
	top_ten_chac = []
	for k in range(len(chac_dic[chac])):                                          
		for i in chacter:             
			if i in chac_dic[chac][k]:
				if i not in Kyle_dic: 
					Kyle_dic[i]=1
				else:    
					Kyle_dic[i]+=1


	top10_chac = sorted(((v,k) for k,v in Kyle_dic.items()), reverse=True)[0:10]
	#print(top10_chac)
	print('random peom: from {} to {}\n'.format(chac, random.choice(top10_chac)[1]))
	result = random.choice(top10_chac)[1]
	
	return result