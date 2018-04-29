def viz(character_quotes_parameters_df):

	import matplotlib.pyplot  as plt
	import numpy as np
	import math

	plt.figure(1)
	plt.plot(character_quotes_parameters_df['Total Word Count'], character_quotes_parameters_df['Lexical Diversity'],'ro',color='green')
	plt.xlabel('Total Word Count', fontsize=16)
	plt.ylabel('Lexical Diversity', fontsize=16)
	plt.savefig('1')
	# plt.show()

	plt.figure(2)
	plt.plot(np.log(character_quotes_parameters_df['Total Word Count'].astype(float)), np.log(character_quotes_parameters_df['Lexical Diversity'].astype(float)),'ro',color='red')
	plt.xlabel('Ln Total Word Count', fontsize=16)
	plt.ylabel('Ln Lexical Diversity', fontsize=16)
	plt.savefig('2')
	# plt.show()

	plt.figure(3)
	plt.plot(character_quotes_parameters_df['Total Word Count'], character_quotes_parameters_df['Unique Words'],'ro',color='yellow')
	plt.xlabel('Total Word Count', fontsize=16)
	plt.ylabel('Unique Word Count', fontsize=16)
	plt.savefig('3')
	# plt.show()

	plt.figure(4)
	plt.plot(np.log(character_quotes_parameters_df['Total Word Count'].astype(float)), np.log(character_quotes_parameters_df['Unique Words'].astype(float)),'ro',color='blue')
	plt.xlabel('Ln Total Word Count', fontsize=16)
	plt.ylabel('Ln Unique Word Count', fontsize=16)
	plt.savefig('4')
	# plt.show()