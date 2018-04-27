def viz(character_quotes_parameters_df):

	import matplotlib.pyplot  as plt
	import numpy as np
	plt.plot(character_quotes_parameters_df['Total Word Count'], character_quotes_parameters_df['Lexical Diversity'],'ro')

	plt.xlabel('Total Word Count', fontsize=16)
	plt.ylabel('Lexical Diversity', fontsize=16)

	plt.show()

	import math
	plt.plot(np.log(character_quotes_parameters_df['Total Word Count'].astype(float)), np.log(character_quotes_parameters_df['Lexical Diversity'].astype(float)),'ro')

	plt.xlabel('Ln Total Word Count', fontsize=16)
	plt.ylabel('Ln Lexical Diversity', fontsize=16)

	plt.show()
	
	plt.plot(character_quotes_parameters_df['Total Word Count'], character_quotes_parameters_df['Unique Words'],'ro')

	plt.xlabel('Total Word Count', fontsize=16)
	plt.ylabel('Unique Word Count', fontsize=16)

	plt.show()

	plt.plot(np.log(character_quotes_parameters_df['Total Word Count'].astype(float)), np.log(character_quotes_parameters_df['Unique Words'].astype(float)),'ro')

	plt.xlabel('Ln Total Word Count', fontsize=16)
	plt.ylabel('Ln Unique Word Count', fontsize=16)

	plt.show()