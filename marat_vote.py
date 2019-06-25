#!/usr/bin/env python
# -*- coding=utf-8 -*-

"""
Parses the justifications made by speakers as they voted on Marat.
Runs topic modeling on those justifications using two different methods.
"""

from bs4 import BeautifulSoup
import pickle
import regex as re
import pandas as pd
from pandas import *
import numpy as np
import collections
from collections import Counter
import string
import gensim
from gensim import corpora, models
import scipy
from scipy.sparse import coo_matrix
import lda
from processing_functions import remove_diacritic
from make_ngrams import remove_stopwords

### NOTE FOR MYSELF
# The two types of topic modeling are different functions. Once is commented out at all times in the parseFile function

# Parses the marat_vote xml file
def parseFile():
	votes = {}
	justifications = []
	votes_model2 = {}
	file = open('marat.xml', "r")
	contents = file.read()
	contents = re.sub(r'(<p>(?:DÉPARTEMENT|DEPARTEMENT|DÉPARTEMENE)[\s\S]{1,35}<\/p>)', '', contents)
	soup = BeautifulSoup(contents, 'lxml')
	# Look at all speaker tags in the XML
	for talk in soup.find_all('sp'):
		speaker = talk.find('speaker').get_text()
		speaker = remove_diacritic(speaker).decode('utf-8')
		speaker = speaker.replace(".","")

		# Find all the text by looking at paragraph tags
		speech = talk.find_all('p')
		text = ""
		full_speech = ""
		for section in speech:
			text = text + section.get_text()
		full_speech = remove_diacritic(text).decode('utf-8')
		full_speech = full_speech.replace('\n', '').replace('\t', '').replace('\r','')
		full_speech = re.sub(r'([ ]{2,})', ' ', full_speech)

		### Both of the following if statements are for topic modeling but are used for different approaches to the topic modeling
		# Restrict to justifications longer than 30 characters for purposes of topic modeling
		if len(full_speech) > 30:
			justifications.append(full_speech)

		votes[speaker] = full_speech

		if len(full_speech) > 30:
			votes_model2[speaker] = full_speech

	# Two topic model functions
	runTopicModel(justifications)
	#topicModel(votes_model2)

	df = pd.DataFrame.from_dict(votes, orient = 'index')
	writer = pd.ExcelWriter('Marat_Justifications.xlsx')
	df.to_excel(writer)
	writer.save()
	file.close()

# Cleans the text of the speech and removes stopwords
def clean(just_speech):
	stopwords_from_file = open('FrenchStopwords.txt', 'r')
	lines = stopwords_from_file.readlines()
	french_stopwords = []
	for line in lines:
		word = line.split(',')
		#remove returns and new lines at the end of stop words so the parser catches matches
		#also remove accents so the entire analysis is done without accents
		word_to_append = remove_diacritic(unicode(word[0].replace("\n","").replace("\r",""), 'utf-8'))
		french_stopwords.append(word_to_append)

	just_speech = just_speech.replace("%"," ").replace("\\"," ").replace("^", " ").replace("=", " ").replace("]"," ").replace("\""," ").replace("``", " ").replace("-"," ").replace("[", " ").replace("{"," ").replace("$", " ").replace("~"," ").replace("-"," ").replace("}", " ").replace("&"," ").replace(">"," ").replace("#"," ").replace("/"," ").replace("\`"," ").replace("'"," ").replace("*", " ").replace("`", " ").replace(";"," ").replace("?"," ").replace(",", " ").replace(":"," ").replace("."," ").replace("("," ").replace(")"," ")
	clean_text = remove_stopwords(just_speech.lower(), french_stopwords)
	clean_text = clean_text.replace("marat", " ").replace("accusation"," ")
	return clean_text

# Runs a topic model with LDA
def runTopicModel(justifications):
	clean_speeches = [clean(justification).split() for justification in justifications]
	dictionary = corpora.Dictionary(clean_speeches)
	doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_speeches]
	Lda = gensim.models.ldamodel.LdaModel
	number_of_topics = 7
	number_of_words = 15
	ldamodel = Lda(doc_term_matrix, num_topics = number_of_topics, id2word = dictionary, passes = 1000)
	results = ldamodel.print_topics(num_topics = number_of_topics, num_words = number_of_words)
	filename = "topic_modeling_" + str(number_of_topics) + "_numtopics_" + str(number_of_words) + "_numwords.txt"
	file = open(filename, 'w')
	for topic in results:
		file.write(topic[1] + "\n\n")
	file.close()
	print(results)

# Runs a topic model with a COO matrix
def topicModel(justifications):
	clean_speeches = {}
	for speaker in justifications:
		clean_speeches[speaker] = clean(justifications[speaker]).split()
	n_nonzero = 0
	vocab = set()
	for votes in clean_speeches.values():
		unique_terms = set(votes)
		vocab |= unique_terms
		n_nonzero += len(unique_terms)
	docnames = list(clean_speeches.keys())

	docnames = np.array(docnames)
	vocab = np.array(list(vocab))

	vocab_sorter = np.argsort(vocab)

	ndocs = len(docnames)
	nvocab = len(vocab)

	data = np.empty(n_nonzero, dtype=np.intc)
	rows = np.empty(n_nonzero, dtype=np.intc)
	cols = np.empty(n_nonzero, dtype=np.intc)

	ind = 0

	for docname, terms in clean_speeches.items():
		term_indices = vocab_sorter[np.searchsorted(vocab, terms, sorter = vocab_sorter)]

		uniq_indices, counts = np.unique(term_indices, return_counts=True)
		n_vals = len(uniq_indices)
		ind_end = ind + n_vals

		data[ind:ind_end] = counts
		cols[ind:ind_end] = uniq_indices
		doc_idx = np.where(docnames == docname)
		rows[ind:ind_end] = np.repeat(doc_idx, n_vals)

		ind = ind_end

	dtm = coo_matrix((data, (rows, cols)), shape = (ndocs, nvocab), dtype = np.intc)

	number_of_topics = 7
	n_top_words = 15

	model = lda.LDA(n_topics = number_of_topics, n_iter = 1000, random_state = 1)
	model.fit(dtm)
	topic_word = model.topic_word_

	filename = "topic_modeling2_" + str(number_of_topics) + "_numtopics_" + str(n_top_words) + "_numwords.txt"
	file = open(filename, 'w')
	for i, topic_dist in enumerate(topic_word):
		topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
		file.write('Topic {}: {}'.format(i, ' '.join(topic_words)) + "\n")
		print('Topic {}: {}'.format(i, ' '.join(topic_words)))
	file.close()



if __name__ == '__main__':
    import sys
    parseFile()