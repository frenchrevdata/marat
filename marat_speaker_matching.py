#!/usr/bin/env python
# -*- coding=utf-8 -*-

"""
Does the majority of the data parsing of the XML files and outputs two critical files -- raw_speeches and speechid_to_speaker.
It also keeps track of errors in the XML files.
"""

from bs4 import BeautifulSoup
import unicodedata
import os
import csv
import pickle
import regex as re
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.util import ngrams
import collections
from collections import Counter
import os
import gzip
from make_ngrams import compute_ngrams
import xlsxwriter
from processing_functions import remove_diacritic, load_speakerlist, write_to_excel


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

		# ### Both of the following if statements are for topic modeling but are used for different approaches to the topic modeling
		# # Restrict to justifications longer than 30 characters for purposes of topic modeling
		# if len(full_speech) > 30:
		# 	justifications.append(full_speech)

		votes[speaker] = full_speech

	# 	if len(full_speech) > 30:
	# 		votes_model2[speaker] = full_speech

	# # Two topic model functions
	# runTopicModel(justifications)
	# #topicModel(votes_model2)

	# df = pd.DataFrame.from_dict(votes, orient = 'index')
	# writer = pd.ExcelWriter('Marat_Justifications.xlsx')
	# df.to_excel(writer)
	# writer.save()
	# file.close()
	return votes


def get_bigrams(names_matching, justifications): 
	votes = pd.read_excel('Conventionnels_mod.xlsx')
	indv_bigrams = {}
	for indv in justifications:
		indv_bigrams[indv] = compute_ngrams(justifications[indv], 2)
	yes = Counter()
	no = Counter()
	group_yes = 0
	group_no = 0
	for speaker in indv_bigrams:
		if speaker in names_matching:
			speaker_full_name = names_matching[speaker]
			group = ""
			for i, name in enumerate(votes["Name"]):
				name = remove_diacritic(name).decode('utf-8')
				if speaker_full_name == name:
					group = votes["Group"].iloc[i]
			if group == "yes":
				group_yes += 1
				yes += indv_bigrams[speaker]
			elif group == "no":
				group_no += 1
				no += indv_bigrams[speaker]
	yes_group = {k:v for k,v in yes.items()}
	df_yes = pd.DataFrame.from_dict(yes_group, orient = "index")
	write_to_excel(df_yes, "yes_counts.xlsx")

	no_group = {k:v for k,v in no.items()}
	df_no = pd.DataFrame.from_dict(no_group, orient = "index")
	write_to_excel(df_no, "no_counts.xlsx")


# def load_votes(speakernames):
# 	pd_list = pd.read_excel(speakernames)

# 	pd_list = pd_list.set_index('Name')
# 	speakers = pd_list.index.tolist()
# 	for speaker in speakers:
# 		ind = speakers.index(speaker)
# 		speakers[ind] = remove_diacritic(speaker).decode('utf-8')
# 	pd_list.index = speakers
# 	return pd_list

def load_speakerlist(speakernames):
	pd_list = pd.read_excel(speakernames)

	pd_list = pd_list.set_index('Names')
	speakers = pd_list.index.tolist()
	for speaker in speakers:
		ind = speakers.index(speaker)
		speakers[ind] = remove_diacritic(speaker).decode('utf-8')
	# pd_list.index = speakers
	return speakers

if __name__ == '__main__':
	import sys
	lastname_speaker_list = load_speakerlist('Marat_Justifications Last names.xlsx')
	fullname_speaker_list = load_speakerlist('Conventionnels full names.xlsx')
	name_matching = {}
	names_not_matched = set()
	for name in lastname_speaker_list:
		for fullname in fullname_speaker_list:
			if fullname.find(name) != -1:
				name_matching[name] = fullname
	for name in lastname_speaker_list:
		if name not in name_matching:
			names_not_matched.add(name + "\n")
	justifications = parseFile()
	get_bigrams(name_matching, justifications)
	# w = csv.writer(open("name_matching.csv", "w"))
	# for key, val in name_matching.items():
	# 	w.writerow([key, val])
	# file = open('speakers_not_matched.txt', 'w')
	# for item in sorted(names_not_matched):
	# 	file.write(item)
	# file.close()


    
       
   	
