import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize
import os
import pandas.io.json as pd_json
from nltk.corpus import stopwords
from collections import Counter,defaultdict
import string
import textstat
import statistics
import pickle

#HOUSEKEEPING
stop_words = stopwords.words('english')
punct_removal_table = {ord(char): None for char in string.punctuation}

#read author-uni file
author_university_df = pd.read_csv("author_university_list.csv").drop_duplicates(subset=['affiliation'])

#read uni-score file
university_score_df = pd.read_csv("csrankings.csv").drop_duplicates(subset=['institute'])

def load_data_files_into_raw_df(conference_directory_path,data_split_type):

    #parsed pdfs
    directory_in_string = conference_directory_path + '/' + data_split_type +'/parsed_pdfs/'
    directory_content = os.fsencode(directory_in_string)
    paper_data_df = pd.DataFrame(columns=['words_from_top_200_titles'])
    for file in os.listdir(directory_content):
        filename = os.fsdecode(file)
        file_dict = defaultdict()
        with open(os.path.join(directory_in_string, filename),encoding="utf8") as file:
            #df = pd.read_json(file,orient='columns')
            data = json.load(file)
            paper_id = data['name']
            paper_metadata = data['metadata']

            #ABSTRACT
            abstract_text = paper_metadata['abstractText']

            #TITLE
            title = paper_metadata['title'].lower()
            with open('WordsFromTitleofTop200Papers.txt', 'r') as f:
                top_200_titles_words_counter = Counter([word for word in f.read().translate(punct_removal_table).lower().split() if word not in stop_words])
                #TAKE TOP 5%
                top_200_titles_vocab = [key for key,value in top_200_titles_words_counter.most_common(int(0.05*len(top_200_titles_words_counter))).items()]
            #1. IF TITLE CONTAINS ATLEAST 2 WORDS
            if set(title.split()).intersection(set(top_200_titles_vocab)) > 0:
                file_dict['words_from_top_200_title'] = True
            #2. TITLE LENTGTH
            file_dict['title_length'] = textstat.lexicon_count(title, removepunct=True)
            #3. TITLE COMPLEXITY
            flesch = 1/textstat.flesch_reading_ease(title)
            dale_chall = textstat.dale_chall_readability_score(title)
            file_dict['title_complexity'] = (flesch + dale_chall)/2

            #AUTHORS
            #1. NUMBER OF AUTHORS
            file_dict['number_of_authors'] = paper_metadata['authors']
            #2. AUTHOR AFFILIATION

            author_emails = [email.split('@')[1] for email in paper_metadata['emails']]
            #get list of unis from email ids
            author_universities = []
            '''for email in author_emails:
            try:
                author_universities.append(author_university_df[author_university_df['homepage'].str.contains(institute)]['affiliation']
            except Exception as e:
                author_unis = '''
            
            #REFERENCES
            references_list,ref_mentions_list = paper_metadata['references'],paper_metadata['referenceMentions']
            #1. NUM OF REFRERENCES
            file_dict['num_of_references'] = len(references_list)
            #2. MOST RECENT REFERENCE YEAR
            ref_years_list = [ref_dict['year'] for ref_dict in references_list]
            file_dict['most_recent_ref_year'] = max(ref_years_list)
            #3. AVG LENGTH OF REF MENTION
            file_dict['avg_len_of_ref_mention'] = statistics.mean([ref_dict['endOffset'] - ref_dict['startOffset'] for ref_dict in ref_mentions_list])
            #4. NUMBER OF RECENT REFERENCES (current recent ref behnchmark = 4)
            file_dict['num_of_recent_references'] = sum([1 for year in ref_years_list if paper_metadata['year']-year<4])
                        
            author_emails = paper_metadata['emails']
            author_institutes = [email.split('@')[1] for email in author_emails]
            #read author-uni file
            author_university_df = pd.read_csv("data/author_university_list.csv")
            unique_unis = author_university_df['affiliation'].unique()
            author_university_df_unique_unis = author_university_df.loc[author_university_df['affiliation'].isin(unique_unis)]
            #get list of unis from email ids

            #read uni-score file
            university_score_df = pd.read_csv("csrankings.csv").drop_duplicates(subset=['institute'])

            #load the affiliation_dict
            with open("processed_data/affiliation_dict", "rb") as input_file:
                affiliation_dict = pickle.load(input_file)

            #use this https://stackoverflow.com/questions/10018679/python-find-closest-string-from-a-list-to-another-string
            #to find the closest match in of email in affiliation_dict.keys()
            #same can be used for university
            #jo error ayega to ayega

    return paper_data_df

raw_df_content = load_data_files_into_raw_df('data/iclr_2017','train')