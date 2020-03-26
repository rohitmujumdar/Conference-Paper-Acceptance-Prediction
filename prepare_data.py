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

#HOUSEKEEPING
stop_words = stopwords.words('english')
punct_removal_table = {ord(char): None for char in string.punctuation}

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
            file_dict['title_complexity'] = flesch + dale_chall
            
            #AUTHORS
            #1. NUMBER OF AUTHORS
            file_dict['number_of_authors'] = paper_metadata['authors']
            #2. AUTHOR AFFILIATION
            author_emails = paper_metadata['emails']
            data = pd.read_csv("author_university_list.csv") 
                

            
            
    return paper_data_df
                
raw_df_content = load_data_files_into_raw_df('data/iclr_2017','train')