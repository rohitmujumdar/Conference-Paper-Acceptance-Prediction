import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize
import os
import pandas.io.json as pd_json
from collections import Counter,defaultdict
import string
import textstat
import statistics
import pickle
import torch
from transformers import *
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from tqdm import tqdm
import difflib

def load_data_files_into_raw_df(data_path, data_split_category, email_institute_affiliation_mapper, tfidf_matrix):
    stop_words = stopwords.words('english')
    punct_removal_table = {ord(char): None for char in string.punctuation}

    MODELS = {"BertModel" : (BertModel,       BertTokenizer,       'bert-base-uncased'),
            "OpenAIGPTModel" : (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
            "GPT2Model" : (GPT2Model,       GPT2Tokenizer,       'gpt2'),
            "TransfoXLModel" : (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
            "XLNetModel" : (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
            "XLMModel" : (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
            "DistilBertModel" : (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
            "RobertaModel" : (RobertaModel,    RobertaTokenizer,    'roberta-base'),
            "XLMRobertaModel" : (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base')}
    model_class, tokenizer_class, pretrained_weights = MODELS['BertModel']
    #Load pretrained model/tokenizer
    print("LOADING TRANSFORMER MODEL ............")
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    nlp = pipeline('feature-extraction')
    print("TRANSFORMER MODEL LOADED!")

    with open(os.path.join(data_path,'WordsFromTitleofTop200Papers.txt'),encoding="utf8") as f:
        top_200_titles_words_counter = Counter([word for word in f.read().translate(punct_removal_table).lower().split() if word not in stop_words])
        #TAKE TOP 5%
        top_200_titles_vocab = [key for key,value in top_200_titles_words_counter.most_common(int(0.05*len(top_200_titles_words_counter)))]

    directory_in_string = data_path + '/iclr_2017/' + str(data_split_category) + '/parsed_pdfs'
    #directory_in_string = data_path + '/iclr_2017/train/parsed_pdfs'
    #review_directory_in_string = data_path + '/iclr_2017/train/reviews'
    review_directory_in_string = data_path + '/iclr_2017/' + str(data_split_category) + '/reviews'
    directory_content = os.fsencode(directory_in_string)
    list_of_file_dicts,paper_data_df = [],pd.DataFrame()
    file_number = 0
    print("FEATURIZING DATA ............")
    for file in tqdm(os.listdir(directory_content)):
        filename = os.fsdecode(file)
        file_dict = defaultdict()
        #print("\nOpening file : ",filename)
        with open(os.path.join(directory_in_string, filename),encoding="utf8") as file:
            data = json.load(file)
            file_dict['paper_id'] = data['name']
            paper_metadata = data['metadata']

            ################################ ABSTRACT ###################################################
            #print("Extracting abstract features....")
            abstract_text = paper_metadata['abstractText'].translate(punct_removal_table).lower()

            #1. BERT et al encoding
            '''input_ids = torch.tensor([tokenizer.encode(abstract_text)]).unsqueeze(0)
            outputs = model(input_ids)
            last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
            file_dict['attention_based_encoding'] = last_hidden_states[0][0]'''
            file_dict['feature_extraction_encoding'] = nlp(abstract_text)[0][0] #DistilBERT

            #2. TFIDF ENCODING
            file_dict['tfidf_encoding'] = tfidf_matrix.toarray()[file_number]
            file_number = file_number + 1

            #title = paper_metadata['title'].lower()
            #3. IF ABSTRACT CONTAINS ATLEAST 2 WORDS FROM TOP 200 TITLES
            if len(set(abstract_text.split()).intersection(set(top_200_titles_vocab))) > 2:
                file_dict['words_from_top_200_title'] = True
            else:
                file_dict['words_from_top_200_title'] = False

            #4. ABSTRACT LENTGTH
            file_dict['abstract_length'] = textstat.lexicon_count(abstract_text, removepunct=True)

            #5. ABSTRACT COMPLEXITY
            flesch = 1/textstat.flesch_reading_ease(abstract_text)
            dale_chall = textstat.dale_chall_readability_score(abstract_text)
            file_dict['abstract_complexity'] = (flesch + dale_chall)/2

            #6. ABSTRACT NOVELTY
            if len({'outperforms','state-of-the-art','state of the art'}.intersection(set(abstract_text.split()))) > 0:
                file_dict['abstract_novelty'] = True
            else:
                file_dict['abstract_novelty'] = False
            ################################ TITLE ###################################################

            ################################ AUTHORS ###################################################
            #print("Extracting authors features.....")
            #1. NUMBER OF AUTHORS
            reported_num_of_authors = len(paper_metadata['authors'])
            if reported_num_of_authors == 0:
                file_dict['number_of_authors'] = 2 #AVG?
            else:
                file_dict['number_of_authors'] = reported_num_of_authors

            #2. AUTHOR AFFILIATION SCORE
            author_emails = paper_metadata['emails']
            author_institutes = [email.split('@')[1] for email in author_emails]
            research_strength_score = 0
            if len(author_institutes) != 0:
                for institute in author_institutes:
                    closest_matches = difflib.get_close_matches(institute, email_institute_affiliation_mapper.keys())
                    if len(closest_matches) == 0:
                        #if there are no closest matches it's not from a university from our list,
                        #which means its either a strange university or a corporate company.
                        #either way its safe to give it a HIGH SCORE - even if the paper is not from a good source
                        #other parameters will take care of it
                        research_strength_score = research_strength_score + 60 #DECISION!
                    else:
                        affiliated_unis_from_string_match = [email_institute_affiliation_mapper[match] for match in closest_matches]
                        sub_score = 0
                        for affiliated_uni in affiliated_unis_from_string_match:
                            try:
                                sub_score = sub_score + university_score_df.loc[university_score_df.institute ==  affiliated_uni, 'count'].values[0]
                            except Exception as e:
                                #if it's an excpetion this is a university alright,
                                #but doesnt fall in our 100 unis of csraniking scores
                                #this means its mostly not a top uni, so it is safe to give it a LOW SCORE
                                sub_score = sub_score + 6 #DECISION!
                        closest_matches_avg_score = sub_score/len(affiliated_unis_from_string_match)
                        research_strength_score = research_strength_score + closest_matches_avg_score
                file_dict['research_strength_score'] = research_strength_score/len(author_institutes)
            else:
                file_dict['research_strength_score'] = 0
            ################################ AUTHORS ###################################################

            ################################ REFERENCES ###################################################
            #print("Extracting references features.....")
            references_list,ref_mentions_list = paper_metadata['references'],paper_metadata['referenceMentions']
            #1. NUM OF REFRERENCES
            file_dict['num_of_references'] = len(references_list)

            #2. MOST RECENT REFERENCE YEAR
            ref_years_list = [ref_dict['year'] for ref_dict in references_list]
            file_dict['most_recent_ref_year'] = max(ref_years_list)

            #3. AVG LENGTH OF REF MENTION
            if len(ref_mentions_list) != 0:
                file_dict['avg_len_of_ref_mention'] = statistics.mean([ref_dict['endOffset'] - ref_dict['startOffset'] for ref_dict in ref_mentions_list])
            else:
                file_dict['avg_len_of_ref_mention'] = 0

            #4. NUMBER OF RECENT REFERENCES (current recent ref behnchmark = 4)
            file_dict['num_of_recent_references'] = sum([1 for year in ref_years_list if paper_metadata['year']-year<4])
            ################################ REFERENCES ###################################################

            ################################ CONTENT ###################################################
            #print("Extracting content features.....")
            #content housekeeping
            sections = paper_metadata['sections']
            if sections is None:
                continue
            else:
                section_content = ''
                for section in sections:
                    section_content = section_content + " " + section['text'].translate(punct_removal_table).lower()
                file_dict['contains_githib_link'],file_dict['contains_appendix'] = False,False

                #1. NUMBER OF SECTIONS
                file_dict['number_of_sections'] = len(sections)

                #2. CONTAINS GITHUB LINK
                for section in sections:
                    if 'github' in section['text'].lower():
                        file_dict['contains_githib_link'] = True
                        break

                #3. READABILITY
                flesch_score,dale_chall_score = 0,0
                for section in sections:
                    flesch_score = flesch_score + textstat.flesch_reading_ease(section['text'])
                    dale_chall_score = dale_chall_score + textstat.dale_chall_readability_score(section['text'])
                flesch_score,dale_chall_score = flesch_score/file_dict['number_of_sections'],dale_chall_score/file_dict['number_of_sections']
                file_dict['content_complexity'] = ((1/flesch_score) + dale_chall_score)/2

                #4. CONTAINS APPENDIX
                for section in sections:
                    if section['heading'] is not None:
                        if 'APPENDIX' in section['heading'] or section['heading'].split()[0] in set(string.ascii_uppercase):
                            file_dict['contains_appendix'] = True
                            break

                #5. NUMBER OF UNIQUE WORDS
                file_dict['number_of_unique_words'] = len(Counter(section_content))
            ################################ CONTENT ###################################################

        list_of_file_dicts.append(file_dict)
            
    features_df = pd.DataFrame(list_of_file_dicts)
    return features_df

def collect_data_labels(features_df, data_path, data_split_category):

    print("COLLECTING DATA LABELS ............")
    accepted_values = []
    review_directory_in_string = data_path + '/iclr_2017/' + str(data_split_category) + '/reviews'
    for index, row in tqdm(features_df.iterrows()):
        filename = row['paper_id'].replace(".pdf","") + '.json'
        with open(os.path.join(review_directory_in_string, filename),encoding="utf8") as file:
            data = json.load(file)
            accepted_values.append(data['accepted'])
    features_df["accepted"] = accepted_values
    return features_df


def build_and_save_tfidf_model(data_path,data_split_category):
    #collect corpus
    punct_removal_table = {ord(char): None for char in string.punctuation}
    corpus = []
    directory_in_string = data_path + '/iclr_2017/' + str(data_split_category) + '/parsed_pdfs'
    directory_content = os.fsencode(directory_in_string)
    for file in tqdm(os.listdir(directory_content)):
        filename = os.fsdecode(file)
        with open(os.path.join(directory_in_string, filename),encoding="utf8") as file:
            paper_metadata = json.load(file)['metadata']
            abstract_text = paper_metadata['abstractText'].translate(punct_removal_table)
            corpus.append(abstract_text)
    stop_words = stopwords.words('english')
    vectorizer = TfidfVectorizer(analyzer='word', stop_words = stop_words)
    X = vectorizer.fit_transform(corpus)
    feature_names  = vectorizer.get_feature_names()
    return X,feature_names

def build_affiliation_dictionary(author_university_df,university_score_df):

    interim_mapper,mapper = {},{}
    for index, row in author_university_df.iterrows():
        #name,affiliation,homepage,scholarid
        link = row['homepage']
        try:
            if "http" in link:
                base = link.split("/")[2]
            else:
                base = link.split("/")[0]

            if "www" in base:
                base = ".".join(base.split(".")[1:])

            if base in interim_mapper:
                if not row["affiliation"] in interim_mapper[base]:
                    interim_mapper[base][row["affiliation"]] = 1
                else:
                    interim_mapper[base][row["affiliation"]] += 1
            else:
                interim_mapper[base] = {}
                interim_mapper[base][row["affiliation"]] = 1

        except Exception as e:
            print("Homepage : ",row['homepage']," defaulted!")

    # for base in interim_mapper:
    #     if len( interim_mapper[base].keys()) > 1:
    #         print(base, interim_mapper[base])

    #interim_mapper now has the email ids, and number of times, they were refferred to as an affiliations.
    #we ignore if number of affliations are more than 4 as then they are generic ids like gmail.com and shoudn't be mapped
    #else we take the maximum

    for base in interim_mapper:
        if len(interim_mapper[base].keys()) < 2:
            #calculate the maximum referred affiliation
            max_count = 0
            max_ff = ""
            for aff in interim_mapper[base]:
                if interim_mapper[base][aff] > max_count:
                    max_count = interim_mapper[base][aff]
                    max_ff = aff
            mapper[base] = aff

    '''with open("processed_data/affiliation_dict", "wb") as output_file:
        pickle.dump(mapper, output_file)'''

    return mapper


#def read_reviews(filename):
