import pandas as pd
import os
from prepare_data import build_affiliation_dictionary,load_data_files_into_raw_df,build_and_save_tfidf_model,collect_data_labels

'''def build_paper_data_csv(data_path = 'data',data_split_category):
	#read author-affiliation file from csrankings hidden page 
	author_university_df = pd.read_csv(os.path.join(data_path,"author_university_list.csv"))
	#read university-research score file from csrankings main page
	university_score_df = pd.read_csv(os.path.join(data_path,"csrankings.csv")).drop_duplicates(subset=['institute'])
	email_institute_affiliation_mapper = build_affiliation_dictionary(author_university_df,university_score_df)
	tfidf_matrix = build_and_save_tfidf_model(data_path,data_split_category)
	features_df = load_data_files_into_raw_df(data_path, data_split_category, email_institute_affiliation_mapper, tfidf_matrix)
	paper_data_df = collect_data_labels(features_df, data_path, data_split_category)
	paper_data_df.to_csv(os.path.join(data_path,"paper_data_ready_for_use.csv"),index=False)
'''
#read author-affiliation file from csrankings hidden page

data_path,data_split_category = 'data','train'
pickle_file_name = 'paper_data_' + str(data_split_category) + '.pkl'
#pickle_file_name = 'paper_data_tfidf_' + str(data_split_category) + '.pkl'
author_university_df = pd.read_csv(os.path.join(data_path,"author_university_list.csv"))
#read university-research score file from csrankings main page
university_score_df = pd.read_csv(os.path.join(data_path,"csrankings.csv")).drop_duplicates(subset=['institute'])
email_institute_affiliation_mapper = build_affiliation_dictionary(author_university_df,university_score_df)
tfidf_matrix = build_and_save_tfidf_model(data_path,data_split_category)[0]
features_df = load_data_files_into_raw_df(data_path, data_split_category, email_institute_affiliation_mapper, tfidf_matrix)
paper_data_df = collect_data_labels(features_df, data_path, data_split_category)
paper_data_df.to_pickle(os.path.join(data_path,pickle_file_name))