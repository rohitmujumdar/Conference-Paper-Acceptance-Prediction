import pandas as pd
import os

data_path = ''
#read author-affiliation file from csrankings hidden page 
author_university_df = pd.read_csv(os.path.join(data_path,"author_university_list.csv"))
#read university-research score file from csrankings main page
university_score_df = pd.read_csv(os.path.join(data_path,"csrankings.csv")).drop_duplicates(subset=['institute'])
email_institute_affiliation_mapper = build_affiliation_dictionary(author_university_df,university_score_df)
tfidf_matrix = build_and_save_tfidf_model()
paper_data_df = load_data_files_into_raw_df()
print(paper_data_df)