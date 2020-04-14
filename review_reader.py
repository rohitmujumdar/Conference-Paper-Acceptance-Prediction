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

def mean(l):
  if len(l) == 0:
    return np.nan
  else:
    return float(sum(l))/len(l)

def read_reviews(directory):
  list_of_dict = []
  extra_info = {}
  for filename in tqdm(os.listdir(directory)):
    with open(directory+"/"+filename) as file:
      review_data = json.load(file)
      extra_info_dict = {
          "comments": [],
          "MEANINGFUL_COMPARISON": [],
          "SOUNDNESS_CORRECTNESS": [],
          "ORIGINALITY": [],
          "CLARITY": [],
          "RECOMMENDATION": [],
          "REVIEWER_CONFIDENCE": [],
          "abstract": review_data["abstract"],
          "result": review_data["accepted"],
          "paper_title": review_data["title"]
      }
      for review in review_data["reviews"]:
        if "MEANINGFUL_COMPARISON" in review:
          extra_info_dict["MEANINGFUL_COMPARISON"].append(int(review["MEANINGFUL_COMPARISON"]))
        if "comments" in review and len(review["comments"]) > 0:
          extra_info_dict["comments"].append(review["comments"])
        if "SOUNDNESS_CORRECTNESS" in review:
          extra_info_dict["SOUNDNESS_CORRECTNESS"].append(int(review["SOUNDNESS_CORRECTNESS"]))
        if "ORIGINALITY" in review:
          extra_info_dict["ORIGINALITY"].append(int(review["ORIGINALITY"]))
        if "CLARITY" in review:
          extra_info_dict["CLARITY"].append(int(review["CLARITY"]))
        if "RECOMMENDATION" in review:
          extra_info_dict["RECOMMENDATION"].append(int(review["RECOMMENDATION"]))
        if "REVIEWER_CONFIDENCE" in review:
          extra_info_dict["REVIEWER_CONFIDENCE"].append(int(review["REVIEWER_CONFIDENCE"]))

      temp_dict = {
          "id": filename.replace("json", "pdf"),
          "abstract": review_data["abstract"],
          "result": review_data["accepted"],
          "MEANINGFUL_COMPARISON": mean(extra_info_dict["MEANINGFUL_COMPARISON"]),
          "SOUNDNESS_CORRECTNESS": mean(extra_info_dict["SOUNDNESS_CORRECTNESS"]),
          "ORIGINALITY": mean(extra_info_dict["ORIGINALITY"]),
          "CLARITY": mean(extra_info_dict["CLARITY"]),
          "RECOMMENDATION": mean(extra_info_dict["RECOMMENDATION"]),
          "REVIEWER_CONFIDENCE": mean(extra_info_dict["REVIEWER_CONFIDENCE"]),
          "paper_title": review_data["title"]
      }
      extra_info[filename.replace("json", "pdf")] = extra_info_dict
      list_of_dict.append(temp_dict)

  df = pd.DataFrame(list_of_dict)
  return df, extra_info