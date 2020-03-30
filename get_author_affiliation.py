import pandas as pd
import pickle

df = pd.read_csv("author_university_list.csv")
interim_mapper = {}
mapper = {}
for index, row in df.iterrows():
    # name,affiliation,homepage,scholarid
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

with open("processed_data/affiliation_dict", "wb") as output_file:
    pickle.dump(mapper, output_file)