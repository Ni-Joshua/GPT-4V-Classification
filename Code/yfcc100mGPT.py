import os
from openai import AzureOpenAI
import pandas as pd
import json
import numpy as np
from PIL import Image
from imageio import imread
import copy
import random
# from llama_index.llms.azure_openai import AzureOpenAI
llm = AzureOpenAI(
    # model="gpt-4-vision-preview",
    # deployment_name="GPT-4V",
    api_key="",
    azure_endpoint='',
    api_version="",
    # api_version="2024-02-01-preview",
)

#dataset loading
yfcc = pd.read_csv("yfcc100m_geolocated_inat2017species.csv")
df = []
cols = ["url", "scientific_name", "lat", "lon", "GPT_response"]
# specieslist = ''
# for j in range(0, len(yfcc.index)):
#     if(not yfcc.iloc[j]['label'] in specieslist):
#         specieslist = specieslist +"(" +str(j+1) + ") " + yfcc.iloc[j]['label']+ ", "
# print(specieslist)
random.seed(42)
i = 0
while i < 100:
    print(i)
    index = random.randint(0,len(yfcc.index)-1, )
    temp = []
    #Mapping image to row for url and correct response
    row = yfcc.iloc[index]
    lat = row['latitude']
    lon = row['longitude']
    temp.append(row["Flickr URL"])
    temp.append(row["label"])
    temp.append(lat)
    temp.append(lon)
    #Checking if image exists
    try:
        x = imread(row["Flickr URL"])        
    except Exception:
        response = "Image DNE"
        temp.append(response)
        continue
    #Prompting model and Allowing code to finish even if false flagged
    i+=1
    try: 
        response = llm.chat.completions.create(
            model ="GPT-4V",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": "There is a depiction of a species in the image. The image was taken at the latitude " +str(lat) + " degrees and the longitude " +str(lon) + " degrees. Please identify only the scientific name of the species from the information provided with no further explanation"},
                {
                "type": "image_url",
                "image_url": {
                    "url":  row['Flickr URL']
                },
                },
            ],
            }
        ],
        max_tokens=50
        )
    except Exception:
        response = "Exception Error"
        temp.append(response)
        df.append(temp)
        if(i%100 == 0):
            df2 = np.array(copy.deepcopy(df))
            df2 = pd.DataFrame(df2, columns = cols)
            df2.to_csv("yfcc100M_GPTFinished" + str(i) + ".csv", sep='\t', index=False)
        continue
    temp.append(response.choices[0].message.content)
    df.append(temp)
    # Creating save states(in case program crashes)
    if(i%100 == 0):
        df2 = np.array(copy.deepcopy(df))
        df2 = pd.DataFrame(df2, columns = cols)
        df2.to_csv("yfcc100M_GPTFinished" + str(i) + ".csv", sep='\t', index=False)
# Saving Results
df = np.array(df)
df = pd.DataFrame(df, columns = cols)
print(df)
df.to_csv("yfcc100M_GPTFinished.csv", sep='\t', index=False)
