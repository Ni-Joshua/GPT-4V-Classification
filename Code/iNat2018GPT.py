import os
from openai import AzureOpenAI
import pandas as pd
import json
import numpy as np
from PIL import Image
from imageio import imread
import copy
import random
import base64
# from llama_index.llms.azure_openai import AzureOpenAI

llm = AzureOpenAI(
    # model="gpt-4-vision-preview",
    # deployment_name="GPT-4V",
    api_key="",
    azure_endpoint='',
    api_version="",
    # api_version="2024-02-01-preview",
)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
#dataset loading
valmap2018 = json.load(open("train_val2018/val2018.json"))
valloc2018 = json.load(open("train_val2018/val2018_locations.json"))
valcat2018 = json.load(open("train_val2018/categories.json"))
df = []
cols = ["url", "scientific_name", "lat", "lon", "GPT_response"]

specieslist = ""
# for j in range(0, len(valcat2018)):
#     specieslist = specieslist +"(" +str(j+1) + ") " + valcat2018[j]['name'] + ", "
# print(specieslist)
random.seed(42)
for i in range(0, 100):
    print(i)
    temp = []
    index = random.randint(0, len(valmap2018["images"]))
    imagepath = valmap2018["images"][index]['file_name']
    temp.append(imagepath)
    annotloc = 0
    lat = 0
    lon = 0
    #Mapping variables
    if(valmap2018["images"][index]['id'] == valmap2018["annotations"][index]['id']):
      annotloc = index

    for j in range(0, len(valcat2018)):
      if(valcat2018[j]['id'] == valmap2018["annotations"][annotloc]['category_id']):
        temp.append(valcat2018[j]['name'])

    if(valmap2018["images"][index]['id'] == valloc2018[index]['id']):
        lat = valloc2018[index]['lat']
        lon = valloc2018[index]['lon']
        temp.append(lat)
        temp.append(lon)

    #encding image into base64
    image = encode_image(imagepath)
    #Prompting model and Allowing code to finish even if false flagged
    try: 
        response = llm.chat.completions.create(
            model ="GPT-4V",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": "There is a species in the image. The image was taken at the latitude " +str(lat) + " degrees and the longitude " +str(lon) + " degrees. Please provide only the scientific name of the species with no further explanation"},
                {
                "type": "image_url",
                "image_url": {
                    "url":  f"data:image/jpeg;base64,{image}"
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
            df2.to_csv("iNat2018_GPTFinished" + str(i) + ".csv", sep='\t', index=False)
        continue
    temp.append(response.choices[0].message.content)
    df.append(temp)
    # Creating save states(in case program crashes)
    if(i%100 == 0):
        df2 = np.array(copy.deepcopy(df))
        df2 = pd.DataFrame(df2, columns = cols)
        df2.to_csv("iNat2018_GPTFinished" + str(i) + ".csv", sep='\t', index=False)

# # #Saving Results
df = np.array(df)
df = pd.DataFrame(df, columns = cols)
print(df)
df.to_csv("iNat2018_GPTFinished.csv", sep='\t', index=False)
