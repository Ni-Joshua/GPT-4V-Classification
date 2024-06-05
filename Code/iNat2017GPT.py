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
valmap2017 = json.load(open("train_val_images/val2017.json"))
valloc2017 = json.load(open("train_val_images/val2017_locations.json"))
df = []
cols = ["url", "scientific_name", "lat", "lon", "GPT_response"]

specieslist = ""
for j in range(0, len(valmap2017['categories'])):
    specieslist = specieslist +"(" +str(j+1) + ") " + valmap2017['categories'][j]['name'] + ", "
print(specieslist)
random.seed(42)
for i in range(0, 100):
    print(i)
    temp = []
    index = random.randint(0, len(valmap2017["images"]))
    imagepath = valmap2017["images"][index]['file_name']
    temp.append(imagepath)
    annotloc = 0
    lat = 0
    lon = 0
    #Mapping variables
    if(valmap2017["images"][index]['id'] == valmap2017["annotations"][index]['id']):
      annotloc = index

    for j in range(0, len(valmap2017['categories'])):
      if(valmap2017['categories'][j]['id'] == valmap2017["annotations"][annotloc]['category_id']):
        temp.append(valmap2017['categories'][j]['name'])

    if(valmap2017["images"][index]['id'] == valloc2017[index]['id']):
        lat = valloc2017[index]['lat']
        lon = valloc2017[index]['lon']
        temp.append(lat)
        temp.append(lon)

    #encoding image into base64
    image = encode_image(imagepath)
    #Prompting model and Allowing code to finish even if false flagged
    #Short prompt
    # try: 
    #     response = llm.chat.completions.create(
    #         model ="GPT-4V",
    #     messages=[
    #         {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": "There is a species in the image. The image was taken at the latitude " +str(lat) + " degrees and the longitude " +str(lon) + " degrees. Please provide only the full scientific name of the species with no further explanation"},
    #             {
    #             "type": "image_url",
    #             "image_url": {
    #                 "url":  f"data:image/jpeg;base64,{image}"
    #             },
    #             },
    #         ],
    #         }
    #     ],
    #     max_tokens=50
    #     )
    # except Exception:
    #     response = "Exception Error"
    #     temp.append(response)
    #     df.append(temp)
    #     if(i%100 == 0):
    #         df2 = np.array(copy.deepcopy(df))
    #         df2 = pd.DataFrame(df2, columns = cols)
    #         df2.to_csv("iNat2017_GPTFinished" + str(i) + ".csv", sep='\t', index=False)
    #     continue
    # temp.append(response.choices[0].message.content)
    #long prompt
    try: 
        response = llm.chat.completions.create(
            model ="GPT-4V",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": "There is a species in the image. The image was taken at the latitude " +str(lat) + " degrees and the longitude " +str(lon) + " degrees. Here is a list of possible species options: " +specieslist +"Please give me the most probable species of the given image in the following format: (CLASS ID) SPECIES NAME. Please just give a short answer and do not provide explanations for your choice."},
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
            df2.to_csv("iNat2017_GPTFinished" + str(i) + ".csv", sep='\t', index=False)
        continue
    temp.append(response.choices[0].message.content)
    df.append(temp)
    # Creating save states(in case program crashes)
    if(i%100 == 0):
        df2 = np.array(copy.deepcopy(df))
        df2 = pd.DataFrame(df2, columns = cols)
        df2.to_csv("iNat2017_GPTFinished" + str(i) + ".csv", sep='\t', index=False)

# # #Saving Results
df = np.array(df)
df = pd.DataFrame(df, columns = cols)
print(df)
df.to_csv("iNat2017_GPTFinished.csv", sep='\t', index=False)