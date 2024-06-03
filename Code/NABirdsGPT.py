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
idtoloc_classes = json.load(open("nabirds/nabirds_with_loc_2019.json"))
df = []
cols = ["url", "common_name", "scientific_name", "lat", "lon", "GPT_response"]

#Creating Species list
specieslist = ""
for j in range(0, len(idtoloc_classes['classes'])):
    specieslist = specieslist +"(" +str(j+1) + ") " + idtoloc_classes['classes'][j].split(' (')[0] + ", "
# print(specieslist)
#Getting 100 random test samples
random.seed(42)
for i in range(0, 100):
    print(i)
    index = random.randint(0,len(idtoloc_classes['test'])-1)
    temp = []
    #Obtaining image path and lat lon info
    imagepath = idtoloc_classes['test'][index]['im_path']
    lat = idtoloc_classes['test'][index]['ebird_meta']['lat']
    lon = idtoloc_classes['test'][index]['ebird_meta']['lon']
    temp.append(imagepath)
    temp.append(idtoloc_classes['classes'][idtoloc_classes['test'][index]['class_id']].split(' (')[0])
    temp.append(idtoloc_classes['classes_sci'][idtoloc_classes['test'][index]['class_id']])
    temp.append(lat)
    temp.append(lon)

    #encoding image into base64 so GPT-4V api can read it
    image = encode_image("nabirds/images/" + imagepath)
    #Long Prompt
    #Prompting model and Allowing code to finish even if false flagged
    try: 
        response = llm.chat.completions.create(
            model ="GPT-4V",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": "You are an Ornithologist. There is a bird in the image. The image was taken at the latitude " +str(lat) + " degrees and the longitude " +str(lon) + " degrees. Here is a list of possible species options: " +specieslist +"Please give me the most probable species of the given image in the following format: (CLASS ID) SPECIES NAME. Please just give a short answer and do not provide explanations for your choice."},
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
            df2.to_csv("NAbirdsplus_GPTFinished" + str(i) + ".csv", sep='\t', index=False)
        continue
    #Short Prompt
    # try: 
    #     response = llm.chat.completions.create(
    #         model ="GPT-4V",
    #     messages=[
    #         {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": "You are an Ornithologist. There is a bird in the image. The image was taken at the latitude " +str(lat) + " degrees and the longitude " +str(lon) + " degrees. Please give me only the name of its species and no additional information"},
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
    #         df2.to_csv("NAbirdsplus_GPTFinished" + str(i) + ".csv", sep='\t', index=False)
    #     continue
    temp.append(response.choices[0].message.content)
    df.append(temp)
    # Creating save states(in case program crashes)
    if(i%100 == 0):
        df2 = np.array(copy.deepcopy(df))
        df2 = pd.DataFrame(df2, columns = cols)
        df2.to_csv("NAbirdsplus_GPTFinished" + str(i) + ".csv", sep='\t', index=False)

# #Saving Results
df = np.array(df)
df = pd.DataFrame(df, columns = cols)
print(df)
df.to_csv("NAbirdsplus_GPTFinished.csv", sep='\t', index=False)
