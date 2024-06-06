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
files = pd.read_csv("new-filelist")
valloc = json.load(open("Hosted-Datasets/fmow/fmow-rgb/val/val_location.json"))
locations = json.load(open("Hosted-Datasets/fmow/fmow-rgb/val\category.json"))
df = []
cols = ["url", "category", "GPT_response"]

#Creating Category list
loclist = ""
counter = 1
for key in locations:
    loclist = loclist +"(" + str(counter) + ") " + key + ", "
    counter+=1
print(loclist)
#Getting 100 random test samples
random.seed(42)
for i in range(0, 100):
    print(i)
    index = random.randint(0, len(files)-1)
    temp = []
    #Obtaining image path and lat lon info
    imagepath = ''
    jsonpath = ''
    #Finding correct files
    if("msrgb.jpg" in files.iloc[index]['file']):
        imagepath = files['file'].iloc[index+2]
        jsonpath = files['file'].iloc[index+3]
    elif("msrgb.json" in files.iloc[index]['file']):
        imagepath =files['file'].iloc[index+1]
        jsonpath = files['file'].iloc[index+2]
    elif("_rgb.jpg" in files.iloc[index]['file']):
        imagepath =files['file'].iloc[index]
        jsonpath = files['file'].iloc[index+1]
    elif("_rgb.json" in files.iloc[index]['file']):
        imagepath =files['file'].iloc[index-1]
        jsonpath = files['file'].iloc[index]

    # for 
    # print(jsonpath)
    info = json.load(open(jsonpath))
    temp.append(imagepath)
    temp.append(info['bounding_boxes'][0]['category'])

    #encoding image into base64
    image = encode_image(imagepath)
    #Prompting model and Allowing code to finish even if false flagged
    # try: 
    #     response = llm.chat.completions.create(
    #         model ="GPT-4V",
    #     messages=[
    #         {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": "There is a place shown in this image. Here is a list of names of the possible places: " + loclist +"Please give me the most probable name shown by the given image in the following format: (CLASS ID) LOCATION NAME. Please just give a short answer and do not provide explanations for your choice."},
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
    #         df2.to_csv("fMoW_GPTFinished" + str(i) + ".csv", sep='\t', index=False)
    #     continue
    try: 
        response = llm.chat.completions.create(
            model ="GPT-4V",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": "There is a place shown in this image. Please identify the type of place shown with no further explanation"},
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
            df2.to_csv("fMoW_GPTFinished" + str(i) + ".csv", sep='\t', index=False)
        continue
    temp.append(response.choices[0].message.content)
    df.append(temp)
    # Creating save states(in case program crashes)
    if(i%100 == 0):
        df2 = np.array(copy.deepcopy(df))
        df2 = pd.DataFrame(df2, columns = cols)
        df2.to_csv("fMoW_GPTFinished" + str(i) + ".csv", sep='\t', index=False)

# #Saving Results
df = np.array(df)
df = pd.DataFrame(df, columns = cols)
print(df)
df.to_csv("fMoW_GPTFinished.csv", sep='\t', index=False)

