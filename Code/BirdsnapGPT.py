import os
from openai import AzureOpenAI
import pandas as pd
import json
import numpy as np
from PIL import Image
from imageio import imread
import copy
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
generalinfo = pd.read_csv("birdsnap/birdsnap\images.txt",delim_whitespace=True)
testbirdsnap = pd.read_csv("birdsnap/birdsnap/test_images.txt", delim_whitespace=True)
specinfo = pd.read_csv("birdsnap/birdsnap\species.txt",delimiter="\t")
birdsnapplus = json.load(open("birdsnap/birdsnap_with_loc_2019.json"))

specieslist = ""
for j in range(0, len(specinfo)):
    specieslist = specieslist +"(" +str(j+1) + ") " + specinfo.iloc[j]['common'] + ", "
print(specieslist)

df = []
cols = ["url", "common_name", "scientific_name", 'lat', 'lon', "GPT_response"]
#looping through test set
for i in range(0, len(testbirdsnap.index)):
    print(i)
    temp = []
    #Mapping image to row for url and correct response
    row = generalinfo.loc[generalinfo['path'] == testbirdsnap.iloc[i]['path']]
    specrow = specinfo.loc[specinfo['id'] == np.array(row['species_id'])[0]]
    lat = birdsnapplus['test'][i]['orig_meta']['lat']
    lon = birdsnapplus['test'][i]['orig_meta']['lon']
    temp.append(np.array(row['url'])[0])
    temp.append(np.array(specrow['common'])[0])
    temp.append(np.array(specrow['scientific'])[0])
    temp.append(lat)
    temp.append(lon)
    #Checking if the image exists
    try:
        x = imread(np.array(row['url'])[0])        
    except Exception:
        response = "Image DNE"
        temp.append(response)
        df.append(temp)
        continue
    #Short Prompt
    #Prompting model and Allowing code to finish even if false flagged
    print(np.isnan(lat), np.isnan(lon))
    # if(np.isnan(lat) or np.isnan(lon)):
    #     try: 
    #         response = llm.chat.completions.create(
    #             model ="GPT-4V",
    #         messages=[
    #             {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": "You are an Ornithologist. There is a bird in the image. Please give me only the name of its species and no additional information"},
    #                 {
    #                 "type": "image_url",
    #                 "image_url": {
    #                     "url":  np.array(row['url'])[0],
    #                 },
    #                 },
    #             ],
    #             }
    #         ],
    #         max_tokens=50
    #         )
    #     except Exception:
    #         response = "Exception Error"
    #         temp.append(response)
    #         df.append(temp)
    #         if(i%100 == 0):
    #             df2 = np.array(copy.deepcopy(df))
    #             df2 = pd.DataFrame(df2, columns = cols)
    #             df2.to_csv("Birdsnap_GPTFinishedloc" + str(i) + ".csv", sep='\t', index=False)
    #         continue
    # else:
    #     try: 
    #         response = llm.chat.completions.create(
    #             model ="GPT-4V",
    #         messages=[
    #             {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": "You are an Ornithologist. There is a bird in the image. The image was taken at the latitude " +str(lat) + " degrees and the longitude " +str(lon) + " degrees. Please give me only the name of its species and no additional information"},
    #                 {
    #                 "type": "image_url",
    #                 "image_url": {
    #                     "url":  np.array(row['url'])[0],
    #                 },
    #                 },
    #             ],
    #             }
    #         ],
    #         max_tokens=50
    #         )
    #     except Exception:
    #         response = "Exception Error"
    #         temp.append(response)
    #         df.append(temp)
    #         if(i%100 == 0):
    #             df2 = np.array(copy.deepcopy(df))
    #             df2 = pd.DataFrame(df2, columns = cols)
    #             df2.to_csv("Birdsnap_GPTFinishedloc" + str(i) + ".csv", sep='\t', index=False)
    #         continue
    #Long prompt
    if(np.isnan(lat) or np.isnan(lon)):
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
                        "url":  np.array(row['url'])[0],
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
                df2.to_csv("Birdsnap_GPTFinishedloc" + str(i) + ".csv", sep='\t', index=False)
            continue
    else:
        try: 
            response = llm.chat.completions.create(
                model ="GPT-4V",
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are an Ornithologist. There is a bird in the image. Here is a list of possible species options: " +specieslist +"Please give me the most probable species of the given image in the following format: (CLASS ID) SPECIES NAME. Please just give a short answer and do not provide explanations for your choice."},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url":  np.array(row['url'])[0],
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
                df2.to_csv("Birdsnap_GPTFinishedloc" + str(i) + ".csv", sep='\t', index=False)
            continue
    temp.append(response.choices[0].message.content)
    df.append(temp)
    #Creating save states(in case program crashes)
    if(i%100 == 0):
        df2 = np.array(copy.deepcopy(df))
        df2 = pd.DataFrame(df2, columns = cols)
        df2.to_csv("Birdsnap_GPTFinishedloc" + str(i) + ".csv", sep='\t', index=False)

#Saving Results
df = np.array(df)
df = pd.DataFrame(df, columns = cols)
print(df)
df.to_csv("Birdsnap_GPTFinishedloc.csv", sep='\t', index=False)







# n = 5
# response = llm.chat.completions.create(
#     model ="GPT-4V",
#     n=n,
#   messages=[
#     {
#       "role": "user",
#       "content": [
#         {"type": "text", "text": "You are an Ornithologist. There is a bird in the image. Please give me the names of 10 possible species."},
#         {
#           "type": "image_url",
#           "image_url": {
#             "url": "http://farm4.staticflickr.com/3228/3152147540_3e6cc05900_o.jpg",
#           },
#         },
#       ],
#     }
#   ],
# )
# # response = llm.complete("who is Michael Jordan?")
# for i in range(0, n):
#     print(response.choices[i].message.content)