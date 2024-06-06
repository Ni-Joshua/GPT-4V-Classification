import os
from openai import AzureOpenAI
import pandas as pd
import json
import numpy as np
from PIL import Image
from imageio import imread

def top1acccalc(data, name):
    df = []
    cols = ['lat', 'lon', 'hit']
    #Looping through results
    for i in range(0, len(data.index)):
        # print(i)
        temp = []

        temp.append(data.iloc[i]['lat'])
        temp.append(data.iloc[i]['lon'])
        #Skipping images that do not exist
        if(data.iloc[i]['GPT_response'] == "Image DNE"):
            continue
        #Check for non existent answers
        if(pd.isnull(data.iloc[i]['GPT_response'])):
            temp.append(0)
            continue
        #Checking if common or scientific name is in the response
        if(data.iloc[i]['common_name'] in data.iloc[i]['GPT_response'] or data.iloc[i]['scientific_name'] in data.iloc[i]['GPT_response']):
            temp.append(1)
        else:
            temp.append(0)
        df.append(temp)
    df = np.array(df)
    df = pd.DataFrame(df, columns = cols)
    print(df)
    df.to_csv(name+ "AccEval.csv", sep='\t', index=False)
        

def top1acccalcSingleClass(data, name):
    df = []
    cols = ['lat', 'lon', 'hit']
    #Looping through results
    for i in range(0, len(data.index)):
        # print(i)
        temp = []

        temp.append(data.iloc[i]['lat'])
        temp.append(data.iloc[i]['lon'])
        #Skipping images that do not exist
        if(data.iloc[i]['GPT_response'] == "Image DNE"):
            continue
        #Check for non existent answers
        if(pd.isnull(data.iloc[i]['GPT_response'])):
            temp.append(0)
            continue
        #Checking if common or scientific name is in the response
        if(data.iloc[i]['scientific_name'] in data.iloc[i]['GPT_response']):
            temp.append(1)
        else:
            temp.append(0)
        df.append(temp)
    df = np.array(df)
    df = pd.DataFrame(df, columns = cols)
    print(df)
    df.to_csv(name+ "AccEval.csv", sep='\t', index=False)


def top1acccalcfMoW(data, name):
    df = []
    cols = ['path', 'hit']
    #Looping through results
    for i in range(0, len(data.index)):
        # print(i)
        temp = []
        temp.append(data.iloc[i]['url'])
        #Skipping images that do not exist
        if(data.iloc[i]['GPT_response'] == "Image DNE"):
            continue
        #Check for non existent answers
        if(pd.isnull(data.iloc[i]['GPT_response'])):
            temp.append(0)
            continue
        #Checking if common or scientific name is in the response
        if(data.iloc[i]['category'].replace("_"," ") in data.iloc[i]['GPT_response']):
            temp.append(1)
        else:
            temp.append(0)
        df.append(temp)
    df = np.array(df)
    df = pd.DataFrame(df, columns = cols)
    # print(df)
    df.to_csv(name+ "AccEval.csv", sep='\t', index=False)

# birdsnap = pd.read_csv("BirdsnapGPTlocrerun\Birdsnap_GPTFinishedloc.csv", delimiter='\t')
# birdsnaplong = pd.read_csv("BirdsnapGPTloc(long)\Birdsnap_GPTFinishedloc1000.csv", delimiter = '\t')
# NAbirds = pd.read_csv("NABirds+GPT/NAbirdsplus_GPTFinished.csv", delimiter="\t")
# NAbirdslong = pd.read_csv("NABirds+GPT(Long)/NAbirdsplus_GPTFinished.csv", delimiter="\t")
# yfcc = pd.read_csv("yfcc100MGPT\yfcc100M_GPTFinished.csv", delimiter="\t")
# birdsnapplus = pd.read_csv("Birdsnap+GPT\Birdsnapplus_GPTFinished.csv", delimiter = '\t')
# iNat2017 = pd.read_csv("iNat2017GPT\iNat2017_GPTFinished.csv", delimiter="\t")
# iNat2017long = pd.read_csv("iNat2017GPT(long)\iNat2017_GPTFinished.csv", delimiter="\t")
# iNat2018 = pd.read_csv("INat2018GPT\iNat2018_GPTFinished.csv", delimiter="\t")
# iNat2018long = pd.read_csv("INat2018GPT(long)\iNat2018_GPTFinished.csv", delimiter="\t")
fMoW = pd.read_csv("fMoWGPT(short)/fMoW_GPTFinished.csv", delimiter="\t")

# top1acccalc(birdsnap, "Birdsnap")
# top1acccalc(birdsnapplus, "Birdsnap+")
# top1acccalc(NAbirds, "NAbirds")
# top1acccalcSingleClass(iNat2017, "iNat2017")
# top1acccalcSingleClass(iNat2018, "iNat2018")
# top1acccalcSingleClass(yfcc, "yfcc100M")
top1acccalcfMoW(fMoW, "fMoW")