import os
from openai import AzureOpenAI
import pandas as pd
import json
import numpy as np
from PIL import Image
from imageio import imread
def top1acccalc(data, name):
    truecount = 0
    totalexDNE = 0
    totalexDNEexFF = 0#Total except image does not exist and except false flag
    totalexDNEexNull = 0#Total except image does not exist and except null
    totalexDNEexFFexNull = 0#Total except image DNE, except false flag, except null
    errorcount = 0
    DNE = 0
    Flagged = 0
    #Looping through results
    for i in range(0, len(data.index)):
        # print(i)
        #Skipping images that do not exist
        if(data.iloc[i]['GPT_response'] == "Image DNE"):
            DNE+=1
            continue
        if(data.iloc[i]['GPT_response'] == "Exception Error"):
            Flagged+=1
        totalexDNE+=1
        if(not pd.isnull(data.iloc[i]['GPT_response'])):
            totalexDNEexNull+=1
        if(data.iloc[i]['GPT_response'] != "Exception Error"):
            totalexDNEexFF+=1
            if(not pd.isnull(data.iloc[i]['GPT_response'])):
                totalexDNEexFFexNull+=1

        #Check for non existent answers
        if(pd.isnull(data.iloc[i]['GPT_response'])):
            errorcount+=1
            continue
        #Checking if common or scientific name is in the response
        if(data.iloc[i]['common_name'] in data.iloc[i]['GPT_response'] or data.iloc[i]['scientific_name'] in data.iloc[i]['GPT_response']):
            truecount+=1

    #Print results
    print(name)
    print("Top1acc without image DNE = " + str(truecount/totalexDNE))
    print("Top1acc without image DNE and without false flags = " + str(truecount/totalexDNEexFF))
    print("Top1acc without image DNE and without null = " + str(truecount/totalexDNEexNull))
    print("Top1acc without image DNE and without false flags and without null = " + str(truecount/totalexDNEexFFexNull))

    print("Null answers = " + str(errorcount))
    print("Flagged images = " + str(Flagged))
    print("Image Does Not Exist = " + str(DNE))

def top1acccalcSingleClass(data, name):
    truecount = 0
    totalexDNE = 0
    totalexDNEexFF = 0#Total except image does not exist and except false flag
    totalexDNEexNull = 0#Total except image does not exist and except null
    totalexDNEexFFexNull = 0#Total except image DNE, except false flag, except null
    errorcount = 0
    DNE = 0
    Flagged = 0
    #Looping through results
    for i in range(0, len(data.index)):
        # print(i)
        #Skipping images that do not exist
        if(data.iloc[i]['GPT_response'] == "Image DNE"):
            DNE+=1
            continue
        if(data.iloc[i]['GPT_response'] == "Exception Error"):
            Flagged+=1
        totalexDNE+=1
        if(not pd.isnull(data.iloc[i]['GPT_response'])):
            totalexDNEexNull+=1
        if(data.iloc[i]['GPT_response'] != "Exception Error"):
            totalexDNEexFF+=1
            if(not pd.isnull(data.iloc[i]['GPT_response'])):
                totalexDNEexFFexNull+=1

        #Check for non existent answers
        if(pd.isnull(data.iloc[i]['GPT_response'])):
            errorcount+=1
            continue
        #Checking if scientific name or category is in the response
        try:
            if(data.iloc[i]['scientific_name'] in data.iloc[i]['GPT_response']):
                truecount+=1
        except Exception:
            if(data.iloc[i]['category'] in data.iloc[i]['GPT_response']):
                truecount+=1

    #Print results
    print(name)
    print("Top1acc without image DNE = " + str(truecount/totalexDNE))
    print("Top1acc without image DNE and without false flags = " + str(truecount/totalexDNEexFF))
    print("Top1acc without image DNE and without null = " + str(truecount/totalexDNEexNull))
    print("Top1acc without image DNE and without false flags and without null = " + str(truecount/totalexDNEexFFexNull))

    print("Null answers = " + str(errorcount))
    print("Flagged images = " + str(Flagged))
    print("Image Does Not Exist = " + str(DNE))


birdsnap = pd.read_csv("BirdsnapGPT\Birdsnap_GPTFinished.csv", delimiter='\t')
birdsnaplong = pd.read_csv("Birdsnap(long)GPT\Birdsnap_GPTFinished1100.csv", delimiter = '\t')
NAbirds = pd.read_csv("NABirds+GPT/NAbirdsplus_GPTFinished.csv", delimiter="\t")
NAbirdslong = pd.read_csv("NABirds+GPT(Long)/NAbirdsplus_GPTFinished.csv", delimiter="\t")
yfcc = pd.read_csv("yfcc100MGPT\yfcc100M_GPTFinished.csv", delimiter="\t")
birdsnapplus = pd.read_csv("Birdsnap+GPT\Birdsnapplus_GPTFinished.csv", delimiter = '\t')
iNat2017 = pd.read_csv("iNat2017GPT\iNat2017_GPTFinished.csv", delimiter="\t")
iNat2018 = pd.read_csv("iNat2018GPT\iNat2018_GPTFinished.csv", delimiter="\t")
fMoW = pd.read_csv("fMoWGPT/fMoW_GPTFinished.csv", delimiter="\t")

top1acccalc(birdsnap, "Birdsnap")
top1acccalc(birdsnaplong, "Birdsnaplong")
top1acccalc(birdsnapplus, "Birdsnap+")
top1acccalc(NAbirds, "NAbirds")
top1acccalc(NAbirdslong, "NAbirdslong")
top1acccalcSingleClass(iNat2017, "iNat2017")
top1acccalcSingleClass(iNat2018, "iNat2018")
top1acccalcSingleClass(yfcc, "yfcc100M")
top1acccalcSingleClass(fMoW, "fMoW")
