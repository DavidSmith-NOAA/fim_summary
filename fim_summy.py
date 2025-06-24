# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 09:00:51 2025

@author: david.smith
"""

#import packages
import numpy as np
import pandas as pd
import geopandas as gpd
from zipfile import ZipFile
import urllib.request
import tarfile
import os
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer  # You can also try Luhn, LexRank, etc.
from transformers import pipeline
import nltk
nltk.download('punkt')
import json
import requests
#%%
# Define summarization function
def summarize_text(reviewcomments, summarizer):
    input_length = len(reviewcomments.split())
    max_len = max(20, int(input_length * 0.5))  # summary ~50% of input
    max_len = min(max_len, 60)  # cap at 60 tokens
    summary = summarizer(reviewcomments, max_length=max_len, min_length=10, do_sample=False)
    return summary[0]['summary_text']

#Define the function to extract impact statments from nwps api
def fetch_and_extract_stage_statements(api_url):
    # Make the API request
    response = requests.get(api_url)
    response.raise_for_status()  # Raise an error for bad status codes
    data = response.json()
    
    records = []

    # Extract impacts (flood-related stages and statements)
    for impact in data.get("flood", {}).get("impacts", []):
        records.append({
            "type": "impact",
            "stage": impact.get("stage"),
            "statement": impact.get("statement")
        })


    return pd.DataFrame(records)
#%%
#Set working directory unique to the user
wdir = "C:/Users/david.smith/Documents/Python/fim_summary"
#%%
#Unzip folder with FIM Review Shapefiles.
# loading the zip and creating a zip object.

with ZipFile(wdir+"/NWC_FIM_v5_Reviews.zip", 'r') as zObject:

    # Extracting all the members of the zip 
    # into a specific location.
    zObject.extractall(
        path=wdir+"/NWC_FIM_v5_Reviews")

#%%
#Open FIM Review Shapfiles
points = gpd.read_file(wdir+"/NWC_FIM_v5_Reviews/Point_Location_Reviews.shp" )
polygons = gpd.read_file(wdir+"/NWC_FIM_v5_Reviews/Regional_Reviews.shp" )

#%%
#Determine the polygon centriod and create a new geometry of ponts from the centrod of the polygons
centroids = polygons.centroid
polygons.geometry = centroids

#%%
# Append the points and polygons into one dateframe
# Append the dataframes vertically
fim_reviews = pd.concat([polygons, points], ignore_index=True)

#%%
#Download and read the all gages file from NWPS
url1 = 'https://water.noaa.gov/resources/downloads/reports/nwps_all_gauges_report.csv'

file_Path = os.path.join(wdir, 'nwps_all_gauges_report.csv')
urllib.request.urlretrieve(url1, file_Path)
#%%
all_gages = pd.read_csv('nwps_all_gauges_report.csv', index_col=False)

#%%
#Create geodataframe from all_gages
all_gages= gpd.GeoDataFrame(
    all_gages, geometry=gpd.points_from_xy(all_gages.longitude, all_gages.latitude), crs="EPSG:4326"
)

#%%
#Reproject all_gages to match points and polygons.
all_gages = all_gages.to_crs(epsg=3857)


#%%
#Filter columns by FIMType ['Stage-Based Categorical HAND FIM']
# The purpose of this is to grab all the FIMs that use the SRC and flow. All SRC using FIMs theoretically should all look the same at similar flows.

fim_reviews = fim_reviews.loc[fim_reviews['FIMType'] != 'Stage-Based Categorical HAND FIM']

#%%
#find the nearest NWPS point (obs and frcst) and join it to the fim_reviews dataframe
# If 'index_right' is already in fim_reviews, drop it
if 'index_right' in fim_reviews.columns:
    fim_reviews = fim_reviews.drop(columns='index_right')

# Spatial join: get nearest gage and distance in meters
fim_reviews = gpd.sjoin_nearest(
    fim_reviews,
    all_gages,
    distance_col="distance_meters"
).reset_index(drop=True)

# Convert distance to miles
fim_reviews["distance_miles"] = fim_reviews["distance_meters"] / 1609.34

#%%
#Filter FIM Reviews out that are greater than 5 miles from any gage.
fim_reviews = fim_reviews[fim_reviews["distance_miles"] < 5]


#%%
#Download the latest forecast from NWPS. This will help us get the forecast flows for all the gages. 
#We can then grab all fim reviews with flows comparable and below.

url2 = "https://water.noaa.gov/resources/downloads/shapefiles/national_shapefile_fcst_ffep.tgz"

file_Path = os.path.join(wdir, 'national_shapefile_fcst_ffep.tgz')
urllib.request.urlretrieve(url2, file_Path)


#%%
fname = 'national_shapefile_fcst_ffep.tgz'

if fname.endswith("national_shapefile_fcst_ffep.tgz"):
    tar = tarfile.open(fname, "r:gz")
    tar.extractall()
    tar.close()
elif fname.endswith("tar"):
    tar = tarfile.open(fname, "r:")
    tar.extractall()
    tar.close()
    
forecasts = gpd.read_file(r'national_shapefile_fcst_ffep.dbf')
#%%
#Reproject forecasts to match points and polygons.
forecasts = forecasts.to_crs(epsg=3857)


#%%
#Join forecasts to FIM Reviews on the LID. forecasts lid is "GaugeLID" and fim_reviews LID is called "nws shef id"
# First step will be giving the keys, identical names.
fim_reviews['GaugeLID'] = fim_reviews['nws shef id']

#left merge
fim_reviews = pd.merge(fim_reviews, forecasts, on='GaugeLID', how='left')

#%%
#Filter forecasts out by major and moderate flood
fim_reviews = fim_reviews.loc[(fim_reviews['Status'] == "moderate") | (fim_reviews['Status'] == "major")]

#%%
#Filter out the reviews where the flow is greater than the forecast flow.
fim_reviews['SecValue'] = fim_reviews['SecValue'].astype(float)
#convert kcfs to cfs as fim reiew flows are in cfs
fim_reviews['SecValue'] = fim_reviews['SecValue']*1000
fim_reviews = fim_reviews[fim_reviews['Flow'] < fim_reviews['SecValue']*1.1]


#%%
# Reset series index
fim_reviews = fim_reviews.reset_index(drop=True)

#%%
#Create list of unique LIDs from the fim reviews dataframe

lids = fim_reviews['GaugeLID'].str.split(',\s*').explode().unique().tolist()



#%%
#Summarize the data for each RFC point

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")

reviews_summary = pd.DataFrame()

for i in lids[0:1000]:
    
    try:
        df = fim_reviews[fim_reviews['GaugeLID'] == i]
        
        reviews = pd.DataFrame()
    
        reviews['lid'] = [df['GaugeLID'].head(1).iloc[0]]
        reviews['river'] = [df['location name'].head(1).iloc[0]]
        reviews['forecast_status'] = [df['Status'].head(1).iloc[0]]
        reviews['forecast_stage_ft'] = [df['Forecast'].head(1).iloc[0]]
        reviews['forecast_flow_cfs'] = [df['SecValue'].head(1).iloc[0]]
        reviews['review_count'] = [len(df)]
        reviews['flood_count'] = (df['FloodImpac'] == 'Flood').sum()
        reviews['no_flood_count'] = (df['FloodImpac'] == 'No Flood').sum()
        reviews['inconclusive_count'] = (df['FloodImpac'] == 'Inconclusive').sum()
        
        #Combine waterbodies to list out impacted streams/rivers
        waterbodies = df['WaterFeatu'].str.split(',\s*').explode().unique().tolist()
        # Remove None values
        waterbodies = [item for item in waterbodies if item is not None]
        reviews['waterbodies_impacted'] =  ', '.join(waterbodies) 
    
            
        # Create a single string out of all the review comments for input into summarization model.
        reviewcomments = df['ReviewComm'].str.split(',\s*', regex=True).explode().unique().tolist()
        
        try:
            reviewcomments = ', '.join(reviewcomments)
        except Exception as e:
            print(f"Error joining review comments: {e}")
        
        #Summarize the resultant text string of reviews
        '''Model Type
        Model Architecture: facebook/bart-large-cnn  
        BART (Bidirectional and Auto-Regressive Transformers)
        
        BART (Bidirectional and Auto-Regressive Transformers) is a Transformer model architecture used in 
        natural language processing, particularly for sequence-to-sequence tasks like summarization and translation. 
        It combines the strengths of BERT's bidirectional encoding with GPT's autoregressive decoding. 
        This combination allows BART to effectively grasp context from both directions of the input text and generate 
        coherent, contextually relevant outputs. 
        
        Task: Abstractive Summarization — it generates new sentences to summarize the input, 
        rather than extracting existing ones.'''
        
        # Use a PyTorch model instead of the TensorFlow one
    
        summary = summarize_text(reviewcomments, summarizer)
        reviews['reviews_summary'] = summary
        
        #Pull and summarize Flood Impacts Statements from the NWPS API
        # NWPS API endpoint
        url = "https://api.water.noaa.gov/nwps/v1/gauges/"+i
    
        # Fetch stage and impact statmentsand create the dataframe using the function
        impact_statments = fetch_and_extract_stage_statements(url)
        
        #Select statements that are valid for less than the forecast stage + 0.5ft
        df['Forecast'] = df['Forecast'].astype(float)
        impact_statments = impact_statments[impact_statments['stage'] < (df['Forecast'].head(1).iloc[0]+0.5)]    
        
        # Create a single string out of all the impact statments for input into summarization model.
        impacts = impact_statments['statement'].str.split(',\s*', regex=True).explode().unique().tolist()
        
        try:
            impacts = ', '.join(impacts)
        except Exception as e:
            print(f"Error joining review comments: {e}")
            
        impacts_summary = summarize_text(impacts, summarizer)
        reviews['impacts_statements_summary'] = impacts_summary
        
        
        
        #Add WFO, RFC, state, and county
        reviews['wfo'] = [df['wfo'].head(1).iloc[0]]
        reviews['rfc'] = [df['rfc'].head(1).iloc[0]]
        reviews['state'] = [df['state'].head(1).iloc[0]]
        reviews['county'] = [df['county'].head(1).iloc[0]]
        
        
        reviews_summary = pd.concat([reviews_summary, reviews], ignore_index=True)

    except: 
        pass
    


#%%

reviews_summary.to_csv("reviews_summary.csv")

#%%


