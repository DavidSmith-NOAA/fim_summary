# fim_summary
Automated consolidation and summarization of FIM reviews/evaluations/impacts and impact statements near river gages that are valid with the current RFC forecasts.


FIM Review and Impact Statement Summarization Tool
Author: David Smith
Date Created: April 10, 2025
Description:
This Python script automates the processing and summarization of Flood Inundation Mapping (FIM) review data and related impact statements based on the latest RFC forecasts. The goal is to extract meaningful insights from review shapefiles and NOAA’s NWPS (National Water Prediction Service) data, summarize relevant comments and impacts using NLP techniques, and prepare an output CSV summarizing findings at each gage.

🔧 Features
Downloads and processes review shapefiles and NWPS gage/forecast data

Matches reviews to the nearest NWPS gage within 5 miles

Filters for FIMs relevant to flow-based forecasting

Extracts and summarizes:

Review comments from shapefiles

Impact statements from NOAA’s API

NLP summarization using Facebook's BART (abstractive summarizer)

Output CSV includes forecast details, review statistics, summaries, and metadata

📁 Input Data
Review Shapefiles (zipped)(Downloaded manually):

NWC_FIM_v5_Reviews.zip with:

Point_Location_Reviews.shp

Regional_Reviews.shp

Gage Data:

Downloaded from NWPS: nwps_all_gauges_report.csv

Forecast Shapefile:

From NWPS: national_shapefile_fcst_ffep.tgz

Impact Statements:

Fetched from NWPS API: https://api.water.noaa.gov/nwps/v1/gauges/{lid}

📦 Dependencies
Python >= 3.8

pandas

geopandas

numpy

requests

tarfile

zipfile

sumy (for optional alternate summarization)

transformers (for BART summarization)

nltk (for tokenizer support)

HuggingFace model: facebook/bart-large-cnn

Install with:

bash
Copy
Edit
pip install pandas geopandas numpy requests sumy transformers nltk
Also run:

python
Copy
Edit
import nltk
nltk.download('punkt')
🧠 Output Summary Fields
The output reviews_summary.csv includes the following columns:

Column	Description
lid	Gage ID
river	River name and location
forecast_status	Forecast status (moderate or major)
forecast_stage_ft	Forecasted stage (feet)
forecast_flow_cfs	Forecasted flow (cfs)
review_count	Total reviews for this LID
flood_count	Count of “Flood” reviews
no_flood_count	Count of “No Flood” reviews
inconclusive_count	Count of “Inconclusive” reviews
waterbodies_impacted	Nearby impacted streams/rivers
reviews_summary	Summarized review comments (BART NLP model)
impacts_statements_summary	Summarized stage-based impact statements (BART NLP model)
wfo	Weather Forecast Office
rfc	River Forecast Center
state	State of the LID
county	County of the LID

🧪 Filtering Logic
Only reviews within 5 miles of a NWPS gage are used.

Only gages with a forecast status of moderate or major are used.

Reviews are included only if their flow is < 110% of the forecast flow.

Impact statements are summarized only if stage is < forecast stage + 0.5 ft

📝 Notes
Summarization is performed using HuggingFace’s facebook/bart-large-cnn, which is ideal for abstractive tasks.

Review and impact texts are cleaned, de-duplicated, and joined before being passed to the NLP pipeline.

💾 Output
Final summarized output: reviews_summary.csv

Can be used for flood risk assessments, FIM validation, or communication to stakeholders.
