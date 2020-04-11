import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pprint import pprint
import pandas as pd

#Link to the Google Excel Sheet
scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope)
client = gspread.authorize(creds)
sheet = client.open("SD_Annotation").sheet1

data = sheet.get_all_records()

#Create dataframe from all the data in Excel Sheet
whole_df = pd.DataFrame.from_dict(data)

'''
Create dataframe with three columns Exact_Comments, Comments and Labels.
Exact_Comments are the raw comments without any preprocessing
Comments are to be preprocessed
'''
def df():
	df = pd.DataFrame(columns = ['Exact_Comments','Comments', 'Labels', 'Embedding'])
	df['Exact_Comments'] = df['Comments'] = whole_df["Comments"]
	df["Labels"] = whole_df["Labels"]
	return df

