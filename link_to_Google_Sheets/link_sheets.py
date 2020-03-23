import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pprint import pprint
import pandas as pd

scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope)
client = gspread.authorize(creds)
sheet = client.open("SD_Annotation").sheet1

data = sheet.get_all_records()
# pprint(data)

whole_df = pd.DataFrame.from_dict(data)
# Comments = whole_df["Comments"]
# Labels = whole_df["Labels"]
# print(Comments)

#create dataframe with two columns comments and labels
df = pd.DataFrame(columns = ['Comments, Labels'])
df['Comments'] = whole_df["Comments"]
df["Labels"] = whole_df["Labels"]