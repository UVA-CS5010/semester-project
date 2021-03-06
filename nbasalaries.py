"""
NBASalaries
Google Colab file located at:
https://colab.research.google.com/drive/1iYfVAFHi4aI7tiOoLBOOTCqaGAbhl_p7
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import plot


#######################################
## WNBA SALARY DATA WEBSCRAPING      ##
## 100 highest paid players for 2020 ##
## (data from spotrac.com)           ##
#######################################

URL = 'https://www.spotrac.com/wnba/rankings/average/'
page = requests.get(URL).text

soup = BeautifulSoup(page, 'lxml')

names= []
for name in soup.find_all('a', class_= 'team-name'):
  names.append(name.get_text())

teams=[]
for team in soup.find_all('div', class_= 'rank-position'):
  teams.append(team.get_text())

avg_salaries=[]
for salaries in soup.find_all('span', class_= 'info'):
  avg_salaries.append(salaries.get_text())

pos= []
pos2= []
for p in soup.find_all('td', class_= 'center small'):
  pos.append(p.get_text())
  
for i in pos:
    j = i.replace(' ','').replace('\n', '')
    pos2.append(j)

age= []
for v in pos2:
    if len(v) % 2 == 0:
        age.append(v)
positions= []
for v in pos2:
    if len(v) % 2 != 0:
        positions.append(v)

wnba= 'WNBA'

wnba_salaries = pd.DataFrame(
    {'players': names,
     'League' : wnba,
     'team': teams,
     'position': positions,
     'age': age,
     'average salary': avg_salaries
    })

wnba_salaries




#######################################
## NBA SALARY DATA WEBSCRAPING       ##
## 100 highest paid players for 2020 ##
## (data from spotrac.com)           ##
#######################################

URL = 'https://www.spotrac.com/nba/rankings/average/'
page = requests.get(URL).text

soup = BeautifulSoup(page, 'lxml')

men_names= []
for name in soup.find_all('a', class_= 'team-name'):
  men_names.append(name.get_text())

men_teams=[]
for team in soup.find_all('div', class_= 'rank-position'):
  men_teams.append(team.get_text())

men_avg_salaries=[]
for salaries in soup.find_all('span', class_= 'info'):
  men_avg_salaries.append(salaries.get_text())

men_age= []
for a in soup.find_all('td', class_= 'center xs-hide'):
  men_age.append(a.get_text())



men_pos= []
men_pos2= []
for p in soup.find_all('td', class_= 'center', ):
  men_pos.append(p.get_text())

for i in men_pos:
    j = i.replace(' ','').replace('\n', '')
    men_pos2.append(j)

digits= []
men_positions= []
for ps in men_pos2:
  if ps.isdigit():
    digits.append(ps)
  elif ps== '':
    digits.append(ps)
  else:
    men_positions.append(ps)

nba= 'NBA'

nba_salaries = pd.DataFrame(
    {'players': men_names,
     'league' : nba,
     'team': men_teams,
     'position': men_positions,
     'age': men_age,
     'average salary': men_avg_salaries
    })

nba_salaries




#########################################
## NBA STATS DATA WEBSCRAPING          ##
## Per-game avgs for 2020-21 season    ##
## (data from basketballreference.com) ##
#########################################

url = 'https://www.basketball-reference.com/leagues/NBA_2021_per_game.html'
html_doc = requests.get(url)

#parse the html from site:
parsed_html = BeautifulSoup(html_doc.content, 'html.parser')

#extract specific table we are interested in (per-game stats for each player):
table = parsed_html.find(id='per_game_stats')

##Header:
#Locate the table header, extract header values, and store in list:
table_header = table.find('thead') #html 'thead' element contains all header-related data
header_elements = table_header.find_all('th') #store all 'th' (header) elements from 'thead' element 

headers = [] #initialize empty list to later store headers
for header in header_elements:
    item = header.get_text().strip() #extract each header value (text)
    headers.append(item) #append each header value to list

headers

##Body:
#Locate table body, extract data values, and store in list:
table_body = table.find('tbody') #html 'tbody' element contains all body-related data
body_rows = table_body.find_all('tr') #store all 'tr' (row) elements from 'tbody' element 

rows = [] #initialize empty list to later store data rows
for row in body_rows:
    row_header = row.find('th').get_text() #extract the row's header (season)
    items = row.find_all('td') #extract data values from row
    row = [row_header] #initialize list  w/ row header. Will later populate w/ all values for the specific row
    for item in items: #iterate through the values of the row and store each in row list
        row.append(item.get_text())
    rows.append(row)
    

#Store in DataFrame, using rows for data and headers for column names:
nba_stats = pd.DataFrame(rows, columns = headers) 

#Convert nba_stats column names from list of tuples to list to be able to index dataframe by column names
#columnList = [x[0] for x in nba_stats.columns]
#nba_stats.columns = columnList

#Some players (who switched teams mid-year) are duplicated - have a total record and 
#individual records for each team they played for that year.
#Need to delete these duplicated records and include only the "total" entries
nba_stats = nba_stats.drop_duplicates(['Player'])

#Print player stats:
nba_stats





#########################################
## WNBA STATS DATA UPLOAD (CSV)        ##
## Per-game avgs for 2020 season       ##
## (data from basketballreference.com) ##
#########################################

#from google.colab import drive
#drive.mount('/content/drive')

#Load data from csv downloaded from basketballreference.com
#wnba_stats = pd.read_csv('/content/drive/MyDrive/CS5010/code/project/wnba_stats.csv')
wnba_stats = pd.read_csv('wnba_stats.csv')

#Again, need to delete the duplicated records and include only the "total" entries for players that were on multiple teams that year
wnba_stats = wnba_stats.drop_duplicates(['Player'])

#Print player stats:
wnba_stats




#########################################
## COMBINING STATS & SALARY DATA - NBA ##
#########################################

#Change column name to have column to perform join on:
sal_new_columns = ['PLAYER', 'LEAGUE', 'TEAM', 'POS', 'AGE', 'SALARY']
nba_salaries.columns  = sal_new_columns

stat_new_columns = nba_stats.columns.values
stat_new_columns[1] = 'PLAYER'
nba_stats.columns = stat_new_columns


#Perform join on 'PLAYER' column from both datasets:
nba_df = pd.merge(nba_stats, nba_salaries, on='PLAYER')

#Remove repeated or needless columns:
nba_df.drop(['Rk', 'Tm', 'Pos', 'Age'], axis=1, inplace=True)

nba_df['salary_float'] = nba_df['SALARY'].str.strip('$').replace(',','', regex=True).astype(int) #create column of salaries as float values (to be used for sorting & math)

nba_df.sort_values('salary_float', ascending=False)





##########################################
## COMBINING STATS & SALARY DATA - WNBA ##
##########################################
wnba_stats.columns
#Change column name to have column to perform join on:
w_sal_new_columns = ['PLAYER', 'LEAGUE', 'TEAM', 'POS', 'AGE', 'SALARY']
wnba_salaries.columns  = w_sal_new_columns

w_stat_new_columns = wnba_stats.columns.values
w_stat_new_columns[0] = 'PLAYER'
wnba_stats.columns = w_stat_new_columns

#Perform join on 'PLAYER' column from both datasets:
wnba_df = pd.merge(wnba_stats, wnba_salaries, on='PLAYER')

#Remove repeated or needless columns:
wnba_df.drop(['Team', 'Pos', 'G.1'], axis=1, inplace=True)

wnba_df['salary_float'] = wnba_df['SALARY'].str.strip('$').replace(',','', regex=True).astype(int) #create column of salaries as float values (to be used for sorting & math)

wnba_df.sort_values('salary_float', ascending=False)




#############################
## COMBINE NBA & WNBA DATA ##
#############################

all_df = pd.concat([nba_df, wnba_df]) #combine NBA & WNBA data
all_df


"""
#Data Comparisons & Analysis:
Questions/Goals to Consider:
- Is there a statistically significant difference in salaries b/w NBA and WNBA players? What about in context of salary cap? league revenue?
- Which statistics are the best predictors of an NBA player???s salary? WNBA player's salary?
  - Correlations b/w salary and any specific stats? look at visualizations?
- Which ages, positions are best predictors of NBA/WNBA salary?
- Determine which players have been overvalued/undervalued based on stats & salary
- Predict NBA salaries based on MLR model ??? take in user input for each of the predictors, return predicted salary
"""

##CONSTANTS (may want to feed these in from somewhere later - just hardcoded for now):
nba_salary_cap = 109140000
wnba_salary_cap = 1300000
nba_revenue = 7400000000
wnba_revenue = 60000000


##Aggregate Data Collection/Querying:
nba_avg_sal = np.mean(nba_df['salary_float'][0:50]) #calculate avg salary among 50 highest paid players
nba_top10 = np.sum(nba_df['salary_float'][0:10]) #calculate combined salary of 10 highest paid players
nba_avg_sal_pct_cap = round(((nba_avg_sal / nba_salary_cap) * 100), 2) #calculate avg salary as % of cap (nba)
nba_avg_sal_pct_rev = round(((nba_avg_sal / nba_revenue) * 100), 2) #calculate avg salary as % of revenue (nba)
nba_top10_pct_rev = round(((nba_top10 / nba_revenue) * 100), 2) #calculate salary of 10 highest paid players as % of league's revenue  

wnba_avg_sal = np.mean(wnba_df['salary_float'][0:50]) #calculate avg salary among 50 highest paid players
wnba_top10 = np.sum(wnba_df['salary_float'][0:10]) #calculate combined salary of 10 highest paid players
wnba_avg_sal_pct_cap = round(((wnba_avg_sal / wnba_salary_cap) * 100), 2) #calculate avg salary as % of cap (nba)
wnba_avg_sal_pct_rev = round(((wnba_avg_sal / wnba_revenue) * 100), 2) #calculate avg salary as % of revenue (nba)
wnba_top10_pct_rev = round(((wnba_top10 / wnba_revenue) * 100), 2) #calculate salary of 10 highest paid players as % of league's revenue  

print("NBA aggregate stats:")
print("Avg salary of top 50 NBA players as % of team salary cap: " + str(nba_avg_sal_pct_cap) + "%")
print("Avg salary of top 50 NBA players as % of league's revenue: " + str(nba_avg_sal_pct_rev) + "%")
print("10 highest paid NBA players' combined salary as % of league's revenue: " + str(nba_top10_pct_rev) + "%")
print()
print("WNBA aggregate stats:")
print("Avg salary of top 50 WNBA players as % of team salary cap: " + str(wnba_avg_sal_pct_cap) + "%")
print("Avg salary of top 50 WNBA players as % of league's revenue: " + str(wnba_avg_sal_pct_rev) + "%")
print("10 highest paid WNBA players' combined salary as % of league's revenue: " + str(wnba_top10_pct_rev) + "%")




#########################
## DATA VISUALIZATIONS ##
#########################

## PLAYER'S SALARY VS AGE
nba_df['Age'] = pd.to_numeric(nba_df["Age"]) #Convert age to numeric value (not string)
wnba_df['age'] = pd.to_numeric(wnba_df["age"]) #Convert age to numeric value (not string)

nba_age_fig = px.scatter(nba_df, x='Age', y='salary_float')
plot(nba_age_fig)

wnba_age_fig = px.scatter(wnba_df, x='age', y='salary_float', trendline="lowess")
plot(wnba_age_fig)

all_df['Age'] = pd.to_numeric(all_df["Age"]) #Convert age to numeric value (not string)
age_fig = px.scatter(all_df, x='Age', y='salary_float')
plot(age_fig)
