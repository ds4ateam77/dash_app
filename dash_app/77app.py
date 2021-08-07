import dash
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import os
import base64
from sklearn import datasets
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import export_graphviz
import pydot
################################################################################################
# Load the data 
################################################################################################
#2019 Endorsements
endorsements_2019 = pd.read_csv('data/endorsements_2019_updated.csv')

#player sponsorship 2020 2021
player_sponsorships_2020_2021 = pd.read_csv('data/player_sponsorships_2020_2021.csv')
player_sponsorships_2020_2021 = player_sponsorships_2020_2021.replace(np.nan, 'No Subindustry', regex=True)

#player stats 2020 2021
player_stats_2020_2021 = pd.read_csv('data/2020-2021 NBA Player Stats.csv')

#2020 2021 player data combined
player_2020_2021_count_fg_stats_social_media = pd.read_csv('data/player_2020_2021_count_fg_stats_social_media_csv.csv')

#industry data to plot
player_sponsorships_2020_2021_industry = player_sponsorships_2020_2021.merge(player_stats_2020_2021, on='Player')
industry_fg = player_sponsorships_2020_2021_industry.groupby('Industry').mean().sort_values(by='PTS', ascending=False)['PTS']
industry_fg = pd.DataFrame(industry_fg)
industry_fg["PTS"] = round(industry_fg["PTS"], 1)
industry_toplot = industry_fg.reset_index()
#industry_toplot

#endorsements to plot
endorsements_2019_updated = pd.read_csv('data/endorsements_2019_updated.csv')
endorsements_toplot = pd.DataFrame(endorsements_2019_updated.groupby('Tenure (Years)').mean()['Endorsment (millions)'])
endorsements_toplotindex = endorsements_toplot.reset_index()

#endorsements draft
endorsements_2019_draft = \
pd.DataFrame(endorsements_2019_updated.groupby('Overall Draft Pick').mean()['Endorsment (millions)']).reset_index()

#subindustry data to plot
subindustry_fg = player_sponsorships_2020_2021_industry.groupby('Subindustry').mean().sort_values(by='Age', ascending=False)['Age']
subindustry_fg = pd.DataFrame(subindustry_fg)
subindustry_fg["Age"] = round(subindustry_fg["Age"], 1)
subindustry_toplot = subindustry_fg.reset_index()
#subindustry_toplot

#Industry
#industry = pd.read_csv('data/Industry.csv')
#industry.fillna("No Subindustry", inplace=True)

#Team Valuations
team_valuations = pd.read_csv('data/Team_Valuations_City (1).csv')
team_valuations = team_valuations.rename(columns={"Item": "Team", "Attribute" : "Year", "Value" : "Valuation (in millions)"})
team_valuations = team_valuations[team_valuations["Team"].notna()]
team_valuations["Year"] = team_valuations["Year"].astype("int")
add_zeroes = "000000"
team_valuations["Valuation (in millions)"] = team_valuations["Valuation (in millions)"].map(str) + add_zeroes
team_valuations["Valuation (in millions)"] = team_valuations["Valuation (in millions)"].str.replace(".", "")
convert_dict = {"City": str,
               "Team": str,
               "Valuation (in millions)": int,
               "LAT": int,
               "LONG" : int
               }
team_valuations_ = team_valuations.copy()
team_valuations_ = team_valuations_.astype(convert_dict)
team_valuations_['Team'] = team_valuations_['Team'].astype(str)

#salary data
salaries_endorsements = pd.read_csv('data/Salaries and Endorsements.csv')
salaries_endorsements = salaries_endorsements.dropna(subset=['Player'])
salaries_endorsements = salaries_endorsements.rename(columns={"Oncourt (2017)": "Salary (2017)", "Offcourt (2017)": "Endorsements (2017)", "Salaray (2016)" : "Salary (2016)"})
salaries_endorsements_2021 = salaries_endorsements.dropna(subset=['Salary'])
salaries_endorsements_2017 = salaries_endorsements.dropna(subset=['Salary (2017)'])
salaries_endorsements_2016 = salaries_endorsements.dropna(subset=['Salary (2016)'])
salaries_endorsements_2015 = salaries_endorsements.dropna(subset=['Salary (2015)'])

#Attendance
attendance = pd.read_csv('data/Attendance.csv')
attendance = pd.melt(attendance, id_vars=['Team'])
attendance = attendance.rename(columns={"variable": "Year", "value": "Average Attendance"})
team_list = attendance.Team.unique()

image_filename = 'Landing Page app.png'
def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

#Influence Index
influence = pd.read_csv('data/influence_d.csv')
influence["Unnamed: 0"] = influence["Unnamed: 0"].astype(int)
influence.dtypes
influence = influence.rename(columns={"Unnamed: 0": "Category"})
influence = pd.DataFrame(influence)
#influence = influence.set_index("Category")
influence["Year"] = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
influence = influence.set_index("Year")
influence = influence.drop(columns=['Category'])
influence = influence.reset_index()
influence_long = pd.melt(influence, id_vars=['Year'])

#Followers
player_2020_2021_stats_updated = pd.read_csv('data/player_2020_2021_count_fg_stats_social_media_csv.csv')
player_2020_2021_stats_updated.columns
followers = player_2020_2021_stats_updated[["Player", "Twitter Followers (number)", "Instagram Presence (number)", "Facebook Followers (number)"]]
followers = followers.rename(columns={"Instagram Presence (number)": "Instagram Followers", "Twitter Followers (number)": "Twitter Followers",
                 "Facebook Followers (number)" : "Facebook Followers"})
followers["Twitter Followers"] = followers["Twitter Followers"].replace(np.nan, 'No Twitter Account', regex=True)
followers["Instagram Followers"] = followers["Instagram Followers"].replace(np.nan, 'No Instagram Account', regex=True)
followers["Facebook Followers"] = followers["Facebook Followers"].replace(np.nan, 'No Facebook Account', regex=True)

#player value to cost
endorsements_value_cost_player = endorsements_2019[['Player','Value to cost']]

#sports view
sports_view = pd.read_csv("data/sports_views_tools.csv")
sports_views = sports_view[['year','used facebook to follow/post about this game',
       'used instagram to follow/post about this game',
       'used twitter to follow/post about this game',
                            'used google+ to follow/post about this game',
       'watch sports-related video on youtube',
       'used snapchat to follow/post about this game',
       'used reddit to follow/post about this game',
       'used pinterest to follow/post about this game',
       'used tumblr to follow/post about this game']]
sports_views = sports_views.rename(columns={"used facebook to follow/post about this game": "Used Facebook to follow/post about this game", "used instagram to follow/post about this game": "Used Instagram to follow/post about this game",
                 "used twitter to follow/post about this game" : "Used Twitter to follow/post about this game",
                                           "used snapchat to follow/post about this game": "Used Snapchat to follow/post about this game", "watch sports-related video on youtube": "Watch sports-related video on Youtube",
                 "used google+ to follow/post about this game" : "Used Google+ to follow/post about this game",
                                           "used reddit to follow/post about this game": "Used Reddit to follow/post about this game", "used tumblr to follow/post about this game": "Used Tumblr to follow/post about this game",
                 "used pinterest to follow/post about this game" : "Used Pinterest to follow/post about this game"})
sports_views = pd.melt(sports_views, id_vars=['year'])
sports_views = sports_views.rename(columns={"year": "Year", "variable": "Survey",
                 "value" : "Number of Fans"})
tv_views = sports_view[['year', 'total mobile owners who viewed nba game (s) on tv',
       'number used for game-viewing-related activities related to game viewing on tv',
       #'% using for activities related to game (s) viewing on tv',
       'total nba fans who watched on tv and own a smartphone and/or tablet',
       'use smartphone and/or tablet for game-viewing-related purpose while viewing games on tv',]]
tv_views = tv_views.rename(columns={"total mobile owners who viewed nba game (s) on tv": "Total mobile owners who viewed NBA game(s) on TV", "number used for game-viewing-related activities related to game viewing on tv": "# Used for game-viewing-related activities related to game viewing on TV",
                                          "% using for activities related to game (s) viewing on tv": "% Using for activities related to game(s) viewing on TV", "total nba fans who watched on tv and own a smartphone and/or tablet": "Total NBA fans who watched on TV and own a smartphone and/or tablet",
                "use smartphone and/or tablet for game-viewing-related purpose while viewing games on tv" : "Used smartphone and/or tablet for game-viewing-related purpose while viewing games on TV",
                                        })
tv_views = pd.melt(tv_views, id_vars=['year'])
tv_views = tv_views.rename(columns={"year": "Year", "variable": "Survey",
                 "value" : "Number of Fans"})

#endorsements by team
endorsements_2019_value = pd.DataFrame(endorsements_2019['Team'].value_counts())
endorsements_2019_values = endorsements_2019_value.reset_index()
endorsements_2019_values_r = endorsements_2019_values.rename(columns={"index": "Team", "Team": "Number of Endorsements"})

#value to cost by team
endorsements_value_cost_team = endorsements_2019[['Team','Value to cost']]
endorsements_value_cost_teams = endorsements_value_cost_team.groupby('Team').mean().sort_values(by='Value to cost', ascending=False)
endorsements_value_cost_team = endorsements_value_cost_teams.reset_index()

#popularity 
#brand_share = pd.read_csv('data/BrandShareofMarket.csv')
#sports_drink = brand_share.iloc[119:129]
#sports_drink.reset_index()
#sports_drink.columns = ["Sports Drink", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
#there are columns with nothing in them so I'm dropping them
#sports_drink = sports_drink.drop(["2010", "2011", "2012","2013", "2014", "2020"], axis =1)
#sports_drink["2019"] = sports_drink["2019"].replace("---", "").replace('',np.NaN)
#sports_drink["2015"] = sports_drink["2015"].astype(float)
#sports_drink["2016"] = sports_drink["2016"].astype(float)
#sports_drink["2017"] = sports_drink["2017"].astype(float)
#sports_drink["2018"] = sports_drink["2018"].astype(float)
#sports_drink["2019"] = sports_drink["2018"].astype(float)
#sports_drink = sports_drink.set_index("Sports Drink")
#sports_drink_melt = sports_drink.reset_index()
#sports_drink_melt = pd.melt(sports_drink_melt, id_vars=['Sports Drink'])
#sports_drink_melt = sports_drink_melt.rename(columns={"variable": "Year", "value" : "Percentage Popularity"})

#bball_shoes = brand_share.iloc[180:194]
#bball_shoes.reset_index()
#bball_shoes.columns = ["Basketball Shoes", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
#bball_shoes = bball_shoes.drop(["2010", "2011", "2012","2013", "2014", "2020"], axis =1)
#bball_shoes_f = bball_shoes.copy()
#bball_shoes_f["2015"] = bball_shoes_f["2015"].replace("---", "").replace('', np.NaN)
#bball_shoes_f["2016"] = bball_shoes_f["2016"].replace("---", "").replace('', np.NaN)
#bball_shoes_f["2017"] = bball_shoes_f["2017"].replace("---", "").replace('', np.NaN)
#bball_shoes_f["2018"] = bball_shoes_f["2018"].replace("---", "").replace('', np.NaN)
#bball_shoes_f["2019"] = bball_shoes_f["2019"].replace("---", "").replace('', np.NaN)
#bball_shoes_f = bball_shoes_f.replace(',','',regex=True)
#bball_shoes_f["2015"] = bball_shoes_f["2015"].astype(float)
#bball_shoes_f["2016"] = bball_shoes_f["2016"].astype(float)
#bball_shoes_f["2017"] = bball_shoes_f["2017"].astype(float)
#bball_shoes_f["2018"] = bball_shoes_f["2018"].astype(float)
#bball_shoes_f["2019"] = bball_shoes_f["2019"].astype(float)
#bball_shoes_f = bball_shoes_f.set_index("Basketball Shoes")
#bball_shoes_melt = bball_shoes_f.reset_index()
#bball_shoes_melt = pd.melt(bball_shoes_melt, id_vars=['Basketball Shoes'])
#bball_shoes_melt = bball_shoes_melt.rename(columns={"variable": "Year", "value" : "Percentage Popularity"})

#insurance = brand_share.iloc[268:304]
#insurance.reset_index()
#insurance.columns = ["Insurance", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
#insurance = insurance.drop(["2010", "2011", "2012","2013", "2014", "2020"], axis =1)
#insurance_f = insurance.copy()
#insurance_f["2015"] = insurance_f["2015"].replace("---", "").replace('', np.NaN)
#insurance_f["2016"] = insurance_f["2016"].replace("---", "").replace('', np.NaN)
#insurance_f["2017"] = insurance_f["2017"].replace("--", "").replace('', np.NaN)
#insurance_f["2017"] = insurance_f["2017"].replace(" \t---", "").replace('', np.NaN)
#insurance_f["2017"] = insurance_f["2017"].replace(" ---", "").replace('', np.NaN)
#insurance_f["2017"] = insurance_f["2017"].replace("---", "").replace('', np.NaN)
#insurance_f["2018"] = insurance_f["2018"].replace("---", "").replace('', np.NaN)
#insurance_f["2019"] = insurance_f["2019"].replace("---", "").replace('', np.NaN)
#insurance_f = insurance_f.replace(',','',regex=True)
#insurance_f["2015"] = insurance_f["2015"].astype(float)
#insurance_f["2016"] = insurance_f["2016"].astype(float)
#insurance_f["2017"] = insurance_f["2017"].astype(float)
#insurance_f["2018"] = insurance_f["2018"].astype(float)
#insurance_f["2019"] = insurance_f["2019"].astype(float)
#insurance_f = insurance_f.set_index("Insurance")
#insurance_melt = insurance_f.reset_index()
#insurance_melt = pd.melt(insurance_melt, id_vars=['Insurance'])
#insurance_melt = insurance_melt.rename(columns={"variable": "Year", "value" : "Percentage Popularity"})

#cell = brand_share.iloc[200:216]
#cell.reset_index()
#cell.columns = ["Cell Phone", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
#cell = cell.drop(["2010", "2011", "2012","2013", "2014", "2020"], axis =1)
#cell_f = cell.copy()
#cell_f["2015"] = cell_f["2015"].replace("---", "").replace('', np.NaN)
#cell_f["2016"] = cell_f["2016"].replace("---", "").replace('', np.NaN)
#cell_f["2017"] = cell_f["2017"].replace("--", "").replace('', np.NaN)
#cell_f["2017"] = cell_f["2017"].replace(" \t---", "").replace('', np.NaN)
#cell_f["2017"] = cell_f["2017"].replace(" ---", "").replace('', np.NaN)
#cell_f["2017"] = cell_f["2017"].replace("---", "").replace('', np.NaN)
#cell_f["2018"] = cell_f["2018"].replace("---", "").replace('', np.NaN)
#cell_f["2019"] = cell_f["2019"].replace("---", "").replace('', np.NaN)
#cell_f = cell_f.replace(',','',regex=True)
#cell_f["2015"] = cell_f["2015"].astype(float)
#cell_f["2016"] = cell_f["2016"].astype(float)
#cell_f["2017"] = cell_f["2017"].astype(float)
#cell_f["2018"] = cell_f["2018"].astype(float)
#cell_f["2019"] = cell_f["2019"].astype(float)
#cell_f = cell_f.set_index("Cell Phone")
#cell_melt = cell_f.reset_index()
#cell_melt = pd.melt(cell_melt, id_vars=['Cell Phone'])
#cell_melt = cell_melt.rename(columns={"variable": "Year", "value" : "Percentage Popularity"})

#phone_service = brand_share.iloc[238:262]
#phone_service.reset_index()
#phone_service.columns = ["Phone Service Provider", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
#phone_service = phone_service.drop(["2010", "2011", "2012","2013", "2014", "2020"], axis =1)
#phone_service_f = phone_service.copy()
#phone_service_f["2015"] = phone_service_f["2015"].replace("---", "").replace('', np.NaN)
#phone_service_f["2016"] = phone_service_f["2016"].replace("---", "").replace('', np.NaN)
#phone_service_f["2017"] = phone_service_f["2017"].replace("--", "").replace('', np.NaN)
#phone_service_f["2017"] = phone_service_f["2017"].replace(" \t---", "").replace('', np.NaN)
#phone_service_f["2017"] = phone_service_f["2017"].replace(" ---", "").replace('', np.NaN)
#phone_service_f["2017"] = phone_service_f["2017"].replace("---", "").replace('', np.NaN)
#phone_service_f["2018"] = phone_service_f["2018"].replace("---", "").replace('', np.NaN)
#phone_service_f["2019"] = phone_service_f["2019"].replace("---", "").replace('', np.NaN)
#phone_service_f = phone_service_f.replace(',','',regex=True)
#phone_service_f["2015"] = phone_service_f["2015"].astype(float)
#phone_service_f["2016"] = phone_service_f["2016"].astype(float)
#phone_service_f["2017"] = phone_service_f["2017"].astype(float)
#phone_service_f["2018"] = phone_service_f["2018"].astype(float)
#phone_service_f["2019"] = phone_service_f["2019"].astype(float)
#phone_service_f = phone_service_f.set_index("Phone Service Provider")
#phone_service_melt = phone_service_f.reset_index()
#phone_service_melt = pd.melt(phone_service_melt, id_vars=['Phone Service Provider'])
#phone_service_melt = phone_service_melt.rename(columns={"variable": "Year", "value" : "Percentage Popularity"})


modeling_roi = pd.read_csv('data/endorsements_2019_updated.csv')
modeling_roi
modeling_roi = modeling_roi.dropna()
modeling_roi = modeling_roi.reset_index(drop=True)
modeling_roi = modeling_roi.rename(columns={"Value to cost": "Value to Cost", "Win_shares": "Win Shares",
                 "Minutes_played" : "Minutes Played"})

def color_clusters_g_vc(data, x_col, y_col, clusters):
    iris = datasets.load_iris()
    x=data[[x_col]]
    y=data[[y_col]]
    kmeans = KMeans(n_clusters=clusters)
    kmeans = kmeans.fit(data[[x_col,y_col]])
    labels = kmeans.predict(data[[x_col,y_col]])
    
    fig = px.scatter(modeling_roi, x=x_col, y=y_col, color = labels, hover_data = [x_col, y_col, "Player"], color_continuous_scale= "Bluered_r", title = x_col + ' vs. ' + y_col)
    return fig

sponsorship_model = player_2020_2021_stats_updated.copy()
def color_clusters_ten_sp(data, x_col, y_col, clusters):
    iris = datasets.load_iris()
    x=data[[x_col]]
    y=data[[y_col]]
    kmeans = KMeans(n_clusters=clusters)
    kmeans = kmeans.fit(data[[x_col,y_col]])
    labels = kmeans.predict(data[[x_col,y_col]])
    
    fig = px.scatter(data, x=x_col, y=y_col, color = labels, hover_data = [x_col, y_col, 'Player'], color_continuous_scale= "Bluered_r", title = x_col + ' vs. ' + y_col)
    return fig

data_ts = sponsorship_model[["Tenure (Years)", "Sponsorship", "Player"]].dropna()
data_os = sponsorship_model[["Overall Pick", "Sponsorship", "Player"]].dropna()
data_2pas = sponsorship_model[["2PA", "Sponsorship","Player"]].dropna()
data_fts = sponsorship_model[["FT", "Sponsorship", "Player"]].dropna()
################################################################################################
# plotly vizulaizations
################################################################################################

####
# endorsement locations
endorsement_location = endorsements_2019[['Endorsment (millions)','Division']]
endorsement_location_fig = px.bar(endorsement_location, 
             x='Division', y='Endorsment (millions)', 
             color_discrete_sequence =[('blue')]*len(endorsement_location))

####
#value cost location
value_cost_location = endorsements_2019[['Value to cost','Division']]
value_cost_location_fig = px.bar(value_cost_location, 
             x='Division', y='Value to cost', 
             color_discrete_sequence =[('red')]*len(endorsement_location),
             title='Value to Cost by NBA Division')


#player sponsorship compared to fg average
sponsorship_fg_fig = px.scatter(player_2020_2021_count_fg_stats_social_media, x="FG", y="Sponsorship", color="Sponsorship",
                 size='FG', hover_data=['Player'],color_continuous_scale='rdbu')


#industry compared to fg average
industry_fg_fig = px.scatter(player_2020_2021_count_fg_stats_social_media, x="FG", y="Industry", color="Industry",
                 size='FG', hover_data=['Player'],color_continuous_scale='rdbu', title = "NBA Player Industry Compared to FG Average")


#sub industry compared to fg average
subindustry_fg_fig = px.scatter(player_2020_2021_count_fg_stats_social_media, x="FG", y="Subindustry", color="Subindustry",
                 size='FG', hover_data=['Player'],color_continuous_scale='rdbu')


#Average Points by Sponsorship Industry
#avg_pts_industry_fig = px.bar(industry_toplot, 
             #x='Industry', y='PTS',color_discrete_sequence =[('blue')]*len(industry_toplot),
             #title='Average Points by Sponsorship Industry')

#Average Points by Sponsorship Subindustry
#avg_pts_subindustry_fig = px.bar(subindustry_toplot, 
             #x='Age', y='Subindustry',
             #color_discrete_sequence =[('red')]*len(subindustry_toplot),
             #title='Average Age by Sponsorship Subindustry')

#Industry Tree Map
#industry_tree_fig = px.treemap(industry, path=['Industry', 'Subindustry'], 
                 #values='total', color='Industry')
#industry_tree_fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
#industry_tree_fig.data[0].hovertemplate = '%{label}<br>%{value}'

#player sponsorship and tenure
sponsor_tenure = px.scatter(player_2020_2021_count_fg_stats_social_media, x="Tenure (Years)", y="Sponsorship", hover_data=['Player'],color_continuous_scale='rdbu',)
                #title = "NBA Player Sponsorships by Tenure")

endorsement_tenure = px.scatter(endorsements_toplotindex, x="Tenure (Years)", y="Endorsment (millions)", hover_data=['Endorsment (millions)'],color_continuous_scale='rdbu', 
                title = "NBA Endorsement (millions) by Tenure (Years)")
# Team valuations bubble map
bubble_map = px.scatter_geo(team_valuations_, 
                     lat="LAT", 
                     lon= "LONG", 
                     color="Team",
                     hover_name="Team",
                     color_discrete_sequence = ["blue", "red", "yellow"],
                     #hover_data = {'Team': True, 'City': True, 'LAT':False, 'LONG':False, 'Year': True, 'Valuation (in millions)': True},
                     size="Valuation (in millions)",
                     animation_frame="Year",
                     scope = "usa"
                     )

# Player Sonsorship Table
headerColor = 'grey'
rowEvenColor = 'blue'
rowOddColor = 'red'

sponsor_table = go.Figure(data=[go.Table(
  header=dict(
    values=['<b>Player</b>','<b>Sponsorship</b>','<b>Industry</b>','<b>Subindustry</b>'],
    line_color='darkslategray',
    fill_color=headerColor,
    align=['left','center'],
    font=dict(color='white', size=12)
  ),
  cells=dict(
    values=[
    player_sponsorships_2020_2021["Player"],
    player_sponsorships_2020_2021["Sponsorship"],
    player_sponsorships_2020_2021["Industry"],
    player_sponsorships_2020_2021["Subindustry"]],
    line_color='darkslategray',
    # 2-D list of colors for alternating rows
    fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor]*35],
    align = ['center', 'center', 'center', 'center'],
    font = dict(color = 'white', size = 11)
    ))
])

# Social Media Table

social_table = go.Figure(data=[go.Table(
  header=dict(
    values=['<b>Player</b>','<b>Twitter Followers</b>','<b>Instagram Followers</b>','<b>Facebook Followers</b>'],
    line_color='darkslategray',
    fill_color=headerColor,
    align=['left','center'],
    font=dict(color='white', size=12)
  ),
  cells=dict(
    values=[
    followers["Player"],
    followers["Twitter Followers"],
    followers["Instagram Followers"],
    followers["Facebook Followers"]],
    line_color='darkslategray',
    # 2-D list of colors for alternating rows
    fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor]*35],
    align = ['center', 'center', 'center', 'center'],
    font = dict(color = 'white', size = 11)
    ))
])

# Value to cost
value_to_cost = px.bar(endorsements_value_cost_player, x='Player', y='Value to cost', color_discrete_sequence = ["blue"])

# Stat list
statlist = ["G", "FG%", "3P%", "2P%", "FT%"]

#endrosements draft
endorsements_draft = px.scatter(endorsements_2019_draft, x="Overall Draft Pick", y="Endorsment (millions)", hover_data=['Endorsment (millions)'],color_continuous_scale='rdbu', 
                title = "NBA Endorsement (millions) by Overall Draft Pick")

#fan viewing
fan_views = px.bar(tv_views, x="Year", y="Number of Fans",
             color='Survey', barmode='group',
             color_discrete_sequence = ["blue", "red", "lightblue", "salmon"],
             height=400)

#endorsements by team
team_endorsements = px.bar(endorsements_2019_values_r, x="Team", y="Number of Endorsements",
             color_discrete_sequence=["blue"],
             title='Number of Endorsements by Team')

#value to cost by team
team_value_to_cost = px.bar(endorsements_value_cost_team, x="Team", y="Value to cost",
             color_discrete_sequence=["red"],
             title='Value to Cost by Team')
################################################################################################
# create dash app 
################################################################################################
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.JOURNAL], suppress_callback_exceptions=True)
server = app.server
app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dbc.Navbar(
            children=[
                html.A(
                    # Use row and col to control vertical alignment of logo / brand
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H2('Team 77'),
                                style={'color': "#18408b"}
                            ),
                            dbc.Col(
                                dbc.NavbarBrand(
                                    "Athlete Sponsorships", className="ml-2",
                                    style={'color': "#18408b"}
                                )
                            ),
                        ],
                        no_gutters=True,
                        className="ml-auto flex-nowrap mt-3 mt-md-0",
                        align="center",
                    ),
                    href=app.get_relative_path("/"),
                ),
                dbc.Row(
                    children=[
                        dbc.NavLink("Home", href=app.get_relative_path("/")),
                        dbc.NavLink("Player Stats", href=app.get_relative_path("/player")),
                        dbc.NavLink("Team & League Stats", href=app.get_relative_path("/team")),
                        dbc.NavLink("Market Stats", href=app.get_relative_path("/market")),
                        dbc.NavLink("Discovering Sponsorships", href=app.get_relative_path("/discovering")),
                    ],
                    style={"paddingLeft": "500px", 'color': '#c00001', 'font-weight': 'bold'},
                ),
            ], color='#ffffff'
        ),
        html.Div(id="page-content")
    ])
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page_content(pathname):
    path = app.strip_relative_path(pathname)
    if not path:
        return html.Div([
            html.Img(src=b64_image(image_filename), style={"width": "75%", 
                                                           "object-fit": "fill",
                                                           "display": "block",
                                                           "margin-left": "auto",
                                                           "margin-right": "auto"})
        ], style={"background-color": "#18408b"})
    elif path == "player":
         return dbc.Container(
            [
                dbc.Row([
                    dbc.Col(html.H1("PLAYER STATS"), className='text-center', style={'color': '#ffffff'})
                ]),
                dbc.Row([
                    dbc.Col(html.H1("    "))
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            html.H5("NBA Players & Sponsorships", className="card-title text-center"),
                            dbc.CardBody([
                                dcc.Graph(id="sponsor_table", figure= sponsor_table)
                            ])
                        ], color='#7f7f7f', inverse=True)
                    ]),
                    dbc.Col([
                        dbc.Card([
                            html.H5("Top Earners (Salary and Endorsements) Per Year", className="card-title text-center"),
                            dbc.CardBody([
                                dcc.Graph(id="salary_endorsements"),
                                dcc.Slider(id= "salary_sliders",
                                    min=2015, 
                                    max=2021,
                                    step=None,
                                    marks={
                                        2015: '2015',
                                        2016: '2016',
                                        2017: '2017',
                                        2021: '2021'
                                    },value=2015)
                            ])
                        ], color='#c00001', inverse=True)
                    ])
                ]),
                dbc.Row([
                    dbc.Col(html.H1("    "))
                ]),
                dbc.Row([
                   dbc.Col([
                        dbc.Card([
                            html.H5("NBA Player Sponsorship Compared to FG Average", className="card-title text-center"),
                            html.Div(
                                [
                                    dbc.RadioItems(id="radios", className="btn-group", labelClassName="btn btn-secondary", labelCheckedClassName="active",
                                                   options=[
                                                       {"label": "Sponsorship", "value": "Sponsorship"},
                                                       {"label": "Industry", "value": "Industry"},
                                                       {"label": "Subindustry", "value": "Subindustry"}
                                                   ], value="Sponsorship")
                                ], className="radio-group"),
                            dbc.CardBody([
                                dcc.Graph(id='FG_plot')
                            ]),                             
                        ], color='#c00001', inverse=True)
                    ], width={'size': 6}),
                    dbc.Col([
                        dbc.Card([
                            html.H5("Endorsements based on Tenure and Draft picks", className="card-title text-center"),
                            html.Div(
                                [
                                    dbc.RadioItems(id="endorsement_radios", className="btn-group", labelClassName="btn btn-secondary", labelCheckedClassName="active",
                                                   options=[
                                                       {"label": "Tenure", "value": "Tenure (Years)"},
                                                       {"label": "Draft Pick", "value": "Overall Pick"}
                                                   ], value="Tenure (Years)")
                                ], className="radio-group"),
                            dbc.CardBody([
                              dcc.Graph(id='endorsement_plots')  
                            ])
                        ], color='#4472c4', inverse=True)
                    ], width={'size': 6}),
                ]),
                dbc.Row([
                    dbc.Col(html.H1("    "))
                ]),
                 dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            html.H5("NBA Player's Social Media", className=('card-title text-center')),
                            dbc.CardBody([
                                dcc.Graph(id="social_table", figure = social_table)
                            ])
                        ], color='#7f7f7f', inverse=True)
                    ]),
                     dbc.Col([
                         dbc.Card([
                             html.H5("NBA Player's Social Media Value to Cost", className=('card-title text-center')),
                             dbc.CardBody([
                             dcc.Graph(id="value_to_cost", figure=value_to_cost)
                             ])                         
                         ], color='#c00001', inverse=True)
                     ])
                ]),
                dbc.Row([
                    dbc.Col(html.H1("    "))
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            html.H5("NBA Player Stat Matrix", className=('card-title text-center')),
                            dbc.CardBody([
                                dcc.Dropdown(id="stat_dropdown", options=[{"label": x, "value": x} 
                 for x in statlist], value=statlist, multi=True, style={"color": "black"}),
                                dcc.Graph(id="stat_matrix")
                            ])
                        ], color='#c00001', inverse=True)
                    ])
                ])
            ], style={'background-color': "#18408b"}, fluid=True)
    elif path == "team":
        return dbc.Container(
            [
                dbc.Row([
                    dbc.Col(html.H1("TEAM & LEAGUE STATS"), className='text-center', style={'color': '#ffffff'})
                ]),
                dbc.Row([
                    dbc.Col(html.H1("    "))
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            html.H5("AVERAGE ATTENDANCE BY TEAM", className=('card-title text-center')),
                            dbc.CardBody([
                                dcc.Dropdown(id='team_dropdown', options=[{"label": x, "value": x}
                                                                          for x in team_list],
                                            multi=True, placeholder='Select Teams', style={"color": "black"}),
                                dcc.Graph(id='attendance_plot')
                            ])
                        ], color='#4472c4', inverse=True)
                    ]),
                    dbc.Col([
                        dbc.Card([
                            html.H5("Team Valuation (in millions)", className=('card-title text-center')),
                            dbc.CardBody([
                                dcc.Graph(id='bubble map', figure=bubble_map)
                            ])
                        ], color='#c00001', inverse=True)
                    ])
                ]),
                dbc.Row([
                    dbc.Col(html.H1("    "))
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div(
                                [
                                    dbc.RadioItems(id="endorsement_milli_radios", className="btn-group", labelClassName="btn btn-secondary", labelCheckedClassName="active",
                                                   options=[
                                                       {"label": "Tenure", "value": "Tenure (Years)"},
                                                       {"label": "Draft Pick", "value": "Overall Pick"}
                                                   ], value="Tenure (Years)")
                                ], className="radio-group"),
                                dcc.Graph(id="endorsement_graph", figure = endorsement_tenure)
                            ])
                        ], color='#c00001', inverse=True)
                    ]),
                  dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div(
                                [
                                    dbc.RadioItems(id="team_radios", className="btn-group", labelClassName="btn btn-secondary", labelCheckedClassName="active",
                                                   options=[
                                                       {"label": "Endorsements", "value": "endorsements"},
                                                       {"label": "Value to Cost", "value": "value_to_cost"}
                                                   ], value="endorsements")
                                ], className="radio-group"),
                                dcc.Graph(id="team_bar", figure = team_endorsements)
                            ])
                        ], color='#4472c4', inverse=True)
                   ])  
                ]),
                dbc.Row([
                    dbc.Col(html.H1("    "))
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            html.H5("NBA Fan Viewing Activity", className=('card-title text-center')),
                            dbc.CardBody([
                                dcc.Graph(id="graph_3", figure = fan_views)
                            ])
                        ], color='#7f7f7f', inverse=True)
                   ])
                ])
            ], style={'background-color': "#18408b"}, fluid=True)  
    elif path == "market":
        return dbc.Container([
            dbc.Row([
                dbc.Col(html.H1("MARKET STATS"), className='text-center', style={'color': '#ffffff'})
            ]),
            dbc.Row([
                    dbc.Col(html.H1("    "))
                ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Brand Market Share By Industry", className=('card-title text-center')),
                            html.Div(
                                [
                                dbc.RadioItems(id="pop_radio", className="btn-group", labelClassName="btn btn-secondary",
                                               options=[
                                                   {"label": "Sports Drink", "value": "sports_drink"},
                                                   {"label": "Basketball Shoes", "value": "bball_shoes"},
                                                   {"label": "Insurance", "value": "insurance"},
                                                   {"label": "Cell Phone", "value": "cell_phone"},
                                                   {"label": "Service Provider", "value": "service"}
                                               ], value="sports_drink")
                                ], className="radio-group"),
                            dcc.Graph(id="pop_bar", figure={})
                        ])
                    ], color='#4472c4', inverse=True)
                ])
            ]),
            dbc.Row([
                    dbc.Col(html.H1("    "))
                ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Sports Sponsorship Influence Index", className=('card-title text-center')),
                            dcc.Graph(id="influence_index", figure=px.bar(influence_long, x="Year", y="value",color = "variable"))
                        ])
                    ], color='#7f7f7f', inverse=True)
                ])
            ]),
            dbc.Row([
                    dbc.Col(html.H1("    "))
                ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Fan Social Media Usage", className=('card-title text-center')),
                            dcc.Graph(id="fan_view", figure = px.bar(sports_views, x="Year", y="Number of Fans",
             color='Survey', barmode='group',
             color_discrete_sequence = ["blue", "red", "darkcyan","maroon", "cornflowerblue", "salmon", "deepskyblue", "mediumvioletred", "lightblue"],
             height=400)
                        )
                    ])
                ], color='#c00001', inverse=True)
            ])
            ]),
            
        ], style={'background-color': "#18408b"}, fluid=True)
    elif path == "discovering":
        return dbc.Container([
            dbc.Row([
                dbc.Col(html.H1("DISCOVERING SPONSORSHIPS"), className='text-center', style={'color': '#ffffff'})
            ]),
            dbc.Row([
                    dbc.Col(html.H1("    "))
                ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Value to Cost Clustering", className=('card-title text-center')),
                            html.Div(
                                [
                                dbc.RadioItems(id="cluster_radio", className="btn-group", labelClassName="btn btn-secondary",
                                               options=[
                                                   {"label": "Games Played", "value": "games"},
                                                   {"label": "Win Shares", "value": "win_shares"},
                                                   {"label": "Tenure", "value": "tenure"},
                                                   {"label": "Draft Pick", "value": "draft_pick"}
                                               ], value="games")
                                ], className="radio-group"),
                            dcc.Graph(id="clusters_value")
                        ])
                    ], color='#c00001', inverse=True)
                ])
            ]),
            dbc.Row([
                    dbc.Col(html.H1("    "))
                ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Numer of Sponsorships Clustering", className=('card-title text-center')),
                            html.Div(
                                [
                                dbc.RadioItems(id="cluster_sponsorship_radio", className="btn-group", labelClassName="btn btn-secondary",
                                               options=[
                                                   {"label": "Tenure", "value": "tenure"},
                                                   {"label": "Overall Pick", "value": "overall_pick"},
                                                   {"label": "2 Pointers Attempted", "value": "2pa"},
                                                   {"label": "Free Throws", "value": "ft"}
                                               ], value="tenure")
                                ], className="radio-group"),
                            dcc.Graph(id="clusters_sponsorship")
                        ])
                    ], color='#7f7f7f', inverse=True)
                ])
            ])
        ], style={'background-color': "#18408b"}, fluid=True)
    else:
        return "404"
############################################################################
# CALLBACKS
# FG plot
@app.callback(Output("FG_plot", "figure"), [Input("radios", "value")])
def update_figure(value):
    fig = px.scatter(player_2020_2021_count_fg_stats_social_media, x="FG", y=value, color="Sponsorship",
                 size='FG', hover_data=['Player'],color_continuous_scale='rdbu')
    return fig

# Endorsement plots
@app.callback(Output("endorsement_plots", "figure"), [Input("endorsement_radios", "value")])
def update_endorsement_figure(value):
    fig = sponsor_tenure = px.scatter(player_2020_2021_count_fg_stats_social_media, x=value, y="Sponsorship", hover_data=['Player'],color_continuous_scale='rdbu')
    return fig

#salary endorsements
@app.callback(Output("salary_endorsements", "figure"), [Input("salary_sliders", "value")])
def change_salary_chart(value):
    if value == 2015:
        return px.bar(salaries_endorsements_2015, x="Player", y=["Salary (2015)", "Endorsements (2015)"], 
             color_discrete_sequence=px.colors.qualitative.G10,
             title="2015 Salary and Endorsements")
    if value == 2016:
        return px.bar(salaries_endorsements_2016, x="Player", y=["Salary (2016)", "Endorsements (2016)"], 
             color_discrete_sequence=px.colors.qualitative.G10,
             title="2016 Salary and Endorsements")
    if value == 2017:
        return px.bar(salaries_endorsements_2017, x="Player", y=["Salary (2017)", "Endorsements (2017)"], 
             color_discrete_sequence=px.colors.qualitative.G10,
             title="2017 Salary and Endorsements")
    if value == 2021:
        return px.bar(salaries_endorsements_2021, x="Player", y=["Salary", "Endorsements"], 
             color_discrete_sequence=px.colors.qualitative.G10,
             title="2021 Salary and Endorsements")
    
#Attendance by team
@app.callback(Output("attendance_plot", "figure"), [Input('team_dropdown', 'value')])
def update_attendance_plot(team):
    mask = attendance.Team.isin(team)
    fig = px.line(attendance[mask], x="Year", y="Average Attendance", color='Team', title = "Average Attendance by Team")
    return fig
        
#Stat Matrix
@app.callback(Output("stat_matrix", "figure"), [Input("stat_dropdown", "value")])
def update_stat_matrix(stat):
    fig = px.scatter_matrix(player_stats_2020_2021,
    dimensions=stat,
    hover_data = ["Player", "G", "FG%", "3P%", "2P%", "FT%"],
    color_continuous_scale="bluered",
    color="Rk",
    title="NBA Player Statistics")
    return fig

@app.callback(Output("pop_bar", "figure"), [Input("pop_radio", "value")])
def change_industry_chart(value):
    if value == "sports_drink":
        return px.bar(sports_drink_melt, x='Sports Drink', y='Percentage Popularity', 
             color_discrete_sequence=px.colors.qualitative.G10,
             animation_frame = "Year",
             color = "Sports Drink",
             title="Sports Drink Popularity Over Time")
    if value == "bball_shoes":
        return px.bar(bball_shoes_melt, x='Basketball Shoes', y='Percentage Popularity', 
             color_discrete_sequence=px.colors.qualitative.G10,
             animation_frame = "Year",
             color = "Basketball Shoes",
             title="Basketball Shoes Popularity Over Time")
    if value == "insurance":
        return px.bar(insurance_melt, x='Insurance', y='Percentage Popularity', 
             color_discrete_sequence=px.colors.qualitative.G10,
             animation_frame = "Year",
             color = "Insurance",
             title="Insurance Popularity Over Time")
    if value == "call_phone":
        return px.bar(cell_melt, x='Cell Phone', y='Percentage Popularity', 
             color_discrete_sequence=px.colors.qualitative.G10,
             animation_frame = "Year",
             color = "cell_phone",
             title="Cell Phone Popularity Over Time")
    if value == "service":
        return px.bar(phone_service_melt, x='Phone Service Provider', y='Percentage Popularity', 
             color_discrete_sequence=px.colors.qualitative.G10,
             animation_frame = "Year",
             color = "Phone Service Provider",
             title="Phone Service Provider Popularity Over Time")
    
@app.callback(Output("clusters_value", "figure"), [Input("cluster_radio", "value")])
def change_value_chart(value):
    if value == "games":
        return color_clusters_g_vc(modeling_roi, 'Games', 'Value to Cost',5)
    if value == "win_shares":
        return color_clusters_g_vc(modeling_roi, 'Win Shares', 'Value to Cost',5)
    if value == "tenure":
        return color_clusters_g_vc(modeling_roi, 'Tenure (Years)', 'Value to Cost',5)
    if value == "draft_pick":
        return color_clusters_g_vc(modeling_roi, 'Overall Draft Pick', 'Value to Cost',5)
    
@app.callback(Output("clusters_sponsorship", "figure"), [Input("cluster_sponsorship_radio", "value")])
def change_sponsor_chart(value):
    if value == "tenure":
        return color_clusters_ten_sp(data_ts, 'Tenure (Years)', 'Sponsorship',5)
    if value == "overall_pick":
        return color_clusters_ten_sp(data_os, 'Overall Pick', 'Sponsorship',5)
    if value == "2pa":
        return color_clusters_ten_sp(data_2pas, '2PA', 'Sponsorship',5)
    if value == "ft":
        return color_clusters_ten_sp(data_fts, 'FT', 'Sponsorship',5)
    
@app.callback(Output("endorsement_graph", "figure"), [Input("endorsement_milli_radios", "value")])
def change_sponsor_chart(value):
    if value == "Tenure (Years)":
        return endorsement_tenure
    if value == "Overall Pick":
        return endorsements_draft

@app.callback(Output("team_bar", "figure"), [Input("team_radios", "value")])
def change_sponsor_chart(value):
    if value == "endorsements":
        return team_endorsements
    if value == "value_to_cost":
        return team_value_to_cost    
    
if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port='8050', debug=True, dev_tools_ui=False, dev_tools_props_check=False)