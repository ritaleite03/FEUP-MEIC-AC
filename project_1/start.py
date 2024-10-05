import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# upload the datasets
awards_players = pd.read_csv("data/awards_players.csv")
coaches = pd.read_csv("data/coaches.csv")
players_teams = pd.read_csv("data/players_teams.csv")
players = pd.read_csv("data/players.csv")
series_post = pd.read_csv("data/series_post.csv")
teams_post = pd.read_csv("data/teams_post.csv")
teams = pd.read_csv("data/teams.csv")

# remove columns of the same value in every entry
awards_players = awards_players.loc[:, awards_players.nunique() > 1]
coaches = coaches.loc[:, coaches.nunique() > 1]
players_teams = players_teams.loc[:, players_teams.nunique() > 1]
players = players.loc[:, players.nunique() > 1]
series_post = series_post.loc[:, series_post.nunique() > 1]
teams_post = teams_post.loc[:, teams_post.nunique() > 1]
teams = teams.loc[:, teams.nunique() > 1]

# rename columns for merge
coaches = coaches.rename(columns=lambda x: x + '_coaches' if x not in ['year', 'tmID'] else x)
teams_post = teams_post.rename(columns=lambda x: x + '_post' if x not in ['year', 'tmID'] else x)

# merge datasets teams, teams_post and coaches
teams_coaches_merge = pd.merge(teams, coaches, on=['year', 'tmID'], how='left')
teams_coaches_merge = pd.merge(teams_coaches_merge, teams_post, on=['year', 'tmID'], how='left')

# fill NaN values
teams_coaches_merge['W_post'] = teams_coaches_merge['W_post'].fillna(0)
teams_coaches_merge['L_post'] = teams_coaches_merge['L_post'].fillna(0)

# see which columns have Nan values
rows_with_nan = teams_coaches_merge[teams_coaches_merge.isna().any(axis=1)]
nan_columns = rows_with_nan.columns[rows_with_nan.isna().any()]
print(rows_with_nan[nan_columns].head())
