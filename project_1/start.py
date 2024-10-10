import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# upload the datasets
awards_players = pd.read_csv("data/awards_players.csv")
coaches = pd.read_csv("data/coaches.csv")
players_teams = pd.read_csv("data/players_teams.csv")
players = pd.read_csv("data/players.csv")
series_post = pd.read_csv("data/series_post.csv")
teams_post = pd.read_csv("data/teams_post.csv")
teams = pd.read_csv("data/teams.csv")

# remove columns that have the same value in every entry
awards_players = awards_players.loc[:, awards_players.nunique() > 1]
coaches = coaches.loc[:, coaches.nunique() > 1]
players_teams = players_teams.loc[:, players_teams.nunique() > 1]
players = players.loc[:, players.nunique() > 1]
series_post = series_post.loc[:, series_post.nunique() > 1]
teams_post = teams_post.loc[:, teams_post.nunique() > 1]
teams = teams.loc[:, teams.nunique() > 1]

# change team ID because of the teams that changed their name
mapTeam = {}
for index, row in teams[['tmID', 'franchID']].iterrows():
    mapTeam [row['tmID']] = row['franchID']

coaches['tmID'] = coaches['tmID'].replace(mapTeam) 
players_teams['tmID'] = players_teams['tmID'].replace(mapTeam) 
series_post['tmIDWinner'] = series_post['tmIDWinner'].replace(mapTeam) 
series_post['tmIDLoser'] = series_post['tmIDLoser'].replace(mapTeam) 
teams_post['tmID'] = teams_post['tmID'].replace(mapTeam)
teams['tmID'] = teams['tmID'].replace(mapTeam).drop(columns=['franchID'])


# rename columns for merge
coaches = coaches.rename(columns=lambda x: x + '_coaches' if x not in ['year', 'tmID'] else x)
teams_post = teams_post.rename(columns=lambda x: x + '_post' if x not in ['year', 'tmID'] else x)

# merge datasets teams, teams_post and coaches
teams_coaches_merge = pd.merge(teams, coaches, on=['year', 'tmID'], how='left')
teams_coaches_merge = pd.merge(teams_coaches_merge, teams_post, on=['year', 'tmID'], how='left')

# fill NaN values
teams_coaches_merge['W_post'] = teams_coaches_merge['W_post'].fillna(0)
teams_coaches_merge['L_post'] = teams_coaches_merge['L_post'].fillna(0)

# drop unnecessary columns
teams_coaches_merge_without_playoffs = teams_coaches_merge.drop(columns=['firstRound', 'semis', 'finals', 'W_post', 'L_post', 'rank', 'post_wins_coaches', 'post_losses_coaches'])

# Convert 'playoff' column to numeric
teams_coaches_merge_without_playoffs['playoff'] = teams_coaches_merge_without_playoffs['playoff'].map({'Y': 1, 'N': 0})

# Split training and testing data based on the year
training_data = teams_coaches_merge_without_playoffs[teams_coaches_merge_without_playoffs['year'] < 10]
testing_data = teams_coaches_merge_without_playoffs[teams_coaches_merge_without_playoffs['year'] == 10]

# Drop the target column and apply one-hot encoding to the features
X_train = pd.get_dummies(training_data.drop(columns=['playoff']), drop_first=True)
y_train = training_data['playoff']

X_test = pd.get_dummies(testing_data.drop(columns=['playoff']), drop_first=True)

# Align the train and test sets
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Create and train the Decision Tree model
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Make predictions on the test set
y_pred = decision_tree.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(testing_data['playoff'], y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Extrair as importâncias das features
feature_importances = decision_tree.feature_importances_

# Criar um DataFrame para organizar as importâncias
feature_importances_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})

# Ordenar as features pela importância
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

# Exibir as 10 features mais importantes
# print(feature_importances_df.head(10))
# Visualizar as importâncias das features
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Importance', y='Feature', data=feature_importances_df.head(10))
# plt.title('Top 10 Most Important Features')
# plt.show()
