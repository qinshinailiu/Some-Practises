import pandas as pd 

data = pd.read_csv('melb_data.csv')

cols_to_use = ['Rooms','Distance','Landsize','BuildingArea','YearBuilt']
X = data[cols_to_use]
y = data.Price

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[
			('preprocessor',SimpleImputer()),
			('model',RandomForestRegressor(n_estimators = 50,random_state = 0))
	])

from sklearn.model_selection import cross_val_score

scores = -1 * cross_val_score(my_pipeline,X,y,
							  cv = 5,
							  scoring = 'neg_mean_absolute_error')
print('MAE scores:\n',scores)
print("Average MAE score (across experiments):")
print(scores.mean())