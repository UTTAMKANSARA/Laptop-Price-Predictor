import numpy as np
import pandas as pd

# --- CELL ---
df = pd.read_csv('laptop_data.csv')

# --- CELL ---
df.head()

# --- CELL ---
df.info()

# --- CELL ---
df.duplicated().sum()

# --- CELL ---
df.isnull().sum()

# --- CELL ---
df.drop(columns = ['Unnamed: 0'],inplace = True)

# --- CELL ---
df.head()

# --- CELL ---
df['Ram'] = df['Ram'].str.replace('GB','')
df['Weight'] = df['Weight'].str.replace('kg','')

# --- CELL ---
df.head()

# --- CELL ---
df['Ram'] = df['Ram'].astype('int32')
df['Weight'] = df['Weight'].astype('float32')

# --- CELL ---
df.info()

# --- CELL ---
import seaborn as sns
sns.distplot(df['Price'])

# --- CELL ---
df['Company'].value_counts().plot(kind= 'bar')

# --- CELL ---
import matplotlib.pyplot as plt
sns.barplot(x = df['Company'], y  = df['Price'])
plt.xticks(rotation = 'vertical')
plt.show()

# --- CELL ---
df['TypeName'].value_counts().plot(kind ='bar')

# --- CELL ---
import matplotlib.pyplot as plt
sns.barplot(x = df['TypeName'], y  = df['Price'])
plt.xticks(rotation = 'vertical')
plt.show()

# --- CELL ---
sns.distplot(x = df['Inches'])

# --- CELL ---
sns.scatterplot(x = df['Inches'],y=df['Price'])

# --- CELL ---
df['ScreenResolution'].value_counts()

# --- CELL ---
df['Touchscreen'] = df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)

# --- CELL ---
df.sample(5)

# --- CELL ---
df['Touchscreen'].value_counts()

# --- CELL ---
sns.barplot(x = df['Touchscreen'],y = df['Price'])

# --- CELL ---
df['Ips'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)


# --- CELL ---
df.head()

# --- CELL ---
df['Ips'].value_counts()

# --- CELL ---
sns.barplot(x =df['Ips'],y = df['Price'])

# --- CELL ---
new = df['ScreenResolution'].str.split('x',n = 1,expand = True)

# --- CELL ---
df['X_res'] = new[0]
df['Y_res'] = new[1]

# --- CELL ---
df.head()

# --- CELL ---
df['X_res'] = df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])

# --- CELL ---
df.head()

# --- CELL ---
df['X_res'] = df['X_res'].astype('int')
df['Y_res'] = df['Y_res'].astype('int')

# --- CELL ---
df.info()

# --- CELL ---
df.corr(numeric_only = True)['Price']

# --- CELL ---
df['ppi']=(((df['X_res']**2)+(df['Y_res']**2))**0.5/df['Inches']).astype('float')

# --- CELL ---
df.head()

# --- CELL ---
df.corr(numeric_only = True)['Price']

# --- CELL ---
df.drop(columns = ['ScreenResolution'] ,inplace = True)

# --- CELL ---
df.drop(columns = ['X_res','Y_res','Inches'] ,inplace = True)

# --- CELL ---
df.head()

# --- CELL ---
df['Cpu'].value_counts()

# --- CELL ---
df['Cpu name']=df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))

# --- CELL ---
df.head()

# --- CELL ---
df['Cpu name'].value_counts()

# --- CELL ---
def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'

# --- CELL ---
df['Cpu Brand'] = df['Cpu name'].apply(fetch_processor)

# --- CELL ---
df.head()

# --- CELL ---
df.sample(5)

# --- CELL ---
df['Cpu Brand'].value_counts().plot(kind = 'bar')

# --- CELL ---
sns.barplot(x = df['Cpu Brand'],y = df['Price'])
plt.xticks(rotation ='vertical')

# --- CELL ---
df.drop(columns = ['Cpu','Cpu name'],inplace = True)

# --- CELL ---
df.head()

# --- CELL ---
df['Ram'].value_counts()

# --- CELL ---
sns.barplot(x = df['Ram'],y = df['Price'])

# --- CELL ---
df['Memory'].value_counts()

# --- CELL ---
df.info()

# --- CELL ---
df['Memory'] = df['Memory'].astype(str).replace('\.0','',regex = True)
df['Memory'] = df['Memory'].str.replace('GB','')
df['Memory'] = df['Memory'].str.replace('TB','000')
new = df['Memory'].str.split("+",n = 1,expand = True)


df['first'] = new[0]
df['first'] = df["first"].str.strip()

df['second'] = new[1]





# --- Process the 'first' part ---
# Create 1/0 flags instead of keeping the string
df['Layer1HDD'] = df['first'].apply(lambda x: 1 if "HDD" in str(x) else 0)
df['Layer1SSD'] = df['first'].apply(lambda x: 1 if "SSD" in str(x) else 0)
df['Layer1Hybrid'] = df['first'].apply(lambda x: 1 if "Hybrid" in str(x) else 0)
df['Layer1Flash_Storage'] = df['first'].apply(lambda x: 1 if "Flash Storage" in str(x) else 0)

# Now convert 'first' to integer safely
df['first'] = df['first'].str.replace(r'\D', '', regex=True).astype(int)

# --- Process the 'second' part ---
df['second'].fillna("0", inplace=True)

# Create 1/0 flags for the second storage component
df['Layer2HDD'] = df['second'].apply(lambda x: 1 if "HDD" in str(x) else 0)
df['Layer2SSD'] = df['second'].apply(lambda x: 1 if "SSD" in str(x) else 0)
df['Layer2Hybrid'] = df['second'].apply(lambda x: 1 if "Hybrid" in str(x) else 0)
df['Layer2Flash_Storage'] = df['second'].apply(lambda x: 1 if "Flash Storage" in str(x) else 0)

# Now convert 'second' to integer
df['second'] = df['second'].str.replace(r'\D', '', regex=True).astype(int)

# --- The math will now work (Integer * 0 or 1) ---
df["HDD"] = (df['first'] * df["Layer1HDD"] + df["second"] * df["Layer2HDD"])
df["SSD"] = (df['first'] * df["Layer1SSD"] + df["second"] * df["Layer2SSD"])
df["Hybrid"] = (df['first'] * df["Layer1Hybrid"] + df["second"] * df["Layer2Hybrid"])
df["Flash_Storage"] = (df['first'] * df["Layer1Flash_Storage"] + df["second"] * df["Layer2Flash_Storage"])






# df['Layer1HDD'] = df['first'].apply(lambda x:x if "HDD" in x else 0)
# df['Layer1SSD'] = df['first'].apply(lambda x:x if "SSD" in x else 0)
# df['Layer1Hybrid'] = df['first'].apply(lambda x:x if "Hybrid" in x else 0)
# df['Layer1Flash_Storage'] = df['first'].apply(lambda x:x if "Flash Storage" in x else 0)


# df['first'] = df['first'].str.replace(r'\D','',regex=True).astype(int)

# df['second'].fillna("0",inplace =True)


# df['Layer2HDD'] = df['second'].apply(lambda x:x if "HDD" in x else 0)
# df['Layer2SSD'] = df['second'].apply(lambda x:x if "SSD" in x else 0)
# df['Layer2Hybrid'] = df['second'].apply(lambda x:x if "Hybrid" in x else 0)
# df['Layer2Flash_Storage'] = df['second'].apply(lambda x:x if "Flash Storage" in x else 0)


# df['second'] = df['second'].str.replace(r'\D','',regex=True).astype(int)

# # df["first"] = df["first"].astype(int)
# # df['second'] = df["second"].astype(int)


# df["HDD"] = (df['first']*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
# df["SSD"] = (df['first']*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
# df["Hybrid"] = (df['first']*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
# df["Flash Storage"] = (df['first']*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])

df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage'],inplace=True)

# --- CELL ---
df.head()

# --- CELL ---
df.sample(5)

# --- CELL ---
df.corr(numeric_only =True)['Price']

# --- CELL ---
df.drop(columns =['Hybrid','Flash_Storage'],inplace = True)

# --- CELL ---
df.head()

# --- CELL ---
df['Gpu'].value_counts()

# --- CELL ---
df['Gpu brand']=df['Gpu'].apply(lambda x:x .split()[0])

# --- CELL ---
df['Gpu brand'].value_counts()

# --- CELL ---
df = df[df['Gpu brand'] != 'ARM']

# --- CELL ---
df['Gpu brand'].value_counts()

# --- CELL ---
sns.barplot(x = df['Gpu brand'],y = df['Price'],estimator = np.median)


# --- CELL ---
df.drop(columns = ['Gpu'],inplace =True)

# --- CELL ---
df.head()

# --- CELL ---
df['OpSys'].value_counts()

# --- CELL ---
sns.barplot(x = df['OpSys'],y = df['Price']) 
plt.xticks(rotation = 'vertical')

# --- CELL ---
def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 s':
        return 'Windows'
    elif inp == 'macOS'or inp == 'MAc OS X':
        return 'Mac'
    else:
        return 'Other/No OS/Linux'

# --- CELL ---
df['OS'] = df['OpSys'].apply(cat_os)

# --- CELL ---
df.head()

# --- CELL ---
df.drop(columns = ['OpSys'], inplace = True)

# --- CELL ---
df.head()

# --- CELL ---
sns.barplot(x = df['OS'],y = df['Price'])

# --- CELL ---
sns.distplot(df['Weight'])

# --- CELL ---
sns.scatterplot(x = df['Weight'],y=df['Price'])

# --- CELL ---
df.corr(numeric_only = True)['Price']

# --- CELL ---
sns.heatmap(df.corr(numeric_only = True))

# --- CELL ---
sns.distplot(np.log(df['Price']))

# --- CELL ---
df.drop(columns=['Memory'],inplace= True)

# --- CELL ---
df.head()

# --- CELL ---
X = df.drop(columns = ['Price'])
Y = np.log(df['Price'])

# --- CELL ---
X

# --- CELL ---
Y

# --- CELL ---
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =0.15,random_state =2)

# --- CELL ---
X_train

# --- CELL ---
Y_train

# --- CELL ---
# !pip install xgboost

# --- CELL ---
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# --- CELL ---

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = LinearRegression()

pipe = Pipeline([('step1',step1),('step2',step2)])

pipe.fit(X_train,Y_train)
y_pred = pipe.predict(X_test)

print('R2 score',r2_score(Y_test,y_pred))
print('MAE',mean_absolute_error(Y_test,y_pred))

# --- CELL ---
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = Ridge(alpha=10)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,Y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(Y_test,y_pred))
print('MAE',mean_absolute_error(Y_test,y_pred))

# --- CELL ---
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = Lasso(alpha=0.001)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,Y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(Y_test,y_pred))
print('MAE',mean_absolute_error(Y_test,y_pred))

# --- CELL ---
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = KNeighborsRegressor(n_neighbors=3)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,Y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(Y_test,y_pred))
print('MAE',mean_absolute_error(Y_test,y_pred))

# --- CELL ---
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth=8)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,Y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(Y_test,y_pred))
print('MAE',mean_absolute_error(Y_test,y_pred))

# --- CELL ---
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = SVR(kernel='rbf',C=10000,epsilon=0.1)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,Y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(Y_test,y_pred))
print('MAE',mean_absolute_error(Y_test,y_pred))

# --- CELL ---
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,Y_train)
y_pred = pipe.predict(X_test)

print('R2 score',r2_score(Y_test,y_pred))
print('MAE',mean_absolute_error(Y_test,y_pred))

# --- CELL ---
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = ExtraTreesRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=None,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,Y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(Y_test,y_pred))
print('MAE',mean_absolute_error(Y_test,y_pred))

# --- CELL ---
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = AdaBoostRegressor(n_estimators=15,learning_rate=1.0)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,Y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(Y_test,y_pred))
print('MAE',mean_absolute_error(Y_test,y_pred))


# --- CELL ---
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = GradientBoostingRegressor(n_estimators=500)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,Y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(Y_test,y_pred))
print('MAE',mean_absolute_error(Y_test,y_pred))

# --- CELL ---
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = XGBRegressor(n_estimators=45,max_depth=5,learning_rate=0.5)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,Y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(Y_test,y_pred))
print('MAE',mean_absolute_error(Y_test,y_pred))

# --- CELL ---

from sklearn.ensemble import VotingRegressor,StackingRegressor

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')


rf = RandomForestRegressor(n_estimators=350,random_state=3,max_samples=None,max_features=0.75,max_depth=15)
gbdt = GradientBoostingRegressor(n_estimators=100,max_features=0.5)
xgb = XGBRegressor(n_estimators=25,learning_rate=0.3,max_depth=5)
et = ExtraTreesRegressor(n_estimators=100,random_state=3,max_samples=None,max_features=0.75,max_depth=10)

step2 = VotingRegressor([('rf', rf), ('gbdt', gbdt), ('xgb',xgb), ('et',et)],weights=[5,1,1,1])

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,Y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(Y_test,y_pred))
print('MAE',mean_absolute_error(Y_test,y_pred))

# --- CELL ---
from sklearn.ensemble import VotingRegressor,StackingRegressor

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')


estimators = [
    ('rf', RandomForestRegressor(n_estimators=350,random_state=3,max_samples=None,max_features=0.75,max_depth=15)),
    ('gbdt',GradientBoostingRegressor(n_estimators=100,max_features=0.5)),
    ('xgb', XGBRegressor(n_estimators=25,learning_rate=0.3,max_depth=5))
]

step2 = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=100))

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,Y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(Y_test,y_pred))
print('MAE',mean_absolute_error(Y_test,y_pred))

# --- CELL ---
import pickle as pl
pl.dump(df,open('df.pkl','wb'))
pl.dump(pipe,open('pipe.pkl','wb'))

# --- CELL ---
df

# --- CELL ---
X_train

# --- CELL ---
Y_train

# --- CELL ---
import pickle
pipe = pickle.load(open('pipe.pkl','rb'))

# Print the final step of your pipeline
print("Final Estimator:", pipe.steps[-1][0])
print("Model Type:", type(pipe.steps[-1][1]))