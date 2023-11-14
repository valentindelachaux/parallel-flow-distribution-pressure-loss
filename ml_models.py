from sklearn.ensemble import RandomForestRegressor
import pandas as pd

df_testings = pd.read_excel("G:\Drive partagés\BU04-Innovation\PVT-PL-model\Outputs\Tests-model-simplification\V4.5_1MPE_testings.xlsx")
X = df_testings[['QF', 'QF_out', 'alpha']].to_numpy()
yin = df_testings['DPin'].to_numpy()
yout = df_testings['DPout'].to_numpy()
yx = df_testings['DPx'].to_numpy()

model_in = RandomForestRegressor() 
model_in.fit(X, yin)

model_out = RandomForestRegressor() 
model_out.fit(X, yout)

model_x = RandomForestRegressor() 
model_x.fit(X, yx)
