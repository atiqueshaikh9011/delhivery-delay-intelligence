import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES

df=pd.read_csv("data/modelling_table.csv"); X=df[FEATURES]; y=df.delivery_hours
Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.3,random_state=42)
Xv,Xte,yv,yte=train_test_split(Xtmp,ytmp,test_size=.5,random_state=42)
sc=StandardScaler().fit(Xtr); trs=sc.transform(Xtr); tes=sc.transform(Xte)
pd.DataFrame({"metric":["train_rows","val_rows","test_rows","train_scaled_max_abs_mean","test_scaled_max_abs_mean"],"value":[len(Xtr),len(Xv),len(Xte),np.abs(trs.mean(0)).max(),np.abs(tes.mean(0)).max()]}).round(4).to_csv("outputs/part_b_checks.csv",index=False)

def metrics(name,split,yt,p):
    mse=mean_squared_error(yt,p); return {"model":name,"split":split,"MAE":mean_absolute_error(yt,p),"MSE":mse,"RMSE":np.sqrt(mse),"R2":r2_score(yt,p)}
rows=[metrics("mean_baseline","val",yv,np.repeat(ytr.mean(),len(yv)))]
depths=[1,2,3,4,6,8,12,None]; dr=[]
for d in depths:
    m=DecisionTreeRegressor(max_depth=d,random_state=42).fit(Xtr,ytr)
    dr.append({"max_depth":"unlimited" if d is None else d,"train_rmse":np.sqrt(mean_squared_error(ytr,m.predict(Xtr))),"val_rmse":np.sqrt(mean_squared_error(yv,m.predict(Xv)))})
pd.DataFrame(dr).round(4).to_csv("outputs/part_b_tree_depth.csv",index=False)
best_depth=min(dr,key=lambda r:(r["val_rmse"],999 if r["max_depth"]=="unlimited" else int(r["max_depth"]))) ["max_depth"]
best_depth=None if best_depth=="unlimited" else int(best_depth)
models={"linear_regression":Pipeline([("scaler",StandardScaler()),("model",LinearRegression())]),"decision_tree":DecisionTreeRegressor(max_depth=best_depth,random_state=42),"random_forest":RandomForestRegressor(n_estimators=100,random_state=42),"gradient_boosting":GradientBoostingRegressor(n_estimators=100,learning_rate=.1,random_state=42)}
fitted={}
for n,m in models.items(): fitted[n]=m.fit(Xtr,ytr); rows.append(metrics(n,"val",yv,m.predict(Xv)))
lrrows=[]
for lr in [.01,.1,1.0]:
    m=GradientBoostingRegressor(n_estimators=100,learning_rate=lr,random_state=42).fit(Xtr,ytr); p=m.predict(Xv); lrrows.append({"learning_rate":lr,"val_rmse":np.sqrt(mean_squared_error(yv,p)),"val_r2":r2_score(yv,p)})
pd.DataFrame(lrrows).round(4).to_csv("outputs/part_b_gb_learning_rate.csv",index=False)
rank=sorted(rows[1:],key=lambda r:r["RMSE"]); top=[r["model"] for r in rank[:2]]; Xtv=pd.concat([Xtr,Xv]); ytv=pd.concat([ytr,yv]); kf=KFold(n_splits=5,shuffle=True,random_state=42)
cv=[]
for n in top:
    s=cross_val_score(models[n],Xtv,ytv,cv=kf); cv.append({"model":n,"mean_r2":s.mean(),"std_r2":s.std()})
pd.DataFrame(cv).round(4).to_csv("outputs/part_b_cv.csv",index=False)
winner=rank[0]["model"]; rows.append(metrics(winner,"test",yte,fitted[winner].predict(Xte)))
pd.DataFrame(rows).round(4).to_csv("outputs/part_b_metrics.csv",index=False)
