import json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS

df=pd.read_csv("data/modelling_table.csv"); X=df[FEATURES]; y=df.is_delayed
Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.3,random_state=42,stratify=y)
Xv,Xte,yv,yte=train_test_split(Xtmp,ytmp,test_size=.5,random_state=42,stratify=ytmp)
def row(name,yt,p):
    tn,fp,fn,tp=confusion_matrix(yt,p,labels=[0,1]).ravel(); return {"model":name,"TN":tn,"FP":fp,"FN":fn,"TP":tp,"accuracy":accuracy_score(yt,p),"precision":precision_score(yt,p,zero_division=0),"recall":recall_score(yt,p,zero_division=0),"f1":f1_score(yt,p,zero_division=0)}
rows=[row("blind_all_on_time",yv,np.zeros(len(yv),dtype=int))]
log=Pipeline([("scaler",StandardScaler()),("model",LogisticRegression(max_iter=1000))]).fit(Xtr,ytr); rows.append(row("logistic_regression",yv,log.predict(Xv)))
best=None
for d in [2,3,4,5,6,8,10]:
    m=DecisionTreeClassifier(criterion="gini",max_depth=d,random_state=42).fit(Xtr,ytr); r=row(f"decision_tree_depth_{d}",yv,m.predict(Xv));
    if best is None or r["f1"]>best[0]: best=(r["f1"],d,r)
rows.append(best[2])
rf=RandomForestClassifier(n_estimators=100,random_state=42).fit(Xtr,ytr); rows.append(row("random_forest_default",yv,rf.predict(Xv)))
pd.DataFrame(rows).round(4).to_csv("outputs/part_c_models.csv",index=False)
sel=SelectKBest(f_classif,k="all").fit(Xtr,ytr); fs=sel.scores_; ranks=pd.Series(fs,index=FEATURES).rank(ascending=False,method="min").astype(int)
pd.DataFrame([{"feature":f,"f_score":fs[i],"f_rank":ranks[f],"rf_importance":rf.feature_importances_[i],"is_noise":f in NOISE_COLUMNS} for i,f in enumerate(FEATURES)]).round(4).to_csv("outputs/part_c_feature_scores.csv",index=False)
ks=[]
for k in [5,7,10,12,17]:
    p=Pipeline([("selector",SelectKBest(f_classif,k=k)),("model",RandomForestClassifier(n_estimators=100,random_state=42))]).fit(Xtr,ytr); ks.append({"k":k,"val_f1":f1_score(yv,p.predict(Xv))})
pd.DataFrame(ks).round(4).to_csv("outputs/part_c_selectk.csv",index=False)
selector=SelectKBest(f_classif,k=7).fit(Xtr,ytr); A=selector.transform(Xtr); V=selector.transform(Xv)
grid={"n_estimators":[50,100,150],"max_depth":[4,6,8],"min_samples_leaf":[1,5,10]}
gs=GridSearchCV(RandomForestClassifier(random_state=42),grid,cv=5).fit(A,ytr)
rs=RandomizedSearchCV(RandomForestClassifier(random_state=42),grid,n_iter=9,cv=5,random_state=42).fit(A,ytr)
def tune_row(name,m,cands,folds,bestacc,params): return {"search":name,"candidates":cands,"cv_folds":folds,"total_fits":cands*folds if folds else 1,"best_params":json.dumps(params,sort_keys=True) if isinstance(params,dict) else params,"best_cv_accuracy":bestacc,"val_f1":f1_score(yv,m.predict(V))}
defm=RandomForestClassifier(n_estimators=100,random_state=42).fit(A,ytr)
tunes=[tune_row("grid_search",gs.best_estimator_,27,5,gs.best_score_,gs.best_params_),tune_row("randomized_search",rs.best_estimator_,9,5,rs.best_score_,rs.best_params_),tune_row("default_forest_same_7_features",defm,1,0,np.nan,"defaults")]
pd.DataFrame(tunes).round(4).to_csv("outputs/part_c_tuning.csv",index=False)
priority={"default_forest_same_7_features":0,"grid_search":1,"randomized_search":2}; win=sorted(tunes,key=lambda r:(-r["val_f1"],priority[r["search"]]))[0]
if win["search"]=="grid_search": params=gs.best_params_
elif win["search"]=="randomized_search": params=rs.best_params_
else: params={"n_estimators":100,"max_depth":None,"min_samples_leaf":1}
final=Pipeline([("selector",SelectKBest(f_classif,k=7)),("model",RandomForestClassifier(random_state=42,**params))]).fit(Xtr,ytr)
prob=final.predict_proba(Xv)[:,1]; th=[]
for t in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:
    r=row("x",yv,(prob>=t).astype(int)); r.pop("model"); r={"threshold":t,**r}; th.append(r)
pd.DataFrame(th).round(4).to_csv("outputs/part_c_thresholds.csv",index=False)
elig=[r["threshold"] for r in th if r["recall"]>=.8]; threshold=max(elig) if elig else .1
ptest=(final.predict_proba(Xte)[:,1]>=threshold).astype(int); tr=row("x",yte,ptest); tr.pop("model"); pd.DataFrame([{ "threshold":threshold,**tr}]).round(4).to_csv("outputs/part_c_final_test.csv",index=False)
pd.DataFrame([{"selector_k":7,"n_estimators":params.get("n_estimators",100),"max_depth":"unlimited" if params.get("max_depth") is None else params["max_depth"],"min_samples_leaf":params.get("min_samples_leaf",1),"threshold":threshold,"chosen_from":win["search"]}]).to_csv("outputs/part_c_final_config.csv",index=False)
