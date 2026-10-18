import joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from src.shipment_prep import FEATURES, clean_shipments

df=pd.read_csv("data/modelling_table.csv"); X=df[FEATURES]; y=df.is_delayed
Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.3,random_state=42,stratify=y); Xv,Xte,yv,yte=train_test_split(Xtmp,ytmp,test_size=.5,random_state=42,stratify=ytmp)
cfg=pd.read_csv("outputs/part_c_final_config.csv").iloc[0]; md=None if str(cfg.max_depth)=="unlimited" else int(cfg.max_depth)
p=Pipeline([("selector",SelectKBest(f_classif,k=int(cfg.selector_k))),("model",RandomForestClassifier(random_state=42,n_estimators=int(cfg.n_estimators),max_depth=md,min_samples_leaf=int(cfg.min_samples_leaf)))]).fit(Xtr,ytr)
joblib.dump(p,"models/delay_model.pkl"); pd.DataFrame({"threshold":[cfg.threshold]}).to_csv("models/delay_threshold.csv",index=False); pd.DataFrame({"feature":FEATURES}).to_csv("models/feature_order.csv",index=False)
raw=pd.read_csv("data/raw_shipments.csv"); _,med=clean_shipments(raw); pd.DataFrame([{"column":k,"median":v} for k,v in med.items()]).to_csv("models/impute_values.csv",index=False)
r=joblib.load("models/delay_model.pkl"); a=p.predict_proba(Xte)[:,1]; b=r.predict_proba(Xte)[:,1]; pd.DataFrame({"metric":["test_rows_scored","reload_max_abs_probability_diff","threshold"],"value":[len(Xte),np.max(np.abs(a-b)),cfg.threshold]}).to_csv("outputs/part_d_reload_check.csv",index=False)
