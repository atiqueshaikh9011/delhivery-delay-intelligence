import joblib, pandas as pd
from src.shipment_prep import clean_shipments, build_features

raw=pd.read_csv("data/new_shipments.csv"); hubs=pd.read_csv("data/hubs.csv"); meddf=pd.read_csv("models/impute_values.csv"); med=dict(zip(meddf["column"],meddf["median"])); clean,_=clean_shipments(raw,med); feat=build_features(clean,hubs)
order=pd.read_csv("models/feature_order.csv").feature.tolist(); model=joblib.load("models/delay_model.pkl"); threshold=float(pd.read_csv("models/delay_threshold.csv").threshold.iloc[0]); feat["delay_probability"]=model.predict_proba(feat[order])[:,1].round(4); feat["flagged"]=(feat.delay_probability>=threshold).astype(int)
seg=pd.read_csv("outputs/hub_segments.csv")[["hub_id","segment"]]; out=feat.merge(seg,on="hub_id",how="left",sort=False); out["action"]=out.apply(lambda r:"No action" if not r.flagged else ("Escalate to hub manager" if r.segment=="Escalate" else "Proactive customer call"),axis=1); out[["shipment_id","hub_id","segment","delay_probability","flagged","action"]].to_csv("outputs/new_shipment_actions.csv",index=False)
