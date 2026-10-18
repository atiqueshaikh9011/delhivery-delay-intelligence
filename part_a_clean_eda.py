import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES

os.makedirs("outputs", exist_ok=True)
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
dedup = raw.drop_duplicates().copy()
checks = {
    "raw_rows": len(raw), "duplicate_rows_removed": len(raw)-len(dedup), "rows_after_dedup": len(dedup),
    "missing_distance_km_before": int(dedup.distance_km.isna().sum()),
    "missing_parcel_weight_before": int(dedup.parcel_weight_kg.isna().sum()),
    "sentinel_weight_rows": int((dedup.parcel_weight_kg > 100).sum()),
    "missing_declared_value_before": int(dedup.declared_value.isna().sum()),
    "missing_payment_mode_before": int(dedup.payment_mode.isna().sum()),
    "service_type_spellings_before": int(dedup.service_type.nunique()),
}
clean, medians = clean_shipments(raw)
checks["service_type_spellings_after"] = int(clean.service_type.nunique())
model = build_features(clean, hubs)
model["is_delayed"] = (model.delivery_hours > model.promised_hours).astype(int)
model.to_csv("data/modelling_table.csv", index=False)
checks.update({
    "median_distance_km": medians["distance_km"], "median_parcel_weight_kg": medians["parcel_weight_kg"],
    "median_declared_value_inr": medians["declared_value_inr"], "rows_after_hub_merge": len(model),
    "nulls_in_features_and_target": int(model[FEATURES+["delivery_hours","is_delayed"]].isna().sum().sum()),
    "delayed_count": int(model.is_delayed.sum()), "delay_rate": model.is_delayed.mean(),
    "declared_value_mean_inr": model.declared_value_inr.mean(), "declared_value_median_inr": model.declared_value_inr.median(),
    "log_declared_value_mean": model.log_declared_value.mean(), "log_declared_value_median": model.log_declared_value.median(),
})
q1, q3 = model.delivery_hours.quantile([.25,.75]); iqr=q3-q1; lo=q1-1.5*iqr; hi=q3+1.5*iqr
checks.update({"hours_q1":q1,"hours_q3":q3,"hours_lower_bound":lo,"hours_upper_bound":hi,"hours_outlier_count":int(((model.delivery_hours<lo)|(model.delivery_hours>hi)).sum())})
pd.DataFrame([{"metric":k,"value":round(float(v),4) if isinstance(v,(float,np.floating)) else v} for k,v in checks.items()]).to_csv("outputs/part_a_checks.csv",index=False)

model["distance_band"] = pd.cut(model.distance_km,bins=[0,200,400,600,2000],labels=["0-200","201-400","401-600","601+"],right=True)
rows=[]
for dim in ["service_type","hub_tier","distance_band"]:
    for group,g in model.groupby(dim, observed=False):
        rows.append({"dimension":dim,"group":group,"shipments":len(g),"delayed":int(g.is_delayed.sum()),"delay_rate":round(g.is_delayed.mean(),4),"mean_hours":round(g.delivery_hours.mean(),4)})
pd.DataFrame(rows).to_csv("outputs/part_a_delay_by_group.csv",index=False)
corr=model.select_dtypes(include=np.number).corr().round(4)
corr.to_csv("outputs/part_a_correlations.csv",index=True)
