import itertools, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df=pd.read_csv("data/modelling_table.csv")
profile=df.groupby("hub_id",sort=True).agg(shipments=("shipment_id","count"),mean_delivery_hours=("delivery_hours","mean"),delay_rate=("is_delayed","mean"),mean_distance_km=("distance_km","mean"),mean_n_stops=("n_stops","mean")).reset_index().sort_values("hub_id")
profile.round(4).to_csv("outputs/hub_profile.csv",index=False)
cols=["mean_delivery_hours","delay_rate","mean_distance_km","mean_n_stops"]; Z=StandardScaler().fit_transform(profile[cols])
def ratio(labels):
    same=[]; diff=[]
    for i,j in itertools.combinations(range(len(Z)),2):
        d=np.linalg.norm(Z[i]-Z[j]); (same if labels[i]==labels[j] else diff).append(d)
    return np.mean(diff)/np.mean(same)
seps=[]
for k in [2,3,4,5]:
    lab=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(Z); sizes=sorted(pd.Series(lab).value_counts().tolist(),reverse=True); seps.append({"K":k,"across_over_within":ratio(lab),"cluster_sizes":"-".join(map(str,sizes))})
pd.DataFrame(seps).round(4).to_csv("outputs/part_d_separation.csv",index=False)
labels=KMeans(n_clusters=3,random_state=42,n_init=10).fit_predict(Z); profile["cluster"]=labels
order=profile.groupby("cluster").delay_rate.mean().sort_values().index.tolist(); names={order[0]:"Standard",order[1]:"Watch",order[2]:"Escalate"}; profile["segment"]=profile.cluster.map(names)
hubs=pd.read_csv("data/hubs.csv"); seg=profile[["hub_id","segment"]].merge(hubs[["hub_id","hub_city","hub_tier"]],on="hub_id",how="left"); seg.to_csv("outputs/hub_segments.csv",index=False)
sp=profile.groupby("segment").agg(hubs=("hub_id","count"),mean_delivery_hours=("mean_delivery_hours","mean"),delay_rate=("delay_rate","mean"),mean_distance_km=("mean_distance_km","mean"),mean_n_stops=("mean_n_stops","mean")).reindex(["Standard","Watch","Escalate"]).reset_index(); sp.round(4).to_csv("outputs/segment_profile.csv",index=False)
implied={1:"Standard",2:"Watch",3:"Escalate"}; mm=seg[seg.apply(lambda r:r.segment!=implied[r.hub_tier],axis=1)]; mm.to_csv("outputs/tier_vs_segment_mismatch.csv",index=False)
