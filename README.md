# Delhivery Last-Mile Delay Intelligence Pipeline

A reproducible offline ML pipeline for synthetic parcel data. It estimates delivery hours, predicts delay risk before a parcel is late, segments hubs into three operating playbooks, packages the final classifier, and scores a new batch into operational actions.

## How to run

From a fresh clone, run these commands from the repository root, in order:

```bash
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

`generate_data.py` must print `24 1224 48`. `package_model.py` regenerates `models/delay_model.pkl`. The committed pickle loads reliably only with the scikit-learn version pinned in `requirements.txt`; if an installed version differs, regenerate the pickle by running `python package_model.py` rather than relying on the committed artifact.

## Data findings

- Express has the highest delay rate: 119 of 243 shipments, or **0.4897**. Standard is 0.2056 and Economy is 0.0857.
- Delay rate rises strongly with hub tier: **0.0400** for Tier 1, **0.2150** for Tier 2, and **0.4400** for Tier 3.
- Distance shows an even steeper pattern: delay rate is **0.0219** at 0–200 km, **0.1368** at 201–400 km, **0.4364** at 401–600 km, and **0.8523** above 600 km.
- The three strongest correlations with `delivery_hours` by absolute value are `promised_hours` **+0.8372**, `distance_km` **+0.4013**, and `n_stops` **+0.3247**.
- `declared_value_inr` is right-skewed: its filled mean is **INR 1,533.0542** versus median **INR 1,121.5000**. After natural-log transformation the mean and median are **7.0277** and **7.0224**, much closer, so the transform reduces skew.

## Model results (regression)

The mean baseline has validation R² **-0.0212**, meaning it is slightly worse than simply explaining no variance around the validation-set mean. Linear regression wins the single validation split with RMSE **4.8912 hours** and R² **0.9313**; its final test R² is **0.9493**.

The depth diagnostic shows high bias at depth 1: train RMSE **11.6334** and validation RMSE **12.1324** are both high. Depth 6 is the best tree with validation RMSE **6.2796**. The unlimited tree has train RMSE **0.0000** but validation RMSE **7.8368**, a **7.8368-hour** generalization gap that indicates high variance/overfitting.

For gradient boosting, learning rate **0.10** is best with validation R² **0.9306**. At 0.01, learning is too slow and reaches only **0.7127**; at 1.0, the updates are too aggressive and performance falls to **0.8440**. Cross-validation ranks gradient boosting higher (mean R² **0.9353**) than linear regression (**0.9261**) even though linear regression narrowly wins the pinned validation split. For the assignment's prescribed final-model rule I ship linear regression, because the final model must be the candidate with the lowest validation RMSE; in a production selection exercise I would investigate the CV advantage before locking the algorithm.

## Model results (classification)

The blind all-on-time baseline has validation accuracy **0.7667**, but recall is 0 because it misses all 42 delayed validation parcels, so its apparently good accuracy is operationally useless.

The five bookkeeping/noise columns contribute about **0.2245** of the default forest's total importance, yet all five rank outside the top seven by ANOVA F-score and each has F-score below 10. The `k` experiment supports the pinned seven-feature choice: validation F1 is **0.6988** (k=5), **0.8333** (k=7), **0.7857** (k=10), **0.8049** (k=12), and **0.7848** (k=17).

Tuning does not beat the untuned forest on this validation set. Grid search reaches F1 **0.7750**, randomized search **0.7848**, while the default forest on the same seven selected features reaches **0.8333**. With only 180 validation rows, small differences can be unstable; this is a reason to prefer repeated/CV evidence in a real deployment rather than repeatedly optimizing against one small holdout.

The operations rule chooses threshold **0.5**, the highest tested threshold with validation recall at least 0.80. At 0.5, validation precision is **0.7955** and recall is **0.8333**. On the untouched test set, precision is **0.6889**, recall **0.7561**, and F1 **0.7209**. Test recall is lower than validation recall, and the test set contains only 41 delayed parcels, so that difference should not be over-interpreted.

## Operations recommendation

The `Escalate` segment contains nine hubs: HUB13 Nagpur (delay rate **0.4600**), HUB15 Kochi (**0.3600**), HUB17 Patna (**0.4600**), HUB18 Ranchi (**0.3800**), HUB19 Guwahati (**0.6200**), HUB20 Raipur (**0.4800**), HUB21 Bhubaneswar (**0.4000**), HUB23 Siliguri (**0.4400**) and HUB24 Agartala (**0.5400**). These hubs should receive the highest-intensity operating playbook.

Four tier labels disagree with observed operating behavior: HUB07 Kolkata is Tier 1 but `Watch`; HUB13 Nagpur and HUB15 Kochi are Tier 2 but `Escalate`; HUB22 Dehradun is Tier 3 but `Watch`. Tier should therefore be treated as descriptive metadata, not a substitute for measured behavior when choosing the playbook.

At threshold **0.5**, test precision **0.6889** means roughly **6.9 of every 10** flagged parcels are truly delayed, while recall **0.7561** means roughly **7.6 of every 10** delayed parcels are caught. In the 48-parcel scoring batch, **8** parcels are flagged and **6** of those require `Escalate to hub manager` because their hubs are in the Escalate segment.

K remains fixed at 3 because operations needs exactly three playbooks and the solution is well separated: the across/within ratio is **5.3928** with cluster sizes **9-8-7**. Although the ratio rises to **6.3047** at K=4 and **6.8866** at K=5, those solutions create clusters smaller than six hubs (5 and 4 hubs for K=4; 4, 3 and 2 for K=5), which are too small to justify separate operating playbooks.

## Production plan

**Readiness mapping.** Accuracy is evidenced by held-out regression/classification metrics. Reliability is supported by the reload check and deterministic seeds. Reproducibility comes from pinned dependencies, generated data, fixed seeds and ordered scripts. Performance is supported by compact offline sklearn models and batch scoring of the 48-row input without external services. Maintainability comes from centralizing constants, cleaning and feature construction in `src/shipment_prep.py`. Consistency is enforced because both training and `predict_delay.py` use the same `clean_shipments` and `build_features` functions. Monitorability is supported by storing threshold, feature order, segment outputs and explicit prediction actions as auditable CSV artifacts.

**Monitoring triggers.** Track weekly delivered-parcel delay rate and investigate if it changes by more than **5 percentage points** from the relevant four-week baseline. Track weekly flagged share and investigate a shift of more than **7 percentage points** from the recent baseline. Once outcomes mature, calculate weekly recall and trigger review if recall falls below **0.70** for two consecutive weeks. Also monitor missingness for the three imputed numeric inputs and investigate if any exceeds **5%** in a week.

**Retraining and drift.** Retrain monthly initially, and also trigger retraining review when a monitoring threshold is breached. The 15–28 September sale period must be compared with comparable sale-event periods using `sale_event_flag`, not with ordinary weeks. Drift here would include sustained changes in distance, stops, hub mix, rain, service mix, missingness, delay rate, or the relationship between these inputs and delayed outcomes.

## Limitations

The dataset is synthetic and cannot capture every operational behavior of a real logistics network. The classification test set contains only 41 delayed parcels, so precision and recall estimates have meaningful sampling uncertainty. The alert threshold is tuned on one 180-row validation split, making it sensitive to that split. Hub segmentation is based on only 24 hub-level observations and four aggregate measures. Finally, the current pipeline uses a fixed feature definition and does not model temporal drift beyond the engineered sale/weekend/dispatch-time signals.

