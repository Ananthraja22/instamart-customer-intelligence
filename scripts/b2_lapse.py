"""Part B, tasks 4-8: lapse label and features, leakage-safe preprocessing, tuning, evaluation,
class-balance experiment, interpretation, persistence and the illustrative retention list.

Leakage guards:
  * every behavioural feature is computed only from orders dated before the cutoff;
  * Part A imputed some delivery times from whole-window medians, which include September. Those
    imputed values are recomputed here from measured pre-cutoff times (see add_delivery_feature);
  * encoders and scalers live inside a Pipeline, so GridSearchCV refits them per training fold;
  * segment labels are never read when building features.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample

from common import CLEAN, OUTPUTS, ensure_dirs, md_table, section

CUTOFF = pd.Timestamp("2025-09-01")
WINDOW_END = pd.Timestamp("2025-10-01")
NUMERIC = ["recency", "frequency", "monetary", "avg_discount_pct", "cancellation_rate",
           "avg_delivery_minutes", "age", "tenure_days"]
NOMINAL = ["city", "gender", "favourite_category"]


def add_delivery_feature(completed_pre):
    """Average delivery time per customer, with no post-cutoff information.

    Measured times are kept. Times that Part A imputed from a whole-window category median are
    replaced by the category median of measured times before the cutoff.
    """
    measured = completed_pre[completed_pre["delivery_imputed"] == 0]
    medians = measured.groupby("category")["delivery_minutes"].median()
    minutes = completed_pre["delivery_minutes"].where(completed_pre["delivery_imputed"] == 0,
                                                      completed_pre["category"].map(medians))
    assert minutes.notna().all(), "a pre-cutoff delivery time could not be filled"
    return minutes.groupby(completed_pre["customer_id"]).mean().rename("avg_delivery_minutes"), medians


def build_dataset():
    orders = pd.read_csv(f"{CLEAN}/orders.csv", parse_dates=["order_ts"])
    customers = pd.read_csv(f"{CLEAN}/customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv(f"{CLEAN}/products.csv")
    orders = orders.merge(products[["product_id", "category"]], on="product_id", validate="many_to_one")

    pre = orders[orders["order_ts"] < CUTOFF]
    assert pre["order_ts"].max() < CUTOFF, "a post-cutoff order reached the feature window"
    completed_pre = pre[pre["status"] == "completed"]
    september = orders[(orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < WINDOW_END)
                       & (orders["status"] == "completed")]

    data = completed_pre.groupby("customer_id").agg(
        last_order=("order_ts", "max"), frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"), avg_discount_pct=("discount_pct", "mean"))
    data["recency"] = (CUTOFF - data["last_order"]).dt.days
    data = data.drop(columns="last_order")

    status_counts = pre.groupby("customer_id")["status"].value_counts().unstack(fill_value=0)
    data["cancellation_rate"] = (status_counts["cancelled"] / status_counts.sum(axis=1)).reindex(data.index)

    delivery, medians = add_delivery_feature(completed_pre)
    data = data.join(delivery)

    # Ties for favourite category break alphabetically so the feature is deterministic.
    favourite = (completed_pre.groupby(["customer_id", "category"]).size().rename("n").reset_index()
                 .sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
                 .drop_duplicates("customer_id").set_index("customer_id")["category"])
    data["favourite_category"] = favourite

    data = data.join(customers.set_index("customer_id")[["city", "age", "gender", "signup_date"]])
    data["tenure_days"] = (CUTOFF - data["signup_date"]).dt.days
    data["lapsed"] = (~data.index.isin(september["customer_id"])).astype(int)
    data = data.drop(columns="signup_date").sort_index()
    return data, medians


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), NOMINAL),
    ])


def lapsed_metrics(y_true, y_pred):
    return {"precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)}


def clean_name(name):
    return name.split("__", 1)[1]


def main():
    ensure_dirs(OUTPUTS)
    data, pre_cutoff_medians = build_dataset()

    section("Task 4: cohort and label")
    print(f"cohort {len(data):,}; lapsed {data['lapsed'].sum()} ({data['lapsed'].mean() * 100:.2f}%)")
    assert len(data) == 1156 and data["lapsed"].sum() == 443
    print("pre-cutoff delivery medians:", pre_cutoff_medians.to_dict())

    X = data[NUMERIC + NOMINAL]
    y = data["lapsed"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"train {len(X_train)}; test {len(X_test)}; train lapsed {y_train.sum()}; test lapsed {y_test.sum()}")
    assert (len(X_train), len(X_test)) == (924, 232)

    section("Task 5: leakage-safe preprocessing")
    assert X_train.isna().sum().sum() == 0 and X_test.isna().sum().sum() == 0, "missing feature value"
    print("missing values across all feature columns: 0")

    section("Task 6a: baseline (always predict not lapsed)")
    baseline = np.zeros(len(y_test), dtype=int)
    base = lapsed_metrics(y_test, baseline)
    print(f"test accuracy {accuracy_score(y_test, baseline) * 100:.1f}%; lapsed precision {base['precision']:.3f}, "
          f"recall {base['recall']:.3f}, F1 {base['f1']:.3f}")

    section("Task 6b-c: grid searches (fitted on training rows only)")
    lr_search = GridSearchCV(
        Pipeline([("prep", make_preprocessor()), ("clf", LogisticRegression(max_iter=1000))]),
        {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_train, y_train)
    rf_search = GridSearchCV(
        Pipeline([("prep", make_preprocessor()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]),
        {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1").fit(X_train, y_train)
    best_lr, best_rf = lr_search.best_estimator_, rf_search.best_estimator_
    print(f"logistic regression: best {lr_search.best_params_}; CV F1 {lr_search.best_score_:.3f}; "
          f"fits {len(lr_search.cv_results_['params']) * 5}")
    print(f"random forest: best {rf_search.best_params_}; CV F1 {rf_search.best_score_:.3f}; "
          f"fits {len(rf_search.cv_results_['params']) * 5}")

    print(md_table(pd.DataFrame({"C": [str(p["clf__C"]) for p in lr_search.cv_results_["params"]],
                                 "cv_f1": lr_search.cv_results_["mean_test_score"].round(4)}), floatfmt="{:.4f}"))

    freq_idx = NUMERIC.index("frequency")
    learned = best_lr.named_steps["prep"].named_transformers_["num"].mean_[freq_idx]
    print(f"scaler mean of frequency (training rows only): {learned:.4f}; full-cohort mean: {X['frequency'].mean():.4f}")

    section("Task 6d: untouched test set")
    results = {}
    rows = []
    for name, model in [("logistic_regression", best_lr), ("random_forest", best_rf)]:
        pred = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        m = lapsed_metrics(y_test, pred)
        results[name] = m
        rows.append({"model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp, "precision": round(m["precision"], 3),
                     "recall": round(m["recall"], 3), "f1": round(m["f1"], 3)})
    print(md_table(pd.DataFrame(rows), floatfmt="{:.3f}"))
    selected_name = max(results, key=lambda n: results[n]["f1"])
    selected = {"logistic_regression": best_lr, "random_forest": best_rf}[selected_name]
    print(f"selected: {selected_name}")
    assert results[selected_name]["f1"] >= 0.55, "selected model is below the required test F1"

    section("Task 7: class-balance experiment (training rows only)")
    train_lapsed = X_train[y_train == 1]
    n_extra = int((y_train == 0).sum() - (y_train == 1).sum())
    extra = resample(train_lapsed, replace=True, n_samples=n_extra, random_state=42)
    X_balanced = pd.concat([X_train, extra])
    y_balanced = pd.concat([y_train, pd.Series(1, index=extra.index, name=y_train.name)])
    print(f"balanced training counts: {y_balanced.value_counts().to_dict()}")
    balanced_lr = clone(best_lr).fit(X_balanced, y_balanced)
    before, after = results["logistic_regression"], lapsed_metrics(y_test, balanced_lr.predict(X_test))
    tn, fp, fn, tp = confusion_matrix(y_test, balanced_lr.predict(X_test)).ravel()
    print(md_table(pd.DataFrame([
        {"metric": k, "unbalanced": round(before[k], 3), "balanced": round(after[k], 3),
         "change": round(after[k] - before[k], 3)} for k in ["precision", "recall", "f1"]]), floatfmt="{:.3f}"))
    print(f"balanced confusion: TN {tn}, FP {fp}, FN {fn}, TP {tp}")

    section("Task 8a: interpretation")
    rf_names = [clean_name(n) for n in best_rf.named_steps["prep"].get_feature_names_out()]
    importance = pd.Series(best_rf.named_steps["clf"].feature_importances_, index=rf_names).sort_values(ascending=False)
    print(md_table(importance.head(5).round(3).rename("importance").rename_axis("feature").reset_index(), floatfmt="{:.3f}"))
    lr_names = [clean_name(n) for n in best_lr.named_steps["prep"].get_feature_names_out()]
    coef = pd.Series(best_lr.named_steps["clf"].coef_[0], index=lr_names)
    print(md_table(coef.reindex(coef.abs().sort_values(ascending=False).index).round(3).rename("coefficient")
                   .rename_axis("feature").reset_index(), floatfmt="{:.3f}"))

    section("Task 8b: persistence")
    model_path = f"{OUTPUTS}/lapse_model.joblib"
    joblib.dump(selected, model_path)
    reloaded = joblib.load(model_path)
    identical = np.array_equal(selected.predict(X_test), reloaded.predict(X_test))
    print(f"saved {selected_name} to outputs/lapse_model.joblib; reloaded predictions identical: {identical}")
    assert identical

    section("Task 8c: retention list (illustrative)")
    segments = pd.read_csv(f"{OUTPUTS}/customer_segments.csv")
    premium_ids = set(segments.loc[segments["is_premium"], "customer_id"])
    scored = pd.DataFrame({"customer_id": X_test.index,
                           "lapse_probability": reloaded.predict_proba(X_test)[:, 1].round(4),
                           "actually_lapsed": y_test.to_numpy()})
    premium_test = scored[scored["customer_id"].isin(premium_ids)]
    retention = premium_test.sort_values(["lapse_probability", "customer_id"], ascending=[False, True]).head(10)
    retention.to_csv(f"{OUTPUTS}/retention_list.csv", index=False)
    print(f"premium customers in test set: {len(premium_test)}, of whom truly lapsed: {int(premium_test['actually_lapsed'].sum())}; "
          f"listed: {len(retention)}; truly lapsed among listed: {int(retention['actually_lapsed'].sum())}")
    print(md_table(retention, floatfmt="{:.4f}"))
    print("\nwritten: outputs/lapse_model.joblib, outputs/retention_list.csv")


if __name__ == "__main__":
    main()
