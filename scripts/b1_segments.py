"""Part B, tasks 1-3: RFM table, K-means model selection, named segments, PCA, customer_segments.csv."""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from common import CLEAN, OUTPUTS, ensure_dirs, md_table, section

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FEATURES = ["recency", "frequency", "monetary"]
FINAL_K = 4
ACTIONS = {
    "Premium High-Value": "Give priority support and early access to new electronics; avoid blanket discounts on customers who already pay full value.",
    "Frequent Loyalists": "Offer a staples subscription and cross-sell personal care to raise basket size, since visit frequency is already high.",
    "Occasional Shoppers": "Send category-specific reminders and small bundles to lift them from a few orders to a monthly habit.",
    "Dormant Customers": "Run one capped win-back coupon, then stop contacting those who do not respond.",
}


def build_rfm():
    orders = pd.read_csv(f"{CLEAN}/orders.csv", parse_dates=["order_ts"])
    done = orders[orders["status"] == "completed"]
    rfm = done.groupby("customer_id").agg(last_order=("order_ts", "max"), frequency=("order_id", "nunique"),
                                          monetary=("order_amount", "sum")).reset_index()
    rfm["recency"] = (SNAPSHOT - rfm["last_order"]).dt.days
    rfm["monetary"] = rfm["monetary"].round(2)
    return rfm[["customer_id", "recency", "frequency", "monetary"]].sort_values("customer_id").reset_index(drop=True)


def name_segments(profile):
    """Names come from each segment's numbers, never from cluster ids, which are arbitrary labels."""
    remaining = profile.copy()
    names = {}
    premium = remaining["monetary"].idxmax()
    names[premium] = "Premium High-Value"
    remaining = remaining.drop(premium)
    loyal = remaining["frequency"].idxmax()
    names[loyal] = "Frequent Loyalists"
    remaining = remaining.drop(loyal)
    dormant = remaining["recency"].idxmax()
    names[dormant] = "Dormant Customers"
    remaining = remaining.drop(dormant)
    names[remaining.index[0]] = "Occasional Shoppers"
    return names, premium


def main():
    ensure_dirs(OUTPUTS)
    rfm = build_rfm()

    section("Task 1: RFM table")
    print(f"customers: {len(rfm):,}")
    print(f"recency  median {rfm['recency'].median():.0f}, min {rfm['recency'].min()}, max {rfm['recency'].max()}")
    print(f"frequency mean {rfm['frequency'].mean():.2f}, max {rfm['frequency'].max()}")
    print(f"monetary mean {rfm['monetary'].mean():,.2f}, max {rfm['monetary'].max():,.2f}")
    assert len(rfm) == 1175

    scaled = StandardScaler().fit_transform(rfm[FEATURES])

    section("Task 2: K-means inertia and silhouette")
    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaled)
        rows.append({"K": k, "inertia": round(km.inertia_, 1), "silhouette": round(silhouette_score(scaled, km.labels_), 3)})
    table = pd.DataFrame(rows)
    table["inertia_drop_pct"] = (-table["inertia"].pct_change() * 100).round(1)
    print(md_table(table, floatfmt="{:.3f}"))

    final = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(scaled)
    rfm["segment_id"] = final.labels_

    section(f"Task 3: segment profile for K={FINAL_K}")
    profile = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                            frequency=("frequency", "mean"), monetary=("monetary", "mean"))
    names, premium = name_segments(profile)
    rfm["segment_name"] = rfm["segment_id"].map(names)
    rfm["is_premium"] = rfm["segment_id"] == premium

    shown = profile.round(2).reset_index()
    shown.insert(1, "segment_name", shown["segment_id"].map(names))
    shown["action"] = shown["segment_name"].map(ACTIONS)
    print(md_table(shown))
    assert rfm.loc[rfm["is_premium"], "segment_id"].nunique() == 1
    assert profile.loc[premium, "monetary"] == profile["monetary"].max()

    pca = PCA(n_components=2, random_state=42).fit(scaled)
    ratios = pca.explained_variance_ratio_
    section("Task 3c: PCA")
    print(f"explained_variance_ratio_: {ratios[0]:.3f}, {ratios[1]:.3f}; total {ratios.sum():.3f}")
    loadings = pd.DataFrame(pca.components_, columns=FEATURES, index=["PC1", "PC2"]).round(3).reset_index()
    print(md_table(loadings, floatfmt="{:.3f}"))

    columns = ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]
    rfm[columns].to_csv(f"{OUTPUTS}/customer_segments.csv", index=False)
    print("\nwritten: outputs/customer_segments.csv")


if __name__ == "__main__":
    main()
