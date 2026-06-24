"""
TASK 4: Sales Prediction 
================================================================

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams['figure.dpi'] = 110
np.random.seed(202)


n = 320
influencer_spend_k = np.round(np.random.uniform(0, 80, n), 1)     # $ thousands
paid_ads_spend_k = np.round(np.random.uniform(0, 150, n), 1)
content_pieces = np.random.randint(2, 40, n)                      # number of posts/reels/videos
reach_k = np.round(np.random.uniform(20, 500, n), 1)               # thousands reached
campaign_type = np.random.choice(['Awareness', 'Conversion', 'Hybrid'], n, p=[0.3, 0.35, 0.35])
platform_primary = np.random.choice(['Instagram', 'YouTube', 'Multi-platform'], n, p=[0.4, 0.3, 0.3])


sales = (
    3.0
    + 0.085 * influencer_spend_k**0.95
    + 0.032 * paid_ads_spend_k
    + 0.11 * content_pieces
    + 0.026 * reach_k
    + 0.00045 * influencer_spend_k * reach_k / 50
    + np.random.normal(0, 1.1, n)
)
campaign_adj = pd.Series(campaign_type).map({'Awareness': 0.2, 'Conversion': 1.4, 'Hybrid': 0.9}).values
platform_adj = pd.Series(platform_primary).map({'Instagram': 0.5, 'YouTube': 0.3, 'Multi-platform': 1.1}).values
sales = np.clip(sales + campaign_adj + platform_adj, 1, None).round(2)

df = pd.DataFrame({
    'influencer_spend_k': influencer_spend_k,
    'paid_ads_spend_k': paid_ads_spend_k,
    'content_pieces': content_pieces,
    'reach_k': reach_k,
    'campaign_type': campaign_type,
    'platform_primary': platform_primary,
    'sales_units_k': sales
})
df.to_csv('digital_campaign_data.csv', index=False)

print("="*60)
print("DATASET OVERVIEW — Digital Campaign Sales Dataset")
print("="*60)
print(df.head())
print("\nShape:", df.shape)
print("\nMissing values:", df.isnull().sum().sum())
print("\nStatistics:\n", df.describe())
print("\nCorrelation with sales:\n",
      df[['influencer_spend_k','paid_ads_spend_k','content_pieces','reach_k','sales_units_k']]
      .corr()['sales_units_k'].sort_values(ascending=False))


fig, ax = plt.subplots(figsize=(9, 7))
type_colors = {'Awareness': '#F2A93B', 'Conversion': '#3D9970', 'Hybrid': '#5B6EE1'}
for ctype, color in type_colors.items():
    sub = df[df.campaign_type == ctype]
    ax.scatter(sub['influencer_spend_k'], sub['reach_k'],
               s=sub['sales_units_k']*8, alpha=0.5, color=color,
               edgecolor='black', linewidth=0.3, label=ctype)

ax.set_xlabel("Influencer Spend ($k)")
ax.set_ylabel("Reach (thousands)")
ax.set_title("Spend Efficiency Map — bubble size = sales generated")
ax.legend(title="Campaign Type", loc='upper left')
plt.tight_layout()
plt.savefig('campaign_spend_efficiency_bubble.png', bbox_inches='tight')
plt.close()


df_enc = pd.get_dummies(df, columns=['campaign_type', 'platform_primary'], drop_first=True)
X = df_enc.drop(columns=['sales_units_k'])
y = df_enc['sales_units_k']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=202)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=8, random_state=202),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=202),
}

print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = {"model": model, "mae": mae, "rmse": rmse, "r2": r2}
    print(f"{name:20s} | MAE: {mae:.3f}k units | RMSE: {rmse:.3f}k units | R²: {r2:.4f}")

best_name = max(results, key=lambda n: results[n]["r2"])
best_model = results[best_name]["model"]
print(f"\nBest model: {best_name} (R² = {results[best_name]['r2']:.4f})")


base = X_train.mean()
scenarios = {}
for ch in ['influencer_spend_k', 'paid_ads_spend_k', 'content_pieces']:
    boosted = base.copy()
    boosted[ch] = boosted[ch] * 1.20
    base_pred = best_model.predict(scaler.transform(pd.DataFrame([base])))[0]
    boosted_pred = best_model.predict(scaler.transform(pd.DataFrame([boosted])))[0]
    scenarios[ch] = boosted_pred - base_pred

print("\n" + "="*60)
print("ROI SIMULATION: Sales lift from +20% increase in each lever")
print("="*60)
for ch, lift in scenarios.items():
    print(f"{ch:22s} -> +{lift:.3f}k units in sales")
best_lever = max(scenarios, key=scenarios.get)
print(f"\nInsight: '{best_lever}' gives the strongest marginal sales lift — "
      f"prioritize incremental budget there.")

print("\nDone. Files saved ")
