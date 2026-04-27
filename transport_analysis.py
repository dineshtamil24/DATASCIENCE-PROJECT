"""
Public Transport Usage Analysis
Dataset: Simulated from Kaggle-style public transport data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
})
PALETTE = ['#1a6faf', '#e05c2a', '#2ea84e', '#9b59b6', '#f39c12', '#16a085']

# ── 1. Generate Synthetic Dataset ────────────────────────────────────────────
np.random.seed(42)
n = 500

transport_modes = ['Bus', 'Metro', 'Train', 'Tram', 'Ferry', 'BRT']
cities          = ['Chennai', 'Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad']
days_of_week    = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
time_slots      = ['Morning Peak','Afternoon','Evening Peak','Night']
seasons         = ['Summer','Monsoon','Winter','Spring']

mode_weights    = [0.30, 0.25, 0.20, 0.12, 0.05, 0.08]
passenger_base  = {'Bus':320,'Metro':480,'Train':410,'Tram':180,'Ferry':95,'BRT':260}
fare_map        = {'Bus':15,'Metro':30,'Train':45,'Tram':20,'Ferry':35,'BRT':18}

data = {
    'Mode'         : np.random.choice(transport_modes, n, p=mode_weights),
    'City'         : np.random.choice(cities, n),
    'Day'          : np.random.choice(days_of_week, n),
    'Time_Slot'    : np.random.choice(time_slots, n),
    'Season'       : np.random.choice(seasons, n),
    'Delay_Min'    : np.abs(np.random.normal(5, 8, n)).astype(int),
    'Satisfaction' : np.clip(np.random.normal(3.5, 0.9, n), 1, 5).round(1),
    'CO2_Saved_kg' : np.random.uniform(0.5, 12, n).round(2),
}

data['Passengers'] = [
    int(passenger_base[m] * np.random.uniform(0.6, 1.4)) for m in data['Mode']
]
data['Fare_INR'] = [fare_map[m] + np.random.randint(-5, 10) for m in data['Mode']]
data['Revenue_INR'] = [p * f for p, f in zip(data['Passengers'], data['Fare_INR'])]

df = pd.DataFrame(data)

print("Dataset shape:", df.shape)
print("\nBasic Statistics:")
print(df.describe(include='all').to_string())

# ── 2. PLOT 1 – Bar: Ridership by Transport Mode ─────────────────────────────
mode_agg = df.groupby('Mode')['Passengers'].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(mode_agg.index, mode_agg.values, color=PALETTE, edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, mode_agg.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('Average Ridership by Transport Mode', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Transport Mode', fontsize=11)
ax.set_ylabel('Avg Passengers per Trip', fontsize=11)
ax.set_ylim(0, mode_agg.max() * 1.15)
plt.tight_layout()
plt.savefig('/home/claude/plot1_ridership_by_mode.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 1 saved.")

# ── 3. PLOT 2 – Grouped Bar: Ridership by City & Mode ────────────────────────
city_mode = df.groupby(['City','Mode'])['Passengers'].mean().unstack()

fig, ax = plt.subplots(figsize=(11, 5.5))
city_mode.plot(kind='bar', ax=ax, color=PALETTE, edgecolor='white', linewidth=0.6, width=0.75)
ax.set_title('Average Ridership by City and Transport Mode', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('City', fontsize=11)
ax.set_ylabel('Avg Passengers', fontsize=11)
ax.legend(title='Mode', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
ax.tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig('/home/claude/plot2_city_mode_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 2 saved.")

# ── 4. PLOT 3 – Bar: Revenue by Transport Mode ───────────────────────────────
rev_agg = df.groupby('Mode')['Revenue_INR'].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(rev_agg.index, rev_agg.values / 1e6, color=PALETTE, edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, rev_agg.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'₹{val/1e6:.2f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('Total Revenue by Transport Mode (INR Millions)', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Transport Mode', fontsize=11)
ax.set_ylabel('Revenue (₹ Millions)', fontsize=11)
ax.set_ylim(0, rev_agg.max() / 1e6 * 1.18)
plt.tight_layout()
plt.savefig('/home/claude/plot3_revenue_by_mode.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 3 saved.")

# ── 5. PLOT 4 – Bar: Delay by Time Slot ──────────────────────────────────────
slot_order = ['Morning Peak','Afternoon','Evening Peak','Night']
delay_agg  = df.groupby('Time_Slot')['Delay_Min'].mean().reindex(slot_order)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(delay_agg.index, delay_agg.values, color=['#e05c2a','#f39c12','#e05c2a','#1a6faf'],
              edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, delay_agg.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{val:.1f} min', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title('Average Delay by Time Slot', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Time Slot', fontsize=11)
ax.set_ylabel('Avg Delay (minutes)', fontsize=11)
ax.set_ylim(0, delay_agg.max() * 1.2)
plt.tight_layout()
plt.savefig('/home/claude/plot4_delay_by_slot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 4 saved.")

# ── 6. PLOT 5 – Stacked Bar: Ridership by Season & Mode ──────────────────────
season_mode = df.groupby(['Season','Mode'])['Passengers'].sum().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(10, 5.5))
season_mode.plot(kind='bar', stacked=True, ax=ax, color=PALETTE, edgecolor='white', linewidth=0.5, width=0.6)
ax.set_title('Total Ridership by Season and Transport Mode', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Season', fontsize=11)
ax.set_ylabel('Total Passengers', fontsize=11)
ax.legend(title='Mode', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
ax.tick_params(axis='x', rotation=15)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
plt.tight_layout()
plt.savefig('/home/claude/plot5_season_stacked.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 5 saved.")

# ── 7. PLOT 6 – Heatmap: Satisfaction by Day & Mode ──────────────────────────
day_order = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
heat_data = df.groupby(['Day','Mode'])['Satisfaction'].mean().unstack()
heat_data = heat_data.reindex(day_order)

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(heat_data, annot=True, fmt='.2f', cmap='YlOrRd', linewidths=0.5,
            ax=ax, cbar_kws={'label':'Avg Satisfaction (1–5)'})
ax.set_title('Passenger Satisfaction Heatmap (Day × Mode)', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Transport Mode', fontsize=11)
ax.set_ylabel('Day of Week', fontsize=11)
plt.tight_layout()
plt.savefig('/home/claude/plot6_satisfaction_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 6 saved.")

# ── 8. PLOT 7 – K-Means Clustering ───────────────────────────────────────────
features = df[['Passengers','Fare_INR','Delay_Min','Satisfaction','CO2_Saved_kg']]
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

fig, ax = plt.subplots(figsize=(9, 5.5))
cluster_colors = {0:'#1a6faf', 1:'#e05c2a', 2:'#2ea84e'}
cluster_labels = {0:'High-Cap Efficient', 1:'Budget Short-Haul', 2:'Premium Reliable'}
for c in [0,1,2]:
    mask = df['Cluster'] == c
    ax.scatter(df.loc[mask,'Passengers'], df.loc[mask,'Satisfaction'],
               color=cluster_colors[c], label=f'Cluster {c}: {cluster_labels[c]}',
               alpha=0.65, s=40, edgecolors='none')
ax.set_title('K-Means Clustering: Passengers vs Satisfaction', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Passengers per Trip', fontsize=11)
ax.set_ylabel('Satisfaction Score', fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('/home/claude/plot7_kmeans_cluster.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 7 saved.")

# ── 9. Summary Statistics ─────────────────────────────────────────────────────
print("\n─── Summary Table ───")
summary = df.groupby('Mode').agg(
    Trips       =('Mode','count'),
    Avg_Riders  =('Passengers','mean'),
    Avg_Fare    =('Fare_INR','mean'),
    Total_Rev   =('Revenue_INR','sum'),
    Avg_Delay   =('Delay_Min','mean'),
    Avg_Sat     =('Satisfaction','mean'),
    CO2_Saved   =('CO2_Saved_kg','sum'),
).round(2)
print(summary.to_string())

print("\nAll plots generated successfully.")
