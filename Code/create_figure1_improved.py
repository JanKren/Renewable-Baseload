# -*- coding: utf-8 -*-
"""
Improved Figure 1: European Wind Baseload Analysis (2015-2024)
Shows both the positive (non-zero baseload) and critical (declining efficiency) findings

Panels:
a) Hourly capacity factor time series (existing)
b) 10-year evolution showing BOTH minimum and baseload ratio
c) Baseload conversion efficiency - the key skeptical finding

Author: Jan Kren
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os

# Configuration
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures_Paper")
os.makedirs(FIGURE_DIR, exist_ok=True)

# Data from Table 1 in the paper
data = {
    'year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'countries': [27, 27, 27, 27, 28, 28, 28, 29, 29, 29],
    'minimum_gw': [6.7, 5.5, 8.2, 7.1, 8.4, 8.8, 7.7, 11.1, 9.9, 10.0],
    'mean_gw': [32.5, 32.0, 38.2, 39.9, 45.7, 50.4, 49.3, 54.7, 59.8, 61.0],
    'cv': [0.41, 0.41, 0.42, 0.45, 0.41, 0.44, 0.45, 0.44, 0.45, 0.47],
    'capacity_gw': [142, 154, 169, 189, 205, 220, 236, 255, 272, 300]  # WindEurope data
}

df = pd.DataFrame(data)

# Calculate baseload ratio (minimum / capacity)
df['baseload_ratio'] = df['minimum_gw'] / df['capacity_gw'] * 100

# Calculate marginal conversion (change in minimum / change in capacity from previous year)
df['delta_min'] = df['minimum_gw'].diff()
df['delta_cap'] = df['capacity_gw'].diff()
df['marginal_conversion'] = df['delta_min'] / df['delta_cap'] * 100

# Overall marginal conversion (2015 to 2024)
total_delta_min = df['minimum_gw'].iloc[-1] - df['minimum_gw'].iloc[0]
total_delta_cap = df['capacity_gw'].iloc[-1] - df['capacity_gw'].iloc[0]
overall_marginal = total_delta_min / total_delta_cap * 100

print(f"Total capacity added: {total_delta_cap} GW")
print(f"Total baseload added: {total_delta_min} GW")
print(f"Overall marginal conversion: {overall_marginal:.1f}%")
print(f"GW capacity needed per 1 GW baseload: {total_delta_cap/total_delta_min:.1f}")

# Create figure
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
                      'xtick.labelsize': 11, 'ytick.labelsize': 11})
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel a: Minimum and Mean production with dual y-axis
ax1 = axes[0]
ax1_twin = ax1.twinx()

# Bar chart for minimum
bars = ax1.bar(df['year'], df['minimum_gw'], color='#d62728', alpha=0.8, label='Minimum (baseload)')
ax1.set_ylabel('Minimum Production [GW]', color='#d62728')
ax1.tick_params(axis='y', labelcolor='#d62728')
ax1.set_ylim(0, 15)

# Line for mean
ax1_twin.plot(df['year'], df['mean_gw'], 'o-', color='#1f77b4', linewidth=2, markersize=8, label='Mean')
ax1_twin.set_ylabel('Mean Production [GW]', color='#1f77b4')
ax1_twin.tick_params(axis='y', labelcolor='#1f77b4')
ax1_twin.set_ylim(0, 75)

# Add trend line for minimum
z = np.polyfit(df['year'], df['minimum_gw'], 1)
p = np.poly1d(z)
ax1.plot(df['year'], p(df['year']), '--', color='#d62728', alpha=0.5, linewidth=1.5)
ax1.annotate(f'+{z[0]:.2f} GW/year', xy=(2017, 13), fontsize=12, color='#d62728')

# Add trend line and annotation for mean production
z_mean = np.polyfit(df['year'], df['mean_gw'], 1)
p_mean = np.poly1d(z_mean)
ax1_twin.plot(df['year'], p_mean(df['year']), '--', color='#1f77b4', alpha=0.5, linewidth=1.5)
ax1_twin.annotate(f'+{z_mean[0]:.1f} GW/year', xy=(2019, 68), fontsize=12, color='#1f77b4')

ax1.set_xlabel('Year')
ax1.set_title('a) 10-Year Evolution', fontweight='bold')
ax1.set_xticks(df['year'][::2])

# Panel b: Baseload ratio (declining!)
ax2 = axes[1]

colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df)))  # Red to green reversed (red = worse)
bars = ax2.bar(df['year'], df['baseload_ratio'], color=colors, edgecolor='black', linewidth=0.5)

# Add trend line
z2 = np.polyfit(df['year'], df['baseload_ratio'], 1)
p2 = np.poly1d(z2)
ax2.plot(df['year'], p2(df['year']), 'k--', linewidth=2, label=f'Trend: {z2[0]:.2f}%/year')

# Annotate start and end values
ax2.annotate(f'{df["baseload_ratio"].iloc[0]:.1f}%',
             xy=(2015, df['baseload_ratio'].iloc[0] + 0.15),
             ha='center', fontsize=13, fontweight='bold')
ax2.annotate(f'{df["baseload_ratio"].iloc[-1]:.1f}%',
             xy=(2024, df['baseload_ratio'].iloc[-1] + 0.4),
             ha='center', fontsize=13, fontweight='bold')

ax2.axhline(y=df['baseload_ratio'].mean(), color='gray', linestyle=':', alpha=0.7)

ax2.set_xlabel('Year')
ax2.set_ylabel('Baseload Ratio [%]')
ax2.set_title('b) Baseload as % of Capacity', fontweight='bold')
ax2.set_ylim(0, 6)
ax2.set_xticks(df['year'][::2])
ax2.legend(loc='upper right', fontsize=11)

# Add annotation for the decline
decline = df['baseload_ratio'].iloc[0] - df['baseload_ratio'].iloc[-1]
ax2.annotate(f'↓ {decline:.1f}% decline',
             xy=(2019.5, 2.0), fontsize=14, color='#d62728', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='#d62728', alpha=0.8))

# Panel c: Marginal conversion efficiency
ax3 = axes[2]

# Show the key finding: capacity added vs baseload added
capacity_added = [df['capacity_gw'].iloc[-1] - df['capacity_gw'].iloc[0]]
baseload_added = [df['minimum_gw'].iloc[-1] - df['minimum_gw'].iloc[0]]

# Create grouped bar showing the stark contrast
x_pos = [0, 1]
heights = [capacity_added[0], baseload_added[0]]
colors = ['#1f77b4', '#d62728']
labels = ['Capacity\nadded', 'Baseload\nadded']

bars = ax3.bar(x_pos, heights, color=colors, width=0.6, edgecolor='black', linewidth=1)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(labels, fontsize=13)
ax3.set_ylabel('GW')

# Add value labels on bars
for bar, val in zip(bars, heights):
    ax3.annotate(f'{val:.1f} GW',
                 xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 3),
                 ha='center', fontsize=14, fontweight='bold')

ax3.set_title('c) Marginal Conversion', fontweight='bold')
ax3.set_ylim(0, 180)

# Add the key ratio
ratio_text = f"Conversion rate:\n{overall_marginal:.1f}%"
ax3.annotate(ratio_text, xy=(0.5, 100), fontsize=14, fontweight='bold',
             ha='center', color='#d62728',
             bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='#d62728', alpha=0.9))

# Add implication
ax3.annotate(f"Need ~{total_delta_cap/total_delta_min:.0f} GW capacity\nper 1 GW baseload",
             xy=(0.5, 60), fontsize=10, ha='center', style='italic',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()

# Save
plt.savefig(os.path.join(FIGURE_DIR, 'figure1_baseload.pdf'),
            bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(FIGURE_DIR, 'figure1_baseload.png'),
            bbox_inches='tight', dpi=300)
print(f"Saved to {FIGURE_DIR}")
plt.close()
