import json

file_path = r"c:\Users\jdelhoyo\PhD\Study cases\Genissiat\RV Characterization\repo-github\Clone\riparian-vegetation\Riparian_Forest_Analysis3WRhone(NIndex) 3 Clean VariablesSubBasin.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and fix the cell
for cell in nb['cells']:
    if cell.get('id') == 'b65dada0':
        code = """# ===================== PASO 3 VISUALIZATION: BOXPLOTS BY RANKING =====================
print('\\n' + '='*120)
print('BOXPLOTS: Predictor Distributions Ranked by Evidence Strength')
print('='*120)


# ==================== DEAD_WOOD BOXPLOTS ====================
print(f'\\n✓ Dead_Wood Model: {len(df_rank_dw)} predictors')

# Variable name mapping for better display with units
var_name_map = {
    'Standing_Dead_Trees': 'Standing Dead Trees (m)',
    'Height_IQR': 'Height IQR (m)',
    'Invasive_Ab': 'Invasive sp',
    'P50_Height': 'Height p50 (m)',
}

# Build palette list for Dead_Wood (values 1-4)
dw_palette_for_boxplot = [dw_class_colors.get(i, '#cccccc') for i in [1, 2, 3, 4]]

fig_dw, axes_dw = plt.subplots(2, 3, figsize=(18, 13))
axes_dw = axes_dw.flatten()

for idx, (i, row) in enumerate(df_rank_dw.iterrows()):
    if idx >= len(axes_dw):
        break
    
    pred = row['Predictor']
    rho = row['Spearman_rho']
    p_spear = row['Spearman_p']
    p_kw = row['Kruskal_p']
    ranking_pos = row['Ranking_position']
    evidence = row['Evidence_class']
    
    # Prepare data
    plot_data = df[[dead_wood_response, pred]].dropna()
    n_samples = len(plot_data)
    
    if n_samples > 0:
        # Create boxplot with custom palette
        sns.boxplot(data=plot_data, x=dead_wood_response, y=pred, ax=axes_dw[idx],
                   palette=dw_palette_for_boxplot, order=['1', '2', '3', '4'],
                   width=0.6, linewidth=1.5, patch_artist=True)
        
        # Overlay individual points
        sns.stripplot(data=plot_data, x=dead_wood_response, y=pred, ax=axes_dw[idx],
                     order=['1', '2', '3', '4'], size=5, alpha=0.6,
                     color='black', jitter=True, edgecolor='gray', linewidth=0.5)
        
        axes_dw[idx].set_xlabel('Dead_Wood Value (1-4)', fontsize=13)
        axes_dw[idx].set_ylabel(pred, fontsize=14)
        
        # Format p-values with <0.001 threshold
        spear_p_str = '<0.001' if p_spear < 0.001 else f'{p_spear:.3f}'
        kw_p_str = '<0.001' if p_kw < 0.001 else f'{p_kw:.3f}'
        
        # Get formatted variable name (with units) or default to pred with dashes replaced by spaces
        pred_display = var_name_map.get(pred, pred.replace('_', ' '))
        
        # Set title with variable name and statistics on two lines
        title_text = f'{pred_display}\\nρ = {rho:.3f} ({spear_p_str}) | KW p = {kw_p_str} | n = {n_samples}'
        axes_dw[idx].set_title(title_text, fontsize=14, fontweight='bold')
        axes_dw[idx].grid(axis='y', alpha=0.3)

# Hide unused axes
for idx in range(len(df_rank_dw), len(axes_dw)):
    axes_dw[idx].set_visible(False)

plt.suptitle(' ', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print(f'\\n✓ Dead_Wood boxplots created: {len(df_rank_dw)} predictors displayed')

# ==================== LW_PRESENCE BOXPLOTS ====================
print(f'\\n✓ LW_Presence Model: {len(df_rank_lw)} predictors')

# Build palette list for LW_Presence (values 1-4)
lw_palette_for_boxplot = [lw_class_colors.get(i, '#cccccc') for i in [1, 2, 3, 4]]

fig_lw, axes_lw = plt.subplots(3, 3, figsize=(18, 15))
axes_lw = axes_lw.flatten()

ax_idx = 0  # Separate counter for axes to avoid empty subplots when skipping predictors

for i, row in df_rank_lw.iterrows():
    pred = row['Predictor']
    
    # Skip Distance to outlet from LW_Presence visualization
    if pred == 'Distance to outlet (km)':
        continue
    
    # Skip Regeneration from LW_Presence visualization
    if pred == 'Regeneration':
        continue
    
    if ax_idx >= len(axes_lw):
        break
    
    rho = row['Spearman_rho']
    p_spear = row['Spearman_p']
    p_kw = row['Kruskal_p']
    ranking_pos = row['Ranking_position']
    evidence = row['Evidence_class']
    
    # Prepare data
    plot_data = df[[lw_response, pred]].dropna()
    n_samples = len(plot_data)
    
    if n_samples > 0:
        # Create boxplot with custom palette
        sns.boxplot(data=plot_data, x=lw_response, y=pred, ax=axes_lw[ax_idx],
                   palette=lw_palette_for_boxplot, order=['1', '2', '3', '4'],
                   width=0.6, linewidth=1.5, patch_artist=True)
        
        # Overlay individual points
        sns.stripplot(data=plot_data, x=lw_response, y=pred, ax=axes_lw[ax_idx],
                     order=['1', '2', '3', '4'], size=5, alpha=0.6,
                     color='black', jitter=True, edgecolor='gray', linewidth=0.5)
        
        axes_lw[ax_idx].set_xlabel('LW_Presence Value (1-4)', fontsize=14)
        axes_lw[ax_idx].set_ylabel(pred, fontsize=14)
        
        # Format p-values
        spear_p_str = '<0.001' if p_spear < 0.001 else f'{p_spear:.3f}'
        kw_p_str = '<0.001' if p_kw < 0.001 else f'{p_kw:.3f}'
        
        # Get formatted variable name (with units) or default to pred with dashes replaced by spaces
        pred_display = var_name_map.get(pred, pred.replace('_', ' '))
        
        # Set title with variable name and statistics on two lines
        title_text = f'{pred_display}\\nρ = {rho:.3f} ({spear_p_str}) | KW p = {kw_p_str} | n = {n_samples}'
        axes_lw[ax_idx].set_title(title_text, fontsize=14, fontweight='bold')
        axes_lw[ax_idx].grid(axis='y', alpha=0.3)
        
        ax_idx += 1


# Hide unused axes
for idx in range(ax_idx, len(axes_lw)):
    axes_lw[idx].set_visible(False)

plt.suptitle(' ', 
             fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print(f'\\n✓ LW_Presence boxplots created: {len(df_rank_lw)} predictors displayed')
print(f'\\n✓ All visualizations completed. Compare distributions with ranking evidence.')"""
        
        # Split into source array with proper line endings
        lines = code.split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
        print(f"Fixed cell - {len(cell['source'])} lines")

# Save the fixed notebook
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✓ Notebook saved successfully")
