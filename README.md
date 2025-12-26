# Riparian Vegetation Characterization

- **Full title**
- **Authors**: 
- **Published**
- **Contact**

## Abstact
Analysis of riparian forest structure and dynamics in the Arve and Valserine river basins. This project examines relationships between forest structural variables and large wood presence, with implications for river habitat characterization and management.

The objective is to understand how forest structure variables relate to:
- **Dead Wood** - Presence indicator of dead wood in the riparian zone
- **Large Wood (LW) Presence** - Presence indicator of large wood in the river

## Data Source
- Dataset: RV_For_RF2.xlsx
- Basins: Arve and Valserine (Rhone basin excluded)
- Samples: Multiple locations across different sub-basins
- Variables: 30+ forest structural, hydromorphological, and compositional features

## Usage
- Open `data-analysis.ipynb` to run the full analysis pipeline
- All visualizations are generated in-line with correlation heatmaps, scatter plots, and feature importance charts

## Requirements
- pandas, numpy, matplotlib, seaborn
- scikit-learn (RandomForest, StandardScaler)
- skrub (TableReport)