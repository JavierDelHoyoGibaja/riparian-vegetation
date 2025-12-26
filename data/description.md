# Project description

This dataset contains field and derived data describing instream large wood presence (LW) and riparian and geomorphical conditions along several river reaches in Alpine and pre-Alpine catchments.

Data are organised by river reach and riparian units located on both river banks. Each reach includes two riparian units (Left and Right bank), allowing the comparison of bank-specific conditions while retaining shared reach-scale information.

The main objective of the dataset is to understand how predictor variables relate to the target variables, considering not only local conditions (n) but also upstream conditions (n-1). Because rivers function as connected systems, processes occurring in upstream reaches can influence downstream instream wood presence.

This upstream–downstream dependency is explicitly considered only when the target variable is LW_Presence, as instream wood is transported within the river network and is therefore  part of a cascading longitudinal process.

When Dead_Wood is used as the target variable, only local riparian conditions (n) are considered, since this wood originates from the adjacent forest and is not transported by the river.

The dataset is intended for statistical and machine-learning analyses, particularly Random Forest models, and for the development of simplified decision rules to support river management and restoration planning.

# Variables description

All rows correspond to field sample points collected along several river reaches.

Each sample area (defined by Id_Place and Reach) contains two sample points, one on each river bank. These points are identified by Id_Data and Cod_Plg, and the corresponding side of the river is given by Bank (Left / Right).

Some variables (forest data) differ between river banks, while others (river geomorphological data) are shared by both sides of the same sample area/reach. For this reason, the analytical unit is Cod_Plg, but it is essential to consider whether a variable is:

	(S) : Shared — same value for Left and Right banks
	(NS) : Not shared — bank-specific value

This distinction is explicitly indicated for each variable below.

## Categorical variables 

**Id_RipUnit** : Unique identifier of the RipUnit (bank-specific)

**Id_Reach** : Unique identifier of the river reach

**Basin** : Main river basin

**Sub_Basin** : iver or tributary where the reach is located (see Figure "Graph scheme")

**Reach** : Sample point - River reach.

**Bank** : For each Reach, side of the river bank where the riparian unit is located (Left or Right)

**RipUnit** : Riparian Unit. Sampling unit associated with a given Reach and Bank


## Target variable

**LW_Presence** : (S) *Ordinal 1-4*. Amount of instream large wood found in the river reach. 1 = no wood; 4 = high wood presence. 
This is the main target variable in the modelling framework.


## Target variable and predictor variable

**Dead_Wood** : (NS) *Ordinal 1-4*. How much wood was found on the floodplain at that sample point. 1 = no dead wood, 4 = high abundance. 
This variable is used as a predictor when LW_Presence is the target variable. In complementary analyses, Dead_Wood is also analysed as a target variable, but LW_Presence is excluded from the predictor set in those cases, as the relationship between instream wood and floodplain dead wood is not assumed to be bidirectional.


## Predictor variables

**Sinuosity** : (S) *Continuous*. Channel sinuosity of the river reach.

**Width_Mean** : (S) *Continuous*. Mean channel width of the reach.

**Gradient (%)** : (S) *Continuous*. Mean longitudinal slope of the river reach

**SPI / Width** : (S) *Continuous*. Stream Power Index normalized by mean channel width.

**SPI** : (S) *Continuous*. Stream Power Index of the reach.

**Lentgh (m)** : (S) *Continuous*. Length of the river reach.

**Lat_Connectivity** : (NS) *Ordinal 1-4*. Lateral connectivity between the floodplain and the river channel. 1 =channalized, 4 = natural connectivity.

**Standing_Dead_Trees** : (NS) *Ordinal 1-4*. Abundance of standing dead trees within the riparian unit. 1 means no one, 4 means a lot.

**Regeneration** : (NS) *Ordinal 1-4*. Regeneration status. 1 = low, 4= high.

**Basal_Area (m2/ha)** : (NS) *Continuous*. Total basal area of riparian trees within the riparian unit

**P50_Height** : (NS) *Continuous*. Median (50th percentile) tree height derived from structural data.

**Height_IQR** : (NS) *Continuous*. Interquartile range of tree heights, representing vertical structural heterogeneity.

**HardWood_Ab** : (NS) *Binary*. Presence (1) or absence (0) of hardwood species within the riparian unit.

**SoftWood_Ab** : (NS) *Discrete / Continuous*. Number of softwood species present within the riparian unit.

**Pioneers_Ab** : (NS) *Discrete / Continuous*. Number of pioneer species present within the riparian unit.

**Shrubs_Ab**  : (NS) *Discrete / Continuous*. Number of shrub species present within the riparian unit.

**Brambles_Ab** : (NS) *Discrete / Continuous*. Number of bramble species present within the riparian unit.

**Invasive_Ab** : (NS) *Discrete / Continuous*. Number of invasive species present within the riparian unit.

**MaxDiamClass** : (NS) *Ordinal*. Maximum tree diameter class observed within the riparian unit. 1 means maximum a range of 10-20cm, 5 means over 50cm of diameter

**DiamComplex** : (NS) *Ordinal*. Index describing the heterogeneity of tree diameter classes.

**StructuralIndex** : (NS) *Continuous*. Composite index integrating riparian structural and ecological diversity.






