# ============================================================
# BEST-SUBSET ORDINAL MODELS WITH EFFECTIVE LATERAL EXPOSURE (ELE)
#
# Purpose:
#   - Recalculate Effective Lateral Exposure (ELE) class.
#   - Screen candidate predictors bivariately.
#   - Fit many parsimonious ordinal logistic models for FLW and CLW.
#   - Compare candidate models using AIC, BIC, and delta AIC/BIC.
#   - Keep original models as reference.
#   - Diagnose the RF-CDD / StructuralIndex coefficient scale.
#
# Responses:
#   FLW = Dead_Wood
#   CLW = LW_Presence
#
# Recommended use:
#   Run this cell after df has been loaded.
# ============================================================

import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import spearmanr, kruskal, somersd
from statsmodels.miscmodels.ordinal_model import OrderedModel

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. OUTPUT DIRECTORY
# ------------------------------------------------------------
try:
    OUT_DIR
except NameError:
    OUT_DIR = Path.cwd() / "best_subset_ordinal_models_with_ELE"
    OUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# 2. SETTINGS
# ------------------------------------------------------------
STANDARDIZE_CONTINUOUS = True
MAX_PREDICTORS = 4
BIVARIATE_P_THRESHOLD = 0.25

# Keep these predictors in candidate pools even if bivariate p is weak,
# because they are conceptually central or were in the original models.
FORCE_KEEP_FLW = [
    "StandingDead_num",
    "BasalArea_model",
    "Regeneration_num",
    "RFCDD_model",
    "ELE_num"
]

FORCE_KEEP_CLW = [
    "SPIwidth_model",
    "Sinuosity_model",
    "FLW_num",
    "StandingDead_num",
    "ELE_num"
]

# Avoid reporting overly complex models as the final choice unless AIC improvement is large.
PREFER_BIC_FOR_PARSIMONY = True

VALID_RESPONSE_CLASSES = [1, 2, 3, 4]

# ------------------------------------------------------------
# 3. COLUMN NAMES AND COLUMN DISCOVERY
# ------------------------------------------------------------
FLW_COL = "Dead_Wood"
CLW_COL = "LW_Presence"

VALLEY_COL = "ValleyConfinIndex"
LAT_COL = "Lat_Connectivity"

NAT_CLASS_COL = "NaturalLatAccommodationClass"
ART_CAP_COL = "ArtificialExposureCap"
EFF_CLASS_COL = "EffLatExpClass"
DOWNGRADE_COL = "ClassDowngrade"

STANDING_COL = "Standing_Dead_Trees"
REGEN_COL = "Regeneration"
INVASIVE_COL = "Invasive_Ab"

SINUOSITY_COL = "Sinuosity"
SPI_COL = "SPI / Width"
GRADIENT_COL = "Gradient (%)"
WIDTH_COL = "Width_Mean"
DISTANCE_COL = "Distance to outlet (km)"
HEIGHT_COL = "P50_Height"
HEIGHT_IQR_COL = "Height_IQR"

BASAL_CANDIDATES = [
    "Basal_Area (m2/ha)",
    "Basal_Area",
    "Basal area",
    "Basal Area"
]

RFCDD_CANDIDATES = [
    "RF-CDD Index",
    "RF_CDD_Index",
    "StructuralIndex",
    "Structural Index"
]


def find_existing_col(candidates, dataframe):
    for col in candidates:
        if col in dataframe.columns:
            return col
    return None


BASAL_COL = find_existing_col(BASAL_CANDIDATES, df)
RFCDD_COL = find_existing_col(RFCDD_CANDIDATES, df)

if BASAL_COL is None:
    raise ValueError(f"Could not find basal area column. Tried: {BASAL_CANDIDATES}")

if RFCDD_COL is None:
    raise ValueError(f"Could not find RF-CDD / Structural index column. Tried: {RFCDD_CANDIDATES}")

base_required = [
    FLW_COL,
    CLW_COL,
    VALLEY_COL,
    LAT_COL,
    STANDING_COL,
    REGEN_COL,
    SINUOSITY_COL,
    SPI_COL,
    BASAL_COL,
    RFCDD_COL
]

missing = [col for col in base_required if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("Column mapping used in this analysis:")
print(f"  FLW response: {FLW_COL}")
print(f"  CLW response: {CLW_COL}")
print(f"  Basal area: {BASAL_COL}")
print(f"  RF-CDD / Structural index column actually used: {RFCDD_COL}")
print(f"  Valley confinement/accommodation proxy: {VALLEY_COL}")
print(f"  Lateral constraint class: {LAT_COL}")

# ------------------------------------------------------------
# 4. CREATE EFFECTIVE LATERAL EXPOSURE (ELE) CLASS
# ------------------------------------------------------------
data = df.copy()

for col in base_required:
    data[col] = pd.to_numeric(data[col], errors="coerce")


def classify_natural_accommodation(value):
    """
    Natural lateral accommodation class from ValleyConfinIndex.

    1 = confined / low natural accommodation
    2 = partly confined / moderate natural accommodation
    3 = unconfined / high natural accommodation
    """
    if pd.isna(value):
        return np.nan
    if value <= 2:
        return 1
    if value <= 4:
        return 2
    return 3


def artificial_exposure_cap(value):
    """
    Artificial lateral-exposure cap from Lat_Connectivity.

    Lat_Connectivity:
      4 -> cap 3
      3 -> cap 2
      2 -> cap 2
      1 -> cap 1
    """
    if pd.isna(value):
        return np.nan

    value = int(value)

    if value == 4:
        return 3
    if value == 3:
        return 2
    if value == 2:
        return 2
    if value == 1:
        return 1

    return np.nan


data[NAT_CLASS_COL] = data[VALLEY_COL].apply(classify_natural_accommodation)
data[ART_CAP_COL] = data[LAT_COL].apply(artificial_exposure_cap)
data[EFF_CLASS_COL] = data[[NAT_CLASS_COL, ART_CAP_COL]].min(axis=1)

data[NAT_CLASS_COL] = data[NAT_CLASS_COL].astype("Int64")
data[ART_CAP_COL] = data[ART_CAP_COL].astype("Int64")
data[EFF_CLASS_COL] = data[EFF_CLASS_COL].astype("Int64")
data[DOWNGRADE_COL] = data[NAT_CLASS_COL] - data[EFF_CLASS_COL]

df[NAT_CLASS_COL] = data[NAT_CLASS_COL]
df[ART_CAP_COL] = data[ART_CAP_COL]
df[EFF_CLASS_COL] = data[EFF_CLASS_COL]
df[DOWNGRADE_COL] = data[DOWNGRADE_COL]

ele_dist = (
    data[EFF_CLASS_COL]
    .value_counts(dropna=False)
    .sort_index()
    .rename_axis(EFF_CLASS_COL)
    .reset_index(name="n")
)

ele_dist["percent"] = 100 * ele_dist["n"] / ele_dist["n"].sum()

print("\nEffective Lateral Exposure (ELE) class distribution:")
display(ele_dist)

print("\nNumber of downgraded observations:")
print(f"{(data[DOWNGRADE_COL] > 0).sum()} / {len(data)}")

# ------------------------------------------------------------
# 5. MODEL DATASET
# ------------------------------------------------------------
optional_cols = [
    INVASIVE_COL,
    GRADIENT_COL,
    WIDTH_COL,
    DISTANCE_COL,
    HEIGHT_COL,
    HEIGHT_IQR_COL
]

existing_optional_cols = [col for col in optional_cols if col in data.columns]

all_cols_for_model = list(dict.fromkeys(base_required + existing_optional_cols + [EFF_CLASS_COL]))

model_data = data[all_cols_for_model].copy()

for col in model_data.columns:
    model_data[col] = pd.to_numeric(model_data[col], errors="coerce")

# Response and ordinal score variables
model_data["FLW_num"] = model_data[FLW_COL]
model_data["CLW_num"] = model_data[CLW_COL]
model_data["StandingDead_num"] = model_data[STANDING_COL]
model_data["Regeneration_num"] = model_data[REGEN_COL]
model_data["ELE_num"] = model_data[EFF_CLASS_COL]

if INVASIVE_COL in model_data.columns:
    model_data["Invasive_num"] = model_data[INVASIVE_COL]

# Continuous variables
continuous_map = {
    "BasalArea_model": BASAL_COL,
    "RFCDD_model": RFCDD_COL,
    "Sinuosity_model": SINUOSITY_COL,
    "SPIwidth_model": SPI_COL
}

if GRADIENT_COL in model_data.columns:
    continuous_map["Gradient_model"] = GRADIENT_COL

if WIDTH_COL in model_data.columns:
    continuous_map["Width_model"] = WIDTH_COL

if DISTANCE_COL in model_data.columns:
    continuous_map["Distance_model"] = DISTANCE_COL

if HEIGHT_COL in model_data.columns:
    continuous_map["HeightP50_model"] = HEIGHT_COL

if HEIGHT_IQR_COL in model_data.columns:
    continuous_map["HeightIQR_model"] = HEIGHT_IQR_COL

for new_col, original_col in continuous_map.items():
    if STANDARDIZE_CONTINUOUS:
        sd = model_data[original_col].std()

        if pd.isna(sd) or sd == 0:
            model_data[new_col] = np.nan
        else:
            model_data[new_col] = (
                model_data[original_col] - model_data[original_col].mean()
            ) / sd
    else:
        model_data[new_col] = model_data[original_col]

# ELE categorical dummies, reference = class 1
ele_dummies = pd.get_dummies(
    model_data["ELE_num"].astype("Int64"),
    prefix="ELE",
    drop_first=True
).astype(float)

model_data = pd.concat([model_data, ele_dummies], axis=1)
ELE_DUMMY_COLS = list(ele_dummies.columns)

# ------------------------------------------------------------
# 6. DISPLAY NAMES
# ------------------------------------------------------------
term_display = {
    "SPIwidth_model": "SPI / Width",
    "Sinuosity_model": "Sinuosity",
    "Gradient_model": "Gradient",
    "Width_model": "Mean channel width",
    "Distance_model": "Distance to outlet",
    "StandingDead_num": "Standing dead trees",
    "FLW_num": "FLW",
    "CLW_num": "CLW",
    "BasalArea_model": "Basal area",
    "Regeneration_num": "Regeneration",
    "RFCDD_model": "RF-CDD Index",
    "HeightP50_model": "Tree height P50",
    "HeightIQR_model": "Tree height IQR",
    "Invasive_num": "Invasive species",
    "ELE_num": "Effective Lateral Exposure (ELE) class",
    "ELE_2": "ELE class 2",
    "ELE_3": "ELE class 3"
}

# ------------------------------------------------------------
# 7. RF-CDD / STRUCTURAL INDEX DIAGNOSTICS
# ------------------------------------------------------------
print("\nRF-CDD / Structural index diagnostics:")
rf_diag = model_data[[RFCDD_COL, "RFCDD_model"]].describe().T
display(rf_diag)

if RFCDD_COL != "RF-CDD Index":
    print(
        "\nWARNING: The model is labelling this variable as 'RF-CDD Index', "
        f"but the column actually used is: {RFCDD_COL}. "
        "Check that this is the intended index before reporting coefficients."
    )

# ------------------------------------------------------------
# 8. BIVARIATE SCREENING
# ------------------------------------------------------------

def bivariate_continuous_vs_ordinal(dataframe, predictor, response):
    sub = dataframe[[predictor, response]].dropna().copy()
    sub = sub[sub[response].isin(VALID_RESPONSE_CLASSES)].copy()

    if len(sub) < 5 or sub[predictor].nunique() < 2 or sub[response].nunique() < 2:
        return np.nan, np.nan, np.nan, len(sub)

    rho, p_spear = spearmanr(sub[predictor], sub[response])

    groups = [
        g[predictor].dropna().values
        for _, g in sub.groupby(response)
        if len(g[predictor].dropna()) > 0
    ]

    if len(groups) >= 2:
        try:
            _, p_kw = kruskal(*groups)
        except Exception:
            p_kw = np.nan
    else:
        p_kw = np.nan

    return rho, p_spear, p_kw, len(sub)


def bivariate_ordinal_vs_ordinal(dataframe, predictor, response):
    sub = dataframe[[predictor, response]].dropna().copy()
    sub = sub[sub[response].isin(VALID_RESPONSE_CLASSES)].copy()

    if len(sub) < 5 or sub[predictor].nunique() < 2 or sub[response].nunique() < 2:
        return np.nan, np.nan, len(sub)

    try:
        sd = somersd(sub[predictor].astype(int), sub[response].astype(int))
        return sd.statistic, sd.pvalue, len(sub)

    except Exception:
        return np.nan, np.nan, len(sub)


continuous_predictors = list(continuous_map.keys())
ordinal_predictors = [
    "StandingDead_num",
    "Regeneration_num",
    "ELE_num"
]

if "Invasive_num" in model_data.columns:
    ordinal_predictors.append("Invasive_num")

# Candidate pools before bivariate screening
flw_candidate_pool = [
    predictor for predictor in (
        continuous_predictors +
        ordinal_predictors
    )
    if predictor != "FLW_num"
]

# Do not use CLW as a predictor of FLW in the main selection, because it reverses the conceptual direction.
flw_candidate_pool = [p for p in flw_candidate_pool if p != "CLW_num"]

clw_candidate_pool = [
    predictor for predictor in (
        continuous_predictors +
        ordinal_predictors +
        ["FLW_num"]
    )
    if predictor != "CLW_num"
]

# Remove duplicates
flw_candidate_pool = list(dict.fromkeys(flw_candidate_pool))
clw_candidate_pool = list(dict.fromkeys(clw_candidate_pool))


def run_bivariate_screen(response_label, response_col, candidate_pool):
    rows = []

    for predictor in candidate_pool:
        if predictor not in model_data.columns:
            continue

        if predictor in continuous_predictors:
            stat, p1, p2, n = bivariate_continuous_vs_ordinal(
                model_data,
                predictor,
                response_col
            )

            p_for_screen = np.nanmin([p1, p2]) if not (pd.isna(p1) and pd.isna(p2)) else np.nan

            rows.append(
                {
                    "Response": response_label,
                    "Predictor": predictor,
                    "Predictor_display": term_display.get(predictor, predictor),
                    "Type": "continuous",
                    "Statistic": stat,
                    "p_Spearman": p1,
                    "p_Kruskal": p2,
                    "p_Somers": np.nan,
                    "p_for_screen": p_for_screen,
                    "n": n
                }
            )

        else:
            stat, p, n = bivariate_ordinal_vs_ordinal(
                model_data,
                predictor,
                response_col
            )

            rows.append(
                {
                    "Response": response_label,
                    "Predictor": predictor,
                    "Predictor_display": term_display.get(predictor, predictor),
                    "Type": "ordinal",
                    "Statistic": stat,
                    "p_Spearman": np.nan,
                    "p_Kruskal": np.nan,
                    "p_Somers": p,
                    "p_for_screen": p,
                    "n": n
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values("p_for_screen", na_position="last").reset_index(drop=True)
    return out


biv_flw = run_bivariate_screen("FLW", "FLW_num", flw_candidate_pool)
biv_clw = run_bivariate_screen("CLW", "CLW_num", clw_candidate_pool)

biv_all = pd.concat([biv_flw, biv_clw], ignore_index=True)

print("\nBivariate screening: FLW")
display(biv_flw)

print("\nBivariate screening: CLW")
display(biv_clw)

# ------------------------------------------------------------
# 9. BUILD SCREENED MODEL POOLS
# ------------------------------------------------------------

def build_screened_pool(biv_table, forced_predictors):
    selected = biv_table[
        (biv_table["p_for_screen"] <= BIVARIATE_P_THRESHOLD)
    ]["Predictor"].dropna().tolist()

    selected = selected + [
        predictor for predictor in forced_predictors
        if predictor in model_data.columns
    ]

    selected = list(dict.fromkeys(selected))

    return selected


flw_pool_screened = build_screened_pool(biv_flw, FORCE_KEEP_FLW)
clw_pool_screened = build_screened_pool(biv_clw, FORCE_KEEP_CLW)

print("\nScreened FLW model pool:")
print([term_display.get(p, p) for p in flw_pool_screened])

print("\nScreened CLW model pool:")
print([term_display.get(p, p) for p in clw_pool_screened])

# ------------------------------------------------------------
# 10. MODEL FITTING FUNCTION
# ------------------------------------------------------------

def fit_ordered_model(df_model, response, predictors, model_name, response_label):
    cols = [response] + predictors
    tmp = df_model[cols].copy()

    for col in cols:
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

    tmp = tmp.dropna().copy()
    tmp = tmp[tmp[response].isin(VALID_RESPONSE_CLASSES)].copy()

    if tmp.empty:
        return None, None, None

    y = tmp[response].astype(int)
    X = tmp[predictors].astype(float)

    zero_var_cols = [
        col for col in X.columns
        if X[col].nunique(dropna=True) < 2
    ]

    if zero_var_cols:
        X = X.drop(columns=zero_var_cols)
        predictors = [
            predictor for predictor in predictors
            if predictor not in zero_var_cols
        ]

    if X.shape[1] == 0:
        return None, None, None

    try:
        mod = OrderedModel(
            endog=y,
            exog=X,
            distr="logit"
        )

        res = mod.fit(
            method="bfgs",
            disp=False,
            maxiter=1000
        )

    except Exception:
        try:
            mod = OrderedModel(
                endog=y,
                exog=X,
                distr="logit"
            )

            res = mod.fit(
                method="lbfgs",
                disp=False,
                maxiter=1000
            )

        except Exception:
            return None, None, None

    coef_table = pd.DataFrame(
        {
            "Response": response_label,
            "Model": model_name,
            "Term": res.params.index,
            "Coefficient": res.params.values,
            "Std_Error": res.bse.values,
            "z": res.params.values / res.bse.values,
            "p_value": res.pvalues.values
        }
    )

    coef_table["Term_display"] = (
        coef_table["Term"]
        .map(term_display)
        .fillna(coef_table["Term"])
    )

    predictor_coef_table = coef_table[
        coef_table["Term"].isin(predictors)
    ].copy()

    summary_row = {
        "Response": response_label,
        "Model": model_name,
        "n": tmp.shape[0],
        "n_predictors": len(predictors),
        "Predictor_list": predictors,
        "Predictors": " + ".join(
            [term_display.get(predictor, predictor) for predictor in predictors]
        ),
        "AIC": res.aic,
        "BIC": res.bic,
        "LogLik": res.llf,
        "Converged": res.mle_retvals.get("converged", np.nan)
    }

    return res, summary_row, predictor_coef_table

# ------------------------------------------------------------
# 11. GENERATE ALL COMBINATIONS
# ------------------------------------------------------------

def generate_combinations(pool, max_predictors):
    combinations = []

    for k in range(1, min(max_predictors, len(pool)) + 1):
        for combo in itertools.combinations(pool, k):
            combinations.append(list(combo))

    return combinations


flw_combinations = generate_combinations(flw_pool_screened, MAX_PREDICTORS)
clw_combinations = generate_combinations(clw_pool_screened, MAX_PREDICTORS)

# Add original and conceptually important models explicitly
manual_models = {
    "FLW_original": {
        "Response": "FLW",
        "Response_col": "FLW_num",
        "Predictors": [
            "StandingDead_num",
            "BasalArea_model",
            "Regeneration_num",
            "RFCDD_model"
        ]
    },
    "FLW_original_plus_ELE": {
        "Response": "FLW",
        "Response_col": "FLW_num",
        "Predictors": [
            "StandingDead_num",
            "BasalArea_model",
            "Regeneration_num",
            "RFCDD_model",
            "ELE_num"
        ]
    },
    "FLW_core_forest": {
        "Response": "FLW",
        "Response_col": "FLW_num",
        "Predictors": [
            "StandingDead_num",
            "BasalArea_model"
        ]
    },
    "CLW_original": {
        "Response": "CLW",
        "Response_col": "CLW_num",
        "Predictors": [
            "SPIwidth_model",
            "Sinuosity_model",
            "StandingDead_num",
            "FLW_num"
        ]
    },
    "CLW_original_plus_ELE": {
        "Response": "CLW",
        "Response_col": "CLW_num",
        "Predictors": [
            "SPIwidth_model",
            "Sinuosity_model",
            "StandingDead_num",
            "FLW_num",
            "ELE_num"
        ]
    },
    "CLW_SPI_FLW": {
        "Response": "CLW",
        "Response_col": "CLW_num",
        "Predictors": [
            "SPIwidth_model",
            "FLW_num"
        ]
    },
    "CLW_SPI_FLW_ELE": {
        "Response": "CLW",
        "Response_col": "CLW_num",
        "Predictors": [
            "SPIwidth_model",
            "FLW_num",
            "ELE_num"
        ]
    },
    "CLW_replace_StandingDead_with_ELE": {
        "Response": "CLW",
        "Response_col": "CLW_num",
        "Predictors": [
            "SPIwidth_model",
            "Sinuosity_model",
            "FLW_num",
            "ELE_num"
        ]
    }
}

# ------------------------------------------------------------
# 12. FIT BEST-SUBSET MODELS
# ------------------------------------------------------------
all_results = {}
summary_rows = []
coef_tables = []

# FLW exhaustive combinations
for i, predictors in enumerate(flw_combinations, start=1):
    model_name = f"FLW_subset_{i:03d}"

    res, summary_row, coef_table = fit_ordered_model(
        model_data,
        "FLW_num",
        predictors,
        model_name,
        "FLW"
    )

    all_results[model_name] = res

    if summary_row is not None:
        summary_rows.append(summary_row)

    if coef_table is not None:
        coef_tables.append(coef_table)

# CLW exhaustive combinations
for i, predictors in enumerate(clw_combinations, start=1):
    model_name = f"CLW_subset_{i:03d}"

    res, summary_row, coef_table = fit_ordered_model(
        model_data,
        "CLW_num",
        predictors,
        model_name,
        "CLW"
    )

    all_results[model_name] = res

    if summary_row is not None:
        summary_rows.append(summary_row)

    if coef_table is not None:
        coef_tables.append(coef_table)

# Manual models
for model_name, spec in manual_models.items():
    predictors = [
        p for p in spec["Predictors"]
        if p in model_data.columns
    ]

    res, summary_row, coef_table = fit_ordered_model(
        model_data,
        spec["Response_col"],
        predictors,
        model_name,
        spec["Response"]
    )

    all_results[model_name] = res

    if summary_row is not None:
        summary_rows.append(summary_row)

    if coef_table is not None:
        coef_tables.append(coef_table)

model_compare = pd.DataFrame(summary_rows)

if model_compare.empty:
    raise ValueError("No models were successfully fitted.")

# Remove duplicate predictor sets within each response to avoid repeated rows
model_compare["Predictor_key"] = model_compare["Predictor_list"].apply(
    lambda x: tuple(x)
)

model_compare = (
    model_compare
    .sort_values(["Response", "AIC"])
    .drop_duplicates(subset=["Response", "Predictor_key"], keep="first")
    .reset_index(drop=True)
)

for response_label in model_compare["Response"].unique():
    mask = model_compare["Response"] == response_label

    model_compare.loc[mask, "delta_AIC"] = (
        model_compare.loc[mask, "AIC"] -
        model_compare.loc[mask, "AIC"].min()
    )

    model_compare.loc[mask, "delta_BIC"] = (
        model_compare.loc[mask, "BIC"] -
        model_compare.loc[mask, "BIC"].min()
    )

model_compare = (
    model_compare
    .sort_values(["Response", "AIC"])
    .reset_index(drop=True)
)

all_coef = pd.concat(coef_tables, ignore_index=True)

# ------------------------------------------------------------
# 13. OUTPUT TABLES
# ------------------------------------------------------------
print("\nTop 15 FLW models by AIC:")
top_flw = (
    model_compare[model_compare["Response"] == "FLW"]
    .sort_values("AIC")
    .head(15)
)

display(
    top_flw[
        [
            "Response",
            "Model",
            "n",
            "n_predictors",
            "AIC",
            "delta_AIC",
            "BIC",
            "delta_BIC",
            "Predictors"
        ]
    ]
)

print("\nTop 15 CLW models by AIC:")
top_clw = (
    model_compare[model_compare["Response"] == "CLW"]
    .sort_values("AIC")
    .head(15)
)

display(
    top_clw[
        [
            "Response",
            "Model",
            "n",
            "n_predictors",
            "AIC",
            "delta_AIC",
            "BIC",
            "delta_BIC",
            "Predictors"
        ]
    ]
)

print("\nModels with delta_AIC <= 2:")
display(
    model_compare[model_compare["delta_AIC"] <= 2][
        [
            "Response",
            "Model",
            "n",
            "n_predictors",
            "AIC",
            "delta_AIC",
            "BIC",
            "delta_BIC",
            "Predictors"
        ]
    ].sort_values(["Response", "AIC"])
)

print("\nBest models by BIC:")
best_bic = (
    model_compare
    .sort_values(["Response", "BIC"])
    .groupby("Response")
    .head(5)
)

display(
    best_bic[
        [
            "Response",
            "Model",
            "n",
            "n_predictors",
            "AIC",
            "delta_AIC",
            "BIC",
            "delta_BIC",
            "Predictors"
        ]
    ]
)

# ------------------------------------------------------------
# 14. COEFFICIENT TABLES FOR SELECTED MODELS
# ------------------------------------------------------------
selected_models = []

for response_label in ["FLW", "CLW"]:
    sub = model_compare[model_compare["Response"] == response_label].copy()

    best_aic_model = sub.sort_values("AIC").iloc[0]["Model"]
    best_bic_model = sub.sort_values("BIC").iloc[0]["Model"]

    selected_models.extend([best_aic_model, best_bic_model])

selected_models.extend([
    "FLW_original",
    "FLW_core_forest",
    "FLW_original_plus_ELE",
    "CLW_original",
    "CLW_SPI_FLW",
    "CLW_SPI_FLW_ELE",
    "CLW_replace_StandingDead_with_ELE"
])

selected_models = list(dict.fromkeys(selected_models))

selected_coef = all_coef[all_coef["Model"].isin(selected_models)].copy()

selected_coef["Coefficient_rounded"] = selected_coef["Coefficient"].round(3)
selected_coef["p_value_display"] = selected_coef["p_value"].apply(
    lambda p: "<0.01" if p < 0.01 else f"{p:.3f}"
)

print("\nSelected model coefficients:")
display(
    selected_coef[
        [
            "Response",
            "Model",
            "Term_display",
            "Coefficient_rounded",
            "Std_Error",
            "z",
            "p_value_display"
        ]
    ].sort_values(["Response", "Model", "Term_display"])
)

# ------------------------------------------------------------
# 15. RF-CDD COEFFICIENT INTERPRETATION AID
# ------------------------------------------------------------
rf_rows = selected_coef[selected_coef["Term"] == "RFCDD_model"].copy()

if not rf_rows.empty:
    print("\nRF-CDD coefficient interpretation aid:")
    rf_sd_original = model_data[RFCDD_COL].std()
    rf_range_original = model_data[RFCDD_COL].max() - model_data[RFCDD_COL].min()

    print(f"Original RF-CDD / structural index column used: {RFCDD_COL}")
    print(f"Original scale SD: {rf_sd_original:.4f}")
    print(f"Original scale range: {rf_range_original:.4f}")
    print(f"Continuous predictors standardized in this run: {STANDARDIZE_CONTINUOUS}")

    rf_show = rf_rows[
        [
            "Response",
            "Model",
            "Coefficient",
            "p_value"
        ]
    ].copy()

    rf_show["Odds_ratio_per_model_unit"] = np.exp(rf_show["Coefficient"])
    rf_show["Coefficient"] = rf_show["Coefficient"].round(4)
    rf_show["p_value"] = rf_show["p_value"].apply(
        lambda p: "<0.01" if p < 0.01 else f"{p:.3f}"
    )
    rf_show["Odds_ratio_per_model_unit"] = rf_show["Odds_ratio_per_model_unit"].round(3)

    display(rf_show)

# ------------------------------------------------------------
# 16. SAVE OUTPUTS
# ------------------------------------------------------------
biv_all.to_csv(
    OUT_DIR / "bivariate_screening_FLW_CLW.csv",
    index=False
)

model_compare.to_csv(
    OUT_DIR / "best_subset_model_comparison_FLW_CLW_with_ELE.csv",
    index=False
)

all_coef.to_csv(
    OUT_DIR / "best_subset_model_coefficients_FLW_CLW_with_ELE.csv",
    index=False
)

selected_coef.to_csv(
    OUT_DIR / "selected_model_coefficients_FLW_CLW_with_ELE.csv",
    index=False
)

ele_dist.to_csv(
    OUT_DIR / "ELE_class_distribution_best_subset_models.csv",
    index=False
)

print(f"\nOutputs saved in: {OUT_DIR}")

# ------------------------------------------------------------
# 17. DECISION AID
# ------------------------------------------------------------
print("\nDecision aid:")
print(
    "1) Use AIC to identify the best predictive/fit-supported model among parsimonious candidates."
)
print(
    "2) Use BIC to avoid overfitting and to prefer simpler models when AIC differences are small."
)
print(
    "3) Treat models with delta_AIC <= 2 as broadly competitive; prefer the simpler and more interpretable one."
)
print(
    "4) If ELE improves CLW but not FLW, this supports using ELE in the CLW model only."
)
print(
    "5) Do not interpret ELE as direct evidence of hydraulic connectivity or transfer direction."
)
