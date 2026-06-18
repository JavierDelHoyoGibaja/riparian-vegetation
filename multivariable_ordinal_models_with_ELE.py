# ============================================================
# MULTIVARIABLE ORDINAL MODELS WITH EFFECTIVE LATERAL EXPOSURE (ELE) CLASS
#
# Responses:
#   1) FLW = Dead_Wood
#   2) CLW = LW_Presence
#
# This cell:
#   - Creates Effective Lateral Exposure (ELE) class
#   - Fits several ordinal logistic models for FLW and CLW
#   - Compares models using AIC, BIC and log-likelihood
#   - Extracts coefficients and p-values
#   - Runs selected likelihood-ratio tests
#
# ELE class is a conservative contextual descriptor.
# It is not a direct measure of hydraulic connectivity, bank erosion,
# inundation frequency, floodplain relative elevation, or wood recruitment.
# ============================================================

import numpy as np
import pandas as pd
import scipy.stats as stats
from pathlib import Path
from statsmodels.miscmodels.ordinal_model import OrderedModel

# ------------------------------------------------------------
# 1. OUTPUT DIRECTORY
# ------------------------------------------------------------
try:
    OUT_DIR
except NameError:
    OUT_DIR = Path.cwd() / "ordinal_models_with_ELE"
    OUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# 2. COLUMN NAMES
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
SINUOSITY_COL = "Sinuosity"
SPI_COL = "SPI / Width"

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

required_cols = [
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

missing = [col for col in required_cols if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("Using columns:")
print(f"  FLW: {FLW_COL}")
print(f"  CLW: {CLW_COL}")
print(f"  Basal area: {BASAL_COL}")
print(f"  RF-CDD / Structural index: {RFCDD_COL}")
print(f"  ELE natural component: {VALLEY_COL}")
print(f"  ELE artificial constraint component: {LAT_COL}")

# ------------------------------------------------------------
# 3. CREATE EFFECTIVE LATERAL EXPOSURE (ELE) CLASS
# ------------------------------------------------------------
data = df.copy()

for col in required_cols:
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

# Write back to df for later use
df[NAT_CLASS_COL] = data[NAT_CLASS_COL]
df[ART_CAP_COL] = data[ART_CAP_COL]
df[EFF_CLASS_COL] = data[EFF_CLASS_COL]
df[DOWNGRADE_COL] = data[DOWNGRADE_COL]

print("\nEffective Lateral Exposure (ELE) class distribution:")

ele_dist = (
    data[EFF_CLASS_COL]
    .value_counts(dropna=False)
    .sort_index()
    .rename_axis(EFF_CLASS_COL)
    .reset_index(name="n")
)

ele_dist["percent"] = 100 * ele_dist["n"] / ele_dist["n"].sum()

display(ele_dist)

print("\nNumber of downgraded observations:")
print(f"{(data[DOWNGRADE_COL] > 0).sum()} / {len(data)}")

# ------------------------------------------------------------
# 4. MODEL SETTINGS
# ------------------------------------------------------------
STANDARDIZE_CONTINUOUS = False
RUN_ELE_CATEGORICAL_SENSITIVITY = True
VALID_RESPONSE_CLASSES = [1, 2, 3, 4]

# ------------------------------------------------------------
# 5. PREPARE MODEL DATA
# ------------------------------------------------------------
model_data = data[
    [
        FLW_COL,
        CLW_COL,
        STANDING_COL,
        REGEN_COL,
        EFF_CLASS_COL,
        BASAL_COL,
        RFCDD_COL,
        SINUOSITY_COL,
        SPI_COL
    ]
].copy()

for col in model_data.columns:
    model_data[col] = pd.to_numeric(model_data[col], errors="coerce")

model_data["FLW_num"] = model_data[FLW_COL]
model_data["CLW_num"] = model_data[CLW_COL]
model_data["StandingDead_num"] = model_data[STANDING_COL]
model_data["Regeneration_num"] = model_data[REGEN_COL]
model_data["ELE_num"] = model_data[EFF_CLASS_COL]

if STANDARDIZE_CONTINUOUS:
    model_data["BasalArea_model"] = (
        model_data[BASAL_COL] - model_data[BASAL_COL].mean()
    ) / model_data[BASAL_COL].std()

    model_data["RFCDD_model"] = (
        model_data[RFCDD_COL] - model_data[RFCDD_COL].mean()
    ) / model_data[RFCDD_COL].std()

    model_data["Sinuosity_model"] = (
        model_data[SINUOSITY_COL] - model_data[SINUOSITY_COL].mean()
    ) / model_data[SINUOSITY_COL].std()

    model_data["SPIwidth_model"] = (
        model_data[SPI_COL] - model_data[SPI_COL].mean()
    ) / model_data[SPI_COL].std()

else:
    model_data["BasalArea_model"] = model_data[BASAL_COL]
    model_data["RFCDD_model"] = model_data[RFCDD_COL]
    model_data["Sinuosity_model"] = model_data[SINUOSITY_COL]
    model_data["SPIwidth_model"] = model_data[SPI_COL]

# ELE as categorical dummies. Reference category = ELE class 1.
ele_dummies = pd.get_dummies(
    model_data["ELE_num"].astype("Int64"),
    prefix="ELE",
    drop_first=True
).astype(float)

model_data = pd.concat([model_data, ele_dummies], axis=1)
ELE_DUMMY_COLS = list(ele_dummies.columns)

print("\nELE dummy columns for categorical sensitivity:")
print(ELE_DUMMY_COLS)

# ------------------------------------------------------------
# 6. DISPLAY NAMES
# ------------------------------------------------------------
term_display = {
    "SPIwidth_model": "SPI / Width",
    "Sinuosity_model": "Sinuosity",
    "StandingDead_num": "Standing dead trees",
    "FLW_num": "FLW",
    "CLW_num": "CLW",
    "BasalArea_model": "Basal area",
    "Regeneration_num": "Regeneration",
    "RFCDD_model": "RF-CDD Index",
    "ELE_num": "Effective Lateral Exposure (ELE) class",
    "ELE_2": "ELE class 2",
    "ELE_3": "ELE class 3"
}

# ------------------------------------------------------------
# 7. FITTING FUNCTION
# ------------------------------------------------------------

def fit_ordered_model(df_model, response, predictors, model_name, response_label):
    """
    Fit ordinal logistic regression using statsmodels OrderedModel.
    No intercept is added because OrderedModel estimates thresholds internally.
    """
    cols = [response] + predictors
    tmp = df_model[cols].copy()

    for col in cols:
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

    tmp = tmp.dropna().copy()
    tmp = tmp[tmp[response].isin(VALID_RESPONSE_CLASSES)].copy()

    if tmp.empty:
        print(f"\nModel failed: {model_name}. Empty model dataframe.")
        return None, None, None

    y = tmp[response].astype(int)
    X = tmp[predictors].astype(float)

    zero_var_cols = [
        col for col in X.columns
        if X[col].nunique(dropna=True) < 2
    ]

    if zero_var_cols:
        print(f"\nWarning: dropping zero-variance predictors in {model_name}: {zero_var_cols}")
        X = X.drop(columns=zero_var_cols)
        predictors = [
            predictor for predictor in predictors
            if predictor not in zero_var_cols
        ]

    if X.shape[1] == 0:
        print(f"\nModel failed: {model_name}. No valid predictors.")
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

    except Exception as first_error:
        print(f"\nFirst fit failed for {model_name}. Trying lbfgs.")
        print(first_error)

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

        except Exception as second_error:
            print(f"\nModel failed: {model_name}")
            print(second_error)
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
# 8. CANDIDATE MODELS
# ------------------------------------------------------------

flw_models = {
    "FLW_M0_original": [
        "StandingDead_num",
        "BasalArea_model",
        "Regeneration_num",
        "RFCDD_model"
    ],
    "FLW_M1_original_plus_ELE": [
        "StandingDead_num",
        "BasalArea_model",
        "Regeneration_num",
        "RFCDD_model",
        "ELE_num"
    ],
    "FLW_M2_core_forest": [
        "StandingDead_num",
        "BasalArea_model"
    ],
    "FLW_M3_core_forest_plus_ELE": [
        "StandingDead_num",
        "BasalArea_model",
        "ELE_num"
    ],
    "FLW_M4_replace_Regeneration_with_ELE": [
        "StandingDead_num",
        "BasalArea_model",
        "RFCDD_model",
        "ELE_num"
    ],
    "FLW_M5_replace_RFCDD_with_ELE": [
        "StandingDead_num",
        "BasalArea_model",
        "Regeneration_num",
        "ELE_num"
    ],
    "FLW_M6_BasalArea_ELE": [
        "BasalArea_model",
        "ELE_num"
    ]
}

clw_models = {
    "CLW_M0_original": [
        "SPIwidth_model",
        "Sinuosity_model",
        "StandingDead_num",
        "FLW_num"
    ],
    "CLW_M1_original_plus_ELE": [
        "SPIwidth_model",
        "Sinuosity_model",
        "StandingDead_num",
        "FLW_num",
        "ELE_num"
    ],
    "CLW_M2_SPI_FLW": [
        "SPIwidth_model",
        "FLW_num"
    ],
    "CLW_M3_SPI_FLW_ELE": [
        "SPIwidth_model",
        "FLW_num",
        "ELE_num"
    ],
    "CLW_M4_replace_StandingDead_with_ELE": [
        "SPIwidth_model",
        "Sinuosity_model",
        "FLW_num",
        "ELE_num"
    ],
    "CLW_M5_replace_FLW_with_ELE": [
        "SPIwidth_model",
        "Sinuosity_model",
        "StandingDead_num",
        "ELE_num"
    ],
    "CLW_M6_SPI_FLW_StandingDead_ELE": [
        "SPIwidth_model",
        "FLW_num",
        "StandingDead_num",
        "ELE_num"
    ],
    "CLW_M7_SPI_ELE": [
        "SPIwidth_model",
        "ELE_num"
    ]
}

if RUN_ELE_CATEGORICAL_SENSITIVITY and len(ELE_DUMMY_COLS) > 0:
    flw_models.update(
        {
            "FLW_C1_original_plus_ELEcat": [
                "StandingDead_num",
                "BasalArea_model",
                "Regeneration_num",
                "RFCDD_model"
            ] + ELE_DUMMY_COLS,
            "FLW_C2_core_forest_plus_ELEcat": [
                "StandingDead_num",
                "BasalArea_model"
            ] + ELE_DUMMY_COLS
        }
    )

    clw_models.update(
        {
            "CLW_C1_original_plus_ELEcat": [
                "SPIwidth_model",
                "Sinuosity_model",
                "StandingDead_num",
                "FLW_num"
            ] + ELE_DUMMY_COLS,
            "CLW_C2_SPI_FLW_ELEcat": [
                "SPIwidth_model",
                "FLW_num"
            ] + ELE_DUMMY_COLS,
            "CLW_C3_replace_StandingDead_with_ELEcat": [
                "SPIwidth_model",
                "Sinuosity_model",
                "FLW_num"
            ] + ELE_DUMMY_COLS
        }
    )

# ------------------------------------------------------------
# 9. FIT ALL MODELS
# ------------------------------------------------------------
all_results = {}
summary_rows = []
coef_tables = []

for model_name, predictors in flw_models.items():
    res, summary_row, coef_table = fit_ordered_model(
        df_model=model_data,
        response="FLW_num",
        predictors=predictors,
        model_name=model_name,
        response_label="FLW"
    )

    all_results[model_name] = res

    if summary_row is not None:
        summary_rows.append(summary_row)

    if coef_table is not None:
        coef_tables.append(coef_table)

for model_name, predictors in clw_models.items():
    res, summary_row, coef_table = fit_ordered_model(
        df_model=model_data,
        response="CLW_num",
        predictors=predictors,
        model_name=model_name,
        response_label="CLW"
    )

    all_results[model_name] = res

    if summary_row is not None:
        summary_rows.append(summary_row)

    if coef_table is not None:
        coef_tables.append(coef_table)

model_compare = pd.DataFrame(summary_rows)

if model_compare.empty:
    raise ValueError("No models were successfully fitted.")

model_compare["delta_AIC"] = np.nan
model_compare["delta_BIC"] = np.nan

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
# 10. DISPLAY MODEL COMPARISON
# ------------------------------------------------------------
print("\n==============================")
print("MODEL COMPARISON BY RESPONSE")
print("==============================")

display(
    model_compare[
        [
            "Response",
            "Model",
            "n",
            "n_predictors",
            "AIC",
            "delta_AIC",
            "BIC",
            "delta_BIC",
            "LogLik",
            "Converged",
            "Predictors"
        ]
    ]
)

print("\n==============================")
print("PREDICTOR COEFFICIENTS")
print("==============================")

coef_display = all_coef.copy()

coef_display["Coefficient"] = coef_display["Coefficient"].round(4)
coef_display["Std_Error"] = coef_display["Std_Error"].round(4)
coef_display["z"] = coef_display["z"].round(3)

coef_display["p_value_display"] = coef_display["p_value"].apply(
    lambda p: "<0.01" if p < 0.01 else f"{p:.3f}"
)

display(
    coef_display[
        [
            "Response",
            "Model",
            "Term_display",
            "Coefficient",
            "Std_Error",
            "z",
            "p_value_display"
        ]
    ]
)

# ------------------------------------------------------------
# 11. BEST MODELS AND COMPETITIVE MODELS
# ------------------------------------------------------------
print("\n==============================")
print("BEST MODELS AND COMPETITIVE MODELS")
print("==============================")

for response_label in ["FLW", "CLW"]:
    sub = model_compare[
        model_compare["Response"] == response_label
    ].copy().sort_values("AIC")

    print(f"\n{response_label}: best model by AIC")
    display(sub.head(1))

    print(f"\n{response_label}: models with delta_AIC <= 2")
    display(sub[sub["delta_AIC"] <= 2])

# ------------------------------------------------------------
# 12. LIKELIHOOD-RATIO TESTS
# ------------------------------------------------------------

def lr_test(reduced_name, full_name):
    reduced = all_results.get(reduced_name)
    full = all_results.get(full_name)

    if reduced is None or full is None:
        return {
            "Reduced_model": reduced_name,
            "Full_model": full_name,
            "LR_stat": np.nan,
            "df_diff": np.nan,
            "p_value": np.nan
        }

    lr_stat = 2 * (full.llf - reduced.llf)
    df_diff = len(full.params) - len(reduced.params)
    p_value = stats.chi2.sf(lr_stat, df_diff)

    return {
        "Reduced_model": reduced_name,
        "Full_model": full_name,
        "LR_stat": lr_stat,
        "df_diff": df_diff,
        "p_value": p_value
    }


lr_comparisons = [
    ("FLW_M0_original", "FLW_M1_original_plus_ELE"),
    ("FLW_M2_core_forest", "FLW_M3_core_forest_plus_ELE"),
    ("CLW_M0_original", "CLW_M1_original_plus_ELE"),
    ("CLW_M2_SPI_FLW", "CLW_M3_SPI_FLW_ELE"),
    ("CLW_M7_SPI_ELE", "CLW_M3_SPI_FLW_ELE")
]

lr_table = pd.DataFrame(
    [lr_test(reduced, full) for reduced, full in lr_comparisons]
)

lr_table["LR_stat"] = lr_table["LR_stat"].round(4)
lr_table["p_value_display"] = lr_table["p_value"].apply(
    lambda p: "NA" if pd.isna(p) else ("<0.01" if p < 0.01 else f"{p:.3f}")
)

print("\n==============================")
print("LIKELIHOOD-RATIO TESTS")
print("==============================")

display(
    lr_table[
        [
            "Reduced_model",
            "Full_model",
            "LR_stat",
            "df_diff",
            "p_value_display"
        ]
    ]
)

# ------------------------------------------------------------
# 13. COMPACT MANUSCRIPT-LIKE TABLE
# ------------------------------------------------------------
best_flw_model = (
    model_compare[model_compare["Response"] == "FLW"]
    .sort_values("AIC")
    .iloc[0]["Model"]
)

best_clw_model = (
    model_compare[model_compare["Response"] == "CLW"]
    .sort_values("AIC")
    .iloc[0]["Model"]
)

models_for_compact_output = [
    "FLW_M0_original",
    best_flw_model,
    "CLW_M0_original",
    best_clw_model
]

compact_table = all_coef[
    all_coef["Model"].isin(models_for_compact_output)
].copy()

compact_table["Coefficient"] = compact_table["Coefficient"].round(3)
compact_table["p_value_numeric"] = compact_table["p_value"]
compact_table["p_value"] = compact_table["p_value"].apply(
    lambda p: "<0.01" if p < 0.01 else f"{p:.3f}"
)

compact_table = (
    compact_table[
        [
            "Response",
            "Model",
            "Term_display",
            "Coefficient",
            "p_value",
            "p_value_numeric"
        ]
    ]
    .sort_values(["Response", "Model", "Term_display"])
)

print("\n==============================")
print("COMPACT MANUSCRIPT-LIKE COEFFICIENT TABLE")
print("==============================")

display(compact_table)

# ------------------------------------------------------------
# 14. SAVE OUTPUTS
# ------------------------------------------------------------
model_compare.to_csv(
    OUT_DIR / "model_comparison_FLW_CLW_with_ELE.csv",
    index=False
)

all_coef.to_csv(
    OUT_DIR / "model_coefficients_FLW_CLW_with_ELE.csv",
    index=False
)

lr_table.to_csv(
    OUT_DIR / "likelihood_ratio_tests_FLW_CLW_with_ELE.csv",
    index=False
)

compact_table.to_csv(
    OUT_DIR / "compact_model_table_FLW_CLW_with_ELE.csv",
    index=False
)

ele_dist.to_csv(
    OUT_DIR / "ELE_class_distribution_for_models.csv",
    index=False
)

print(f"\nOutputs saved in: {OUT_DIR}")

# ------------------------------------------------------------
# 15. AUTOMATIC INTERPRETATION AID
# ------------------------------------------------------------
print("\n==============================")
print("INTERPRETATION AID")
print("==============================")

for response_label in ["FLW", "CLW"]:
    sub = model_compare[
        model_compare["Response"] == response_label
    ].sort_values("AIC")

    best = sub.iloc[0]

    print(
        f"\n{response_label}: best AIC model = {best['Model']} "
        f"(AIC = {best['AIC']:.2f}, delta_AIC = {best['delta_AIC']:.2f})."
    )

    ele_models = all_coef[
        (all_coef["Response"] == response_label) &
        (all_coef["Term"].str.contains("ELE", na=False))
    ].copy()

    if ele_models.empty:
        print(f"No ELE term was estimated for {response_label}.")

    else:
        print(f"ELE terms for {response_label}:")

        ele_show = ele_models[
            [
                "Model",
                "Term_display",
                "Coefficient",
                "p_value"
            ]
        ].copy()

        ele_show["Coefficient"] = ele_show["Coefficient"].round(4)
        ele_show["p_value"] = ele_show["p_value"].apply(
            lambda p: "<0.01" if p < 0.01 else f"{p:.3f}"
        )

        display(ele_show.sort_values(["Model", "Term_display"]))

print(
    "\nDecision rule:\n"
    "1) If a model with ELE has lower AIC/BIC and ELE has a clear coefficient, report it as a sensitivity model.\n"
    "2) If adding ELE does not improve AIC/BIC or ELE is non-significant after main predictors, keep the original model and report ELE as contextual/bivariate sensitivity.\n"
    "3) Prefer the simpler model when delta_AIC <= 2 unless the more complex model is needed to address the lateral-exposure question.\n"
    "4) Do not interpret ELE as direct evidence of hydraulic connectivity or wood transfer direction."
)
