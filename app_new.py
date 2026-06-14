import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt

st.set_page_config(page_title="Smart Analytics Predictor", page_icon="💰", layout="wide")

st.markdown("""
<style>
.app-page-title {
    font-size: 30px;
    font-weight: 800;
    margin: 0 0 12px 0;
    line-height: 1.2;
}
.regression-info-box {
    margin-top: 10px;
    padding: 12px 16px;
    border-left: 5px solid #E8B89D;
    border-radius: 9px;
    background: rgba(249, 213, 192, 0.14);
    color: #F6DDCF;
    font-size: 15px;
    font-weight: 600;
}
.correlation-status {
    margin-bottom: 8px;
    color: #9FE3D5;
    font-size: 15px;
    font-weight: 800;
}
.compact-info-box {
    display: inline-flex;
    align-items: center;
    margin: 8px 0 10px 0;
    padding: 9px 13px;
    border: 1px solid rgba(147, 197, 253, 0.28);
    border-radius: 18px;
    background: rgba(30, 64, 175, 0.20);
    color: #E6F2FF;
    font-size: 14px;
    font-weight: 650;
}
.compact-alert-box {
    display: inline-flex;
    align-items: center;
    margin: 4px 0 10px 0;
    padding: 9px 13px;
    border-radius: 18px;
    font-size: 14px;
    font-weight: 650;
}
.alert-high {
    border: 1px solid rgba(248, 113, 113, 0.38);
    background: rgba(127, 29, 29, 0.22);
    color: #FEE2E2;
}
.alert-moderate {
    border: 1px solid rgba(251, 191, 36, 0.38);
    background: rgba(120, 53, 15, 0.20);
    color: #FEF3C7;
}
.alert-normal {
    border: 1px solid rgba(52, 211, 153, 0.35);
    background: rgba(6, 78, 59, 0.20);
    color: #D1FAE5;
}
.message-highlight-model {
    color: #C4B5FD;
    font-weight: 850;
}
.message-highlight-moderate {
    color: #FBBF24;
    font-weight: 850;
}
.regression-page-title {
    display: inline-block;
    padding: 7px 15px;
    border-left: 5px solid #86C5B5;
    border-radius: 10px;
    background: linear-gradient(90deg, rgba(168, 230, 207, 0.18), rgba(176, 224, 230, 0.08));
    color: #D5F2E8;
}
.app-subheading {
    display: inline-block;
    margin: 8px 0 10px 0;
    padding: 5px 12px;
    border-left: 4px solid #B8A1E3;
    border-radius: 7px;
    background: linear-gradient(90deg, rgba(216, 196, 240, 0.15), rgba(190, 227, 230, 0.06));
    color: #E3D5F5;
    font-size: 20px;
    font-weight: 750;
    line-height: 1.2;
}
.centered-subheading-wrap {
    text-align: center;
}
div[data-testid="stPlotlyChart"] {
    background: rgba(148, 163, 184, 0.08);
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

def load_sample_data():
    return pd.read_csv("sample_main_dataset_5y.csv")

def load_category_data():
    return pd.read_csv("sample_category_dataset_5y.csv")

def standardize_columns(df):
    df = df.copy()
    df = df.rename(columns={
        "Travel Days": "Travel_Days",
        "Family Size": "Family_Size"
    })

    

    return df

def fix_month_column(df):
    df = df.copy()
    col = df["Month"]

    if pd.api.types.is_numeric_dtype(col):
        parsed_excel = pd.to_datetime(
            col,
            unit="D",
            origin="1899-12-30",
            errors="coerce"
        )

        if parsed_excel.dt.year.between(2000, 2035).mean() >= 0.8:
            df["Month"] = parsed_excel
        else:
            df["Month"] = pd.to_datetime(col, errors="coerce")

    else:
        df["Month"] = pd.to_datetime(
            col,
            errors="coerce",
            dayfirst=False
        )

    # Final date validation
    valid_date_ratio = df["Month"].notna().mean()
    valid_year_ratio = df["Month"].dt.year.between(2000, 2035).mean()

    if valid_date_ratio < 0.8 or valid_year_ratio < 0.8:
        st.error(
            "Selected date column does not appear to contain valid dates. "
            "Please use a proper Month/Date column."
        )
        st.stop()

    return df


def assign_season(m):
    if m in [3,4,5,6]:
        return "Summer"
    elif m in [7,8,9]:
        return "Monsoon"
    else:
        return "Winter"

def engineer_features(df):
    df = df.copy()
    df = fix_month_column(df)
    df = df.sort_values("Month").reset_index(drop=True)

    df["Year"] = df["Month"].dt.year
    df["Month_No"] = df["Month"].dt.month
    df["YearMonth"] = df["Month"].dt.strftime("%b %Y")
    df["Season"] = df["Month_No"].apply(assign_season)
    if "Income" in df.columns:
        df["Expense_to_Income_Ratio"] = df["Expense"] / df["Income"]
    else:
        df["Expense_to_Income_Ratio"] = 0
    df["High_Expense_Flag"] = (df["Expense"] > df["Expense"].mean()).astype(int)
    df["Previous_Month_Expense"] = df["Expense"].shift(1).fillna(df["Expense"].mean())

    return df
def detect_seasonality(y, seasonal_lag=12, threshold=0.45):
    if len(y) < seasonal_lag * 2:
        return False
    values = y.dropna().astype(float)
    x = np.arange(len(values))
    trend = np.polyval(np.polyfit(x, values.values, 1), x)
    detrended = pd.Series(values.values - trend, index=values.index)
    corr = detrended.autocorr(lag=seasonal_lag)
    if pd.isna(corr):
        return False
    return corr >= threshold

FORECAST_MAPE_MARGIN = 5
MIN_MONTHS_HW = 24
MIN_MONTHS_FORECAST = 6

def prepare_forecast_series(df):
    forecast_cols = ["Month", "Expense"]
    if "Household_ID" in df.columns:
        forecast_cols.append("Household_ID")

    monthly = df[forecast_cols].copy()
    monthly["Month"] = pd.to_datetime(monthly["Month"], errors="coerce")
    monthly["Expense"] = pd.to_numeric(monthly["Expense"], errors="coerce")
    monthly = monthly.dropna(subset=["Month", "Expense"])
    monthly["Month"] = monthly["Month"].dt.to_period("M").dt.to_timestamp()

    if "Household_ID" in monthly.columns:
        household_monthly = (
            monthly
            .dropna(subset=["Household_ID"])
            .groupby(["Month", "Household_ID"], as_index=False)["Expense"]
            .sum()
        )
        households_per_month = household_monthly.groupby("Month")["Household_ID"].nunique().median()
        agg_func = "mean" if households_per_month > 1 else "sum"
        monthly = household_monthly.groupby("Month", as_index=True)["Expense"].agg(agg_func).to_frame()
    else:
        rows_per_month = monthly.groupby("Month").size().median()
        agg_func = "sum" if rows_per_month > 2 else "mean"
        monthly = monthly.groupby("Month", as_index=True)["Expense"].agg(agg_func).to_frame()

    monthly = monthly.sort_index()
    monthly = monthly.asfreq("MS")
    return monthly["Expense"]

def _fit_forecast_model(y, model_label):
    if model_label == "Holt":
        return Holt(y).fit()
    return ExponentialSmoothing(
        y, trend="add", seasonal="add", seasonal_periods=12
    ).fit()

def _forecast_mape(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    nonzero = np.abs(actual) > 1e-9
    if not nonzero.any():
        return np.nan
    return float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)

def evaluate_forecast_models(y):
    rows = []
    fitted_models = {}
    validation_size = min(12, max(3, len(y) // 5))
    train_y = y.iloc[:-validation_size]
    validation_y = y.iloc[-validation_size:]

    for model_label in ["Holt", "Holt-Winters"]:
        try:
            validation_model = _fit_forecast_model(train_y, model_label)
            validation_pred = validation_model.forecast(validation_size)
            mape = _forecast_mape(validation_y.values, validation_pred.values)
            fitted_models[model_label] = _fit_forecast_model(y, model_label)
            rows.append({
                "Model": model_label,
                "MAPE (%)": round(mape, 2),
                "Forecast Accuracy (%)": round(max(0.0, 100.0 - mape), 2),
            })
        except Exception:
            fitted_models[model_label] = None
            rows.append({
                "Model": model_label,
                "MAPE (%)": None,
                "Forecast Accuracy (%)": None,
            })
    return pd.DataFrame(rows), fitted_models

def select_forecast_model(y, fitted_models, model_metrics_df, mape_margin=FORECAST_MAPE_MARGIN):
    holt_model = fitted_models.get("Holt")
    hw_model = fitted_models.get("Holt-Winters")

    holt_row = model_metrics_df[model_metrics_df["Model"] == "Holt"]
    hw_row = model_metrics_df[model_metrics_df["Model"] == "Holt-Winters"]
    holt_mape = holt_row.iloc[0]["MAPE (%)"] if not holt_row.empty else None
    hw_mape = hw_row.iloc[0]["MAPE (%)"] if not hw_row.empty else None

    has_seasonality = detect_seasonality(y)

    if holt_model is None:
        return None, "Moving Average fallback", 999.0, "Fallback used: Holt model could not be fitted."

    if (
        len(y) >= MIN_MONTHS_HW
        and has_seasonality
        and hw_model is not None
        and holt_mape is not None
        and hw_mape is not None
        and hw_mape <= holt_mape - mape_margin
    ):
        reason = (
            "Holt-Winters Model Selected"
        )
        return hw_model, "Holt-Winters", hw_mape, reason

    if len(y) < MIN_MONTHS_HW:
        reason = f"Holt selected: fewer than {MIN_MONTHS_HW} months (Holt-Winters needs ~2 years)."
    elif not has_seasonality:
        reason = "Holt selected: no strong seasonality detected."
    elif hw_model is None or hw_mape is None:
        reason = "Holt selected: Holt-Winters could not be fitted."
    elif holt_mape is not None and (holt_mape - hw_mape) < mape_margin:
        reason = "Holt selected: Holt-Winters validation improvement (< 5 percentage points) too small."
    else:
        reason = "Holt selected: simpler and more stable for this dataset."

    return holt_model, "Holt", holt_mape if holt_mape is not None else 999.0, reason

def forecast_expense(df, periods=12):
    y = prepare_forecast_series(df)

    future = pd.date_range(
        y.index.max() + pd.DateOffset(months=1),
        periods=periods,
        freq="MS"
    )

    residual_std = float(y.diff().dropna().std()) if len(y) > 2 else 1000
    empty_metrics = pd.DataFrame(columns=["Model", "MAPE (%)", "Forecast Accuracy (%)"])

    if len(y) < MIN_MONTHS_FORECAST:
        vals = [float(y.mean())] * periods
        model_name = "Moving Average fallback"
        mape = 999.0
        selection_reason = f"Moving average used: less than {MIN_MONTHS_FORECAST} months of data."
        model_metrics_df = empty_metrics
    else:
        model_metrics_df, fitted_models = evaluate_forecast_models(y)
        model, model_name, mape, selection_reason = select_forecast_model(
            y, fitted_models, model_metrics_df
        )
        if model is None:
            vals = [float(y.tail(3).mean())] * periods
            model_name = "Moving Average fallback"
            mape = 999.0
        else:
            try:
                vals = model.forecast(periods).values
                fitted_residuals = (y - model.fittedvalues).dropna()
                if len(fitted_residuals) > 1:
                    residual_std = float(fitted_residuals.std())
            except Exception:
                vals = [float(y.tail(3).mean())] * periods
                model_name = "Moving Average fallback"
                mape = 999.0
                selection_reason = "Fallback used: selected model failed during forecasting."

    out = pd.DataFrame({
        "Month": future,
        "Forecast_Expense": vals
    })
    horizon_scale = np.sqrt(np.arange(1, periods + 1))
    out["Lower_Bound"] = out["Forecast_Expense"] - 1.645 * residual_std * horizon_scale
    out["Upper_Bound"] = out["Forecast_Expense"] + 1.645 * residual_std * horizon_scale
    out["Lower_Bound"] = out["Lower_Bound"].clip(lower=0)
    out["Model_Used"] = model_name
    out["MAPE"] = mape

    return {
        "forecast": out,
        "selection_reason": selection_reason,
    }

VARIANCE_THRESHOLD = 1e-6

def find_high_correlation_pairs(df, features, threshold=0.80):
    if len(features) < 2:
        return pd.DataFrame(columns=["Feature 1", "Feature 2", "Correlation"])

    corr = df[features].corr().abs()
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            feature_1 = corr.columns[i]
            feature_2 = corr.columns[j]
            value = corr.iloc[i, j]
            if value >= threshold:
                pairs.append({
                    "Feature 1": feature_1,
                    "Feature 2": feature_2,
                    "Correlation": round(value, 2)
                })
    return pd.DataFrame(pairs)

DASHBOARD_EXCLUDE_COLS = [
    "Month",
    "Expense",
    "Household_ID",
    "Household_Type_Code",
    "Household",
]

def format_feature_label(feature_name):
    return feature_name.replace("_", " ")

def format_input_label(feature_name):
    acronym_labels = {
        "EMI": "EMI",
        "ID": "ID",
    }
    words = feature_name.replace("_", " ").split()
    return " ".join(acronym_labels.get(word.upper(), word.title()) for word in words)

def detect_date_columns(df):
    priority_names = ("month", "date", "period", "datetime", "timestamp", "yearmonth", "year_month")
    
    candidates = []

    for col in df.columns:
        normalized = col.lower().replace(" ", "_")
        if any(name in normalized for name in priority_names):
            candidates.append(col)

    for col in df.columns:
        if col in candidates:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        parsed = pd.to_datetime(series.head(min(30, len(series))), errors="coerce")
        if parsed.notna().mean() >= 0.7:
            candidates.append(col)

    return list(dict.fromkeys(candidates)) or list(df.columns)

def detect_target_columns(df, date_col):
    preferred = (
        "expense", "revenue", "sales", "cost", "amount",
        "value", "price", "target", "claims", "spending",
    )
    candidates = []

    for col in df.columns:
        if col == date_col:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() < max(3, int(len(df) * 0.5)):
            continue
        if numeric.nunique() <= 1:
            continue
        candidates.append(col)

    def sort_key(column_name):
        lowered = column_name.lower()
        for index, keyword in enumerate(preferred):
            if keyword in lowered:
                return (0, index, column_name)
        return (1, 0, column_name)

    candidates.sort(key=sort_key)
    return candidates

def get_top_numeric_drivers(df, target="Expense", exclude_cols=None, top_n=4):
    exclude_cols = exclude_cols or DASHBOARD_EXCLUDE_COLS
    target_series = pd.to_numeric(df[target], errors="coerce")
    drivers = []

    for col in df.columns:
        if col in exclude_cols or col == target:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].nunique() <= 1:
            continue

        feature_series = pd.to_numeric(df[col], errors="coerce")
        corr = feature_series.corr(target_series)
        if pd.notna(corr):
            drivers.append((col, abs(corr)))

    drivers.sort(key=lambda item: item[1], reverse=True)
    return [col for col, _ in drivers[:top_n]]

def build_driver_chart_df(df, feature, target="Expense"):
    chart_df = df[[feature]].copy()
    chart_df[target] = pd.to_numeric(df[target], errors="coerce")
    chart_df[feature] = pd.to_numeric(chart_df[feature], errors="coerce")
    chart_df = chart_df.dropna()

    if chart_df.empty:
        return chart_df, feature

    if chart_df[feature].nunique() <= 8:
        result = chart_df.groupby(feature, as_index=False)[target].mean()
        result[feature] = result[feature].astype(str)
        return result, feature

    n_bins = min(5, chart_df[feature].nunique())
    if n_bins < 2:
        result = chart_df.groupby(feature, as_index=False)[target].mean()
        return result, feature

    chart_df["_range"] = pd.qcut(chart_df[feature], q=n_bins, duplicates="drop")
    result = (
        chart_df
        .groupby("_range", observed=False)
        .agg(**{target: (target, "mean")}, _value=(feature, "median"))
        .reset_index(drop=True)
    )
    result["_value"] = result["_value"].round(0).map(lambda value: f"{value:,.0f}")
    return result, "_value"

def calculate_vif_values(df, features):
    """Calculate VIF diagnostics without changing the fitted model or features."""
    clean = df[features].replace([np.inf, -np.inf], np.nan).dropna()
    vif_values = {}

    for feature in features:
        other_features = [col for col in features if col != feature]
        if not other_features or clean[feature].var() <= VARIANCE_THRESHOLD:
            vif_values[feature] = 1.0
            continue

        auxiliary_model = LinearRegression().fit(
            clean[other_features], clean[feature]
        )
        auxiliary_r2 = auxiliary_model.score(
            clean[other_features], clean[feature]
        )
        vif_values[feature] = (
            np.inf if auxiliary_r2 >= 0.999999 else 1.0 / (1.0 - auxiliary_r2)
        )

    return vif_values


def interpret_regression_coefficient(
    feature,
    coeff,
    model_df,
    features,
    target_label="Expense",
    vif_values=None,
    corr_threshold=0.80,
):
    """Build pattern-aware coefficient guidance without altering model results."""
    label = format_feature_label(feature)
    target_text = target_label.lower()
    feature_series = pd.to_numeric(model_df[feature], errors="coerce")
    target_series = pd.to_numeric(model_df[target_label], errors="coerce")
    raw_corr = feature_series.corr(target_series)
    vif = (vif_values or {}).get(feature, np.nan)

    correlation_matrix = model_df[features].corr().abs()
    correlated_peers = [
        format_feature_label(peer)
        for peer in features
        if peer != feature
        and pd.notna(correlation_matrix.loc[feature, peer])
        and correlation_matrix.loc[feature, peer] >= corr_threshold
    ]

    raw_direction = "positively" if raw_corr >= 0 else "negatively"
    coefficient_direction = "increase" if coeff >= 0 else "decrease"
    sign_reversal = (
        pd.notna(raw_corr)
        and abs(raw_corr) >= 0.05
        and abs(coeff) > 1e-9
        and np.sign(raw_corr) != np.sign(coeff)
    )
    high_vif = pd.notna(vif) and vif >= 5
    severe_vif = pd.notna(vif) and vif >= 10
    # Show detailed interpretation only for predictors that are part of a
    # highly correlated feature pair. VIF and sign reversal add context, but
    # do not independently cause a predictor to appear in this section.
    needs_caution = bool(correlated_peers)

    if not needs_caution:
        return None

    bullets = [f"**{label}**"]

    if pd.notna(raw_corr):
        if abs(raw_corr) < 0.10:
            bullets.append(
                f"- **{label}** has a weak raw relationship with **{target_label}** "
                f"(r = {raw_corr:.2f})."
            )
        else:
            bullets.append(
                f"- **{label}** and **{target_label}** are {raw_direction} correlated "
                f"overall (r = {raw_corr:.2f})."
            )

    bullets.append(
        f"- After controlling for the other predictors, **{label}** has a conditional "
        f"coefficient of **{coeff:,.2f}**: a one-unit increase is associated with an "
        f"approximate **₹{abs(coeff):,.2f} {coefficient_direction}** in predicted {target_text}."
    )

    if sign_reversal:
        bullets.append(
            "- The raw relationship and conditional coefficient have opposite directions, "
            "which indicates a sign reversal."
        )

    if high_vif:
        severity = "Severe" if severe_vif else "High"
        vif_text = "infinite" if np.isinf(vif) else f"{vif:.2f}"
        bullets.append(
            f"- {severity} multicollinearity (VIF = {vif_text}) may make this coefficient unstable."
        )
    elif correlated_peers:
        bullets.append(
            f"- **{label}** is highly correlated with **{', '.join(correlated_peers)}**, "
            "so its individual coefficient should be interpreted cautiously."
        )

    bullets.append(
        f"- This result does **not** imply that changing {label.lower()} causes "
        f"{target_text} to {'increase' if coeff >= 0 else 'decrease'}."
    )

    return "\n\n".join(bullets)


def prepare_model_data(df):

    df = df.copy()

    # Check required columns
    required_cols = ["Month", "Expense"]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        st.error(
            f"Missing required column(s): {', '.join(missing_cols)}"
        )
        st.stop()

    # Convert Month column to datetime
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")

    # Validate Month values
    if df["Month"].isna().sum() > 0:
        st.error(
            "Month column contains invalid date values. Please use formats like Jan-2021, 2021-01, or 01-Jan-2021."
        )
        st.stop()

    df = df.sort_values("Month").reset_index(drop=True)

    # Convert all non-date columns to numeric wherever possible
    for col in df.columns:
        if col != "Month":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    exclude_cols = [
        "Month",
        "Expense",
        "Household_ID",
        "Household_Type_Code",
        "Household"
    ]

    features = [
        col for col in df.columns
        if col not in exclude_cols
        and pd.api.types.is_numeric_dtype(df[col])
        and df[col].nunique() > 1
    ]

    features = [
        col for col in features
        if df[col].var() > VARIANCE_THRESHOLD
    ]

    df = df.dropna(subset=features + ["Expense"])

    return df, features
def train_models(df):
    df, features = prepare_model_data(df)

    if len(df) < 8 or len(features) == 0:
        return None

    X = df[features]
    y = df["Expense"]

    # Train-test split
    # Uses 80% data for training and 20% data for testing
    split_index = int(len(df) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    evaluation_reg = LinearRegression().fit(X_train, y_train)
    evaluation_rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)

    reg_pred = evaluation_reg.predict(X_test)
    rf_pred = evaluation_rf.predict(X_test)

    # Refit on all available observations for coefficient analysis and new predictions.
    reg = LinearRegression().fit(X, y)
    rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, y)

    return {
        "features": features,
        "reg": reg,
        "rf": rf,
        "X_test": X_test,
        "y_test": y_test,
        "test_dates": df["Month"].iloc[split_index:],
        "reg_pred": reg_pred,
        "rf_pred": rf_pred,
        "reg_mae": mean_absolute_error(y_test, reg_pred),
        "rf_mae": mean_absolute_error(y_test, rf_pred),
        "reg_r2": r2_score(y_test, reg_pred),
        "rf_r2": r2_score(y_test, rf_pred),
    }

def build_model_comparison_df(res):
    comparison_df = pd.DataFrame({
        "Month": pd.to_datetime(res["test_dates"]).values,
        "Actual": res["y_test"].values,
        "Regression Prediction": res["reg_pred"],
        "Random Forest Prediction": res["rf_pred"],
    })
    return (
        comparison_df
        .groupby("Month", as_index=False)[
            ["Actual", "Regression Prediction", "Random Forest Prediction"]
        ]
        .mean()
        .sort_values("Month")
    )

def alerts(df, fc):
    latest = df.sort_values("Month").iloc[-1]
    msgs = []

    if latest["Expense"] > fc.iloc[0]["Forecast_Expense"]:
        msgs.append("🟠 Latest expense is higher than next forecast.")

    if latest["Expense"] > df["Expense"].mean() * 1.2:
        msgs.append("🟡 Spending is unusually high compared to average.")

    if "Festival" in df.columns and latest["Festival"] == 1:
        msgs.append("🎉 Festival month detected. Extra budget may be needed.")

    if "Travel_Days" in df.columns and latest["Travel_Days"] >= 4:
        msgs.append("✈️ High travel days detected.")

    if "Emergency" in df.columns and latest["Emergency"] == 1:
        msgs.append("🚨 Emergency month detected. Check medical or miscellaneous expenses.")

    return msgs or ["🟢 Spending pattern looks normal."]


st.sidebar.markdown("""
<style>
.sidebar-control-label {
    margin: 14px 0 7px 0;
    padding: 8px 11px;
    border-left: 4px solid #B8A1E3;
    border-radius: 8px;
    background: linear-gradient(90deg, rgba(216, 196, 240, 0.22), rgba(190, 227, 230, 0.10));
    color: #E3D5F5;
    font-size: 15px;
    font-weight: 800;
    letter-spacing: 0.3px;
}
section[data-testid="stSidebar"] details > summary {
    margin-top: 10px;
    padding: 8px 11px;
    border-left: 4px solid #B8A1E3;
    border-radius: 8px;
    background: linear-gradient(90deg, rgba(216, 196, 240, 0.22), rgba(190, 227, 230, 0.10));
    color: #E3D5F5;
    font-size: 15px;
    font-weight: 800;
    letter-spacing: 0.3px;
    cursor: pointer;
}
section[data-testid="stSidebar"] details > summary svg {
    display: none;
}
section[data-testid="stSidebar"] details > summary::after {
    content: "";
    margin-left: auto;
    width: 7px;
    height: 7px;
    border-right: 2px solid #FFFFFF;
    border-bottom: 2px solid #FFFFFF;
    transform: rotate(45deg);
    transform-origin: center;
    transition: transform 0.2s ease;
}
section[data-testid="stSidebar"] details[open] > summary::after {
    transform: rotate(225deg);
}
section[data-testid="stSidebar"] div[data-baseweb="select"] *,
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] label,
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"],
section[data-testid="stSidebar"] div[role="radiogroup"] label,
section[data-testid="stSidebar"] input[type="radio"] {
    cursor: pointer !important;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-control-label">Upload Dataset</div>', unsafe_allow_html=True)
uploaded_main = st.sidebar.file_uploader(
    "Upload Main Dataset CSV",
    type="csv",
    label_visibility="collapsed"
)
#uploaded_category = st.sidebar.file_uploader("Upload Category Dataset CSV", type="csv")

sample_df = load_sample_data()
sample_category_df = load_category_data()

if uploaded_main:
    df_raw = pd.read_csv(uploaded_main)
    data_source = "Uploaded Main Dataset "
else:
    df_raw = sample_df
    data_source = "Sample Dataset "

df_raw = standardize_columns(df_raw)
date_candidates = detect_date_columns(df_raw) 
date_default = date_candidates.index("Month") if "Month" in date_candidates else 0
st.sidebar.markdown('<div class="sidebar-control-label">Select the Date Column</div>', unsafe_allow_html=True)
date_col = st.sidebar.selectbox(
    "Select the date column",
    date_candidates,
    index=date_default,
    label_visibility="collapsed"
)
has_valid_date = False

try:
    test_dates = pd.to_datetime(
        df_raw[date_col],
        errors="coerce"
    )

    has_valid_date = (
        test_dates.notna().mean() >= 0.8
        and test_dates.dt.year.between(2000, 2035).mean() >= 0.8
    )

except Exception:
    has_valid_date = False

target_candidates = detect_target_columns(df_raw, date_col)
if not target_candidates:
    st.error("No suitable numeric target column found in the dataset.")
    st.stop()

target_default = target_candidates.index("Expense") if "Expense" in target_candidates else 0
st.sidebar.markdown('<div class="sidebar-control-label">Select the Target Variable</div>', unsafe_allow_html=True)
target_col = st.sidebar.selectbox(
    "Select the target variable",
    target_candidates,
    index=target_default,
    label_visibility="collapsed"
)

target_label = format_feature_label(target_col)
app_title = f"Smart {target_label} Predictor"
predictor_page = f"{target_label} Predictor"

df = df_raw.copy()
if date_col != "Month":
    df = df.rename(columns={date_col: "Month"})

if target_col != "Expense":
    if "Expense" in df.columns:
        df = df.drop(columns=["Expense"])

    df = df.rename(columns={target_col: "Expense"})
df = fix_month_column(df)

if df["Month"].isna().any():
    st.error("Some date values could not be converted. Please check the selected date column format.")
    st.write(df[df["Month"].isna()])
    st.stop()

_, detected_features = prepare_model_data(df)
if detected_features:
    with st.sidebar.expander(f"Detected Features ({len(detected_features)})"):
        for feature in detected_features:
            st.markdown(f"- {format_input_label(feature)}")
else:
    st.sidebar.info("No numeric features detected.")

if uploaded_main:
    category_df = pd.DataFrame()
    category_source = "No Category Dataset"
else:
    category_df = sample_category_df
    category_source = "Sample Category Dataset "

pages = [
    "Dashboard",
    "Dataset",
    "Regression",
    "Random Forest",
    "Model Comparison",
    predictor_page,
]

if has_valid_date:
    pages.insert(2, "Forecasting")

st.sidebar.markdown('<div class="sidebar-control-label">Menu</div>', unsafe_allow_html=True)
page = st.sidebar.radio("Menu", pages, label_visibility="collapsed")


if page == "Dashboard":
    st.markdown(
        f'<div class="app-page-title regression-page-title">📊 {app_title}</div>',
        unsafe_allow_html=True
    )

    dashboard_series = prepare_forecast_series(df).dropna()
    fc = forecast_expense(df)["forecast"]
    show_forecast_kpi = (
    not fc.empty and
    fc.iloc[0]["MAPE"] <= 50
)
    
    st.markdown("""
    <style>
    
    .kpi-card {
        min-height: 92px;
        padding: 12px 8px;
        border-radius: 12px;
        border: 1px solid rgba(216, 196, 240, 0.42);
        background: linear-gradient(135deg, rgba(216, 196, 240, 0.10), rgba(190, 227, 230, 0.05));
        text-align: center;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.10);
    }

    .kpi-title {
        color: #E3D5F5;
        font-size: 16px;
        font-weight: 750;
        margin-bottom: 7px;
        letter-spacing: 0.3px;
    }
    
    .kpi-value1 {
        font-size: 23px;
        color: #38BDF8;   /* Blue */
        font-weight: 900;
    }

    .kpi-value2 {
        font-size: 23px;
        color: #A78BFA;   /* Purple */
        font-weight: 900;
    }

    .kpi-value3 {
        font-size: 23px;
        color: #34D399;   /* Green */
        font-weight: 900;
    }

    .kpi-value4 {
        font-size: 23px;
        color: #FBBF24;   /* Gold */
        font-weight: 900;
    }
   
    div[data-testid="stPlotlyChart"] {
        border: 1.5px solid #CBD5E1;
        border-radius: 14px;
        padding: 10px;
        overflow: hidden;
        background-color: transparent;
    }

    div[data-testid="stPlotlyChart"] > div {
        border-radius: 14px;
    }
   
    </style>
    """, unsafe_allow_html=True)

    if show_forecast_kpi:
        c1, c2, c3, c4 = st.columns(4)
    else:
        c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total {target_label}</div>
            <div class="kpi-value1">₹{dashboard_series.sum():,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Average {target_label}</div>
            <div class="kpi-value2">₹{dashboard_series.mean():,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Latest Monthly {target_label}</div>
            <div class="kpi-value3">₹{dashboard_series.iloc[-1]:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    if show_forecast_kpi:
        with c4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Next Forecast</div>
                <div class="kpi-value4">₹{fc.iloc[0]['Forecast_Expense']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    
             

    # -----------------------------
    # Row 1: Monthly Trend + Regression Vs Random Forest
    # -----------------------------

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        
        trend_df = dashboard_series.rename("Expense").reset_index()

        fig_line = px.line(
            trend_df,
            x="Month",
            y="Expense",
            markers=True,
            title=f"Monthly {target_label} Trend"
        )

        fig_line.update_layout(
            height=320,
            title_x=0.42,
            title_xanchor="center",
            margin=dict(l=10, r=10, t=80, b=10),
            xaxis_title="Month",
            yaxis_title=target_label
        )
        fig_line.update_traces(
            line=dict(color="#10B981", width=3),      # Emerald Green line
            marker=dict(color="#FBBF24", size=7)      # Gold markers
        )
        fig_line.update_xaxes(
            type="date",
            tickformat="%b-%Y",
            dtick="M3",
            showticklabels=False
        )

        st.plotly_chart(fig_line, use_container_width=True) 

    with row1_col2:

        res = train_models(df)

        if res:

            comparison_df = build_model_comparison_df(res)

            fig_model_combo = go.Figure()

            fig_model_combo.add_trace(go.Bar(
                x=comparison_df["Month"],
                y=comparison_df["Actual"],
                name=f"Actual {target_label}",
                marker_color="#A7C7E7",
                width=10 * 24 * 60 * 60 * 1000
            ))

            fig_model_combo.add_trace(go.Scatter(
                x=comparison_df["Month"],
                y=comparison_df["Regression Prediction"],
                name="Regression Prediction",
                mode="lines+markers",
                line=dict(color="#F97316", width=3),
                marker=dict(size=6)
            ))

            fig_model_combo.add_trace(go.Scatter(
                x=comparison_df["Month"],
                y=comparison_df["Random Forest Prediction"],
                name="Random Forest Prediction",
                mode="lines+markers",
                line=dict(color="#10B981", width=3),
                marker=dict(size=6)
            ))

            fig_model_combo.update_layout(
            title=dict(
                text="Regression vs Random Forest",
                font=dict(size=20),
                x=0.5,
                xanchor="center"
            ),
            height=320,
            margin=dict(l=10, r=10, t=80, b=10),
            xaxis_title="Test Month",
            yaxis_title=target_label,
            template="plotly_white",
            barmode="overlay"
        )

        st.plotly_chart(fig_model_combo, use_container_width=True)

    

    # -----------------------------
    # Rows 2+: Top numeric drivers (auto-selected by correlation with Expense)
    # -----------------------------

    driver_bar_colors = ["#B0E0E6", "#CDB4DB", "#FFD6A5", "#F4C2C2", "#BDE0FE", "#A8E6CF"]
    top_drivers = get_top_numeric_drivers(df, top_n=4)

    if top_drivers:
        for row_start in range(0, len(top_drivers), 2):
            row_cols = st.columns(2)
            for col_idx, feature in enumerate(top_drivers[row_start:row_start + 2]):
                with row_cols[col_idx]:
                    chart_df, x_col = build_driver_chart_df(df, feature)
                    if chart_df.empty:
                        continue

                    feature_label = format_feature_label(feature)
                    fig_driver = px.bar(
                        chart_df,
                        x=x_col,
                        y="Expense",
                        title=f"{target_label} vs {feature_label}",
                        text_auto=".2s"
                    )

                    color = driver_bar_colors[(row_start + col_idx) % len(driver_bar_colors)]
                    fig_driver.update_traces(marker_color=color, width=0.45)
                    fig_driver.update_layout(
                        height=300,
                        title_x=0.42,
                        title_xanchor="center",
                        margin=dict(l=10, r=10, t=60, b=10),
                        showlegend=False,
                        bargap=0.35,
                        yaxis=dict(range=[0, chart_df["Expense"].max() * 1.15])
                    )

                    if x_col == "_value":
                        fig_driver.update_xaxes(
                            type="category",
                            tickangle=0,
                            title_text=feature_label
                        )
                    elif chart_df[x_col].nunique() > 4:
                        fig_driver.update_xaxes(tickangle=45)

                    st.plotly_chart(fig_driver, use_container_width=True)

elif page == "Dataset":
    st.markdown(
        '<div class="app-page-title regression-page-title">📂 Dataset</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"**Source:** {data_source}  |  "
        f"**Date :** {date_col}  |  "
        f"**Target Variable:** {target_label}"
    )
    display_df = df_raw.copy()

    if "Month" in display_df.columns:
        display_df["Month"] = pd.to_datetime(display_df["Month"]).dt.strftime("%b-%Y")

    st.dataframe(display_df, use_container_width=True,height=800,hide_index=True)

    
    
elif page == "Forecasting":
   
    st.markdown(
        f'<div class="app-page-title regression-page-title">📈 {target_label} Forecasting</div>',
        unsafe_allow_html=True
    )
    periods = 12

    fc_result = forecast_expense(df, periods)
    fc = fc_result["forecast"]
    selection_reason = fc_result["selection_reason"]
    best_mape = fc.iloc[0]["MAPE"]

    st.markdown(
        f'<div class="compact-info-box">ⓘ&nbsp; '
        f'<span class="message-highlight-model">{selection_reason.split(":", 1)[0]}</span>'
        f'{":" + selection_reason.split(":", 1)[1] if ":" in selection_reason else ""}</div>',
        unsafe_allow_html=True
    )

    df_chart = (
        prepare_forecast_series(df)
        .dropna()
        .rename("Expense")
        .reset_index()
    )

    fc_chart = fc.copy()
    fc_chart["Month"] = pd.to_datetime(fc_chart["Month"], errors="coerce")

    forecast_display = fc[["Month","Forecast_Expense", "Upper_Bound", "Lower_Bound"]].copy()

    if "Month" in forecast_display.columns:
        forecast_display["Month"] = pd.to_datetime(
        forecast_display["Month"]
        ).dt.strftime("%b-%Y")
    forecast_display[[
        "Forecast_Expense", 
        "Upper_Bound", 
        "Lower_Bound"
    ]] = (
        forecast_display[[
            "Forecast_Expense", 
            "Upper_Bound", 
            "Lower_Bound"
        ]]
        .round(0)
        .astype(int)
    )

    if best_mape > 50:
        st.warning("No reliable forecasting model found.")

        st.error("""
        Forecasting graph is not displayed because the dataset contains
        high volatility, irregular expense spikes, and weak trend/seasonality.
        """)

    else:
        styled_forecast = (
            forecast_display.style
            .hide(axis="index")
            .set_table_styles([
                {
                    "selector": "th.col_heading",
                    "props": [
                        ("font-size", "18px"),
                        ("font-weight", "bold"),
                        ("color", "white"),
                        ("background-color", "#1E293B"),
                        ("padding", "7px 10px"),
                        ("text-align", "left")
                    ]
                },
                {
                    "selector": "td",
                    "props": [
                        ("font-size", "15px"),
                        ("padding", "7px 10px")
                    ]
                }
            ])
        )
        st.table(styled_forecast)
        st.markdown(
            f'<div class="app-subheading">{target_label} Forecast</div>',
            unsafe_allow_html=True
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_chart["Month"], y=df_chart["Expense"], mode="lines+markers", name="Actual", line=dict(color="#1E40AF", width=3),      marker=dict(color="#FBBF24", size=7)))
        fig.add_trace(go.Scatter(x=fc_chart["Month"], y=fc_chart["Forecast_Expense"], mode="lines+markers", name="Forecast", line=dict(color="red", width=3),marker=dict(color="pink", size=7)))

        fig.add_trace(go.Scatter(x=fc_chart["Month"], y=fc_chart["Upper_Bound"], mode="lines", name="Upper Bound", line=dict(dash="dash", color="orange")))
        fig.add_trace(go.Scatter(x=fc_chart["Month"], y=fc_chart["Lower_Bound"], mode="lines", name="Lower Bound", line=dict(dash="dash", color="green")))

        fig.update_layout(
            height=450,
            xaxis_title="Month",
            yaxis_title=target_label,
            margin=dict(t=30, b=0, l=10, r=10),
            xaxis=dict(
                range=[df_chart["Month"].min(), fc_chart["Month"].max()]
            )
                
        )
                
        fig.update_xaxes(type="date", tickformat="%b-%Y", dtick="M3")

        st.plotly_chart(fig, use_container_width=True)
        
          

        

        st.markdown('<div class="app-subheading">🚨 Forecast Alerts</div>', unsafe_allow_html=True)
        latest_forecast = fc["Forecast_Expense"].iloc[-1]
        avg_target = prepare_forecast_series(df).dropna().mean()

        if latest_forecast > avg_target * 1.25:
            st.markdown(
                f'<div class="compact-alert-box alert-high">High alert: latest forecasted '
                f'{target_label.lower()} is ₹{latest_forecast:,.0f}, which is more than 25% above the average.</div>',
                unsafe_allow_html=True
            )

        elif latest_forecast > avg_target * 1.10:
            st.markdown(
                f'<div class="compact-alert-box alert-moderate"><span class="message-highlight-moderate">'
                f'Moderate alert:</span>&nbsp; latest forecasted '
                f'{target_label.lower()} is ₹{latest_forecast:,.0f}, which is more than 10% above the average.</div>',
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                f'<div class="compact-alert-box alert-normal">{target_label} is within normal range. '
                f'Latest forecast is ₹{latest_forecast:,.0f}.</div>',
                unsafe_allow_html=True
            )
    st.caption(
        "Note: The 12-month forecast is intended for annual planning guidance. "
        "It is based only on the historical monthly expense pattern and assumes no "
        "major structural change or unexpected shock. The bounds are approximate uncertainty ranges."
    )

elif page == "Regression":
    st.markdown(
        '<div class="app-page-title regression-page-title">📘 Multiple Linear Regression</div>',
        unsafe_allow_html=True
    )
    res = train_models(df)

    if res:
        st.markdown("""
        <style>
        .metric-card {
            min-height: 92px;
            padding: 12px 8px;
            border: 1px solid rgba(216, 196, 240, 0.42);
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(216, 196, 240, 0.10), rgba(190, 227, 230, 0.05));
            text-align: center;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.10);
        }

        .metric-title {
            color: #E3D5F5;
            font-size: 16px;
            font-weight: 750;
            margin-bottom: 7px;
            letter-spacing: 0.3px;
        }

        .metric-mae {
            font-size: 23px;
            font-weight: 900;
            color: #38BDF8;
        }

        .metric-r2 {
            font-size: 23px;
            font-weight: 900;
            color: #34D399;
        }
        </style>
        """, unsafe_allow_html=True)

        m1, m2 = st.columns(2)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">MAE</div>
                <div class="metric-mae">₹{res['reg_mae']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">R²</div>
                <div class="metric-r2">{res['reg_r2']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

        df_model, _ = prepare_model_data(df)
        st.markdown('<div class="app-subheading">Highly Correlated Feature Pairs</div>', unsafe_allow_html=True)
        high_corr_df = find_high_correlation_pairs(df_model, res["features"], threshold=0.80)
        if not high_corr_df.empty:
            st.dataframe(high_corr_df, use_container_width=True, hide_index=True)
        else:
            st.markdown(
                '<div class="correlation-status">No highly correlated feature pairs found.</div>',
                unsafe_allow_html=True
            )

        coef = pd.DataFrame({
            "Feature": res["features"],
            "Coefficient": res["reg"].coef_
        }).sort_values("Coefficient", ascending=False).reset_index(drop=True)
        
        styled_coef = (
            coef.style
            .format({"Coefficient": "{:.2f}"})
            .set_table_styles([
                {
                    "selector": "th.col_heading",
                    "props": [
                        ("font-size", "18px"),
                        ("font-weight", "bold"),
                        ("color", "white"),
                        ("background-color", "#1E293B"),
                        ("padding", "7px 10px"),
                        ("text-align", "left")
                    
                    ]   
                },
                {
                    "selector": "td.row_heading",
                    "props": [
                        ("background-color", "transparent"),   # removes blue
                        ("color", "white"),           
                        ("font-size", "14px"),
                        ("padding", "7px")
                    ]
                },
                {
                    "selector": "td",
                    "props": [
                        ("font-size", "15px"),
                        ("padding", "7px 10px")
                    ]
                }
            ])    
        )

        st.table(styled_coef)
        
        reg = res["reg"]
        coeffs = reg.coef_
        intercept = reg.intercept_

        st.markdown('<div class="app-subheading">Regression Equation</div>', unsafe_allow_html=True)

        equation = f"{target_label} = {intercept:.1f}"
        for feature, coeff in zip(res["features"], coeffs):
            equation += f" + ({coeff:.1f} × {feature})"

        st.code(equation)

        if not high_corr_df.empty:
            correlated_features = sorted(set(
                high_corr_df["Feature 1"].tolist()
                + high_corr_df["Feature 2"].tolist()
            ))
            sign_reversals = []
            for feature, coeff in zip(res["features"], coeffs):
                if feature not in correlated_features:
                    continue
                raw_corr = df_model[feature].corr(df_model[target_label])
                if (
                    pd.notna(raw_corr)
                    and abs(raw_corr) >= 0.05
                    and abs(coeff) > 1e-9
                    and np.sign(raw_corr) != np.sign(coeff)
                ):
                    sign_reversals.append(
                        f"{format_feature_label(feature)} ({coeff:,.2f})"
                    )

            feature_text = ", ".join(
                format_feature_label(feature)
                for feature in correlated_features
            )
            reversal_text = (
                f" Sign reversal detected for {', '.join(sign_reversals)}."
                if sign_reversals else ""
            )
            st.markdown(
                '<div class="app-subheading">Interpret with Caution</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**{feature_text}** are highly correlated, so their coefficients may be "
                f"unstable.{reversal_text} These coefficients should not be interpreted causally."
            )
        else:
            st.markdown(
                '<div class="app-subheading">Interpretation</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                "No major multicollinearity concern was detected. The coefficients are more "
                "stable for interpretation."
            )
        

        st.markdown(
            f'<div class="regression-info-box">📘 Regression estimates associations with '
            f'{target_label.lower()}, not causal effects. Coefficients should be interpreted '
            'alongside holdout MAE and R².</div>',
            unsafe_allow_html=True
        )
    else:
        st.warning("Not enough data.")
    

elif page == "Random Forest":
    st.markdown(
        '<div class="app-page-title regression-page-title">🌳 Random Forest Prediction</div>',
        unsafe_allow_html=True
    )
    res = train_models(df)

    if res:
        st.markdown("""
        <style>
        .metric-card {
            min-height: 92px;
            padding: 12px 8px;
            border: 1px solid rgba(216, 196, 240, 0.42);
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(216, 196, 240, 0.10), rgba(190, 227, 230, 0.05));
            text-align: center;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.10);
        }

        .metric-title {
            color: #E3D5F5;
            font-size: 16px;
            font-weight: 750;
            margin-bottom: 7px;
            letter-spacing: 0.3px;
        }

        .metric-mae {
            font-size: 23px;
            font-weight: 900;
            color: #F59E0B;
        }

        .metric-r2 {
            font-size: 23px;
            font-weight: 900;
            color: #EC4899;
        }
        </style>
        """, unsafe_allow_html=True)

        m1, m2 = st.columns(2)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">MAE</div>
                <div class="metric-mae">₹{res['rf_mae']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">R²</div>
                <div class="metric-r2">{res['rf_r2']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="app-subheading">Feature Importance</div>', unsafe_allow_html=True)

        importance_df = pd.DataFrame({
            "Feature": res["features"],
            "Importance": res["rf"].feature_importances_
        })

        importance_df["Importance_Percentage"] = importance_df["Importance"] * 100
        importance_df = importance_df.sort_values("Importance_Percentage", ascending=True)

        pastel_bar_colors = [
            "#BDE0FE",  # pastel blue
            "#D8B4FE",  # pastel lavender
            "#A8E6CF",  # pastel mint
            "#FFB3C6",  # pastel pink
            "#FFD6A5",  # pastel peach
            "#CAFFBF",  # pastel green
            "#FDFFB6",  # pastel lemon
            "#9BF6FF",  # pastel cyan
            "#FFC6FF",  # pastel magenta
            "#E2ECE9",  # pastel sage
            "#FDE2E4",  # pastel rose
            "#C7CEEA",  # pastel periwinkle
        ]
        bar_colors = [
            pastel_bar_colors[i % len(pastel_bar_colors)]
            for i in range(len(importance_df))
        ]

        fig_imp = px.bar(
            importance_df,
            x="Importance_Percentage",
            y="Feature",
            orientation="h",
            text=importance_df["Importance_Percentage"].round(1).astype(str) + "%",
        )

        fig_imp.update_traces(
            marker_color=bar_colors,
            textposition="outside",
            width=0.35
        )

        fig_imp.update_layout(
            height=max(500, 55 * len(importance_df)),
            width=880,
            xaxis=dict(
                title="Importance (%)",
                range=[0, 100]
            ),
            yaxis_title="",
            margin=dict(l=80, r=40, t=15, b=30),
            showlegend=False
        )

        st.plotly_chart(fig_imp, use_container_width=False)
        importance_df["Importance_Percentage"] = importance_df["Importance"] * 100
        importance_df = importance_df.sort_values("Importance_Percentage", ascending=False)
        st.markdown('<div class="app-subheading">Interpretation</div>', unsafe_allow_html=True)

        for i, row in importance_df.head(5).iterrows():
            st.markdown(
                f"- **{row['Feature']}** accounts for approximately **{row['Importance_Percentage']:.1f}%** "
                "of the model's feature importance."
            )

        st.markdown(
            f'<div class="regression-info-box">🌳 Random Forest identifies the relative importance of '
            f'predictors in {target_label.lower()} prediction. Unlike regression, it does not provide a direct '
            'mathematical equation, but it helps explain which variables the fitted model relies on most. '
            'Importance does not imply causation.</div>',
            unsafe_allow_html=True
        )
        
    else:
            st.warning("Not enough data.")

        
elif page == predictor_page:
    st.markdown(f'<div class="app-page-title">💡 {predictor_page}</div>', unsafe_allow_html=True)

    st.markdown("""
    <style>
    .prediction-card {
        height: 125px;
        padding: 14px;
        border-radius: 16px;
        box-sizing: border-box;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .regression-card {
        background: rgba(56, 189, 248, 0.12);
        border: 2px solid #38BDF8;
    }
    .forest-card {
        background: rgba(52, 211, 153, 0.12);
        border: 2px solid #34D399;
    }
    .prediction-title {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .regression-value {
        color: #38BDF8;
        font-size: 30px;
        font-weight: 800;
    }
    .forest-value {
        color: #34D399;
        font-size: 30px;
        font-weight: 800;
    }
    .recommendation-card {
        height: 125px;
        padding: 14px;
        border-radius: 16px;
        box-sizing: border-box;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        background: linear-gradient(135deg, rgba(167, 139, 250, 0.20), rgba(56, 189, 248, 0.12));
        border: 2px solid #A78BFA;
        box-shadow: 0 6px 20px rgba(167, 139, 250, 0.18);
    }
    .recommendation-label {
        color: #C4B5FD;
        font-size: 18px;
        font-weight: 700;
        letter-spacing: normal;
    }
    .recommendation-model {
        margin-top: 2px;
        font-size: 13px;
        font-weight: 600;
    }
    .recommendation-value {
        color: #FBBF24;
        margin-top: 4px;
        font-size: 30px;
        font-weight: 800;
    }
    div[data-testid="stButton"] {
        display: flex;
        justify-content: center;
    }
    div[data-testid="stButton"] > button {
        min-width: 140px;
        padding: 8px 18px;
        border: 3px solid #E8B89D;
        border-radius: 12px;
        background: rgba(249, 213, 192, 0.22);
        color: #F6DDCF;
        font-family: "Trebuchet MS", "Arial Black", sans-serif;
        font-size: 18px;
        font-weight: 900;
        letter-spacing: 0.8px;
        text-transform: none;
        text-shadow: 0 1px 0 rgba(255, 255, 255, 0.45);
        box-shadow: 0 6px 18px rgba(232, 184, 157, 0.28);
        transition: all 0.2s ease;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: #F6C7AC;
        color: #FFF1E8;
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(232, 184, 157, 0.42);
    }
    div[data-testid="stButton"] > button:active {
        transform: translateY(0);
    }
    div[data-testid="stNumberInput"],
    div[data-testid="stSelectbox"] {
        margin-bottom: 12px;
        padding: 12px 14px 14px 14px;
        border-left: 4px solid #B8A1E3;
        border-radius: 10px;
        background: linear-gradient(90deg, rgba(216, 196, 240, 0.16), rgba(190, 227, 230, 0.10));
    }
    div[data-testid="stNumberInput"] div[data-testid="stWidgetLabel"] p,
    div[data-testid="stSelectbox"] div[data-testid="stWidgetLabel"] p {
        color: #D8C4F0;
        font-size: 17px;
        font-weight: 800;
        letter-spacing: 0.7px;
        text-transform: uppercase;
    }
    div[data-testid="stNumberInput"] input {
        border-radius: 8px;
        font-size: 16px;
        font-weight: 600;
    }
    div[data-testid="stSelectbox"] > div > div {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    res = train_models(df)

    if res:
        st.write("Enter values for the available variables. The predictor automatically changes based on the uploaded dataset.")

        input_values = {}
        cols = st.columns(2)

        for i, feature in enumerate(res["features"]):
            with cols[i % 2]:
                input_label = format_input_label(feature)
                if feature == "Festival":
                    input_values[feature] = st.selectbox(input_label, [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                elif feature == "Emergency":
                    input_values[feature] = st.selectbox(input_label, [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                elif feature == "Household":
                    input_values[feature] = st.selectbox(
                        input_label,
                        [1, 2, 3, 4],
                        format_func=lambda x: {
                            1: "H1 - Single Person",
                            2: "H2 - Couple",
                            3: "H3 - Small Family",
                            4: "H4 - Large Family"
                        }.get(x, str(x))
                    )
                else:
                    default_value = float(df[feature].median()) if feature in df.columns else 0.0
                    input_values[feature] = st.number_input(
                        input_label,
                        value=default_value,
                        step=1.0
                    )

        input_df = pd.DataFrame([input_values])
        input_df = input_df[res["features"]]

        st.markdown("<br>", unsafe_allow_html=True)

        _, button_col, _ = st.columns([3, 1, 3])
        with button_col:
            predict_clicked = st.button(f"Predict {target_label}", use_container_width=True)

        if predict_clicked:
            reg_prediction = res["reg"].predict(input_df)[0]
            rf_prediction = res["rf"].predict(input_df)[0]
            best_model = "Random Forest" if res["rf_mae"] < res["reg_mae"] else "Multiple Linear Regression"
            final_prediction = rf_prediction if best_model == "Random Forest" else reg_prediction

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3, gap="medium")

            with c1:
                st.markdown(f"""
                <div class="prediction-card regression-card">
                    <div class="prediction-title">Regression Prediction</div>
                    <div class="regression-value">₹{reg_prediction:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="recommendation-card">
                    <div class="recommendation-label">Recommended Prediction</div>
                    <div class="recommendation-model">Best model: {best_model}</div>
                    <div class="recommendation-value">₹{final_prediction:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class="prediction-card forest-card">
                    <div class="prediction-title">Random Forest Prediction</div>
                    <div class="forest-value">₹{rf_prediction:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

            st.caption("Note: The input fields are generated dynamically from the numeric variables available in the uploaded dataset.")

    else:
        st.warning("Not enough data or no numeric independent variables available for prediction.")

elif page == "Model Comparison":
    st.markdown(
        '<div class="app-page-title regression-page-title">⚖️ Model Comparison</div>',
        unsafe_allow_html=True
    )
    res = train_models(df)

    if res:
        comp = pd.DataFrame({
            "Model": ["Multiple Linear Regression", "Random Forest"],
            "MAE": [res["reg_mae"], res["rf_mae"]],
            "R²": [res["reg_r2"], res["rf_r2"]]
        })

        st.dataframe(comp, use_container_width=True, hide_index=True)

        best = "Random Forest" if res["rf_mae"] < res["reg_mae"] else "Multiple Linear Regression"
        st.markdown(
            f'<div class="compact-info-box">ⓘ&nbsp; Best model based on lower MAE: {best}</div>',
            unsafe_allow_html=True
        )

        st.markdown("", unsafe_allow_html=True)
        st.markdown(
            '<div class="app-subheading">Regression vs Random Forest</div>',
            unsafe_allow_html=True
        )
        comparison_df = build_model_comparison_df(res)

        fig_compare = go.Figure()

        fig_compare.add_trace(go.Bar(
            x=comparison_df["Month"],
            y=comparison_df["Actual"],
            name=f"Actual {target_label}",
            marker_color="#A7C7E7",
            width=10 * 24 * 60 * 60 * 1000
        ))

        fig_compare.add_trace(go.Scatter(
            x=comparison_df["Month"],
            y=comparison_df["Regression Prediction"],
            name="Regression Prediction",
            mode="lines+markers",
            line=dict(color="#FFA15A", width=3),
            marker=dict(size=8)
        ))

        fig_compare.add_trace(go.Scatter(
            x=comparison_df["Month"],
            y=comparison_df["Random Forest Prediction"],
            name="Random Forest Prediction",
            mode="lines+markers",
            line=dict(color="#34D399", width=3),
            marker=dict(size=8)
        ))

        fig_compare.update_layout(
            xaxis_title="Test Month",
            yaxis_title=target_label,
            template="plotly_white",
            barmode="overlay",
            width=3000,
            height=500,
            yaxis=dict(
                range=[
                    0,
                    max(
                        comparison_df["Actual"].max(),
                        comparison_df["Regression Prediction"].max(),
                        comparison_df["Random Forest Prediction"].max()
                    ) * 1.15
                ]
            )
        )

        st.plotly_chart(fig_compare, use_container_width=False)

        

    else:
        st.warning("Not enough data.")
