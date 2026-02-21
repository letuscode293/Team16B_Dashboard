import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import pearsonr, ttest_ind
from statsmodels.tsa.stattools import acf, pacf

st.set_page_config(
    page_title="Global Warming Analysis Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

NUMERICAL_COLS = [
    "Temperature", "Temp_Anomaly", "CO2_ppm",
    "Sea_Level_mm", "Arctic_Ice_km2", "Population_B", "CO2_Emissions_Gt",
]

PLOT_VARS = [
    "Temp_Anomaly", "CO2_Emissions_Gt", "Temperature",
    "Sea_Level_mm", "Population_B", "Arctic_Ice_km2",
]

VAR_LABELS = {
    "Temp_Anomaly": "Temperature Anomaly (°C)",
    "CO2_Emissions_Gt": "CO2 Emissions (Gt)",
    "Temperature": "Temperature (°C)",
    "Sea_Level_mm": "Sea Level (mm)",
    "Population_B": "Population (Billions)",
    "Arctic_Ice_km2": "Arctic Ice (km²)",
    "CO2_ppm": "CO2 (ppm)",
}


@st.cache_data
def load_data():
    df = pd.read_csv("global_warming_dataset.csv")
    df_processed = df.copy()
    numerical = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if "Year" in numerical:
        numerical.remove("Year")
    for col in numerical:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    categorical = df_processed.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical:
        if df_processed[col].isnull().sum() > 0:
            mode_val = df_processed[col].mode()
            df_processed[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown", inplace=True)
    df_processed = df_processed.drop_duplicates()
    df_processed = df_processed.sort_values(["Year", "Month_Num"]).reset_index(drop=True)
    return df_processed


df = load_data()

yearly_data = df.groupby("Year").agg({
    "Temp_Anomaly": "mean",
    "CO2_Emissions_Gt": "mean",
    "Temperature": "mean",
    "Sea_Level_mm": "mean",
    "Arctic_Ice_km2": "mean",
}).reset_index()

global_temp_by_year = df.groupby("Year")["Temperature"].mean().reset_index()
global_temp_series = global_temp_by_year.set_index("Year")["Temperature"]

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
sections = [
    "Overview",
    "Distributions",
    "Box Plots",
    "Correlation Heatmap",
    "Time Series Trends",
    "Scatter Plots",
    "Temperature Over Time",
    "Trend & First Difference",
    "ACF / PACF",
    "Train-Test Split",
    "Temperature Trend Analysis",
    "CO2 vs Temperature",
    "Period Comparison",
]
section = st.sidebar.radio("Go to", sections)

st.title("Global Warming Analysis Dashboard")
st.markdown("Interactive dashboard consolidating all visualizations from the analysis notebook.")
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
#  1. Overview
# ═══════════════════════════════════════════════════════════════════════════════
if section == "Overview":
    st.header("Dataset Overview")
    st.caption("A high-level summary of the global warming dataset including record count, time span, and descriptive statistics for all numerical features.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Year Range", f"{int(df['Year'].min())} – {int(df['Year'].max())}")
    col3.metric("Features", len(df.columns))

    st.subheader("Descriptive Statistics")
    st.dataframe(df[NUMERICAL_COLS].describe().T.style.format("{:.3f}"), use_container_width=True)

    st.subheader("Sample Data")
    st.dataframe(df.head(20), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  2. Distribution Histograms
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Distributions":
    st.header("Distribution of Key Variables")
    st.caption("Histograms with marginal box plots showing how each climate variable is distributed across all 1,752 monthly observations — useful for spotting skewness and outliers.")
    cols = st.columns(3)
    for i, var in enumerate(PLOT_VARS):
        with cols[i % 3]:
            fig = px.histogram(
                df, x=var, nbins=50,
                title=f"Distribution of {VAR_LABELS.get(var, var)}",
                color_discrete_sequence=["#636EFA"],
                marginal="box",
            )
            fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  3. Box Plots
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Box Plots":
    st.header("Box Plots of Key Variables")
    st.caption("Box plots summarize the median, quartiles, and outliers for each variable — helpful for comparing spread and detecting extreme values at a glance.")
    cols = st.columns(3)
    for i, var in enumerate(PLOT_VARS):
        with cols[i % 3]:
            fig = px.box(
                df, y=var,
                title=f"{VAR_LABELS.get(var, var)}",
                color_discrete_sequence=["#EF553B"],
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  4. Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Correlation Heatmap":
    st.header("Correlation Matrix of Numerical Variables")
    st.caption("Pearson correlation matrix revealing linear relationships between all numerical features — values near +1 or -1 indicate strong positive or negative associations.")
    corr = df[NUMERICAL_COLS].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_masked = corr.where(~mask)

    fig = go.Figure(data=go.Heatmap(
        z=corr_masked.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu_r",
        zmid=0,
        text=corr_masked.round(2).values,
        texttemplate="%{text}",
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(height=650, width=750, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  5. Time Series Trends
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Time Series Trends":
    st.header("Time Series Trends (Yearly Averages)")
    st.caption("Yearly averaged trends for four key climate indicators — showing how temperature anomalies, CO2 emissions, sea level, and Arctic ice extent have evolved over time.")
    configs = [
        ("Temp_Anomaly", "Average Temperature Anomaly Over Time", "red"),
        ("CO2_Emissions_Gt", "Average CO2 Emissions Over Time", "orange"),
        ("Sea_Level_mm", "Average Sea Level Over Time", "royalblue"),
        ("Arctic_Ice_km2", "Arctic Ice Extent Over Time", "green"),
    ]
    cols = st.columns(2)
    for i, (var, title, color) in enumerate(configs):
        with cols[i % 2]:
            fig = px.line(
                yearly_data, x="Year", y=var,
                title=title,
                labels={"Year": "Year", var: VAR_LABELS.get(var, var)},
            )
            fig.update_traces(line_color=color, line_width=2)
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  6. Scatter Plots – Temperature Relationships
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Scatter Plots":
    st.header("Temperature Relationships")
    st.caption("Scatter plots with OLS trend lines exploring how CO2 emissions, population, sea level, and Arctic ice each relate to global temperature.")
    pairs = [
        ("CO2_Emissions_Gt", "darkred"),
        ("Population_B", "green"),
        ("Sea_Level_mm", "red"),
        ("Arctic_Ice_km2", "royalblue"),
    ]
    sample = df.sample(min(800, len(df)), random_state=42)
    cols = st.columns(2)
    for i, (xvar, color) in enumerate(pairs):
        with cols[i % 2]:
            fig = px.scatter(
                sample, x=xvar, y="Temperature",
                trendline="ols",
                title=f"{VAR_LABELS.get(xvar, xvar)} vs Temperature",
                labels={xvar: VAR_LABELS.get(xvar, xvar), "Temperature": "Temperature (°C)"},
                opacity=0.5,
            )
            fig.update_traces(marker=dict(size=4, color=color))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  7. Global Average Temperature Over Time
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Temperature Over Time":
    st.header("Global Average Temperature Over Time")
    st.caption("Year-by-year global average temperature from 1880 to 2025 — the upward curve in recent decades highlights the accelerating warming trend.")
    fig = px.line(
        global_temp_by_year, x="Year", y="Temperature",
        labels={"Temperature": "Temperature (°C)"},
    )
    fig.update_traces(line_color="darkred", line_width=2, mode="lines+markers", marker=dict(size=4))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  8. Trend & First Difference
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Trend & First Difference":
    st.header("Time Series with Trend Line & First Difference")
    st.caption("Top: global temperature overlaid with a linear trend line to quantify the warming rate. Bottom: first difference (year-over-year change) used to assess stationarity for time series modeling.")

    z = np.polyfit(global_temp_series.index.values, global_temp_series.values, 1)
    p = np.poly1d(z)
    diff_temp = global_temp_series.diff().dropna()

    fig = make_subplots(rows=2, cols=1, subplot_titles=[
        "Global Temperature with Trend Line",
        "First Difference of Temperature",
    ], vertical_spacing=0.12)

    fig.add_trace(go.Scatter(
        x=global_temp_series.index, y=global_temp_series.values,
        mode="lines", name="Avg Temperature", line=dict(color="darkred", width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=global_temp_series.index, y=p(global_temp_series.index),
        mode="lines", name="Trend Line", line=dict(color="blue", dash="dash", width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=diff_temp.index, y=diff_temp.values,
        mode="lines", name="First Difference", line=dict(color="darkgreen", width=1.5),
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)

    fig.update_layout(height=700)
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Change (°C)", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  9. ACF / PACF
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "ACF / PACF":
    st.header("Autocorrelation & Partial Autocorrelation")
    st.caption("ACF measures how correlated the temperature series is with its own past values; PACF isolates the direct effect of each lag — together they guide the choice of ARIMA/SARIMA model orders.")

    nlags = st.slider("Number of lags", min_value=10, max_value=60, value=40, step=5)
    series = global_temp_series.dropna()
    acf_vals = acf(series, nlags=nlags, alpha=0.05)
    pacf_vals = pacf(series, nlags=nlags, alpha=0.05)

    acf_y, acf_ci = acf_vals[0], acf_vals[1]
    pacf_y, pacf_ci = pacf_vals[0], pacf_vals[1]

    fig = make_subplots(rows=2, cols=1, subplot_titles=[
        "Autocorrelation Function (ACF)",
        "Partial Autocorrelation Function (PACF)",
    ], vertical_spacing=0.12)

    lags = np.arange(len(acf_y))
    for row_idx, (vals, ci, name) in enumerate([
        (acf_y, acf_ci, "ACF"),
        (pacf_y, pacf_ci, "PACF"),
    ], start=1):
        for lag_i in range(len(vals)):
            fig.add_trace(go.Scatter(
                x=[lag_i, lag_i], y=[0, vals[lag_i]],
                mode="lines", line=dict(color="#636EFA", width=1.5),
                showlegend=False,
            ), row=row_idx, col=1)
        fig.add_trace(go.Scatter(
            x=list(range(len(vals))), y=vals,
            mode="markers", marker=dict(color="#636EFA", size=5),
            name=name, showlegend=False,
        ), row=row_idx, col=1)
        upper = ci[:, 1] - vals
        fig.add_trace(go.Scatter(
            x=list(range(len(vals))), y=upper,
            mode="lines", line=dict(color="rgba(99,110,250,0.2)", width=0),
            showlegend=False,
        ), row=row_idx, col=1)
        fig.add_trace(go.Scatter(
            x=list(range(len(vals))), y=-upper,
            mode="lines", line=dict(color="rgba(99,110,250,0.2)", width=0),
            fill="tonexty", fillcolor="rgba(99,110,250,0.15)",
            showlegend=False,
        ), row=row_idx, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.5, row=row_idx, col=1)

    fig.update_layout(height=700)
    fig.update_xaxes(title_text="Lag", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  10. Train-Test Split Visualization
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Train-Test Split":
    st.header("Train-Test Split Visualization")
    st.caption("Two splitting strategies used to evaluate forecasting models — a year-based split (pre/post 2010) and a chronological 80/20 percentage split, both preserving temporal order.")

    split_year = 2010
    train_ratio = 0.8
    target = "Temperature"

    train_yearly = df[df["Year"] < split_year].groupby("Year")[target].mean()
    test_yearly = df[df["Year"] >= split_year].groupby("Year")[target].mean()

    n_total = len(df)
    train_size = int(n_total * train_ratio)
    train_pct = df.iloc[:train_size]
    test_pct = df.iloc[train_size:]
    train_yearly_pct = train_pct.groupby("Year")[target].mean()
    test_yearly_pct = test_pct.groupby("Year")[target].mean()

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_yearly.index, y=train_yearly.values, name="Train", line=dict(color="blue", width=2)))
        fig.add_trace(go.Scatter(x=test_yearly.index, y=test_yearly.values, name="Test", line=dict(color="red", width=2)))
        fig.add_vline(x=split_year, line_dash="dash", line_color="black", annotation_text=f"Split at {split_year}")
        fig.update_layout(title="Split by Year", height=400, yaxis_title="Temperature (°C)", xaxis_title="Year")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_yearly_pct.index, y=train_yearly_pct.values, name="Train (80%)", line=dict(color="blue", width=2)))
        fig.add_trace(go.Scatter(x=test_yearly_pct.index, y=test_yearly_pct.values, name="Test (20%)", line=dict(color="red", width=2)))
        fig.update_layout(title="Split by Percentage (80/20)", height=400, yaxis_title="Temperature (°C)", xaxis_title="Year")
        st.plotly_chart(fig, use_container_width=True)

    st.info(f"**Year split**: Train = before {split_year} ({len(train_yearly)} years), Test = {split_year}+ ({len(test_yearly)} years)  \n"
            f"**Percentage split**: Train = 80% ({train_size:,} rows), Test = 20% ({n_total - train_size:,} rows)")

# ═══════════════════════════════════════════════════════════════════════════════
#  11. Temperature Trend Analysis
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Temperature Trend Analysis":
    st.header("Temperature Trend Analysis")
    st.caption("Left: scatter plot with a fitted linear regression line quantifying the long-term warming rate. Right: decade-by-decade temperature change showing which decades warmed or cooled.")

    slope, intercept, r_val, p_val, std_err = stats.linregress(
        global_temp_by_year["Year"], global_temp_by_year["Temperature"]
    )

    global_temp_by_year["Decade"] = (global_temp_by_year["Year"] // 10) * 10
    decade_avg = global_temp_by_year.groupby("Decade")["Temperature"].mean().reset_index()
    decade_avg["Change"] = decade_avg["Temperature"].diff()

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=global_temp_by_year["Year"], y=global_temp_by_year["Temperature"],
            mode="markers", name="Avg Temperature",
            marker=dict(color="darkred", size=6, opacity=0.6),
        ))
        fig.add_trace(go.Scatter(
            x=global_temp_by_year["Year"],
            y=slope * global_temp_by_year["Year"] + intercept,
            mode="lines", name=f"Trend (slope={slope:.4f}°C/yr)",
            line=dict(color="blue", dash="dash", width=2),
        ))
        fig.update_layout(title="Global Temperature with Trend Line", height=450, yaxis_title="Temperature (°C)", xaxis_title="Year")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        decade_plot = decade_avg.dropna(subset=["Change"])
        colors = ["green" if v < 0 else "red" for v in decade_plot["Change"]]
        fig = go.Figure(go.Bar(
            x=decade_plot["Decade"], y=decade_plot["Change"],
            marker_color=colors, marker_line_color="black", marker_line_width=1,
        ))
        fig.update_layout(title="Temperature Change by Decade", height=450, yaxis_title="Change (°C)", xaxis_title="Decade")
        st.plotly_chart(fig, use_container_width=True)

    st.success(f"**Linear trend**: slope = {slope:.4f} °C/year | R² = {r_val**2:.4f} | p-value = {p_val:.5f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  12. CO2 Emissions vs Temperature
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "CO2 vs Temperature":
    st.header("CO2 Emissions vs Temperature Analysis")
    st.caption("Left: scatter plot with regression line measuring the direct CO2-temperature relationship. Right: dual-axis time series comparing the parallel rise of CO2 emissions and global temperature.")

    corr, p_val = pearsonr(df["CO2_Emissions_Gt"], df["Temperature"])
    z = np.polyfit(df["CO2_Emissions_Gt"], df["Temperature"], 1)
    p_fit = np.poly1d(z)

    yearly_emissions = df.groupby("Year").agg({"CO2_Emissions_Gt": "mean", "Temperature": "mean"}).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        sample = df.sample(min(800, len(df)), random_state=42)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sample["CO2_Emissions_Gt"], y=sample["Temperature"],
            mode="markers", name="Data",
            marker=dict(color="darkred", size=4, opacity=0.3),
        ))
        x_range = np.linspace(df["CO2_Emissions_Gt"].min(), df["CO2_Emissions_Gt"].max(), 100)
        fig.add_trace(go.Scatter(
            x=x_range, y=p_fit(x_range),
            mode="lines", name=f"Regression (r={corr:.3f})",
            line=dict(color="red", dash="dash", width=2),
        ))
        fig.update_layout(title="CO2 Emissions vs Temperature", height=450,
                          xaxis_title="CO2 Emissions (Gt)", yaxis_title="Temperature (°C)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=yearly_emissions["Year"], y=yearly_emissions["CO2_Emissions_Gt"],
            name="CO2 Emissions", line=dict(color="orange", width=2),
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=yearly_emissions["Year"], y=yearly_emissions["Temperature"],
            name="Avg Temperature", line=dict(color="darkred", width=2),
        ), secondary_y=True)
        fig.update_layout(title="CO2 Emissions & Temperature Over Time", height=450)
        fig.update_yaxes(title_text="CO2 Emissions (Gt)", secondary_y=False)
        fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
        fig.update_xaxes(title_text="Year")
        st.plotly_chart(fig, use_container_width=True)

    st.info(f"**Pearson correlation**: r = {corr:.4f} | p-value = {p_val:.5f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  13. Early vs Recent Period Comparison
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Period Comparison":
    st.header("Early vs Recent Period Temperature Comparison")
    st.caption("A Welch's t-test comparing temperatures from 1900-1960 (early) vs 1980-2023 (recent) — the shaded time series and overlapping histograms visualize the distributional shift.")

    early = df[(df["Year"] >= 1900) & (df["Year"] <= 1960)]["Temperature"]
    recent = df[(df["Year"] >= 1980) & (df["Year"] <= 2023)]["Temperature"]
    t_stat, t_pval = ttest_ind(early, recent, equal_var=False)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=global_temp_by_year["Year"], y=global_temp_by_year["Temperature"],
            mode="lines", name="Avg Temperature", line=dict(color="darkred", width=2),
        ))
        fig.add_vrect(x0=1900, x1=1960, fillcolor="blue", opacity=0.1,
                       annotation_text="Early (1900-1960)", annotation_position="top left")
        fig.add_vrect(x0=1980, x1=2023, fillcolor="red", opacity=0.1,
                       annotation_text="Recent (1980-2023)", annotation_position="top left")
        fig.update_layout(title="Temperature with Period Shading", height=450,
                          yaxis_title="Temperature (°C)", xaxis_title="Year")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=early, name="Early (1900-1960)",
            marker_color="blue", opacity=0.6, nbinsx=30, histnorm="probability density",
        ))
        fig.add_trace(go.Histogram(
            x=recent, name="Recent (1980-2023)",
            marker_color="red", opacity=0.6, nbinsx=30, histnorm="probability density",
        ))
        fig.update_layout(barmode="overlay", title="Temperature Distribution Comparison",
                          height=450, xaxis_title="Temperature (°C)", yaxis_title="Density")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Early Mean", f"{early.mean():.2f} °C")
    c2.metric("Recent Mean", f"{recent.mean():.2f} °C")
    c3.metric("t-statistic", f"{t_stat:.2f}")
    c4.metric("p-value", f"{t_pval:.5f}")
    if t_pval < 0.05:
        st.warning("The difference between early and recent periods is **statistically significant** (p < 0.05).")
    else:
        st.info("The difference is **not statistically significant** (p >= 0.05).")

st.sidebar.divider()
st.sidebar.caption(f"Data: {len(df):,} records | {int(df['Year'].min())}–{int(df['Year'].max())}")
