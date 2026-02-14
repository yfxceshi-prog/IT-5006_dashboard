import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
import io
import gzip

def normalize_raw_url(url: str) -> str:
    u = str(url).strip()
    if "github.com" in u and "/blob/" in u:
        u = u.replace("https://github.com/", "https://raw.githubusercontent.com/")
        u = u.replace("/blob/", "/")
    return u

st.set_page_config(page_title="Chicago Crime Interactive Dashboard", page_icon="📊", layout="wide")
st.markdown("""
    <style>
    .stMetric {background: rgba(255,255,255,0.6); border-radius: 12px; padding: 8px;}
    .stApp {background: linear-gradient(135deg,#f6f8fb 0%,#eef3ff 100%);}
    .title {font-size:28px; font-weight:700; margin-bottom:8px;}
    .subtitle {color:#5a5a5a; margin-bottom:18px;}
    section[data-testid="stSidebar"] {min-width: 360px !important; max-width: 360px !important;}
    section[data-testid="stSidebar"] div[data-baseweb="slider"] {padding-right: 12px;}
    div[data-testid="stMetricValue"] {white-space: nowrap;}
    </style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=True)
def load_csv(src, is_fileobj=False, nrows=None):
    required_kws = [
        "date","occur","incident","crime","primary","offense","category",
        "location","arrest","domestic","latitude","lat","longitude","lon","long",
        "district","ward","community","description"
    ]
    read_params = dict(low_memory=False, on_bad_lines="skip", nrows=nrows,
                       usecols=lambda c: any(k in str(c).lower() for k in required_kws))
    if is_fileobj:
        df = pd.read_csv(src, **read_params)
    else:
        df = pd.read_csv(src, **read_params)
    def pick(kws):
        for c in df.columns:
            s = c.lower().strip()
            for k in kws:
                if k in s:
                    return c
        return None
    date_col = pick(["date","occur","crime","incident"])
    if date_col is not None:
        df["__date__"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df["__date__"] = pd.to_datetime(df.iloc[:,0], errors="coerce")
    if "__date__" in df:
        df["Year"] = df["__date__"].dt.year
        df["Month"] = df["__date__"].dt.month
        df["Hour"] = df["__date__"].dt.hour
        df["DayOfWeek"] = df["__date__"].dt.day_name()
        df["MonthStart"] = df["__date__"].dt.to_period("M").dt.to_timestamp()
    pt_col = pick(["primary type","primary_type","offense","category"])
    if pt_col and "Primary Type" not in df.columns:
        df.rename(columns={pt_col:"Primary Type"}, inplace=True)
    ld_col = pick(["location description","loc_desc"])
    if ld_col and "Location Description" not in df.columns:
        df.rename(columns={ld_col:"Location Description"}, inplace=True)
    arrest_col = pick(["arrest"])
    if arrest_col and "Arrest" not in df.columns:
        df.rename(columns={arrest_col:"Arrest"}, inplace=True)
    domestic_col = pick(["domestic"])
    if domestic_col and "Domestic" not in df.columns:
        df.rename(columns={domestic_col:"Domestic"}, inplace=True)
    lat_col = pick(["latitude","lat"])
    lon_col = pick(["longitude","lon","long"])
    if lat_col and "Latitude" not in df.columns:
        df.rename(columns={lat_col:"Latitude"}, inplace=True)
    if lon_col and "Longitude" not in df.columns:
        df.rename(columns={lon_col:"Longitude"}, inplace=True)
    if "Latitude" in df.columns:
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    if "Longitude" in df.columns:
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    for c in ["Primary Type","Location Description","District","Ward","Community Area","Description"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    if "Arrest" in df.columns:
        df["Arrest"] = df["Arrest"].astype(str).str.lower().map({"true":True,"false":False,"1":True,"0":False}).fillna(np.nan)
    if "Domestic" in df.columns:
        df["Domestic"] = df["Domestic"].astype(str).str.lower().map({"true":True,"false":False,"1":True,"0":False}).fillna(np.nan)
    df = df.dropna(subset=["__date__"])
    return df

def apply_filters(df, s):
    m = pd.Series(True, index=df.index)
    if s["date_range"]:
        m &= (df["__date__"].dt.date >= s["date_range"][0]) & (df["__date__"].dt.date <= s["date_range"][1])
    if s.get("year_range") and "Year" in df.columns:
        m &= (df["Year"] >= s["year_range"][0]) & (df["Year"] <= s["year_range"][1])
    if s["hour_range"]:
        m &= (df["Hour"] >= s["hour_range"][0]) & (df["Hour"] <= s["hour_range"][1])
    if s["types"]:
        m &= df["Primary Type"].isin(s["types"]) if "Primary Type" in df.columns else m
    if s["locs"]:
        m &= df["Location Description"].isin(s["locs"]) if "Location Description" in df.columns else m
    if s["districts"]:
        m &= df["District"].isin(s["districts"]) if "District" in df.columns else m
    if s["areas"]:
        m &= df["Community Area"].isin(s["areas"]) if "Community Area" in df.columns else m
    if s["arrest"] is not None and "Arrest" in df.columns:
        m &= (df["Arrest"] == s["arrest"])
    if s["domestic"] is not None and "Domestic" in df.columns:
        m &= (df["Domestic"] == s["domestic"])
    return df[m]

st.markdown('<div class="title">Chicago Crime Interactive Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Temporal & Spatial Patterns · 2015–2024</div>', unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Data Source")
    src_mode = st.radio("Choose Source", ["Upload CSV","CSV URL"], horizontal=True)
    file_obj = None
    url_text = None
    if src_mode == "Upload CSV":
        file_obj = st.file_uploader("Upload CSV file", type=["csv"])
    else:
        url_text = st.text_input("CSV URL (GitHub Raw or other)")
    load_mode = st.selectbox("Load Mode", ["Preview (20k rows)","Sample (100k rows)","Full dataset"], index=0)
    st.subheader("Filters")
    st.write("Set filters, then click the button below to apply")
    apply_btn = st.button("Apply Filters", type="primary")

nrows = 20000 if "Preview" in load_mode else (100000 if "Sample" in load_mode else None)
if file_obj is not None:
    df = load_csv(file_obj, is_fileobj=True, nrows=nrows)
elif url_text:
    raw_url = normalize_raw_url(url_text)
    try:
        df = load_csv(raw_url, is_fileobj=False, nrows=nrows)
    except Exception:
        st.error("Failed to load CSV from URL. Please paste a direct Raw CSV link (e.g., raw.githubusercontent.com or add ?raw=1).")
        st.stop()
else:
    st.info("Provide a CSV via upload or URL to start.")
    st.stop()

if df.empty or df["__date__"].isna().all():
    st.error("CSV appears invalid or has no parsable date column. Ensure you upload the cleaned CSV and use a Raw URL.")
    st.stop()

min_date_series = df["__date__"].dropna()
min_date = min_date_series.min().date() if not min_date_series.empty else pd.to_datetime("2015-01-01").date()
max_date = min_date_series.max().date() if not min_date_series.empty else pd.to_datetime("2024-12-31").date()
if "Year" in df.columns:
    year_series = pd.to_numeric(df["Year"], errors="coerce").dropna()
    if year_series.empty:
        min_year, max_year = min_date.year, max_date.year
    else:
        min_year, max_year = int(year_series.min()), int(year_series.max())
else:
    min_year, max_year = min_date.year, max_date.year
types = sorted(list(df["Primary Type"].cat.categories)) if "Primary Type" in df.columns else []
locs = sorted(list(df["Location Description"].cat.categories)) if "Location Description" in df.columns else []
districts = sorted(list(df["District"].cat.categories)) if "District" in df.columns else []
areas = sorted(list(df["Community Area"].cat.categories)) if "Community Area" in df.columns else []

with st.sidebar:
    date_range = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    year_range = st.slider("Year Range", min_year, max_year, (min_year, max_year))
    hour_range = st.slider("Hour Range", 0, 23, (0,23))
    if "types_sel" not in st.session_state:
        st.session_state["types_sel"] = types[:5] if types else []
    b_all, b_clear = st.columns(2)
    if b_all.button("Select All", key="btn_types_all") and types:
        st.session_state["types_sel"] = types
    if b_clear.button("Clear", key="btn_types_clear"):
        st.session_state["types_sel"] = []
    types_sel = st.multiselect("Crime Type(s)", types, key="types_sel") if types else []
    locs_sel = st.multiselect("Location Description", locs) if locs else []
    districts_sel = st.multiselect("Police District", districts) if districts else []
    areas_sel = st.multiselect("Community Area", areas) if areas else []
    arrest_sel = st.selectbox("Arrest", ["All","Yes","No"]) if "Arrest" in df.columns else "All"
    domestic_sel = st.selectbox("Domestic", ["All","Yes","No"]) if "Domestic" in df.columns else "All"
    arrest_val = None if arrest_sel=="All" else (True if arrest_sel=="Yes" else False)
    domestic_val = None if domestic_sel=="All" else (True if domestic_sel=="Yes" else False)

sel = {"date_range":date_range,"year_range":year_range,"hour_range":hour_range,"types":types_sel,"locs":locs_sel,"districts":districts_sel,"areas":areas_sel,"arrest":arrest_val,"domestic":domestic_val}
df_sel = apply_filters(df, sel) if apply_btn else df

total_cnt = int(len(df_sel))
arrest_rate = float(df_sel["Arrest"].mean()*100) if "Arrest" in df_sel.columns else np.nan
domestic_rate = float(df_sel["Domestic"].mean()*100) if "Domestic" in df_sel.columns else np.nan

c1, c2, c3, c4 = st.columns([1,1.6,1,1])
c1.metric("Records", f"{total_cnt:,}")
c2.metric("Date Range", f"{sel['date_range'][0].strftime('%Y/%m/%d')} → {sel['date_range'][1].strftime('%Y/%m/%d')}")
c3.metric("Arrest Rate", f"{arrest_rate:.1f}%" if not np.isnan(arrest_rate) else "N/A")
c4.metric("Domestic Share", f"{domestic_rate:.1f}%" if not np.isnan(domestic_rate) else "N/A")

tab1, tab2, tab3 = st.tabs(["Time Trends","Spatial Map","Type Distribution"])

with tab1:
    left, right = st.columns([2,1], gap="large")
    ts = df_sel.groupby("MonthStart").size().reset_index(name="Count")
    fig_ts = px.line(ts, x="MonthStart", y="Count", height=420, markers=True, template="plotly_white")
    fig_ts.update_traces(line=dict(color="#2F64FF", width=2))
    left.plotly_chart(fig_ts, width="stretch")
    if "Primary Type" in df_sel.columns:
        sel_types_line = st.session_state.get("types_sel", [])
        if len(sel_types_line) == 0:
            sel_types_line = list(df_sel["Primary Type"].value_counts().nlargest(6).index)
        ts_types = df_sel.groupby(["MonthStart","Primary Type"], observed=True).size().reset_index(name="Count")
        ts_types = ts_types[ts_types["Primary Type"].isin(sel_types_line)]
        fig_ts_type = px.line(ts_types, x="MonthStart", y="Count", color="Primary Type", height=420, template="plotly_white")
        fig_ts_type.update_layout(legend_title_text="Crime Type")
        left.plotly_chart(fig_ts_type, width="stretch")
    if "DayOfWeek" in df_sel.columns and "Hour" in df_sel.columns:
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        heat = df_sel.groupby(["DayOfWeek","Hour"], observed=True).size().reset_index(name="Count")
        pivot = heat.pivot(index="DayOfWeek", columns="Hour", values="Count").reindex(order)
        fig_heat = px.imshow(pivot, aspect="auto", height=420, color_continuous_scale="Viridis")
        fig_heat.update_layout(template="plotly_white", xaxis_title="Hour", yaxis_title="DayOfWeek")
        right.plotly_chart(fig_heat, width="stretch")

with tab2:
    map_mode = st.radio("Map Mode", ["Density Hexagon","Scatter"], horizontal=True)
    if {"Latitude","Longitude"}.issubset(df_sel.columns):
        df_map = df_sel.dropna(subset=["Latitude","Longitude"])
        if len(df_map) == 0:
            st.info("No valid coordinates in the filtered result; cannot render the map")
        else:
            view_state = pdk.ViewState(latitude=float(df_map["Latitude"].mean()), longitude=float(df_map["Longitude"].mean()), zoom=9, pitch=0)
            if map_mode == "Density Hexagon":
                sample_n = min(150000, len(df_map))
                dhex = df_map.sample(sample_n) if len(df_map) > sample_n else df_map
                layer = pdk.Layer("HexagonLayer", data=dhex, get_position='[Longitude, Latitude]', radius=80, elevation_scale=20, elevation_range=[0,3000], pickable=True, extruded=True)
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text":"Density"}))
            else:
                n = min(50000, len(df_map))
                dsmall = df_map.sample(n) if len(df_map) > n else df_map
                layer = pdk.Layer("ScatterplotLayer", data=dsmall, get_position='[Longitude, Latitude]', get_color='[47,100,255,180]', get_radius=30, pickable=True)
                tooltip = {"text":"{Primary Type}\n{Location Description}\n{__date__}"} if "Primary Type" in df_map.columns else {"text":"{__date__}"}
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))
    else:
        st.info("Missing latitude/longitude columns; cannot render the map")

with tab3:
    cols = st.columns(2, gap="large")
    if "Primary Type" in df_sel.columns:
        top = df_sel["Primary Type"].dropna().value_counts().reset_index()
        top.columns = ["Primary Type","Count"]
        if len(top) > 0:
            fig_bar = px.bar(top.head(15), x="Count", y="Primary Type", orientation="h", height=500, template="plotly_white", color="Count", color_continuous_scale="Blues")
            fig_bar.update_layout(yaxis_categoryorder="total ascending")
            cols[0].plotly_chart(fig_bar, width="stretch")
        else:
            cols[0].info("No type data in filtered result")
    if "Location Description" in df_sel.columns:
        loc_top = df_sel["Location Description"].dropna().value_counts().reset_index()
        loc_top.columns = ["Location Description","Count"]
        if len(loc_top) > 0:
            fig_loc = px.bar(loc_top.head(15), x="Count", y="Location Description", orientation="h", height=500, template="plotly_white", color="Count", color_continuous_scale="Purples")
            fig_loc.update_layout(yaxis_categoryorder="total ascending")
            cols[1].plotly_chart(fig_loc, width="stretch")
        else:
            cols[1].info("No location data in filtered result")

max_rows_download = 500000
if len(df_sel) > max_rows_download:
    st.info(f"Filtered result exceeds {max_rows_download:,} rows. Please narrow the filters or select a smaller year range before downloading.")
else:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb') as gz:
        gz.write(df_sel.to_csv(index=False).encode('utf-8'))
    st.download_button("Download filtered CSV (gzip)", data=buf.getvalue(), file_name="filtered_chicago_crimes.csv.gz", mime="application/gzip")