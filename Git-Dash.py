# ==============================================
# Streamlit Dashboard - Full Version Final (Sidebar Total Documents)
# ==============================================
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

st.set_page_config(layout="wide", page_title="Topic Modeling Dashboard")

# ------------------------------
# Dashboard Title
# ------------------------------
st.title("Topic Modeling Dashboard - Indonesia Environmental Research")

# ------------------------------
# Load data (relative path for GitHub deployment)
# ------------------------------
file_path = "Data.xlsx"  # pastikan Data.xlsx ada di folder yang sama dengan file dashboard.py

try:
    df = pd.read_excel(file_path)
except Exception as e:
    st.error(f"❌ Error loading file: {e}")
    st.stop()

# Pastikan kolom tahun numerik
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# ------------------------------
# Sidebar Filters
# ------------------------------
st.sidebar.header("Filters")
analysis_mode = st.sidebar.radio("Analyze by:", ("Cluster", "Topic"))
include_outlier = st.sidebar.selectbox("Include Outliers?", ["Yes", "No"], index=1)  # default No

# ------------------------------
# Apply Cluster/Topic filter for main analysis
# ------------------------------
filtered = df.copy()
if include_outlier == "No" and "Assigned_Topic" in filtered.columns:
    filtered = filtered[filtered["Assigned_Topic"] != -1]

if analysis_mode == "Cluster":
    col_name = "Cluster Labels"
    available = sorted(filtered[col_name].dropna().unique()) if col_name in filtered.columns else []
    choices = ["All"] + available
    selected = st.sidebar.multiselect("Select Cluster Label:", choices, default=["All"])
else:
    col_name = "Topic Labels"
    available = sorted(filtered[col_name].dropna().unique()) if col_name in filtered.columns else []
    choices = ["All"] + available
    selected = st.sidebar.multiselect("Select Topic Label:", choices, default=["All"])

# Terapkan filter
if "All" not in selected and selected:
    filtered = filtered[filtered[col_name].isin(selected)]

selected_text = f"{'Selected Cluster:' if analysis_mode == 'Cluster' else 'Selected Topic:'} "
selected_text += ", ".join(
    selected) if "All" not in selected else f"Showing all {'clusters' if analysis_mode == 'Cluster' else 'topics'}"
st.markdown(f"**{selected_text}**")

# ------------------------------
# Sidebar Score Card (Total Documents only)
# ------------------------------
total_docs = len(filtered)
st.sidebar.markdown(f"**Total Documents: {total_docs}**")

# ------------------------------
# Sidebar Hyperparameter Information
# ------------------------------
with st.sidebar.expander("Topic Modeling Hyperparameter", expanded=False):
    st.markdown("""
    **Vectorizer Model:**  
    `TfidfVectorizer (max_df=0.95, min_df=1, ngram_range=(1, 2), max_features=10000)`

    **Embedding Model:**  
    `SentenceTransformer (all-mpnet-base-v2)`

    **HDBSCAN Model:**  
    `(min_cluster_size=20, metric='euclidean', cluster_selection_epsilon=0.0)`

    **Topic Model:**  
    `BERTopic (nr_topics=66)`

    **BERTopic Temporal Function:**  
    `topic_model.topics_over_time`

    **Coherence score:**  
    `0.6395`

    **Diversity score:**  
    `0.9197`

    **Outlier Ratio:**  
    `28.11%`

    **Data cut-off date:**  
    `22-11-2025`

    """)

# ==============================================
# 📊 Hierarchical Tree Table + Cluster Bar Chart tanpa kolom "Group"
# ==============================================
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode
import plotly.express as px

with st.expander("Hierarchical Cluster Table & Cluster Document Count", expanded=False):
    # Filter out outliers
    filtered_tree = df.copy()
    if "Assigned_Topic" in filtered_tree.columns:
        filtered_tree = filtered_tree[filtered_tree["Assigned_Topic"] != -1]

    # Pastikan kolom yang dibutuhkan ada
    if {"Cluster Labels", "Topic Labels", "Title"}.issubset(filtered_tree.columns):

        # Hitung jumlah dokumen per topik dan cluster
        summary = (
            filtered_tree.groupby(["Cluster Labels", "Topic Labels"])
            .agg({"Title": "count"})
            .reset_index()
            .rename(columns={"Title": "Document Count"})
        )

        # Urutkan untuk tampilan tree
        summary = summary.sort_values(by=["Cluster Labels", "Document Count"], ascending=[True, False])

        # Tambahkan kolom Cluster untuk ditampilkan sebagai kolom biasa
        summary["Cluster Display"] = summary["Cluster Labels"]

        # Reorder columns: Cluster → Topic → Document Count
        summary = summary[["Cluster Display", "Topic Labels", "Document Count", "Cluster Labels"]]

        # Setup AG Grid
        gb = GridOptionsBuilder.from_dataframe(summary)
        gb.configure_default_column(groupable=True, enableValue=True, enableRowGroup=True)

        # Gunakan Cluster Labels untuk rowGroup (tetap disembunyikan)
        gb.configure_column("Cluster Labels", rowGroup=True, hide=True)

        # Tampilkan kolom lain
        gb.configure_column("Cluster Display", headerName="Cluster")
        gb.configure_column("Topic Labels", headerName="Topic")
        gb.configure_column("Document Count", type=["numericColumn"], headerName="Documents")

        # Konfigurasi tree dan hilangkan kolom "Group"
        gb.configure_grid_options(
            treeData=True,
            animateRows=True,
            groupDefaultExpanded=-1,
            suppressRowClickSelection=True,
            suppressDragLeaveHidesColumns=True,
            domLayout='normal',
            suppressMovableColumns=True,
            groupDisplayType='custom'  # Hilangkan kolom 'Group'
        )

        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_side_bar()
        gb.configure_selection("single")

        grid_options = gb.build()

        AgGrid(
            summary,
            gridOptions=grid_options,
            enable_enterprise_modules=False,
            update_mode=GridUpdateMode.NO_UPDATE,
            height=400,
            fit_columns_on_grid_load=True,
        )

        # ------------------------------
        # Bar chart horizontal: jumlah dokumen per cluster
        # ------------------------------
        cluster_counts = (
            filtered_tree.groupby("Cluster Labels")
            .agg({"Title": "count"})
            .reset_index()
            .rename(columns={"Title": "Document Count"})
            .sort_values(by="Document Count", ascending=True)
        )

        fig_cluster_bar = px.bar(
            cluster_counts,
            y="Cluster Labels",
            x="Document Count",
            orientation='h',
            text="Document Count",
            title="Documents per Cluster",
            labels={"Cluster Labels": "Cluster", "Document Count": "Documents"}
        )
        fig_cluster_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_cluster_bar, use_container_width=True)

    else:
        st.info("Kolom 'Cluster Labels', 'Topic Labels', atau 'Title' tidak ditemukan dalam data.")

# ------------------------------
# Line chart: frequency per year
# ------------------------------
topic_col = "Assigned_Topic" if "Assigned_Topic" in filtered.columns else (
    "Topic Labels" if "Topic Labels" in filtered.columns else None)
if topic_col is None:
    st.error("Cannot find 'Assigned_Topic' or 'Topic Labels'.")
    st.stop()

filtered["Year"] = filtered["Year"].astype(int)

yearly_count = (
    filtered.dropna(subset=["Year", topic_col])
    .groupby(["Year", topic_col])
    .size().reset_index(name="Frequency")
)

hover_meta = filtered.groupby([topic_col, "Year"]).agg({
    "Keywords Over Time": lambda x: "; ".join(x.dropna().astype(str).unique()),
    "Topic Labels": lambda x: "; ".join(x.dropna().astype(str).unique()) if "Topic Labels" in filtered.columns else "",
    "Cluster Labels": lambda x: "; ".join(
        x.dropna().astype(str).unique()) if "Cluster Labels" in filtered.columns else ""
}).reset_index()

yearly_count = yearly_count.merge(hover_meta, on=[topic_col, "Year"], how="left")

st.subheader("Topic Trend and Author Keywords Wordcloud")
fig_line = px.line(
    yearly_count,
    x="Year",
    y="Frequency",
    color="Topic Labels" if "Topic Labels" in yearly_count.columns else topic_col,
    markers=True,
    hover_data={"Year": True,
                "Frequency": True,
                "Topic Labels": True,
                "Cluster Labels": False,
                "Keywords Over Time": False},
    title="Trend and Wordcloud"
)

fig_line.update_traces(
    hovertemplate="<br>".join([
        "Year: %{x}",
        "Frequency: %{y}",
        "Keyword Over Time: %{customdata[2]}"
    ]),
    hoverlabel=dict(
        font_size=24,
        font_family="Arial"
    )
)

fig_line.update_layout(
    xaxis=dict(dtick=1)
)

col_main, col_wc = st.columns([3, 1])
col_main.plotly_chart(fig_line, use_container_width=True)

# ------------------------------
# WordCloud (Author Keywords) - original algorithm
# ------------------------------
if "Author Keywords" in filtered.columns:
    kws_series = filtered["Author Keywords"].dropna().astype(str)
    all_kws = []
    for s in kws_series:
        all_kws.extend([p.strip() for p in s.split(";") if p.strip()])
    if all_kws:
        wc_text = " ".join(all_kws)
        wc = WordCloud(width=500, height=400, background_color="white").generate(wc_text)
        fig_wc, ax_wc = plt.subplots(figsize=(5, 4))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        col_wc.pyplot(fig_wc)
    else:
        col_wc.info("No Author Keywords found")
else:
    col_wc.info("Author Keywords column not found")

# ------------------------------
# Bar charts (Top 10, horizontal, largest on top)
# ------------------------------
st.subheader("Top 10: Authors, Collaborators, Journals/Conferences")
col_a, col_b, col_c = st.columns(3)

# Authors
if "Authors" in filtered.columns:
    authors_exploded = filtered["Authors"].dropna().astype(str).str.split(";").explode().str.strip()
    authors_exploded = authors_exploded[authors_exploded != ""]
    authors_counts = authors_exploded.value_counts().head(10).reset_index()
    authors_counts.columns = ["Author", "Count"]
    authors_counts = authors_counts.sort_values(by="Count", ascending=True)
    fig_a = px.bar(
        authors_counts,
        y="Author",
        x="Count",
        text="Count",
        orientation='h',
        title="Top 10 Authors"
    )
    fig_a.update_traces(textposition="outside")
    col_a.plotly_chart(fig_a, use_container_width=True)
else:
    col_a.info("No Authors data")

# Affiliations
if "Affiliations" in filtered.columns:
    affs_exploded = filtered["Affiliations"].dropna().astype(str).str.split(";").explode().str.strip()
    affs_exploded = affs_exploded[~affs_exploded.str.contains("Badan Riset dan Inovasi Nasional", na=False)]
    affs_exploded = affs_exploded[affs_exploded != ""]
    affs_counts = affs_exploded.value_counts().head(10).reset_index()
    affs_counts.columns = ["Affiliation", "Count"]
    affs_counts = affs_counts.sort_values(by="Count", ascending=True)
    fig_b = px.bar(
        affs_counts,
        y="Affiliation",
        x="Count",
        text="Count",
        orientation='h',
        title="Top 10 Collaborators"
    )
    fig_b.update_traces(textposition="outside")
    col_b.plotly_chart(fig_b, use_container_width=True)
else:
    col_b.info("No Affiliations data")

# Journals
if "Journal Name" in filtered.columns:
    journals = filtered["Journal Name"].dropna().astype(str).str.strip()
    journals = journals[journals != ""]
    journal_counts = journals.value_counts().head(10).reset_index()
    journal_counts.columns = ["Journal", "Count"]
    journal_counts = journal_counts.sort_values(by="Count", ascending=True)
    fig_c = px.bar(
        journal_counts,
        y="Journal",
        x="Count",
        text="Count",
        orientation='h',
        title="Top 10 Journals/Conferences"
    )
    fig_c.update_traces(textposition="outside")
    col_c.plotly_chart(fig_c, use_container_width=True)
else:
    col_c.info("No Journals data")

# ------------------------------
# 10 Representative Documents
# ------------------------------
with st.expander("10 Representative Documents"):
    if "Title" in filtered.columns:
        st.dataframe(filtered[["Title", "Year", "Authors"]].head(10))
    else:
        st.info("Title column not found")

# ------------------------------
# Topic / Cluster Interpretation in Sidebar
# ------------------------------
interp_col = "Topic Interpretation" if analysis_mode == "Topic" else "Cluster Interpretation"
label_col = "Topic Labels" if analysis_mode == "Topic" else "Cluster Labels"

if interp_col in filtered.columns and label_col in filtered.columns:
    interpretations = filtered[[label_col, interp_col]].dropna(subset=[interp_col, label_col]).drop_duplicates()

    # 🔹 Tambahkan Coherence Score hanya untuk Topic mode
    if analysis_mode == "Topic" and "Coherence Score" in filtered.columns:
        coherence_info = (
            filtered.groupby(label_col)["Coherence Score"]
            .first()  # ambil satu nilai per topik
            .reset_index()
        )
        interpretations = interpretations.merge(coherence_info, on=label_col, how="left")

    if not interpretations.empty:
        with st.sidebar.expander("Topic / Cluster Interpretation (Coherence Score)", expanded=False):
            for _, row in interpretations.iterrows():
                if analysis_mode == "Topic" and "Coherence Score" in row:
                    st.markdown(f"**{row[label_col]} - {row[interp_col]} ({row['Coherence Score']:.4f})**")
                else:
                    st.markdown(f"**{row[label_col]} - {row[interp_col]}**")
    else:
        st.sidebar.info("No interpretations available for the selected filter")
else:

    st.sidebar.info(f"Columns {label_col} or {interp_col} not found in data")


