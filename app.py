"""
Document Intelligence: Unstructured ParseQuery - Streamlit Frontend

Document processing pipeline using Databricks AI Functions (ai_parse_document + ai_query)
with MLflow Tracing for full observability.
"""

import streamlit as st
import os
import mlflow
import pandas as pd
from datetime import datetime
from backend import (
    process_file_through_pipeline,
    get_processing_status,
    reset_storage,
    reprocess_file,
    create_initial_file_record,
    get_file_results
)
import json
from config import LOGS_VOLUME_PATH

# Page configuration
st.set_page_config(
    page_title="Document Intelligence: Unstructured ParseQuery",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for improved styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .main-header h1 {
        color: white !important;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0;
    }

    /* Card styling */
    .stCard {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }

    /* Stage cards */
    .stage-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #ddd;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .stage-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    .stage-success {
        background: linear-gradient(135deg, #d4edda 0%, #a8e6cf 100%);
        border-color: #28a745;
    }

    .stage-error {
        background: linear-gradient(135deg, #f8d7da 0%, #ffb3b3 100%);
        border-color: #dc3545;
    }

    .stage-pending {
        background: linear-gradient(135deg, #fff3cd 0%, #ffe066 100%);
        border-color: #ffc107;
    }

    /* Metric cards - aggressive overrides */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 8px !important;
        padding: 0.4rem !important;
        color: white !important;
    }

    div[data-testid="metric-container"] label {
        color: rgba(255,255,255,0.8) !important;
        font-size: 0.6rem !important;
        font-weight: 500 !important;
    }

    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: white !important;
        font-size: 0.75rem !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        white-space: normal !important;
        line-height: 1.1 !important;
    }

    /* Target the inner div that actually contains the text */
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] > div {
        font-size: 0.75rem !important;
        white-space: normal !important;
        word-break: break-word !important;
        line-height: 1.1 !important;
    }

    /* Also target any p or span elements inside */
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] p,
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] span {
        font-size: 0.75rem !important;
        line-height: 1.1 !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }

    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #495057;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }

    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
        background: #f8f9ff;
    }

    .stFileUploader:hover {
        border-color: #764ba2;
        background: #f0f2ff;
    }

    /* Button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.4);
    }

    /* Disabled button styling - faded appearance */
    .stButton > button[kind="primary"]:disabled {
        background: linear-gradient(135deg, #b8c4e8 0%, #c9b3d6 100%);
        opacity: 0.6;
        cursor: not-allowed;
        transform: none;
        box-shadow: none;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 8px;
        font-weight: 500;
    }

    /* Table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }

    .dataframe th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600;
        padding: 12px !important;
    }

    .dataframe td {
        padding: 10px !important;
        border-bottom: 1px solid #e0e0e0;
    }

    .dataframe tr:hover td {
        background: #f8f9ff;
    }

    /* Success/Error/Info boxes */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
        border-radius: 0 8px 8px 0;
    }

    .stError {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 4px solid #dc3545;
        border-radius: 0 8px 8px 0;
    }

    .stInfo {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-left: 4px solid #17a2b8;
        border-radius: 0 8px 8px 0;
    }

    /* Divider styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }

    /* Caption styling */
    .stCaption {
        color: #6c757d;
        font-size: 0.85rem;
    }

    /* Links in tables */
    .dataframe a {
        color: #667eea;
        text-decoration: none;
        font-weight: 500;
    }

    .dataframe a:hover {
        color: #764ba2;
        text-decoration: underline;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #f8f9fa;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    /* Processing indicator */
    .processing-indicator {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 1rem 0;
        animation: pulse 2s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.02); }
    }

    .processing-indicator h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.5rem;
    }

    .processing-indicator p {
        margin: 0;
        opacity: 0.9;
    }

    /* Spinner animation */
    .spinner {
        display: inline-block;
        width: 30px;
        height: 30px;
        border: 3px solid rgba(255,255,255,0.3);
        border-radius: 50%;
        border-top-color: white;
        animation: spin 1s ease-in-out infinite;
        margin-right: 10px;
        vertical-align: middle;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Processing status pulse animation */
    @keyframes processing-pulse {
        0%, 100% {
            transform: scale(1);
            box-shadow: 0 2px 8px rgba(255, 152, 0, 0.4);
        }
        50% {
            transform: scale(1.05);
            box-shadow: 0 4px 16px rgba(255, 152, 0, 0.6);
        }
    }

    /* Processing status badge in table */
    .processing-status-badge {
        display: inline-block;
        background: linear-gradient(135deg, #ff9800 0%, #ff5722 100%);
        color: white !important;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
        animation: processing-pulse 1.5s ease-in-out infinite;
        box-shadow: 0 2px 8px rgba(255, 152, 0, 0.4);
    }

</style>
""", unsafe_allow_html=True)

# MLflow is initialized in backend.py at module level
if "mlflow_initialized" not in st.session_state:
    st.session_state.mlflow_initialized = True
    st.session_state.tracking_uri = "databricks"
    st.session_state.experiment_name = "/Shared/unstructured_parsequery_pipeline"

# Initialize session state
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "current_processing" not in st.session_state:
    st.session_state.current_processing = None

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "just_processed_files" not in st.session_state:
    st.session_state.just_processed_files = []

if "reprocessing_file_id" not in st.session_state:
    st.session_state.reprocessing_file_id = None

if "viewing_results_file_id" not in st.session_state:
    st.session_state.viewing_results_file_id = None

if "last_processing_result" not in st.session_state:
    st.session_state.last_processing_result = None

if "show_trace_info" not in st.session_state:
    st.session_state.show_trace_info = False

# Check for query parameters to trigger results dialog or reprocess
query_params = st.query_params
if "view_results" in query_params:
    file_id = query_params.get("view_results")
    filename = query_params.get("filename", "")
    if file_id:
        st.session_state.viewing_results_file_id = file_id
        st.session_state.viewing_results_filename = filename
        # Clear query params
        st.query_params.clear()
elif "reprocess" in query_params:
    file_id = query_params.get("reprocess")
    filename = query_params.get("filename", "")
    volume_path = query_params.get("volume_path", "")
    if file_id:
        st.session_state.reprocessing_file_id = file_id
        st.session_state.reprocessing_filename = filename
        st.session_state.reprocessing_volume_path = volume_path
        # Clear query params
        st.query_params.clear()

# Cache the status query to reduce SQL calls
@st.cache_data(ttl=30)
def fetch_processing_status():
    try:
        return get_processing_status(limit=12)
    except Exception as e:
        return {"files": [], "error": str(e)}

# Fetch processing status once and reuse
status_data = fetch_processing_status()

# Header with gradient styling - compact version
st.markdown("""
<div style="
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem 2rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    color: white;
    text-align: center;
">
    <h2 style="color: white !important; margin: 0;">Document Intelligence: Unstructured ParseQuery</h2>
    <p style="color: rgba(255,255,255,0.8); margin: 0.25rem 0 0 0; font-size: 0.8rem;">Document Processing Pipeline with MLflow Tracing</p>
    <div style="
        background: rgba(255,255,255,0.2);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        margin-top: 0.75rem;
        display: inline-block;
        backdrop-filter: blur(10px);
    ">
        <span style="font-size: 0.85rem; font-weight: 600; color: white;">Databricks AI Functions</span><span style="font-size: 0.85rem; color: rgba(255,255,255,0.7);"> · </span><span style="font-size: 0.8rem; color: rgba(255,255,255,0.9);">ai_parse_document</span><span style="font-size: 0.8rem; color: rgba(255,255,255,0.7);"> & </span><span style="font-size: 0.8rem; color: rgba(255,255,255,0.9);">ai_query</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Display About and info in sidebar
with st.sidebar:
    # About section at top of sidebar - Professional pipeline visualization
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h4 style="color: #495057; margin: 0 0 0.75rem 0; font-size: 1.1rem;">Pipeline Stages</h4>
        <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); border-left: 3px solid #28a745; border-radius: 0 6px 6px 0; padding: 0.6rem 0.75rem; margin-bottom: 0.5rem;">
            <div style="font-weight: 600; font-size: 0.9rem; color: #155724;">1. Ingest</div>
            <div style="font-size: 0.8rem; color: #495057;">Upload to UC Volume</div>
            <div style="font-size: 0.75rem; color: #28a745; font-weight: 500;">UC Volumes API</div>
        </div>
        <div style="background: linear-gradient(135deg, #cce5ff 0%, #b8daff 100%); border-left: 3px solid #667eea; border-radius: 0 6px 6px 0; padding: 0.6rem 0.75rem; margin-bottom: 0.5rem;">
            <div style="font-weight: 600; font-size: 0.9rem; color: #004085;">2. Parse</div>
            <div style="font-size: 0.8rem; color: #495057;">Text Extraction</div>
            <div style="font-size: 0.75rem; color: #667eea; font-weight: 500;">ai_parse_document</div>
        </div>
        <div style="background: linear-gradient(135deg, #e2d5f1 0%, #d4c5e8 100%); border-left: 3px solid #764ba2; border-radius: 0 6px 6px 0; padding: 0.6rem 0.75rem; margin-bottom: 0.5rem;">
            <div style="font-weight: 600; font-size: 0.9rem; color: #4a235a;">3. Categorize</div>
            <div style="font-size: 0.8rem; color: #495057;">Document Classification</div>
            <div style="font-size: 0.75rem; color: #764ba2; font-weight: 500;">ai_query</div>
        </div>
        <div style="background: linear-gradient(135deg, #e2d5f1 0%, #d4c5e8 100%); border-left: 3px solid #764ba2; border-radius: 0 6px 6px 0; padding: 0.6rem 0.75rem; margin-bottom: 0.5rem;">
            <div style="font-weight: 600; font-size: 0.9rem; color: #4a235a;">4. Extract</div>
            <div style="font-size: 0.8rem; color: #495057;">Entity Extraction</div>
            <div style="font-size: 0.75rem; color: #764ba2; font-weight: 500;">ai_query</div>
        </div>
        <div style="background: linear-gradient(135deg, #e2d5f1 0%, #d4c5e8 100%); border-left: 3px solid #764ba2; border-radius: 0 6px 6px 0; padding: 0.6rem 0.75rem; margin-bottom: 0.5rem;">
            <div style="font-weight: 600; font-size: 0.9rem; color: #4a235a;">5. De-identify</div>
            <div style="font-size: 0.8rem; color: #495057;">PII Masking</div>
            <div style="font-size: 0.75rem; color: #764ba2; font-weight: 500;">ai_query</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Observability section with info button
    obs_col1, obs_col2 = st.columns([3, 1])
    with obs_col1:
        st.markdown("**Observability**")
    with obs_col2:
        if st.button("ℹ️", key="trace_info_btn", help="View trace structure"):
            st.session_state.show_trace_info = True

    st.caption(f"Experiment: `{st.session_state.get('experiment_name', 'unstructured_parsequery_pipeline')}`")


# Main content area - File Upload (left) | Quick Stats (right)
col1, col2 = st.columns([2, 1])

with col1:
    # File upload section with styled header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #dee2e6;
        margin-bottom: 0.5rem;
    ">
        <div style="font-weight: 600; font-size: 1rem; color: #495057; margin-bottom: 0.5rem;">📤 File Upload</div>
        <div style="font-size: 0.75rem; color: #6c757d;">Select documents to process through the pipeline</div>
    </div>
    """, unsafe_allow_html=True)

    # Show completion message if files were just processed
    if st.session_state.just_processed_files:
        for fname in st.session_state.just_processed_files:
            st.success(f"✅ File processed successfully: **{fname}**")
        # Clear the message after showing
        st.session_state.just_processed_files = []

    # File uploader with dynamic key to allow clearing
    uploaded_files = st.file_uploader(
        label="Drop files here or click to browse",
        accept_multiple_files=True,
        help="Upload documents to process through the 5-stage pipeline with MLflow tracing",
        key=f"file_uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.success(f"✓ {len(uploaded_files)} file(s) selected")

        # Display file information
        with st.expander("View selected files", expanded=True):
            for file in uploaded_files:
                file_size_mb = file.size / (1024 * 1024)
                st.text(f"📄 {file.name} ({file_size_mb:.2f} MB)")

with col2:
    # Quick stats from status data
    if "files" in status_data:
        files = status_data["files"]
        completed = len([f for f in files if f.get("status") == "completed"])
        failed = len([f for f in files if f.get("status") == "failed"])
        processing = len([f for f in files if f.get("status") == "processing"])

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            padding: 1rem;
            color: white;
            text-align: center;
            margin-bottom: 0.5rem;
        ">
            <div style="font-weight: 600; font-size: 1rem; margin-bottom: 0.5rem;">Quick Stats</div>
            <div style="display: flex; justify-content: space-around;">
                <div><div style="font-size: 1.5rem; font-weight: bold;">{len(files)}</div><div style="font-size: 0.7rem; opacity: 0.9;">Total</div></div>
                <div><div style="font-size: 1.5rem; font-weight: bold;">{completed}</div><div style="font-size: 0.7rem; opacity: 0.9;">Pass</div></div>
                <div><div style="font-size: 1.5rem; font-weight: bold;">{failed}</div><div style="font-size: 0.7rem; opacity: 0.9;">Fail</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Show last processing result with stage tiles - AFTER the upload/stats columns
if st.session_state.last_processing_result:
    last_result = st.session_state.last_processing_result
    stages_status = last_result.get("stages_status", {})
    result = last_result.get("result", {})
    filename = last_result.get("filename", "")

    # Create a container for the processing result that can be dismissed
    result_container = st.container()
    with result_container:
        # Header with dismiss button
        header_col, dismiss_col = st.columns([5, 1])
        with header_col:
            st.markdown(f"**Last processed: {filename}**")
        with dismiss_col:
            if st.button("✕ Dismiss", key="dismiss_results"):
                st.session_state.last_processing_result = None
                st.rerun()

        # Show stage tiles
        stage_cols = st.columns(5)
        for stage_name, col in zip(
            ["ingest", "parse", "categorize", "extract", "deidentify"],
            stage_cols
        ):
            with col:
                if stage_name in stages_status:
                    stage_result = stages_status[stage_name]
                    if stage_result.get("status") == "success":
                        st.markdown(f"""
                        <div class="stage-card stage-success">
                            <div style="font-size: 1.5rem;">✅</div>
                            <div style="font-weight: 600;">{stage_name.title()}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="stage-card stage-error">
                            <div style="font-size: 1.5rem;">❌</div>
                            <div style="font-weight: 600;">{stage_name.title()}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="stage-card stage-pending">
                        <div style="font-size: 1.5rem;">⏸️</div>
                        <div style="font-weight: 600;">{stage_name.title()}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Show metrics - use same 5-column layout for alignment with tiles
        if result.get("status") == "completed":
            st.write("")  # Small spacer
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Duration", f"{result.get('total_time_seconds', 0):.2f}s")
            with m2:
                if "categorize" in stages_status:
                    category = stages_status["categorize"].get("categorization", {}).get("primary_category", "N/A")
                    st.metric("Category", category)
            with m3:
                if "extract" in stages_status:
                    st.metric("Entities", stages_status["extract"].get("entities_extracted", 0))
            with m4:
                if "deidentify" in stages_status:
                    st.metric("PII Masked", stages_status["deidentify"].get("pii_items_masked", 0))
            # m5 is empty for alignment

        st.markdown("---")

# Process files button
if uploaded_files:
    # Show "Processing..." indicator or button based on state
    if st.session_state.is_processing:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
            border-radius: 8px;
            padding: 0.75rem 2rem;
            text-align: center;
            color: white;
            font-weight: 600;
        ">
            <span class="spinner" style="width: 20px; height: 20px; border-width: 2px;"></span>
            Processing... Please wait
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.button("🚀 Process Files Through Pipeline", type="primary", use_container_width=True):
            st.session_state.is_processing = True
            st.rerun()

# Create a container for processing pipeline that will appear ABOVE the status table
# We create it first so it's positioned above, but populate it after rendering the table
processing_container = st.container()

# Display processing status table - ALWAYS show this
st.divider()
st.subheader("📊 Processing Status Table")

# Create a placeholder for the status table so it can be refreshed during processing
status_table_placeholder = st.empty()

def render_status_table(placeholder):
    """Render the status table into the given placeholder"""
    # Fetch fresh data (bypass cache during processing or reprocessing)
    if st.session_state.is_processing or st.session_state.reprocessing_file_id:
        fresh_status = get_processing_status(limit=12)
    else:
        fresh_status = status_data

    with placeholder.container():
        if "files" in fresh_status and fresh_status["files"]:
            # Get Databricks host and workspace ID from environment
            databricks_host_raw = os.environ.get("DATABRICKS_HOST", "your-workspace.cloud.databricks.com")
            databricks_host = databricks_host_raw if databricks_host_raw.startswith("http") else f"https://{databricks_host_raw}"
            workspace_id = os.environ.get("DATABRICKS_WORKSPACE_ID", "0000000000000000")

            files = fresh_status["files"]
            df_data = []

            for file in files:
                # Build file URL
                volume_path = file.get('volume_path', '')
                file_url = None
                if volume_path:
                    try:
                        parts = volume_path.strip('/').split('/')
                        if len(parts) >= 4 and parts[0] == 'Volumes':
                            catalog = parts[1]
                            schema = parts[2]
                            volume = parts[3]
                            file_in_volume = parts[4] if len(parts) > 4 else file.get("filename", "")
                            file_url = f"{databricks_host}/explore/data/volumes/{catalog}/{schema}/{volume}?o={workspace_id}&filePreviewPath={file_in_volume}"
                    except Exception:
                        pass

                # Build trace URL
                trace_id = file.get('trace_id', '')
                experiment_id = file.get('experiment_id', '')
                trace_url = None
                if trace_id and experiment_id:
                    trace_url = f"{databricks_host}/ml/experiments/{experiment_id}/traces?o={workspace_id}&selectedEvaluationId={trace_id}"

                # Build log file URL
                log_file_path = file.get('log_file_path', '') or ''
                file_id = file.get('file_id', '')
                created_at = file.get('created_at', '')
                log_file_url = None

                if log_file_path:
                    try:
                        log_parts = log_file_path.strip('/').split('/')
                        if len(log_parts) >= 4 and log_parts[0] == 'Volumes':
                            log_catalog = log_parts[1]
                            log_schema = log_parts[2]
                            log_volume = log_parts[3]
                            log_filename = '/'.join(log_parts[4:]) if len(log_parts) > 4 else ''
                            log_file_url = f"{databricks_host}/explore/data/volumes/{log_catalog}/{log_schema}/{log_volume}?o={workspace_id}&filePreviewPath={log_filename}"
                    except Exception:
                        pass
                elif file_id and created_at and LOGS_VOLUME_PATH:
                    try:
                        date_str = created_at[:10]
                        log_parts = LOGS_VOLUME_PATH.strip('/').split('/')
                        if len(log_parts) >= 4 and log_parts[0] == 'Volumes':
                            log_catalog = log_parts[1]
                            log_schema = log_parts[2]
                            log_volume = log_parts[3]
                            subpath = '/'.join(log_parts[4:]) if len(log_parts) > 4 else ''
                            log_filename = f"{subpath}/{date_str}/{file_id}.log" if subpath else f"{date_str}/{file_id}.log"
                            log_file_url = f"{databricks_host}/explore/data/volumes/{log_catalog}/{log_schema}/{log_volume}?o={workspace_id}&filePreviewPath={log_filename}"
                    except Exception:
                        pass

                # Create display values
                filename = file.get("filename", "")
                filename_display = f'<a href="{file_url}" target="_blank">{filename}</a>' if file_url else filename

                if trace_id:
                    trace_id_display = trace_id[:20] + "..." if len(trace_id) > 20 else trace_id
                    if trace_url:
                        trace_id_display = f'<a href="{trace_url}" target="_blank">{trace_id_display}</a>'
                else:
                    trace_id_display = "N/A"

                log_file_display = f'<a href="{log_file_url}" target="_blank">View Log</a>' if log_file_url else "-"

                # Status display
                status = file.get("status", "")
                current_stage = file.get("current_stage", "")

                if status == "completed":
                    status_display = '<span style="color: green;">Pass</span>'
                elif status == "failed":
                    stage_info = f" ({current_stage})" if current_stage else ""
                    status_display = f'<span style="color: red;">Fail{stage_info}</span>'
                elif status == "processing":
                    status_display = f'<span class="processing-status-badge">Processing ({current_stage})</span>'
                else:
                    status_display = status or "-"

                # Actions column - View for completed, Reprocess for failed
                import urllib.parse
                encoded_filename = urllib.parse.quote(filename)
                encoded_volume_path = urllib.parse.quote(volume_path) if volume_path else ""

                if status == "completed":
                    actions_display = f'<a href="?view_results={file_id}&filename={encoded_filename}" target="_self">View</a>'
                elif status == "failed":
                    actions_display = f'<a href="?reprocess={file_id}&filename={encoded_filename}&volume_path={encoded_volume_path}" target="_self">Reprocess</a>'
                else:
                    actions_display = "-"

                # Format timestamps (handle both datetime objects and strings)
                if created_at:
                    if isinstance(created_at, datetime):
                        created_at_display = created_at.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        created_at_display = str(created_at)[:19].replace("T", " ")
                else:
                    created_at_display = "-"

                updated_at = file.get('updated_at', '')
                if updated_at:
                    if isinstance(updated_at, datetime):
                        updated_at_display = updated_at.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        updated_at_display = str(updated_at)[:19].replace("T", " ")
                else:
                    updated_at_display = "-"

                df_data.append({
                    "Filename": filename_display,
                    "Trace ID": trace_id_display,
                    "Log File": log_file_display,
                    "Category": file.get("primary_category", "") or "-",
                    "Entities": file.get("entities_count", 0),
                    "PII Masked": file.get("pii_items_masked", 0),
                    "Status": status_display,
                    "Created": created_at_display,
                    "Updated": updated_at_display,
                    "Actions": actions_display
                })

            df = pd.DataFrame(df_data)
            st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.info("No processing records found")

# Initial render of status table
render_status_table(status_table_placeholder)


# Trace Structure Info Dialog - shows visual representation of MLflow trace hierarchy
@st.dialog("MLflow Trace Structure", width="large")
def show_trace_info_dialog():
    """Display visual representation of trace hierarchy with actual logged keys"""

    st.markdown("""
    Each file generates **1 parent trace** with **5 child spans**. Here's exactly what gets logged:
    """)

    # Parent trace
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 8px; padding: 1rem; margin-bottom: 1rem; color: white;">
        <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.75rem;">process_file_pipeline <span style="opacity: 0.7; font-size: 0.8rem;">(CHAIN)</span></div>
        <div style="font-size: 0.75rem; margin-bottom: 0.75rem;">
            <div style="font-weight: 600; margin-bottom: 0.3rem;">Attributes</div>
            <code style="background: rgba(255,255,255,0.9); color: #333; padding: 2px 6px; border-radius: 3px; margin: 2px;">pipeline_version</code>
            <code style="background: rgba(255,255,255,0.9); color: #333; padding: 2px 6px; border-radius: 3px; margin: 2px;">pipeline_id</code>
            <code style="background: rgba(255,255,255,0.9); color: #333; padding: 2px 6px; border-radius: 3px; margin: 2px;">filename</code>
            <code style="background: rgba(255,255,255,0.9); color: #333; padding: 2px 6px; border-radius: 3px; margin: 2px;">file_size_bytes</code>
            <code style="background: rgba(255,255,255,0.9); color: #333; padding: 2px 6px; border-radius: 3px; margin: 2px;">log_file_path</code>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; font-size: 0.75rem;">
            <div>
                <div style="font-weight: 600; margin-bottom: 0.3rem;">Inputs</div>
                <code style="background: rgba(255,255,255,0.9); color: #333; padding: 2px 6px; border-radius: 3px; margin: 2px;">filename</code>
                <code style="background: rgba(255,255,255,0.9); color: #333; padding: 2px 6px; border-radius: 3px; margin: 2px;">file_size_bytes</code>
                <code style="background: rgba(255,255,255,0.9); color: #333; padding: 2px 6px; border-radius: 3px; margin: 2px;">pipeline_id</code>
            </div>
            <div>
                <div style="font-weight: 600; margin-bottom: 0.3rem;">Outputs</div>
                <code style="background: rgba(255,255,255,0.9); color: #333; padding: 2px 6px; border-radius: 3px; margin: 2px;">status</code>
                <code style="background: rgba(255,255,255,0.9); color: #333; padding: 2px 6px; border-radius: 3px; margin: 2px;">stages_completed</code>
                <code style="background: rgba(255,255,255,0.9); color: #333; padding: 2px 6px; border-radius: 3px; margin: 2px;">total_time_seconds</code>
                <code style="background: rgba(255,255,255,0.9); color: #333; padding: 2px 6px; border-radius: 3px; margin: 2px;">primary_category</code>
                <code style="background: rgba(255,255,255,0.9); color: #333; padding: 2px 6px; border-radius: 3px; margin: 2px;">entities_count</code>
                <code style="background: rgba(255,255,255,0.9); color: #333; padding: 2px 6px; border-radius: 3px; margin: 2px;">pii_items_masked</code>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Child spans in tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["1. Ingest", "2. Parse", "3. Categorize", "4. Extract", "5. De-identify"])

    with tab1:
        st.markdown("""
        **stage_1_ingest** (CHAIN) - UC Volumes API

        **Attributes:** `stage`, `filename`, `catalog`, `schema`, `volume_name`, `file_size_bytes`

        **Inputs:** `filename`, `catalog`, `schema`, `volume_name`, `file_size_bytes`, `overwrite`

        **Outputs:** `status`, `volume_path`, `safe_filename`, `size_bytes`, `file_hash_sha256`
        """)

    with tab2:
        st.markdown("""
        **stage_2_parse** (PARSER) - ai_parse_document

        **Attributes:** `stage`, `parser`, `volume_path`, `sql_warehouse_id`, `statement_id`

        **Inputs:** `volume_path`, `parser`

        **Outputs:** `status`, `text_length`, `text_sample`, `image_output_path`, `pages_count`
        """)

    with tab3:
        st.markdown("""
        **stage_3_categorize** (LLM) - ai_query

        **Attributes:** `stage`, `model`, `request_id`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `finish_reason`

        **Inputs:** `prompt_template`, `model`, `document_text_length`

        **Outputs:** `primary_category`, `primary_confidence`, `primary_justification`, `secondary_category`, `secondary_confidence`
        """)

    with tab4:
        st.markdown("""
        **stage_4_extract** (RETRIEVER) - ai_query

        **Attributes:** `stage`, `model`, `request_id`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `finish_reason`

        **Inputs:** `prompt_template`, `model`, `document_text_length`

        **Outputs:** `entities_count`, `entities` (first 5)
        """)

    with tab5:
        st.markdown("""
        **stage_5_deidentify** (LLM) - ai_query

        **Attributes:** `stage`, `model`, `request_id`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `finish_reason`

        **Inputs:** `prompt_template`, `model`, `document_text_length`

        **Outputs:** `pii_items_masked`, `pii_items` (first 5)
        """)

# Trigger the trace info dialog
if st.session_state.show_trace_info:
    show_trace_info_dialog()
    st.session_state.show_trace_info = False

# Results Viewer Dialog - shows deidentify results in a modal overlay
@st.dialog("De-identification Results", width="large")
def show_results_dialog(file_id: str, filename: str):
    """Display de-identification results in a dialog overlay"""
    import re

    st.markdown(f"**File:** {filename}")

    # Fetch results from backend
    results_data = get_file_results(file_id)

    if "results" in results_data:
        results = results_data["results"]
        deidentify_result_str = results.get("deidentify_result", "")

        if deidentify_result_str:
            deidentify_result = None
            parse_error = None

            # Try multiple parsing strategies
            for attempt in range(3):
                try:
                    if attempt == 0:
                        # First try: clean control characters
                        cleaned_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', deidentify_result_str)
                        deidentify_result = json.loads(cleaned_str)
                    elif attempt == 1:
                        # Second try: also escape problematic characters in string values
                        cleaned_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', deidentify_result_str)
                        # Try to fix common JSON issues
                        cleaned_str = cleaned_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                        deidentify_result = json.loads(cleaned_str)
                    else:
                        # Third try: use ast.literal_eval as fallback
                        import ast
                        cleaned_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', deidentify_result_str)
                        deidentify_result = ast.literal_eval(cleaned_str)
                    break
                except (json.JSONDecodeError, ValueError, SyntaxError) as e:
                    parse_error = e
                    continue

            if deidentify_result:
                # Display PII items found - check both possible locations
                # The stage stores it as deidentification.pii_items
                deidentification = deidentify_result.get("deidentification", {})
                pii_items = deidentification.get("pii_items", [])

                # Also show summary stats
                pii_masked_count = deidentify_result.get("pii_items_masked", len(pii_items))

                if pii_items:
                    st.markdown(f"**PII Items Identified: {pii_masked_count}**")

                    # Create a nice table for PII items
                    pii_data = []
                    for item in pii_items:
                        pii_data.append({
                            "Type": item.get("type", ""),
                            "Original Value": item.get("value", ""),
                            "Strategy": item.get("strategy", ""),
                            "Replacement": item.get("replacement", "")
                        })

                    if pii_data:
                        pii_df = pd.DataFrame(pii_data)
                        st.dataframe(pii_df, use_container_width=True)

                    # Show raw JSON in expander
                    with st.expander("View Raw JSON", expanded=False):
                        st.json(deidentify_result)
                else:
                    st.info("No PII items were identified in this document.")
                    # Still show raw JSON for debugging
                    with st.expander("View Raw JSON", expanded=False):
                        st.json(deidentify_result)
            else:
                # JSON parsing failed - try to extract PII items using regex
                st.warning(f"Note: Full JSON parsing failed, attempting to extract PII items...")

                # Try to find pii_items array in the string
                pii_match = re.search(r'"pii_items"\s*:\s*\[(.*?)\]', deidentify_result_str, re.DOTALL)
                if pii_match:
                    try:
                        # Try to parse just the pii_items array
                        pii_array_str = '[' + pii_match.group(1) + ']'
                        pii_items = json.loads(pii_array_str)

                        if pii_items:
                            st.markdown(f"**PII Items Identified: {len(pii_items)}**")

                            pii_data = []
                            for item in pii_items:
                                pii_data.append({
                                    "Type": item.get("type", ""),
                                    "Original Value": item.get("value", ""),
                                    "Strategy": item.get("strategy", ""),
                                    "Replacement": item.get("replacement", "")
                                })

                            if pii_data:
                                pii_df = pd.DataFrame(pii_data)
                                st.dataframe(pii_df, use_container_width=True)
                        else:
                            st.info("No PII items were identified in this document.")
                    except Exception:
                        st.error(f"Failed to parse deidentify results: {str(parse_error)}")
                        with st.expander("View Raw Data", expanded=False):
                            st.code(deidentify_result_str[:2000] if len(deidentify_result_str) > 2000 else deidentify_result_str)
                else:
                    st.error(f"Failed to parse deidentify results: {str(parse_error)}")
                    with st.expander("View Raw Data", expanded=False):
                        st.code(deidentify_result_str[:2000] if len(deidentify_result_str) > 2000 else deidentify_result_str)
        else:
            st.warning("No de-identification results available for this file.")
    else:
        st.error(results_data.get("error", "Failed to fetch results"))

# Trigger the dialog if viewing results
if st.session_state.viewing_results_file_id:
    file_id = st.session_state.viewing_results_file_id
    filename = st.session_state.get("viewing_results_filename", "")
    show_results_dialog(file_id, filename)
    # Clear the session state after dialog is shown (dialog handles its own close)
    st.session_state.viewing_results_file_id = None
    st.session_state.viewing_results_filename = None

# Processing Pipeline section - runs AFTER status table renders, but displays ABOVE it
if uploaded_files and st.session_state.is_processing:
    with processing_container:
        st.divider()
        st.write("### 📊 Processing Pipeline")

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_files = len(uploaded_files)
        processed_filenames = []

        for idx, uploaded_file in enumerate(uploaded_files):
            # Show prominent processing indicator
            processing_placeholder = st.empty()
            processing_placeholder.markdown(f"""
            <div class="processing-indicator">
                <h3><span class="spinner"></span> Processing File {idx + 1} of {total_files}</h3>
                <p><strong>{uploaded_file.name}</strong></p>
                <p style="font-size: 0.9rem; margin-top: 0.5rem;">Running through 5-stage pipeline with MLflow tracing...</p>
            </div>
            """, unsafe_allow_html=True)

            # Create a container for this file's processing
            with st.container():
                st.markdown(f"#### 📄 {uploaded_file.name}")

                # Read file data to get size, but don't pass bytes to avoid trace bloat
                file_data = uploaded_file.read()
                file_size = len(file_data)

                # Create initial record before processing so file appears in table immediately
                file_id = create_initial_file_record(uploaded_file.name)

                # Refresh status table to show the new file with "Processing (ingest)" status
                render_status_table(status_table_placeholder)

                # Create columns for stage indicators
                stage_cols = st.columns(5)

                # Process through pipeline with MLflow tracing
                try:
                    # Define callback to refresh status table after each stage
                    def refresh_table():
                        render_status_table(status_table_placeholder)

                    # Process file (this creates the parent trace with child spans)
                    result = process_file_through_pipeline(
                        file_bytes=file_data,
                        filename=uploaded_file.name,
                        file_id=file_id,
                        on_stage_update=refresh_table
                    )

                    # Clear processing indicator
                    processing_placeholder.empty()

                    # Update UI with stage results
                    stages_status = result.get("stages", {})

                    for stage_name, col in zip(
                        ["ingest", "parse", "categorize", "extract", "deidentify"],
                        stage_cols
                    ):
                        with col:
                            if stage_name in stages_status:
                                stage_result = stages_status[stage_name]
                                if stage_result.get("status") == "success":
                                    st.markdown(f"""
                                    <div class="stage-card stage-success">
                                        <div style="font-size: 1.5rem;">✅</div>
                                        <div style="font-weight: 600;">{stage_name.title()}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="stage-card stage-error">
                                        <div style="font-size: 1.5rem;">❌</div>
                                        <div style="font-weight: 600;">{stage_name.title()}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="stage-card stage-pending">
                                    <div style="font-size: 1.5rem;">⏸️</div>
                                    <div style="font-weight: 600;">{stage_name.title()}</div>
                                </div>
                                """, unsafe_allow_html=True)

                    # Display results
                    if result.get("status") == "completed":
                        st.success(f"✅ Pipeline completed for {uploaded_file.name}")
                        processed_filenames.append(uploaded_file.name)

                        # Show key metrics - use 5 columns for alignment with tiles
                        m1, m2, m3, m4, m5 = st.columns(5)
                        with m1:
                            st.metric("Duration", f"{result['total_time_seconds']:.2f}s")
                        with m2:
                            if "categorize" in stages_status:
                                category = stages_status["categorize"].get("categorization", {}).get("primary_category", "N/A")
                                st.metric("Category", category)
                        with m3:
                            if "extract" in stages_status:
                                st.metric("Entities", stages_status["extract"].get("entities_extracted", 0))
                        with m4:
                            if "deidentify" in stages_status:
                                st.metric("PII Masked", stages_status["deidentify"].get("pii_items_masked", 0))
                        # m5 is empty for alignment

                        # Add to processed files
                        st.session_state.processed_files.append({
                            "filename": uploaded_file.name,
                            "pipeline_id": result["pipeline_id"],
                            "status": "completed",
                            "timestamp": result["end_time"],
                            "duration": result["total_time_seconds"]
                        })

                    else:
                        st.error(f"❌ Pipeline failed for {uploaded_file.name}")
                        st.error(f"Error: {result.get('error', 'Unknown error')}")

                except Exception as e:
                    processing_placeholder.empty()
                    st.error(f"❌ Failed to process {uploaded_file.name}")
                    st.error(f"Error: {str(e)}")

                st.divider()

            # Update progress
            progress_bar.progress((idx + 1) / total_files)

            # Refresh status table to show updated processing status
            render_status_table(status_table_placeholder)

        status_text.text(f"✨ Processing complete: {len(uploaded_files)} files processed")

    # Store the last result for display after rerun
    if 'result' in dir() and result:
        st.session_state.last_processing_result = {
            "filename": uploaded_file.name if 'uploaded_file' in dir() else "",
            "result": result,
            "stages_status": stages_status if 'stages_status' in dir() else {}
        }

    # Reset processing state and clear uploader
    st.session_state.is_processing = False
    st.session_state.just_processed_files = processed_filenames
    st.session_state.uploader_key += 1
    # Clear cache so new files show up in status table
    st.cache_data.clear()
    st.rerun()

# Reprocessing Pipeline section - runs AFTER status table renders, but displays ABOVE it
if st.session_state.reprocessing_file_id:
    with processing_container:
        file_id = st.session_state.reprocessing_file_id
        volume_path = st.session_state.get("reprocessing_volume_path", "")
        filename = st.session_state.get("reprocessing_filename", "")

        st.divider()
        st.write("### Reprocessing Pipeline")

        # Show processing indicator
        processing_placeholder = st.empty()
        processing_placeholder.markdown(f"""
        <div class="processing-indicator">
            <h3><span class="spinner"></span> Reprocessing File</h3>
            <p><strong>{filename}</strong></p>
            <p style="font-size: 0.9rem; margin-top: 0.5rem;">Running stages 2-5 with new trace...</p>
        </div>
        """, unsafe_allow_html=True)

        try:
            # Define callback to refresh status table after each stage
            def refresh_table_reprocess():
                render_status_table(status_table_placeholder)

            result = reprocess_file(
                file_id=file_id,
                volume_path=volume_path,
                filename=filename,
                on_stage_update=refresh_table_reprocess
            )

            # Clear processing indicator
            processing_placeholder.empty()

            if result.get("status") == "completed":
                st.success(f"✅ Reprocessing completed for {filename}")
            else:
                st.error(f"❌ Reprocessing failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            processing_placeholder.empty()
            st.error(f"❌ Reprocessing failed: {str(e)}")

    # Clear reprocessing state and refresh
    st.session_state.reprocessing_file_id = None
    st.session_state.reprocessing_volume_path = None
    st.session_state.reprocessing_filename = None
    st.cache_data.clear()
    st.rerun()

# Display processed files history
if st.session_state.processed_files:
    st.divider()
    st.subheader("📋 Processing History")

    for file_info in reversed(st.session_state.processed_files):
        with st.expander(f"📄 {file_info['filename']} - {file_info['status']}", expanded=False):
            cols = st.columns(4)
            with cols[0]:
                st.metric("Pipeline ID", file_info['pipeline_id'][:8] + "...")
            with cols[1]:
                st.metric("Status", file_info['status'].title())
            with cols[2]:
                st.metric("Duration", f"{file_info['duration']:.2f}s")
            with cols[3]:
                st.metric("Timestamp", file_info['timestamp'][:19])

# Footer
st.divider()
st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 10px;
    padding: 1.5rem;
    margin-top: 1rem;
">
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem;">
        <div>
            <h5 style="color: #667eea; margin-top: 0;">🔍 MLflow Tracing</h5>
            <ul style="color: #495057; font-size: 0.85rem; margin: 0; padding-left: 1.2rem;">
                <li>URI: <code>{st.session_state.tracking_uri}</code></li>
                <li>Experiment: <code>{st.session_state.get('experiment_name', 'unstructured_parsequery_pipeline')}</code></li>
                <li>1 parent trace + 5 child spans per file</li>
            </ul>
        </div>
        <div>
            <h5 style="color: #764ba2; margin-top: 0;">⚙️ Pipeline Stages</h5>
            <ul style="color: #495057; font-size: 0.85rem; margin: 0; padding-left: 1.2rem;">
                <li><strong>Ingest</strong> - UC Volume Upload</li>
                <li><strong>Parse</strong> - ai_parse_document</li>
                <li><strong>Categorize/Extract/De-identify</strong> - AI Query</li>
            </ul>
        </div>
        <div>
            <h5 style="color: #28a745; margin-top: 0;">📊 Status Tracking</h5>
            <p style="color: #495057; font-size: 0.85rem; margin: 0;">
                Status: <code>unstructured_parsequery.file_processing_status</code><br>
                Results: <code>unstructured_parsequery.results</code>
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
