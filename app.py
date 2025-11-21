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
    get_file_results,
    reset_stuck_processing_files,
    delete_file_record
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

    /* Filename column - narrower with word wrap */
    .dataframe td:first-child {
        max-width: 150px;
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: normal;
    }

    /* Trace ID column (2nd) - narrower */
    .dataframe td:nth-child(2) {
        max-width: 120px;
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: normal;
    }

    /* Entities column (5th) - narrow for numbers */
    .dataframe td:nth-child(5) {
        text-align: center;
    }

    /* PII Masked column (6th) - narrow for numbers */
    .dataframe td:nth-child(6) {
        text-align: center;
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

    /* Full page loading overlay */
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255, 255, 255, 0.95);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 9999;
    }

    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 4px solid rgba(102, 126, 234, 0.2);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s ease-in-out infinite;
        margin-bottom: 1rem;
    }

    .loading-text {
        color: #667eea;
        font-size: 1.1rem;
        font-weight: 500;
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
elif "resume" in query_params:
    # Resume from failed stage
    file_id = query_params.get("resume")
    filename = query_params.get("filename", "")
    volume_path = query_params.get("volume_path", "")
    failed_stage = query_params.get("failed_stage", "")
    if file_id:
        st.session_state.reprocessing_file_id = file_id
        st.session_state.reprocessing_filename = filename
        st.session_state.reprocessing_volume_path = volume_path
        st.session_state.reprocessing_failed_stage = failed_stage
        # Clear query params
        st.query_params.clear()
elif "reprocess" in query_params:
    # Reprocess from scratch (parse stage)
    file_id = query_params.get("reprocess")
    filename = query_params.get("filename", "")
    volume_path = query_params.get("volume_path", "")
    if file_id:
        st.session_state.reprocessing_file_id = file_id
        st.session_state.reprocessing_filename = filename
        st.session_state.reprocessing_volume_path = volume_path
        st.session_state.reprocessing_failed_stage = ""  # Empty means start from parse
        # Clear query params
        st.query_params.clear()
elif "delete" in query_params:
    # Delete corrupt/stuck record
    file_id = query_params.get("delete")
    if file_id:
        result = delete_file_record(file_id)
        if result.get("deleted"):
            st.session_state.delete_success = f"Deleted record: {file_id[:8]}..."
            # Clear cache so table refreshes without the deleted record
            st.cache_data.clear()
        else:
            st.session_state.delete_error = result.get("error", "Failed to delete")
        st.query_params.clear()

# Configurable table row limit
TABLE_ROW_LIMIT = int(os.environ.get("TABLE_ROW_LIMIT", "20"))

# Cache the status query to reduce SQL calls
@st.cache_data(ttl=30)
def fetch_processing_status():
    try:
        return get_processing_status(limit=TABLE_ROW_LIMIT)
    except Exception as e:
        return {"files": [], "error": str(e)}

# Show loading indicator on first load
if "initial_load_complete" not in st.session_state:
    loading_placeholder = st.empty()
    loading_placeholder.markdown("""
    <div class="loading-overlay">
        <div class="loading-spinner"></div>
        <div class="loading-text">Loading processing status...</div>
    </div>
    """, unsafe_allow_html=True)

    # Fetch processing status
    status_data = fetch_processing_status()

    # Mark initial load as complete and clear loading indicator
    st.session_state.initial_load_complete = True
    loading_placeholder.empty()
else:
    # Subsequent loads - fetch without loading indicator
    status_data = fetch_processing_status()

# Header with gradient styling - compact version
st.markdown("""
<div style="
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 0.75rem 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    color: white;
    text-align: center;
">
    <h3 style="color: white !important; margin: 0; font-size: 1.2rem;">Document Intelligence: Unstructured ParseQuery</h3>
    <span style="font-size: 0.75rem; color: rgba(255,255,255,0.8);">Databricks AI Functions</span>
</div>
""", unsafe_allow_html=True)

# Helper to show count badge
def stage_badge(count):
    if count > 0:
        return f'<span style="background: #ff9800; color: white; padding: 2px 6px; border-radius: 10px; font-size: 0.7rem; margin-left: 0.5rem;">{count} file(s) processing</span>'
    return ""

# Function to render sidebar with stage counts
def render_sidebar_stages(data):
    """Render the pipeline stages in sidebar with current counts"""
    stage_counts = {"ingest": 0, "parse": 0, "categorize": 0, "extract": 0, "deidentify": 0}
    total_processing = 0
    if "files" in data:
        for file in data["files"]:
            if file.get("status") == "processing":
                total_processing += 1
                current_stage = file.get("current_stage", "").lower()
                if current_stage in stage_counts:
                    stage_counts[current_stage] += 1

    # Show processing summary if files are being processed
    processing_summary = ""
    if total_processing > 0:
        processing_summary = f'<span style="background: linear-gradient(135deg, #ff9800 0%, #ff5722 100%); color: white; padding: 0.3rem 0.6rem; border-radius: 6px; font-weight: 600; font-size: 0.75rem;">{total_processing} file(s) processing</span>'

    return f"""<div style="margin-bottom: 1rem;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
<h4 style="color: #495057; margin: 0; font-size: 1.1rem;">Pipeline Stages</h4>
{processing_summary}
</div>
<div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); border-left: 3px solid #28a745; border-radius: 0 6px 6px 0; padding: 0.6rem 0.75rem; margin-bottom: 0.5rem;">
<div style="font-weight: 600; font-size: 0.9rem; color: #155724;">1. Ingest{stage_badge(stage_counts["ingest"])}</div>
<div style="font-size: 0.8rem; color: #495057;">Upload to UC Volume</div>
<div style="font-size: 0.75rem; color: #28a745; font-weight: 500;">UC Volumes API</div>
</div>
<div style="background: linear-gradient(135deg, #cce5ff 0%, #b8daff 100%); border-left: 3px solid #667eea; border-radius: 0 6px 6px 0; padding: 0.6rem 0.75rem; margin-bottom: 0.5rem;">
<div style="font-weight: 600; font-size: 0.9rem; color: #004085;">2. Parse{stage_badge(stage_counts["parse"])}</div>
<div style="font-size: 0.8rem; color: #495057;">Text Extraction</div>
<div style="font-size: 0.75rem; color: #667eea; font-weight: 500;">ai_parse_document</div>
</div>
<div style="background: linear-gradient(135deg, #e2d5f1 0%, #d4c5e8 100%); border-left: 3px solid #764ba2; border-radius: 0 6px 6px 0; padding: 0.6rem 0.75rem; margin-bottom: 0.5rem;">
<div style="font-weight: 600; font-size: 0.9rem; color: #4a235a;">3. Categorize{stage_badge(stage_counts["categorize"])}</div>
<div style="font-size: 0.8rem; color: #495057;">Document Classification</div>
<div style="font-size: 0.75rem; color: #764ba2; font-weight: 500;">ai_query</div>
</div>
<div style="background: linear-gradient(135deg, #e2d5f1 0%, #d4c5e8 100%); border-left: 3px solid #764ba2; border-radius: 0 6px 6px 0; padding: 0.6rem 0.75rem; margin-bottom: 0.5rem;">
<div style="font-weight: 600; font-size: 0.9rem; color: #4a235a;">4. Extract{stage_badge(stage_counts["extract"])}</div>
<div style="font-size: 0.8rem; color: #495057;">Entity Extraction</div>
<div style="font-size: 0.75rem; color: #764ba2; font-weight: 500;">ai_query</div>
</div>
<div style="background: linear-gradient(135deg, #e2d5f1 0%, #d4c5e8 100%); border-left: 3px solid #764ba2; border-radius: 0 6px 6px 0; padding: 0.6rem 0.75rem; margin-bottom: 0.5rem;">
<div style="font-weight: 600; font-size: 0.9rem; color: #4a235a;">5. De-identify{stage_badge(stage_counts["deidentify"])}</div>
<div style="font-size: 0.8rem; color: #495057;">PII Masking</div>
<div style="font-size: 0.75rem; color: #764ba2; font-weight: 500;">ai_query</div>
</div>
</div>"""

# Display About and info in sidebar
with st.sidebar:
    # Create placeholder for dynamic stage counts
    sidebar_stages_placeholder = st.empty()
    sidebar_stages_placeholder.markdown(render_sidebar_stages(status_data), unsafe_allow_html=True)

    st.divider()

    # Observability section with info button
    obs_col1, obs_col2 = st.columns([3, 1])
    with obs_col1:
        st.markdown("**Observability**")
    with obs_col2:
        if st.button("ℹ️", key="trace_info_btn", help="View trace structure"):
            st.session_state.show_trace_info = True

    st.caption(f"Experiment: `{st.session_state.get('experiment_name', 'unstructured_parsequery_pipeline')}`")

    st.divider()

    # Reset stuck files button
    st.markdown("**Maintenance**")
    if st.button("Reset Stuck Files", key="reset_stuck_btn", help="Mark all 'processing' files as failed so they can be reprocessed"):
        result = reset_stuck_processing_files()
        if result.get("reset_count", 0) > 0:
            st.success(f"Reset {result['reset_count']} stuck file(s)")
            st.cache_data.clear()
            st.rerun()
        elif "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.info("No stuck files found")


# Initialize session state for storing files during processing
if "files_to_process" not in st.session_state:
    st.session_state.files_to_process = []

# Main content area - File Upload (full width)
# Only show file upload section when not processing
if not st.session_state.is_processing:
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

    # Show delete success/error messages
    if "delete_success" in st.session_state and st.session_state.delete_success:
        st.success(f"✅ {st.session_state.delete_success}")
        st.session_state.delete_success = None
    if "delete_error" in st.session_state and st.session_state.delete_error:
        st.error(f"❌ {st.session_state.delete_error}")
        st.session_state.delete_error = None

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

    # Process files button
    if uploaded_files:
        if st.button("🚀 Process Files Through Pipeline", type="primary", use_container_width=True):
            # Store file data in session state before rerun
            st.session_state.files_to_process = []
            for f in uploaded_files:
                file_data = f.read()
                st.session_state.files_to_process.append({
                    "name": f.name,
                    "data": file_data,
                    "size": len(file_data)
                })
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
        fresh_status = get_processing_status(limit=TABLE_ROW_LIMIT)
        # Also update sidebar stage counts
        sidebar_stages_placeholder.markdown(render_sidebar_stages(fresh_status), unsafe_allow_html=True)
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

                # Build trace URL(s) - trace_id can be comma-separated for resumed files
                trace_id_field = file.get('trace_id', '') or ''
                experiment_id = file.get('experiment_id', '')

                # Parse comma-separated trace IDs
                trace_ids = [tid.strip() for tid in trace_id_field.split(',') if tid.strip()] if trace_id_field else []

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

                if trace_ids:
                    # Build display for each trace ID (most recent first)
                    # Show full ID, CSS will handle wrapping
                    trace_displays = []
                    for i, tid in enumerate(reversed(trace_ids)):
                        if experiment_id:
                            trace_url = f"{databricks_host}/ml/experiments/{experiment_id}/traces?o={workspace_id}&selectedEvaluationId={tid}"
                            trace_displays.append(f'<a href="{trace_url}" target="_blank">{tid}</a>')
                        else:
                            trace_displays.append(tid)

                    if len(trace_displays) == 1:
                        trace_id_display = trace_displays[0]
                    else:
                        # Show most recent trace prominently with "Latest" label, older ones with "Previous" label
                        trace_id_display = f'<span style="font-size:0.7em;color:#28a745;font-weight:600;">Latest:</span> {trace_displays[0]}'
                        older_traces = '<br>'.join([f'<span style="font-size:0.7em;color:#999;">Prev:</span> {t}' for t in trace_displays[1:]])
                        trace_id_display = f'{trace_id_display}<br>{older_traces}'
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
                    # Check if file exists in volume (has volume_path)
                    if volume_path:
                        encoded_current_stage = urllib.parse.quote(current_stage) if current_stage else ""
                        # Resume link - passes failed_stage to resume from that point
                        resume_link = f'<a href="?resume={file_id}&filename={encoded_filename}&volume_path={encoded_volume_path}&failed_stage={encoded_current_stage}" target="_self">Resume</a>'
                        # Reprocess link - no failed_stage means start from scratch (parse)
                        reprocess_link = f'<a href="?reprocess={file_id}&filename={encoded_filename}&volume_path={encoded_volume_path}" target="_self">Reprocess</a>'
                        # Delete link for cleanup
                        delete_link = f'<a href="?delete={file_id}" target="_self" style="color:#dc3545;">Delete</a>'
                        actions_display = f'{resume_link} | {reprocess_link} | {delete_link}'
                    else:
                        # No volume_path means file failed at ingest - can only delete
                        delete_link = f'<a href="?delete={file_id}" target="_self" style="color:#dc3545;">Delete</a>'
                        actions_display = delete_link
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
if st.session_state.is_processing and st.session_state.files_to_process:
    import concurrent.futures
    import threading

    with processing_container:
        files_to_process = st.session_state.files_to_process
        total_files = len(files_to_process)

        # Header showing parallel processing with spinner
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 8px;
            padding: 0.75rem 1.25rem;
            margin-bottom: 0.75rem;
            border: 1px solid #dee2e6;
        ">
            <span style="font-weight: 600; color: #495057; font-size: 1.1rem;">Processing {total_files} file(s) in parallel</span>
            <span style="color: #6c757d; font-size: 0.9rem;"> - Watch status table for progress</span>
            <div style="
                margin-top: 0.75rem;
                height: 4px;
                background: #e9ecef;
                border-radius: 2px;
                overflow: hidden;
            ">
                <div style="
                    height: 100%;
                    width: 30%;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 2px;
                    animation: progress-slide 1.5s ease-in-out infinite;
                "></div>
            </div>
        </div>
        <style>
            @keyframes progress-slide {{
                0% {{ margin-left: 0%; }}
                50% {{ margin-left: 70%; }}
                100% {{ margin-left: 0%; }}
            }}
        </style>
        """, unsafe_allow_html=True)

        # Create initial records for all files first
        file_ids = []
        for file_info in files_to_process:
            file_id = create_initial_file_record(file_info["name"])
            file_ids.append(file_id)

        # Refresh table to show all files as "Processing (ingest)"
        render_status_table(status_table_placeholder)

        # Thread-safe results storage
        results_lock = threading.Lock()
        processing_results = []

        def process_single_file(file_info, file_id):
            """Process a single file - runs in thread"""
            try:
                result = process_file_through_pipeline(
                    file_bytes=file_info["data"],
                    filename=file_info["name"],
                    file_id=file_id,
                    on_stage_update=None,  # Table refresh handled by main thread
                    on_stage_status=None   # No tile updates in parallel mode
                )
                with results_lock:
                    processing_results.append({
                        "filename": file_info["name"],
                        "result": result
                    })
                return result
            except Exception as e:
                with results_lock:
                    processing_results.append({
                        "filename": file_info["name"],
                        "result": {"status": "failed", "error": str(e)}
                    })
                return {"status": "failed", "error": str(e)}

        # Process files in parallel with ThreadPoolExecutor
        max_workers = min(4, total_files)  # Limit concurrent processing

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all files for processing
            futures = {
                executor.submit(process_single_file, file_info, file_id): file_info["name"]
                for file_info, file_id in zip(files_to_process, file_ids)
            }

            # Poll for completion and refresh table
            completed = 0
            progress_placeholder = st.empty()

            import time
            while completed < total_files:
                # Check how many are done
                done_count = sum(1 for f in futures if f.done())
                if done_count > completed:
                    completed = done_count
                    progress_placeholder.markdown(f"**Completed: {completed}/{total_files} files**")

                # Always refresh table to show stage transitions
                render_status_table(status_table_placeholder)

                # Small delay to avoid tight loop
                time.sleep(0.5)

            # Final refresh
            render_status_table(status_table_placeholder)
            progress_placeholder.markdown(f"**✅ All {total_files} files processed**")

        # Collect results
        processed_filenames = []
        for item in processing_results:
            result = item["result"]
            filename = item["filename"]
            if result.get("status") == "completed":
                processed_filenames.append(filename)
                st.session_state.processed_files.append({
                    "filename": filename,
                    "pipeline_id": result.get("pipeline_id", ""),
                    "status": "completed",
                    "timestamp": result.get("end_time", ""),
                    "duration": result.get("total_time_seconds", 0)
                })

    # Reset processing state and clear uploader
    st.session_state.is_processing = False
    st.session_state.just_processed_files = processed_filenames
    st.session_state.files_to_process = []  # Clear stored files
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
        failed_stage = st.session_state.get("reprocessing_failed_stage", "")

        st.divider()
        st.write("### Reprocessing Pipeline")

        # Show processing indicator with resume info
        resume_info = f"Resuming from {failed_stage}" if failed_stage else "Running stages 2-5"
        processing_placeholder = st.empty()
        processing_placeholder.markdown(f"""
        <div class="processing-indicator">
            <h3><span class="spinner"></span> Reprocessing File</h3>
            <p><strong>{filename}</strong></p>
            <p style="font-size: 0.9rem; margin-top: 0.5rem;">{resume_info} with new trace...</p>
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
                failed_stage=failed_stage,
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
    st.session_state.reprocessing_failed_stage = None
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

