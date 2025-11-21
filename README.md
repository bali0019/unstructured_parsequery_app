# Document Intelligence: Unstructured ParseQuery

A document processing pipeline using Databricks AI Functions (ai_parse_document + ai_query) with MLflow Tracing for full observability.

## Overview

This app implements a 5-stage document processing pipeline using **Databricks AI Functions** and demonstrates **MLflow tracing patterns** for tracking document processing at the file level.

### Pipeline Stages

Each uploaded file goes through these stages:

| Stage | Name | AI Function | Description |
|-------|------|-------------|-------------|
| 1 | **Ingest** | UC Volumes API | Upload files to Unity Catalog Volume |
| 2 | **Parse** | `ai_parse_document` | Extract text and structure from documents |
| 3 | **Categorize** | `ai_query` | Classify documents into taxonomy categories |
| 4 | **Extract** | `ai_query` | Extract entities (people, orgs, amounts, dates) |
| 5 | **De-identify** | `ai_query` | Detect and mask PII for compliance |

### MLflow Tracing Structure

For each file processed:
- **1 Parent Trace**: `process_file_pipeline`
  - **5 Child Spans**: One for each stage
    - `stage_1_ingest` (CHAIN)
    - `stage_2_parse` (PARSER)
    - `stage_3_categorize` (LLM)
    - `stage_4_extract` (RETRIEVER)
    - `stage_5_deidentify` (LLM)

Each span logs:
- **Attributes**: filename, stage, model info, token usage
- **Inputs/Outputs**: Stage-specific data (text length, entities, PII items)

## Architecture

### Storage Layer

- **Lakebase (PostgreSQL)**: Status tracking and results storage
  - `unstructured_parsequery.file_processing_status` - Pipeline status for each file
  - `unstructured_parsequery.results` - Stage results as JSON
- **Unity Catalog Volumes**: Document storage and pipeline logs

### App Resources

The app uses these Databricks resources (configured via DAB):

| Resource | Type | Purpose |
|----------|------|---------|
| `source-volume` | UC Volume | Document uploads |
| `parse-warehouse` | SQL Warehouse | Execute `ai_parse_document` |
| `ai-query-endpoint` | Serving Endpoint | Execute `ai_query` for LLM stages |
| `logs-volume` | UC Volume | Pipeline execution logs |
| `database` | Lakebase | PostgreSQL for status/results |

## Deployment

### Prerequisites

- Databricks CLI v0.278.0 or later
- Access to a Databricks workspace with:
  - Unity Catalog enabled
  - SQL Warehouse (existing)
  - Model Serving Endpoint (existing)
  - Lakebase database instance (existing)

### Deploy with Databricks Asset Bundles (Recommended)

DAB automatically creates UC volumes and configures all app resources:

```bash
# 1. Validate the bundle configuration
databricks bundle validate --profile <your-profile>

# 2. Deploy the bundle (creates volumes, app, uploads code)
databricks bundle deploy --profile <your-profile>

# 3. Deploy the app (triggers actual deployment with source code)
databricks apps deploy unstructured-parsequery-app \
  --source-code-path /Workspace/Users/<your-email>/.bundle/unstructured-parsequery-app/dev/files \
  --profile <your-profile>

# 4. Check app status
databricks apps get unstructured-parsequery-app --profile <your-profile>
```

**What gets created:**
- Source Volume: `<catalog>.<schema>.unstructured_parsequery_source`
- Logs Volume: `<catalog>.<schema>.unstructured_parsequery_logs`
- App with all resource permissions configured

**Customize deployment** by overriding variables:
```bash
databricks bundle deploy --profile <your-profile> \
  --var="catalog=main" \
  --var="schema=production" \
  --var="sql_warehouse_id=your-warehouse-id"
```

### Update Existing Deployment

```bash
# Re-deploy after code changes
databricks bundle deploy --profile <your-profile>

# Trigger app deployment with updated code
databricks apps deploy unstructured-parsequery-app \
  --source-code-path /Workspace/Users/<your-email>/.bundle/unstructured-parsequery-app/dev/files \
  --profile <your-profile>
```

### Destroy Deployment

```bash
# Remove all DAB-created resources
databricks bundle destroy --profile <your-profile>
```

## Project Structure

```
unstructured_parsequery_app/
├── app.py                    # Streamlit frontend
├── backend.py                # Pipeline orchestration with MLflow tracing
├── config.py                 # Configuration and AI prompts
├── app.yaml                  # Databricks Apps runtime config
├── databricks.yml            # DAB bundle configuration
├── requirements.txt          # Python dependencies
├── generate_test_pdfs.py     # Generate test PDFs (not deployed)
├── stages/
│   ├── ingest.py            # Stage 1: UC Volume upload
│   ├── parse.py             # Stage 2: ai_parse_document
│   ├── categorize.py        # Stage 3: ai_query for classification
│   ├── extract.py           # Stage 4: ai_query for entity extraction
│   └── deidentify.py        # Stage 5: ai_query for PII detection
├── storage/
│   ├── lakebase_connection.py  # PostgreSQL connection manager
│   ├── status_table.py         # Status table operations (Lakebase)
│   └── results_table.py        # Results table operations (Lakebase)
└── utils/
    ├── oauth.py             # OAuth token management
    └── uc_logger.py         # UC Volume logging handler
```

## Configuration

### Environment Variables

Set in `app.yaml`:

| Variable | Description | Source |
|----------|-------------|--------|
| `MLFLOW_TRACKING_URI` | MLflow tracking | `databricks` |
| `VOLUME_PATH` | Source documents volume | App resource |
| `SQL_WAREHOUSE_ID` | Warehouse for ai_parse_document | App resource |
| `AI_QUERY_ENDPOINT` | Serving endpoint for ai_query | App resource |
| `LOGS_VOLUME_PATH` | Pipeline logs volume | App resource |
| `MLFLOW_EXPERIMENT_NAME` | MLflow experiment | `/Shared/unstructured_parsequery_pipeline` |
| `STATUS_TABLE_NAME` | Lakebase status table | `unstructured_parsequery.file_processing_status` |
| `RESULTS_TABLE_NAME` | Lakebase results table | `unstructured_parsequery.results` |
| `TABLE_ROW_LIMIT` | Max rows in status table | `20` (default) |

### AI Prompts

Prompts for each AI stage are defined in `config.py`:

- `CATEGORIZE_PROMPT` - Document classification into financial taxonomy
- `EXTRACT_PROMPT` - Entity extraction (people, orgs, amounts, etc.)
- `DEIDENTIFY_PROMPT` - PII detection and masking strategy

Prompts can be overridden via environment variables.

## Features

### Streamlit UI

- **File upload**: Multi-file upload with drag-and-drop
- **Parallel processing**: Process multiple files simultaneously (up to 4 concurrent)
- **Live progress**: Real-time status updates in the processing table
- **Status table**: View all processed files with hyperlinks to:
  - Source documents in UC Volume
  - MLflow traces
  - Pipeline logs
- **Results viewer**: Inspect de-identification results
- **Reprocess**: Retry failed files from stage 2 onwards

### MLflow Tracing

Each file creates a complete trace hierarchy:

```
process_file_pipeline (CHAIN)
├── stage_1_ingest (CHAIN)
│   ├── inputs: filename, file_size_bytes
│   └── outputs: volume_path, file_hash_sha256
├── stage_2_parse (PARSER)
│   ├── inputs: volume_path
│   └── outputs: text_length, pages_count
├── stage_3_categorize (LLM)
│   ├── inputs: document_text_length, prompt_template
│   ├── attributes: prompt_tokens, completion_tokens
│   └── outputs: primary_category, confidence
├── stage_4_extract (RETRIEVER)
│   ├── inputs: document_text_length, prompt_template
│   └── outputs: entities_count, entities
└── stage_5_deidentify (LLM)
    ├── inputs: document_text_length, prompt_template
    └── outputs: pii_items_masked, pii_items
```

### Lakebase Storage

- **Low-latency OLTP**: Fast status updates during pipeline execution
- **PostgreSQL compatible**: Standard SQL queries
- **OAuth authentication**: Uses app service principal token
- **Auto-schema creation**: Tables created automatically on first run

## Local Development

### Generate Test PDFs

```bash
# Generate sample financial documents for testing
python generate_test_pdfs.py
```

This creates 5 financial PDFs in `pdfs/` folder for testing the pipeline.

### Force Stage Failure (Testing)

Set in `config.py` or via environment variable:

```python
TEST_FORCE_FAILURE_STAGE = "categorize"  # Force failure at categorize stage
```

Valid values: `ingest`, `parse`, `categorize`, `extract`, `deidentify`, or `None`

## Usage

1. **Upload files**: Select one or more documents (PDF, DOCX, TXT, HTML, MD)
2. **Process**: Click "Process Files Through Pipeline"
3. **Monitor**: Watch real-time progress in the status table (files process in parallel, up to 4 at a time)
4. **View traces**: Click trace ID links to see MLflow traces
5. **View logs**: Click log links to see pipeline execution logs
6. **Review results**: Click "View" to see de-identification results
7. **Reprocess**: Click "Reprocess" on failed files to retry

## Taxonomy Categories

Documents are classified into:

- Loan Application
- Financial Statement
- Investment Document
- Credit Report
- Banking Statement
- Tax Document
- Insurance Policy
- Compliance Document
- Contract Agreement

## Entity Types Extracted

- Person names
- Organizations
- Account numbers
- SSN/Tax IDs
- Monetary amounts
- Dates
- Addresses
- Email/Phone
- Credit scores
- Property descriptions

## PII Types Detected

- Names
- SSN
- Tax ID
- Account/Routing numbers
- Driver's license
- Email/Phone
- Addresses
- Date of birth
- Salary/Income
- Credit scores

## Troubleshooting

### Common Issues

1. **Permission denied for schema**: App service principal needs CREATE on database (use Lakebase with `CAN_CONNECT_AND_CREATE`)

2. **Statement execution timeout**: Increase `AI_QUERY_MAX_TOKENS` or check warehouse availability

3. **Token expired**: Lakebase connection manager auto-refreshes tokens on each connection

4. **Missing environment variables**: Ensure all app resources are configured in Databricks Apps UI or via DAB

### Logs

- Pipeline logs are written to UC Volume at `LOGS_VOLUME_PATH`
- Each file gets a log file: `{date}/{pipeline_id}.log`
- Logs include timestamps, stage names, and detailed execution info

## Requirements

- Python 3.11+
- Databricks SDK >= 0.56.0
- psycopg2-binary (for Lakebase)
- streamlit
- mlflow < 3.6.0
- pandas

See `requirements.txt` for full list.
