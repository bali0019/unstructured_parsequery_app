"""
Configuration module for Unstructured ParseQuery

Manages configurable settings including:
- UC Volume paths
- AI query prompts for categorize, extract, and de-identify stages
- MLflow experiment settings
- Processing parameters
"""

import os

# ============================================================================
# Unity Catalog Volume Configuration
# ============================================================================

# Volume path from environment or default
VOLUME_PATH = os.environ.get("VOLUME_PATH", "/Volumes/catalog/schema/volume")

# Parse volume components
def parse_volume_path(volume_path):
    """Parse UC volume path into components"""
    try:
        path_parts = volume_path.strip("/").split("/")
        if len(path_parts) >= 4 and path_parts[0] == "Volumes":
            return {
                "catalog": path_parts[1],
                "schema": path_parts[2],
                "volume_name": path_parts[3],
                "full_path": volume_path
            }
    except Exception:
        pass
    return None

VOLUME_CONFIG = parse_volume_path(VOLUME_PATH)


# ============================================================================
# MLflow Configuration
# ============================================================================

MLFLOW_EXPERIMENT_NAME = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME",
    "/Shared/unstructured_parsequery_pipeline"
)


# ============================================================================
# AI Query Prompts Configuration
# ============================================================================

# Stage 3: Categorize - Default prompt
CATEGORIZE_PROMPT_DEFAULT = """
Analyze the following financial document and categorize it according to the taxonomy below.
Provide a primary classification with confidence score and justification, and a secondary classification.

Document Content:
{document_text}

Taxonomy Categories:
- Loan Application: Personal loans, mortgages, business loans, credit applications
- Financial Statement: Balance sheets, income statements, cash flow statements, profit & loss
- Investment Document: Portfolio statements, prospectuses, fund reports, investment agreements
- Credit Report: Credit scores, payment history, credit inquiries, debt summaries
- Banking Statement: Account statements, transaction records, deposit confirmations
- Tax Document: Tax returns, W-2s, 1099s, tax assessments, IRS correspondence
- Insurance Policy: Life insurance, property insurance, liability policies, claims documents
- Compliance Document: Regulatory filings, audit reports, KYC documents, risk assessments
- Contract Agreement: Service agreements, purchase agreements, terms and conditions

Respond in JSON format with:
{{
  "primary_category": "category name",
  "primary_confidence": 0.XX,
  "primary_justification": "brief explanation",
  "secondary_category": "category name",
  "secondary_confidence": 0.XX,
  "secondary_justification": "brief explanation"
}}
"""

# Stage 4: Extract Entities - Default prompt
EXTRACT_PROMPT_DEFAULT = """
Extract structured entities from the following financial document.
Identify and extract the following entity types with confidence scores:

Document Content:
{document_text}

Entity Types to Extract:
- person: Individual names (borrowers, account holders, beneficiaries)
- organization: Financial institutions, companies, employers
- account_number: Bank account numbers, loan account numbers, policy numbers
- ssn_tax_id: Social Security Numbers, Tax ID numbers, EIN
- amount: Monetary amounts, loan amounts, account balances, interest rates
- date: Transaction dates, due dates, maturity dates, birth dates
- address: Physical addresses, mailing addresses, property addresses
- email: Email addresses
- phone: Phone numbers, fax numbers
- credit_score: Credit scores, credit ratings
- property: Property descriptions, real estate addresses

Respond in JSON format with:
{{
  "entities": [
    {{
      "type": "entity_type",
      "value": "extracted value",
      "confidence": 0.XX
    }}
  ]
}}
"""

# Stage 5: De-identify - Default prompt
DEIDENTIFY_PROMPT_DEFAULT = """
Identify personally identifiable information (PII) and sensitive financial data in the following document that should be redacted or masked for compliance with privacy regulations (GLBA, CCPA, etc.).

Document Content:
{document_text}

PII and Sensitive Data Categories to Identify:
- person: Individual names (full names, first/last names)
- ssn: Social Security Numbers (XXX-XX-XXXX format)
- tax_id: Tax ID numbers, EIN numbers
- account_number: Bank account numbers, credit card numbers, loan account numbers
- routing_number: Bank routing numbers, ABA numbers
- drivers_license: Driver's license numbers
- email: Email addresses
- phone: Phone numbers, fax numbers
- address: Physical addresses, mailing addresses, property addresses
- date_of_birth: Dates of birth
- salary_income: Salary information, annual income, compensation details
- credit_score: Credit scores, credit ratings
- financial_account_details: Account balances, transaction details

For each PII item found, provide:
1. The type of PII
2. The value to be masked
3. The replacement strategy (REDACT, MASK, GENERALIZE)

Respond in JSON format with:
{{
  "pii_items": [
    {{
      "type": "pii_type",
      "value": "original value",
      "strategy": "REDACT|MASK|GENERALIZE",
      "replacement": "replacement value or pattern"
    }}
  ]
}}
"""

# Allow prompts to be overridden via environment variables
CATEGORIZE_PROMPT = os.environ.get("CATEGORIZE_PROMPT", CATEGORIZE_PROMPT_DEFAULT)
EXTRACT_PROMPT = os.environ.get("EXTRACT_PROMPT", EXTRACT_PROMPT_DEFAULT)
DEIDENTIFY_PROMPT = os.environ.get("DEIDENTIFY_PROMPT", DEIDENTIFY_PROMPT_DEFAULT)


# ============================================================================
# Processing Configuration
# ============================================================================

# Lakebase PostgreSQL table for status tracking
STATUS_TABLE_NAME = os.environ.get(
    "STATUS_TABLE_NAME",
    "unstructured_parsequery.file_processing_status"
)

# Lakebase PostgreSQL table for results tracking
RESULTS_TABLE_NAME = os.environ.get(
    "RESULTS_TABLE_NAME",
    "unstructured_parsequery.results"
)

# UC Volume path for pipeline logs
# LOGS_VOLUME_PATH comes from app resource (base volume path)
# We append the app name and logs subdirectory
_logs_volume_base = os.environ.get("LOGS_VOLUME_PATH")
if _logs_volume_base:
    # Append app subdirectory to the base volume path from resource
    LOGS_VOLUME_PATH = f"{_logs_volume_base}/unstructured-parsequery-app/logs"
else:
    # Fallback default
    LOGS_VOLUME_PATH = f"{VOLUME_PATH}/logs" if VOLUME_PATH else "/Volumes/catalog/schema/volume/logs"

# File processing settings
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "100"))
SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".txt", ".html", ".md"]

# AI Query settings
# AI_QUERY_ENDPOINT comes from app resource (serving endpoint name)
AI_QUERY_ENDPOINT = os.environ.get("AI_QUERY_ENDPOINT")
AI_QUERY_MODEL = AI_QUERY_ENDPOINT or os.environ.get("AI_QUERY_MODEL", "databricks-gpt-5-1")
AI_QUERY_TEMPERATURE = float(os.environ.get("AI_QUERY_TEMPERATURE", "0.0"))
AI_QUERY_MAX_TOKENS = int(os.environ.get("AI_QUERY_MAX_TOKENS", "5000"))

# Databricks serving endpoint base URL
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "https://your-workspace.cloud.databricks.com")

# Ensure DATABRICKS_HOST has https:// prefix (in Databricks Apps it might be just the hostname)
if DATABRICKS_HOST and not DATABRICKS_HOST.startswith(("http://", "https://")):
    DATABRICKS_HOST = f"https://{DATABRICKS_HOST}"

DATABRICKS_BASE_URL = f"{DATABRICKS_HOST}/serving-endpoints"

# SQL Warehouse ID for ai_parse_document execution (comes from app resource)
SQL_WAREHOUSE_ID = os.environ.get("SQL_WAREHOUSE_ID")


# ============================================================================
# Logging Configuration
# ============================================================================

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")


# ============================================================================
# Testing Configuration
# ============================================================================

# Set to a stage name to force failure at that stage for testing
# Valid values: None, "ingest", "parse", "categorize", "extract", "deidentify"
# To enable: set to stage name like "categorize"
# To disable: set to None
TEST_FORCE_FAILURE_STAGE = os.environ.get("TEST_FORCE_FAILURE_STAGE", None)


# ============================================================================
# Debug Helper
# ============================================================================

def print_config():
    """Print current configuration (for debugging)"""
    print("=" * 80)
    print("UNSTRUCTURED PARSEQUERY APP CONFIGURATION")
    print("=" * 80)
    print(f"Volume Path: {VOLUME_PATH}")
    if VOLUME_CONFIG:
        print(f"  Catalog: {VOLUME_CONFIG['catalog']}")
        print(f"  Schema: {VOLUME_CONFIG['schema']}")
        print(f"  Volume: {VOLUME_CONFIG['volume_name']}")
    print(f"\nMLflow Experiment: {MLFLOW_EXPERIMENT_NAME}")
    print(f"Status Table: {STATUS_TABLE_NAME}")
    print(f"Results Table: {RESULTS_TABLE_NAME}")
    print(f"Logs Volume Path: {LOGS_VOLUME_PATH}")
    print(f"Max File Size: {MAX_FILE_SIZE_MB} MB")
    print(f"AI Model: {AI_QUERY_MODEL}")
    print("=" * 80)
