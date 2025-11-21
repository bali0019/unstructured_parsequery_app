"""
Stage 2: Parse - Extract text and structure using AI Parse Document

Uses Databricks ai_parse_document SQL function to extract text, tables, and images.
Executes via SQL warehouse using statement execution API.
Includes MLflow tracing for observability.
"""

import mlflow
from datetime import datetime
from typing import Dict, Any
import logging
import uuid
import time
from databricks.sdk import WorkspaceClient
from config import SQL_WAREHOUSE_ID, VOLUME_CONFIG

logger = logging.getLogger(__name__)


def parse_document(
    volume_path: str
) -> Dict[str, Any]:
    """
    Parse document using Databricks ai_parse_document SQL function.

    Executes SQL statement via SQL warehouse to invoke ai_parse_document.

    Args:
        volume_path: Full path to file in UC volume (e.g., /Volumes/catalog/schema/volume/file.pdf)

    Returns:
        Dictionary with parsed content and metadata
    """
    # Use context manager to control what gets logged to trace (exclude large text content)
    with mlflow.start_span(
        name="stage_2_parse",
        span_type="PARSER",
        attributes={
            "stage": "parse",
            "parser": "ai_parse_document",
            "volume_path": volume_path,
            "sql_warehouse_id": SQL_WAREHOUSE_ID
        }
    ) as span:
        # Set inputs for trace
        span.set_inputs({
            "volume_path": volume_path,
            "parser": "ai_parse_document"
        })

        try:
            logger.info(f"Parsing document: {volume_path}")

            # Check SQL warehouse ID is configured
            if not SQL_WAREHOUSE_ID:
                raise Exception("SQL_WAREHOUSE_ID not configured. Required for ai_parse_document execution.")

            # Parse catalog, schema, volume from the volume_path
            # Format: /Volumes/catalog/schema/volume/file.pdf
            path_parts = volume_path.strip("/").split("/")
            if len(path_parts) < 4 or path_parts[0] != "Volumes":
                raise Exception(f"Invalid volume path format: {volume_path}")

            catalog = path_parts[1]
            schema = path_parts[2]
            volume = path_parts[3]

            logger.info(f"Parsed from volume path - Catalog: {catalog}, Schema: {schema}, Volume: {volume}")

            # Construct image output path (subdirectory in same volume)
            image_output_path = f"/Volumes/{catalog}/{schema}/{volume}/parsed_images/"

            # Construct SQL query with ai_parse_document and extract text
            # Use SQL to parse the JSON structure and extract just the text content
            sql_query = f"""
            WITH parsed_documents AS (
                SELECT
                  path,
                  ai_parse_document(
                    content,
                    map(
                      'imageOutputPath', '{image_output_path}',
                      'descriptionElementTypes', '*'
                    )
                  ) AS parsed
                FROM READ_FILES('{volume_path}', format => 'binaryFile')
            )
            SELECT
              path,
              concat_ws(
                '\\n\\n',
                transform(
                  try_cast(parsed:document:elements AS ARRAY<VARIANT>),
                  element -> try_cast(element:content AS STRING)
                )
              ) AS document_text,
              parsed
            FROM parsed_documents
            WHERE try_cast(parsed:error_status AS STRING) IS NULL
            """

            logger.info(f"Executing SQL with ai_parse_document on warehouse {SQL_WAREHOUSE_ID}")

            # Initialize Workspace Client (uses OAuth credentials from environment)
            w = WorkspaceClient()

            # Execute SQL statement
            stmt = w.statement_execution.execute_statement(
                statement=sql_query,
                warehouse_id=SQL_WAREHOUSE_ID,
                catalog=catalog,
                schema=schema
            )

            logger.info(f"Statement submitted: {stmt.statement_id}")

            # Add statement_id to span attributes
            span.set_attribute("statement_id", stmt.statement_id)

            # Poll until statement completes
            # Statement states: PENDING, RUNNING, SUCCEEDED, FAILED, CANCELED, CLOSED
            max_wait_time = 300  # 5 minutes max
            start_time = time.time()

            # Compare enum values, not strings
            while stmt.status.state.value in ["PENDING", "RUNNING"]:
                if time.time() - start_time > max_wait_time:
                    raise Exception(f"Statement execution timeout after {max_wait_time}s")

                time.sleep(2)  # Poll every 2 seconds
                stmt = w.statement_execution.get_statement(stmt.statement_id)
                logger.debug(f"Statement state: {stmt.status.state.value}")

            # Check for errors - compare enum values
            if stmt.status.state.value == "FAILED":
                error_msg = stmt.status.error.message if stmt.status.error and stmt.status.error.message else "Unknown error"
                logger.error(f"SQL statement failed: {error_msg}")
                if stmt.status.error:
                    logger.error(f"Error code: {stmt.status.error.error_code if hasattr(stmt.status.error, 'error_code') else 'N/A'}")
                raise Exception(f"SQL statement failed: {error_msg}")

            # Fetch results (results are in stmt.result when statement completes successfully)
            if not stmt.result or not stmt.result.data_array:
                raise Exception("No data returned from ai_parse_document")

            logger.info(f"Got {len(stmt.result.data_array)} rows from ai_parse_document")

            # Extract results from SQL query
            # SQL returns: path (column 0), document_text (column 1), parsed (column 2)
            row = stmt.result.data_array[0]  # First row
            path = row[0] if len(row) > 0 else volume_path
            document_text = row[1] if len(row) > 1 else None
            parsed_doc = row[2] if len(row) > 2 else None

            if not document_text:
                raise Exception("No document text extracted from ai_parse_document")

            logger.info(f"Extracted {len(document_text)} characters of text from document")

            # Parse the full ai_parse_document output for metadata (if needed)
            import json as json_module
            parsed_doc_obj = None
            if parsed_doc:
                if isinstance(parsed_doc, str):
                    parsed_doc_obj = json_module.loads(parsed_doc)
                else:
                    parsed_doc_obj = parsed_doc

            # Create pages structure expected by downstream stages (simple format with just text)
            # For simplicity, treat the entire document as a single page
            pages = [{"text": document_text, "page_id": 0}]

            # Construct response with extracted text and metadata (don't log large text to trace)
            parsed_result = {
                "status": "success",
                "volume_path": volume_path,
                "document_text": document_text,  # Extracted text (ready for downstream stages)
                "pages": pages,  # Pages format expected by categorize/extract/deidentify
                "parsed_doc": parsed_doc_obj,  # Full ai_parse_document output for reference
                "image_output_path": image_output_path,
                "statement_id": stmt.statement_id,
                "timestamp": datetime.now().isoformat()
            }

            # Set outputs for trace (include text sample for lineage, not full text)
            text_sample = document_text[:500] + "..." if len(document_text) > 500 else document_text
            span.set_outputs({
                "status": "success",
                "text_length": len(document_text),
                "text_sample": text_sample,
                "image_output_path": image_output_path,
                "pages_count": len(pages)
            })

            logger.info(f"Parse successful via ai_parse_document. Statement: {stmt.statement_id}")
            return parsed_result

        except Exception as e:
            logger.error(f"Parse failed: {str(e)}", exc_info=True)

            # Set error output for trace
            span.set_outputs({
                "error": str(e)
            })

            return {
                "status": "failed",
                "volume_path": volume_path,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
