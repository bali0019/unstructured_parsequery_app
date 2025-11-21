"""
Unstructured ParseQuery Backend with MLflow Tracing

Orchestrates the 5-stage document processing pipeline with file-level MLflow tracing.
Uses mlflow.start_span context managers - parent trace contains 5 child spans (one per stage).
"""

import time
import uuid
from datetime import datetime
from typing import Dict, Any
import mlflow
import os
import logging

# Import stage modules
from stages import (
    ingest_file,
    parse_document,
    categorize_document,
    extract_entities,
    deidentify_document
)

# Import storage
from storage import ProcessingStatusTable, ResultsTable

# Import config
from config import MLFLOW_EXPERIMENT_NAME, VOLUME_CONFIG, print_config, TEST_FORCE_FAILURE_STAGE

# Import logging utilities
from utils.uc_logger import setup_pipeline_logging, cleanup_pipeline_logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug: Print when backend module is loaded
logger.info(f"[BACKEND] Loading backend.py module")
logger.info(f"[BACKEND] MLFLOW_TRACKING_URI env: {os.environ.get('MLFLOW_TRACKING_URI', 'NOT SET')}")

# Set tracking URI - use "databricks" when running inside Databricks workspace
mlflow.set_tracking_uri("databricks")
logger.info(f"[BACKEND] MLflow tracking URI: {mlflow.get_tracking_uri()}")
logger.info(f"[BACKEND] MLflow version: {mlflow.__version__}")

# Set experiment at module level (same pattern that works in notebooks)
exp = mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
logger.info(f"[BACKEND] Experiment set: {exp.name} (ID: {exp.experiment_id})")

# Print configuration
print_config()

# Lazy initialization for tables (avoid blocking on import)
_status_table = None
_results_table = None


def _check_test_failure(stage_name: str):
    """Check if we should force a failure for testing purposes"""
    if TEST_FORCE_FAILURE_STAGE and TEST_FORCE_FAILURE_STAGE.lower() == stage_name.lower():
        logger.error(f"TEST FAILURE: Forcing failure at {stage_name} stage")
        raise Exception(f"TEST FAILURE: Forced failure at {stage_name} stage for testing")

def get_status_table():
    """Get status table with lazy initialization"""
    global _status_table
    if _status_table is None:
        try:
            _status_table = ProcessingStatusTable()
            logger.info("[BACKEND] Status table initialized")
        except Exception as e:
            logger.warning(f"[BACKEND] Could not initialize status table: {e}")
    return _status_table

def get_results_table():
    """Get results table with lazy initialization"""
    global _results_table
    if _results_table is None:
        try:
            _results_table = ResultsTable()
            logger.info("[BACKEND] Results table initialized")
        except Exception as e:
            logger.warning(f"[BACKEND] Could not initialize results table: {e}")
    return _results_table


def create_initial_file_record(filename: str) -> str:
    """
    Create initial file record in status table before processing starts.

    This allows the UI to show the file in the table immediately.

    Args:
        filename: Original filename

    Returns:
        The generated file_id (pipeline_id)
    """
    pipeline_id = str(uuid.uuid4())

    status_table = get_status_table()
    if status_table:
        try:
            status_table.insert_file_record(
                file_id=pipeline_id,
                filename=filename
            )
            # Update to show "processing" status immediately
            status_table.update_file_status(
                file_id=pipeline_id,
                status="processing",
                current_stage="ingest"
            )
            logger.info(f"[BACKEND] Created initial record for {filename} with id {pipeline_id}")
        except Exception as e:
            logger.warning(f"Could not create initial record: {e}")

    return pipeline_id


def process_file_through_pipeline(
    file_bytes: bytes,
    filename: str,
    file_id: str = None,
    on_stage_update: callable = None
) -> Dict[str, Any]:
    """
    Process a single file through the entire 5-stage pipeline.

    This parent trace contains 5 child spans (one per stage).
    Uses mlflow.trace context manager to control what gets logged (metadata only, not bytes).

    Uses OAuth credentials automatically injected by Databricks Apps for authentication.

    Args:
        file_bytes: File content as bytes (NOT logged to trace)
        filename: Original filename
        file_id: Optional pre-created file ID (if provided, skips initial record creation)
        on_stage_update: Optional callback function called after each stage status update

    Returns:
        Dictionary with pipeline results and status
    """
    # Use passed file_id or generate a new one
    pipeline_id = file_id if file_id else str(uuid.uuid4())
    file_size_bytes = len(file_bytes)

    # Set up per-pipeline logging to UC Volume
    uc_log_handler = setup_pipeline_logging(pipeline_id)
    logger.info(f"[PIPELINE {pipeline_id}] Starting pipeline for {filename}")
    if uc_log_handler and uc_log_handler.file_path:
        logger.info(f"[PIPELINE {pipeline_id}] Logs will be written to: {uc_log_handler.file_path}")

    # Start trace with metadata only (no file_bytes)
    with mlflow.start_span(
        name="process_file_pipeline",
        span_type="CHAIN",
        attributes={
            "pipeline_version": "2.0",
            "pipeline_id": pipeline_id,
            "filename": filename,
            "file_size_bytes": file_size_bytes
        }
    ) as span:
        # Set inputs for the trace (shows in MLflow UI request column)
        span.set_inputs({
            "filename": filename,
            "file_size_bytes": file_size_bytes,
            "pipeline_id": pipeline_id
        })

        # Add log file path as attribute for end-to-end lineage
        if uc_log_handler and uc_log_handler.file_path:
            span.set_attribute("log_file_path", uc_log_handler.file_path)

        pipeline_start = time.time()

        results = {
            "pipeline_id": pipeline_id,
            "filename": filename,
            "file_size_bytes": file_size_bytes,
            "start_time": datetime.now().isoformat(),
            "stages": {}
        }

        # Insert initial status record (only if file_id was not pre-created)
        status_table = get_status_table()
        if status_table and not file_id:
            try:
                status_table.insert_file_record(
                    file_id=pipeline_id,
                    filename=filename
                )
            except Exception as e:
                logger.warning(f"Could not insert status record: {e}")

        try:
            # Check volume configuration
            if not VOLUME_CONFIG:
                raise Exception("Volume not configured. Set VOLUME_PATH environment variable.")

            catalog = VOLUME_CONFIG["catalog"]
            schema = VOLUME_CONFIG["schema"]
            volume_name = VOLUME_CONFIG["volume_name"]

            # Stage 1: Ingest - Upload to UC Volume
            # User token is automatically retrieved from Streamlit context
            logger.info(f"[PIPELINE {pipeline_id}] Stage 1: Ingest")
            ingest_result = ingest_file(
                file_bytes=file_bytes,
                filename=filename,
                catalog=catalog,
                schema=schema,
                volume_name=volume_name
            )
            results["stages"]["ingest"] = ingest_result
            _check_test_failure("ingest")

            if ingest_result.get("status") != "success":
                raise Exception(f"Ingest failed: {ingest_result.get('error')}")

            volume_path = ingest_result["volume_path"]

            # Update status with trace info
            # Get trace ID from the current span
            trace_id = None
            if hasattr(span, 'request_id'):
                trace_id = span.request_id
            elif hasattr(span, 'span_id'):
                trace_id = span.span_id

            logger.info(f"[PIPELINE {pipeline_id}] Capturing trace_id: {trace_id}, experiment_id: {exp.experiment_id}")

            # Get log file path for storage
            log_file_path = uc_log_handler.file_path if uc_log_handler else None

            if status_table:
                try:
                    status_table.update_file_status(
                        file_id=pipeline_id,
                        volume_path=volume_path,
                        status="processing",
                        current_stage="parse",
                        stage_ingest_status="completed",
                        trace_id=trace_id,
                        experiment_id=exp.experiment_id,
                        log_file_path=log_file_path
                    )
                    if on_stage_update:
                        on_stage_update()
                except Exception as e:
                    logger.warning(f"Could not update status: {e}")

            # Create result record after ingest (initial row for all stage results)
            results_table = get_results_table()
            if results_table:
                try:
                    results_table.create_result_record(
                        file_id=pipeline_id,
                        trace_id=trace_id,
                        experiment_id=exp.experiment_id,
                        source_volume_path=volume_path
                    )
                except Exception as e:
                    logger.warning(f"Could not create result record: {e}")

            # Stage 2: Parse - Extract text and structure
            logger.info(f"[PIPELINE {pipeline_id}] Stage 2: Parse")
            parse_result = parse_document(volume_path=volume_path)
            results["stages"]["parse"] = parse_result
            _check_test_failure("parse")

            if parse_result.get("status") != "success":
                raise Exception(f"Parse failed: {parse_result.get('error')}")

            # Update status
            if status_table:
                try:
                    status_table.update_file_status(
                        file_id=pipeline_id,
                        current_stage="categorize",
                        stage_parse_status="completed"
                    )
                    if on_stage_update:
                        on_stage_update()
                except Exception as e:
                    logger.warning(f"Could not update status: {e}")

            # Store parse result
            if results_table:
                try:
                    results_table.update_stage_result(
                        file_id=pipeline_id,
                        stage_name="parse",
                        result_data=parse_result
                    )
                except Exception as e:
                    logger.warning(f"Could not store parse result: {e}")

            # Stage 3: Categorize - Classify document
            logger.info(f"[PIPELINE {pipeline_id}] Stage 3: Categorize")
            categorize_result = categorize_document(parsed_data=parse_result)
            results["stages"]["categorize"] = categorize_result
            _check_test_failure("categorize")

            if categorize_result.get("status") != "success":
                raise Exception(f"Categorize failed: {categorize_result.get('error')}")

            # Update status with primary_category
            primary_category = categorize_result.get("categorization", {}).get("primary_category")
            if status_table:
                try:
                    status_table.update_file_status(
                        file_id=pipeline_id,
                        current_stage="extract",
                        stage_categorize_status="completed",
                        primary_category=primary_category
                    )
                    if on_stage_update:
                        on_stage_update()
                except Exception as e:
                    logger.warning(f"Could not update status: {e}")

            # Store categorize result
            if results_table:
                try:
                    results_table.update_stage_result(
                        file_id=pipeline_id,
                        stage_name="categorize",
                        result_data=categorize_result
                    )
                except Exception as e:
                    logger.warning(f"Could not store categorize result: {e}")

            # Stage 4: Extract - Extract entities
            logger.info(f"[PIPELINE {pipeline_id}] Stage 4: Extract")
            extract_result = extract_entities(categorized_data=categorize_result)
            results["stages"]["extract"] = extract_result
            _check_test_failure("extract")

            if extract_result.get("status") != "success":
                raise Exception(f"Extract failed: {extract_result.get('error')}")

            # Update status
            if status_table:
                try:
                    status_table.update_file_status(
                        file_id=pipeline_id,
                        current_stage="deidentify",
                        stage_extract_status="completed"
                    )
                    if on_stage_update:
                        on_stage_update()
                except Exception as e:
                    logger.warning(f"Could not update status: {e}")

            # Store extract result
            if results_table:
                try:
                    results_table.update_stage_result(
                        file_id=pipeline_id,
                        stage_name="extract",
                        result_data=extract_result
                    )
                except Exception as e:
                    logger.warning(f"Could not store extract result: {e}")

            # Stage 5: De-identify - Remove PII
            logger.info(f"[PIPELINE {pipeline_id}] Stage 5: De-identify")
            deidentify_result = deidentify_document(extracted_data=extract_result)
            results["stages"]["deidentify"] = deidentify_result
            _check_test_failure("deidentify")

            if deidentify_result.get("status") != "success":
                raise Exception(f"De-identify failed: {deidentify_result.get('error')}")

            # Store deidentify result
            if results_table:
                try:
                    results_table.update_stage_result(
                        file_id=pipeline_id,
                        stage_name="deidentify",
                        result_data=deidentify_result
                    )
                except Exception as e:
                    logger.warning(f"Could not store deidentify result: {e}")

            # Pipeline completion
            results["status"] = "completed"
            results["end_time"] = datetime.now().isoformat()
            results["total_time_seconds"] = time.time() - pipeline_start
            results["stages_completed"] = 5

            # Mark as completed in status table
            if status_table:
                try:
                    primary_category = categorize_result.get("categorization", {}).get("primary_category")
                    entities_count = extract_result.get("entities_count", 0)
                    pii_items_masked = deidentify_result.get("pii_items_masked", 0)

                    status_table.mark_completed(
                        file_id=pipeline_id,
                        primary_category=primary_category,
                        entities_count=entities_count,
                        pii_items_masked=pii_items_masked
                    )
                    if on_stage_update:
                        on_stage_update()
                except Exception as e:
                    logger.warning(f"Could not mark as completed: {e}")

            logger.info(f"[PIPELINE {pipeline_id}] Completed successfully in {results['total_time_seconds']:.2f}s")

            # Set outputs for the trace (shows in MLflow UI response column)
            span.set_outputs({
                "status": results["status"],
                "stages_completed": results["stages_completed"],
                "total_time_seconds": results["total_time_seconds"],
                "primary_category": primary_category,
                "entities_count": entities_count,
                "pii_items_masked": pii_items_masked
            })

        except Exception as e:
            logger.error(f"[PIPELINE {pipeline_id}] Failed: {str(e)}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)
            results["end_time"] = datetime.now().isoformat()
            results["total_time_seconds"] = time.time() - pipeline_start

            # Mark as failed in status table
            if status_table:
                try:
                    current_stage = results.get("stages", {}).keys()
                    current_stage = list(current_stage)[-1] if current_stage else "ingest"
                    status_table.mark_failed(
                        file_id=pipeline_id,
                        error_message=str(e),
                        current_stage=current_stage
                    )
                except Exception as status_err:
                    logger.warning(f"Could not mark as failed: {status_err}")

            # Set outputs for the trace (shows error in MLflow UI response column)
            span.set_outputs({
                "status": results["status"],
                "error": results.get("error", "")
            })

        # Clean up pipeline logging (flush logs to UC Volume)
        cleanup_pipeline_logging(uc_log_handler)

        return results


def get_processing_status(file_id: str = None, limit: int = 100) -> Dict[str, Any]:
    """
    Get processing status from Lakebase PostgreSQL table

    Args:
        file_id: Specific file ID (returns all if None)
        limit: Maximum records to return

    Returns:
        Dictionary with status records
    """
    status_table = get_status_table()
    logger.info(f"get_processing_status called with file_id={file_id}, limit={limit}")
    logger.info(f"status_table exists: {status_table is not None}")

    if not status_table:
        logger.warning("Status table not available")
        return {"error": "Status table not available"}

    try:
        if file_id:
            logger.info(f"Fetching status for specific file: {file_id}")
            record = status_table.get_file_status(file_id)
            result = {"file": record} if record else {"error": "File not found"}
            logger.info(f"Returning single file result: {result}")
            return result
        else:
            logger.info(f"Fetching all files with limit={limit}")
            records = status_table.get_all_files(limit=limit)
            logger.info(f"get_all_files returned {len(records)} records")
            if records:
                logger.info(f"First record from get_all_files: {records[0]}")
            result = {"files": records, "count": len(records)}
            logger.info(f"Returning result with {result['count']} files")
            return result
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}", exc_info=True)
        return {"error": str(e)}


def reprocess_file(file_id: str, volume_path: str, filename: str, on_stage_update: callable = None) -> Dict[str, Any]:
    """
    Reprocess a failed file through stages 2-5 (parse, categorize, extract, deidentify).

    Skips stage 1 (ingest) since file is already in UC Volume.
    Creates new trace and log file, updates the existing Delta table record.

    Args:
        file_id: Existing file ID to update
        volume_path: Path to file in UC Volume
        filename: Original filename
        on_stage_update: Optional callback function called after each stage status update

    Returns:
        Dictionary with pipeline results and status
    """
    # Use a new pipeline_id for new trace/logs, but update same file_id in Delta
    pipeline_id = str(uuid.uuid4())

    # Set up per-pipeline logging to UC Volume
    uc_log_handler = setup_pipeline_logging(pipeline_id)
    logger.info(f"[REPROCESS {file_id}] Starting reprocess for {filename}")
    logger.info(f"[REPROCESS {file_id}] New pipeline_id for trace/logs: {pipeline_id}")
    if uc_log_handler and uc_log_handler.file_path:
        logger.info(f"[REPROCESS {file_id}] Logs will be written to: {uc_log_handler.file_path}")

    # Start trace with metadata
    with mlflow.start_span(
        name="reprocess_file_pipeline",
        span_type="CHAIN",
        attributes={
            "pipeline_version": "2.0",
            "pipeline_id": pipeline_id,
            "file_id": file_id,
            "filename": filename,
            "volume_path": volume_path,
            "is_reprocess": True
        }
    ) as span:
        # Set inputs for the trace
        span.set_inputs({
            "filename": filename,
            "volume_path": volume_path,
            "file_id": file_id,
            "is_reprocess": True
        })

        # Add log file path as attribute
        if uc_log_handler and uc_log_handler.file_path:
            span.set_attribute("log_file_path", uc_log_handler.file_path)

        pipeline_start = time.time()

        results = {
            "file_id": file_id,
            "pipeline_id": pipeline_id,
            "filename": filename,
            "volume_path": volume_path,
            "start_time": datetime.now().isoformat(),
            "stages": {},
            "is_reprocess": True
        }

        # Get status table
        status_table = get_status_table()
        results_table = get_results_table()

        # Get trace ID from the current span
        trace_id = None
        if hasattr(span, 'request_id'):
            trace_id = span.request_id
        elif hasattr(span, 'span_id'):
            trace_id = span.span_id

        logger.info(f"[REPROCESS {file_id}] New trace_id: {trace_id}, experiment_id: {exp.experiment_id}")

        # Get log file path for storage
        log_file_path = uc_log_handler.file_path if uc_log_handler else None
        logger.info(f"[REPROCESS {file_id}] New log file path: {log_file_path}")

        # Update status to processing with new trace info
        if status_table:
            try:
                status_table.update_file_status(
                    file_id=file_id,
                    status="processing",
                    current_stage="parse",
                    trace_id=trace_id,
                    experiment_id=exp.experiment_id,
                    log_file_path=log_file_path,
                    error_message=None,  # Clear previous error
                    stage_parse_status=None,
                    stage_categorize_status=None,
                    stage_extract_status=None,
                    stage_deidentify_status=None
                )
                if on_stage_update:
                    on_stage_update()
            except Exception as e:
                logger.warning(f"Could not update status: {e}")

        try:
            # Stage 2: Parse - Extract text and structure
            logger.info(f"[REPROCESS {file_id}] Stage 2: Parse")
            parse_result = parse_document(volume_path=volume_path)
            results["stages"]["parse"] = parse_result
            _check_test_failure("parse")

            if parse_result.get("status") != "success":
                raise Exception(f"Parse failed: {parse_result.get('error')}")

            # Update status
            if status_table:
                try:
                    status_table.update_file_status(
                        file_id=file_id,
                        current_stage="categorize",
                        stage_parse_status="completed"
                    )
                    if on_stage_update:
                        on_stage_update()
                except Exception as e:
                    logger.warning(f"Could not update status: {e}")

            # Store parse result
            if results_table:
                try:
                    results_table.update_stage_result(
                        file_id=file_id,
                        stage_name="parse",
                        result_data=parse_result
                    )
                except Exception as e:
                    logger.warning(f"Could not store parse result: {e}")

            # Stage 3: Categorize - Classify document
            logger.info(f"[REPROCESS {file_id}] Stage 3: Categorize")
            categorize_result = categorize_document(parsed_data=parse_result)
            results["stages"]["categorize"] = categorize_result
            _check_test_failure("categorize")

            if categorize_result.get("status") != "success":
                raise Exception(f"Categorize failed: {categorize_result.get('error')}")

            # Update status with primary_category
            primary_category = categorize_result.get("categorization", {}).get("primary_category")
            if status_table:
                try:
                    status_table.update_file_status(
                        file_id=file_id,
                        current_stage="extract",
                        stage_categorize_status="completed",
                        primary_category=primary_category
                    )
                    if on_stage_update:
                        on_stage_update()
                except Exception as e:
                    logger.warning(f"Could not update status: {e}")

            # Store categorize result
            if results_table:
                try:
                    results_table.update_stage_result(
                        file_id=file_id,
                        stage_name="categorize",
                        result_data=categorize_result
                    )
                except Exception as e:
                    logger.warning(f"Could not store categorize result: {e}")

            # Stage 4: Extract - Extract entities
            logger.info(f"[REPROCESS {file_id}] Stage 4: Extract")
            extract_result = extract_entities(categorized_data=categorize_result)
            results["stages"]["extract"] = extract_result
            _check_test_failure("extract")

            if extract_result.get("status") != "success":
                raise Exception(f"Extract failed: {extract_result.get('error')}")

            # Update status
            if status_table:
                try:
                    status_table.update_file_status(
                        file_id=file_id,
                        current_stage="deidentify",
                        stage_extract_status="completed"
                    )
                    if on_stage_update:
                        on_stage_update()
                except Exception as e:
                    logger.warning(f"Could not update status: {e}")

            # Store extract result
            if results_table:
                try:
                    results_table.update_stage_result(
                        file_id=file_id,
                        stage_name="extract",
                        result_data=extract_result
                    )
                except Exception as e:
                    logger.warning(f"Could not store extract result: {e}")

            # Stage 5: De-identify - Remove PII
            logger.info(f"[REPROCESS {file_id}] Stage 5: De-identify")
            deidentify_result = deidentify_document(extracted_data=extract_result)
            results["stages"]["deidentify"] = deidentify_result
            _check_test_failure("deidentify")

            if deidentify_result.get("status") != "success":
                raise Exception(f"De-identify failed: {deidentify_result.get('error')}")

            # Store deidentify result
            if results_table:
                try:
                    results_table.update_stage_result(
                        file_id=file_id,
                        stage_name="deidentify",
                        result_data=deidentify_result
                    )
                except Exception as e:
                    logger.warning(f"Could not store deidentify result: {e}")

            # Pipeline completion
            results["status"] = "completed"
            results["end_time"] = datetime.now().isoformat()
            results["total_time_seconds"] = time.time() - pipeline_start
            results["stages_completed"] = 4  # Stages 2-5

            # Mark as completed in status table
            if status_table:
                try:
                    primary_category = categorize_result.get("categorization", {}).get("primary_category")
                    entities_count = extract_result.get("entities_count", 0)
                    pii_items_masked = deidentify_result.get("pii_items_masked", 0)

                    status_table.mark_completed(
                        file_id=file_id,
                        primary_category=primary_category,
                        entities_count=entities_count,
                        pii_items_masked=pii_items_masked
                    )
                    if on_stage_update:
                        on_stage_update()
                except Exception as e:
                    logger.warning(f"Could not mark as completed: {e}")

            logger.info(f"[REPROCESS {file_id}] Completed successfully in {results['total_time_seconds']:.2f}s")

            # Set outputs for the trace
            span.set_outputs({
                "status": results["status"],
                "stages_completed": results["stages_completed"],
                "total_time_seconds": results["total_time_seconds"],
                "primary_category": primary_category,
                "entities_count": entities_count,
                "pii_items_masked": pii_items_masked
            })

        except Exception as e:
            logger.error(f"[REPROCESS {file_id}] Failed: {str(e)}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)
            results["end_time"] = datetime.now().isoformat()
            results["total_time_seconds"] = time.time() - pipeline_start

            # Mark as failed in status table
            if status_table:
                try:
                    current_stage = results.get("stages", {}).keys()
                    current_stage = list(current_stage)[-1] if current_stage else "parse"
                    status_table.mark_failed(
                        file_id=file_id,
                        error_message=str(e),
                        current_stage=current_stage
                    )
                except Exception as status_err:
                    logger.warning(f"Could not mark as failed: {status_err}")

            # Set outputs for the trace
            span.set_outputs({
                "status": results["status"],
                "error": results.get("error", "")
            })

        # Clean up pipeline logging
        cleanup_pipeline_logging(uc_log_handler)

        return results


def reset_storage():
    """Clear all storage and logs (for testing)"""
    logger.info("Reset storage requested")
    # Note: Lakebase table data is persistent, this is a no-op
    # In production, you might want to implement archival instead of deletion
    pass


def get_file_results(file_id: str) -> Dict[str, Any]:
    """
    Get pipeline results for a specific file from results table.

    Args:
        file_id: File identifier

    Returns:
        Dictionary with results including deidentify_result
    """
    results_table = get_results_table()

    if not results_table:
        logger.warning("Results table not available")
        return {"error": "Results table not available"}

    try:
        results = results_table.get_results(file_id)
        if results:
            return {"results": results}
        else:
            return {"error": "No results found for file"}
    except Exception as e:
        logger.error(f"Error getting file results: {str(e)}", exc_info=True)
        return {"error": str(e)}
