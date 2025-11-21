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
    on_stage_update: callable = None,
    on_stage_status: callable = None
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
        on_stage_status: Optional callback(stage_name, status) for UI stage tile updates

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
            if on_stage_status:
                on_stage_status("ingest", "processing")
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
                if on_stage_status:
                    on_stage_status("ingest", "error")
                raise Exception(f"Ingest failed: {ingest_result.get('error')}")

            if on_stage_status:
                on_stage_status("ingest", "success")

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
            if on_stage_status:
                on_stage_status("parse", "processing")
            parse_result = parse_document(volume_path=volume_path)
            results["stages"]["parse"] = parse_result
            _check_test_failure("parse")

            if parse_result.get("status") != "success":
                if on_stage_status:
                    on_stage_status("parse", "error")
                raise Exception(f"Parse failed: {parse_result.get('error')}")

            if on_stage_status:
                on_stage_status("parse", "success")

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
            if on_stage_status:
                on_stage_status("categorize", "processing")
            categorize_result = categorize_document(parsed_data=parse_result)
            results["stages"]["categorize"] = categorize_result
            _check_test_failure("categorize")

            if categorize_result.get("status") != "success":
                if on_stage_status:
                    on_stage_status("categorize", "error")
                raise Exception(f"Categorize failed: {categorize_result.get('error')}")

            if on_stage_status:
                on_stage_status("categorize", "success")

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
            if on_stage_status:
                on_stage_status("extract", "processing")
            extract_result = extract_entities(categorized_data=categorize_result)
            results["stages"]["extract"] = extract_result
            _check_test_failure("extract")

            if extract_result.get("status") != "success":
                if on_stage_status:
                    on_stage_status("extract", "error")
                raise Exception(f"Extract failed: {extract_result.get('error')}")

            if on_stage_status:
                on_stage_status("extract", "success")

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
            if on_stage_status:
                on_stage_status("deidentify", "processing")
            deidentify_result = deidentify_document(extracted_data=extract_result)
            results["stages"]["deidentify"] = deidentify_result
            _check_test_failure("deidentify")

            if deidentify_result.get("status") != "success":
                if on_stage_status:
                    on_stage_status("deidentify", "error")
                raise Exception(f"De-identify failed: {deidentify_result.get('error')}")

            if on_stage_status:
                on_stage_status("deidentify", "success")

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

            # Determine the failed stage
            completed_stages = list(results.get("stages", {}).keys())
            failed_stage = completed_stages[-1] if completed_stages else "ingest"

            # Determine which stages completed successfully
            stages_completed = len(completed_stages)

            # Mark as failed in status table
            if status_table:
                try:
                    status_table.mark_failed(
                        file_id=pipeline_id,
                        error_message=str(e),
                        current_stage=failed_stage
                    )
                except Exception as status_err:
                    logger.warning(f"Could not mark as failed: {status_err}")

            # Set failure attributes on the span for easy filtering/searching in MLflow
            span.set_attribute("pipeline_status", "failed")
            span.set_attribute("failed_at_stage", failed_stage)
            span.set_attribute("stages_completed", stages_completed)
            span.set_attribute("error_message", str(e)[:500])  # Truncate long errors

            # Set outputs for the trace (shows error in MLflow UI response column)
            span.set_outputs({
                "status": "failed",
                "failed_at_stage": failed_stage,
                "stages_completed": stages_completed,
                "error": str(e),
                "total_time_seconds": results["total_time_seconds"]
            })

        # Flush any pending async trace logging to ensure trace is persisted
        # This is critical for failed traces which may not auto-flush
        try:
            mlflow.flush_trace_async_logging()
            logger.info(f"[PIPELINE {pipeline_id}] Flushed trace to MLflow")
        except Exception as flush_err:
            logger.warning(f"[PIPELINE {pipeline_id}] Could not flush trace: {flush_err}")

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


def reprocess_file(file_id: str, volume_path: str, filename: str, failed_stage: str = None, on_stage_update: callable = None) -> Dict[str, Any]:
    """
    Reprocess a failed file, resuming from the failed stage.

    Skips stage 1 (ingest) since file is already in UC Volume.
    If failed_stage is provided, loads results for completed stages and resumes from that stage.
    Creates new trace and log file, updates the existing Delta table record.

    Args:
        file_id: Existing file ID to update
        volume_path: Path to file in UC Volume
        filename: Original filename
        failed_stage: Stage where processing failed (parse, categorize, extract, deidentify)
        on_stage_update: Optional callback function called after each stage status update

    Returns:
        Dictionary with pipeline results and status
    """
    # Define stage order for determining which stages to skip
    stage_order = ["parse", "categorize", "extract", "deidentify"]

    # If volume_path is empty, try to fetch it from the database
    if not volume_path:
        status_table = get_status_table()
        if status_table:
            try:
                file_record = status_table.get_file_status(file_id)
                if file_record and file_record.get("volume_path"):
                    volume_path = file_record["volume_path"]
                    logger.info(f"[REPROCESS {file_id}] Fetched volume_path from database: {volume_path}")
            except Exception as e:
                logger.warning(f"[REPROCESS {file_id}] Could not fetch volume_path from database: {e}")

        # If still empty, return error
        if not volume_path:
            return {
                "file_id": file_id,
                "filename": filename,
                "status": "failed",
                "error": "Cannot reprocess: volume_path is empty and not found in database"
            }

    # Determine starting stage
    if failed_stage and failed_stage.lower() in stage_order:
        start_stage = failed_stage.lower()
        start_index = stage_order.index(start_stage)
    else:
        # Default to starting from parse if no failed_stage or invalid
        start_stage = "parse"
        start_index = 0

    # Use a new pipeline_id for new trace/logs, but update same file_id in Delta
    pipeline_id = str(uuid.uuid4())

    # Set up per-pipeline logging to UC Volume
    uc_log_handler = setup_pipeline_logging(pipeline_id)
    logger.info(f"[REPROCESS {file_id}] Starting reprocess for {filename}")
    logger.info(f"[REPROCESS {file_id}] Resuming from stage: {start_stage}")
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
            "is_reprocess": True,
            "resume_from_stage": start_stage
        }
    ) as span:
        # Set inputs for the trace
        span.set_inputs({
            "filename": filename,
            "volume_path": volume_path,
            "file_id": file_id,
            "is_reprocess": True,
            "resume_from_stage": start_stage
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

        # Handle trace IDs based on whether this is a resume or fresh reprocess
        # Resume (has failed_stage): append trace IDs to keep history of cached stage reuse
        # Reprocess (no failed_stage): replace trace ID since it's a fresh run from scratch
        existing_trace_ids = ""
        if status_table:
            try:
                file_record = status_table.get_file_status(file_id)
                if file_record:
                    existing_trace_ids = file_record.get("trace_id", "") or ""
                    logger.info(f"[REPROCESS {file_id}] Existing trace_ids: {existing_trace_ids}")
            except Exception as e:
                logger.warning(f"[REPROCESS {file_id}] Could not get existing trace_ids: {e}")

        # Determine if this is a resume (uses cached results) or fresh reprocess
        is_resume = start_index > 0  # If skipping stages, it's a resume

        if is_resume and existing_trace_ids and trace_id:
            # Resume: append new trace_id to existing ones (comma-separated)
            combined_trace_ids = f"{existing_trace_ids},{trace_id}"
            logger.info(f"[REPROCESS {file_id}] Resume mode - appending trace_id")
        else:
            # Fresh reprocess: replace with just the new trace_id
            combined_trace_ids = trace_id
            logger.info(f"[REPROCESS {file_id}] Fresh reprocess mode - replacing trace_id")

        logger.info(f"[REPROCESS {file_id}] Combined trace_ids: {combined_trace_ids}")

        # Add previous trace_ids as attribute to the new trace span for reference (only for resume)
        if is_resume and existing_trace_ids:
            span.set_attribute("previous_trace_ids", existing_trace_ids)

        # Get log file path for storage
        log_file_path = uc_log_handler.file_path if uc_log_handler else None
        logger.info(f"[REPROCESS {file_id}] New log file path: {log_file_path}")

        # Load previous stage results if resuming from a later stage
        parse_result = None
        categorize_result = None
        extract_result = None

        if start_index > 0 and results_table:
            # Load results from previous stages
            try:
                import json
                prev_results = results_table.get_results(file_id)
                if prev_results:
                    # Results are stored as JSON strings, need to parse them
                    if start_index > 0 and prev_results.get("parse_result"):
                        parse_result_str = prev_results["parse_result"]
                        if isinstance(parse_result_str, str):
                            parse_result = json.loads(parse_result_str)
                        else:
                            parse_result = parse_result_str
                        results["stages"]["parse"] = parse_result
                        logger.info(f"[REPROCESS {file_id}] Loaded previous parse result")
                    if start_index > 1 and prev_results.get("categorize_result"):
                        categorize_result_str = prev_results["categorize_result"]
                        if isinstance(categorize_result_str, str):
                            categorize_result = json.loads(categorize_result_str)
                        else:
                            categorize_result = categorize_result_str
                        results["stages"]["categorize"] = categorize_result
                        logger.info(f"[REPROCESS {file_id}] Loaded previous categorize result")
                    if start_index > 2 and prev_results.get("extract_result"):
                        extract_result_str = prev_results["extract_result"]
                        if isinstance(extract_result_str, str):
                            extract_result = json.loads(extract_result_str)
                        else:
                            extract_result = extract_result_str
                        results["stages"]["extract"] = extract_result
                        logger.info(f"[REPROCESS {file_id}] Loaded previous extract result")
            except Exception as e:
                logger.warning(f"Could not load previous results, will run from start: {e}")
                start_index = 0
                start_stage = "parse"

        # Replay cached stages as trace spans (for complete MLflow trace)
        # This creates trace spans for stages that were completed in previous runs
        if start_index > 0:
            logger.info(f"[REPROCESS {file_id}] Replaying {start_index} cached stages as trace spans")

            # Replay Stage 2: Parse (if we're skipping it)
            if start_index > 0 and parse_result:
                with mlflow.start_span(
                    name="stage_2_parse",
                    span_type="PARSER",
                    attributes={
                        "stage": "parse",
                        "is_cached_replay": True,
                        "original_file_id": file_id
                    }
                ) as parse_span:
                    # Log cached inputs
                    parse_span.set_inputs({
                        "volume_path": volume_path,
                        "is_cached_replay": True,
                        "note": "Replayed from previous run"
                    })

                    # Log cached outputs
                    text_length = len(parse_result.get("pages", [{}])[0].get("text", "")) if parse_result.get("pages") else 0
                    pages_count = len(parse_result.get("pages", []))
                    parse_span.set_outputs({
                        "status": parse_result.get("status", "success"),
                        "text_length": text_length,
                        "pages_count": pages_count,
                        "is_cached_replay": True
                    })
                    parse_span.set_attribute("text_length", text_length)
                    parse_span.set_attribute("pages_count", pages_count)
                    logger.info(f"[REPROCESS {file_id}] Replayed parse span (cached)")

            # Replay Stage 3: Categorize (if we're skipping it)
            if start_index > 1 and categorize_result:
                with mlflow.start_span(
                    name="stage_3_categorize",
                    span_type="LLM",
                    attributes={
                        "stage": "categorize",
                        "is_cached_replay": True,
                        "original_file_id": file_id
                    }
                ) as categorize_span:
                    # Log cached inputs
                    categorize_span.set_inputs({
                        "document_text_length": len(parse_result.get("pages", [{}])[0].get("text", "")) if parse_result and parse_result.get("pages") else 0,
                        "is_cached_replay": True,
                        "note": "Replayed from previous run"
                    })

                    # Log cached outputs
                    categorization = categorize_result.get("categorization", {})
                    categorize_span.set_outputs({
                        "status": categorize_result.get("status", "success"),
                        "primary_category": categorization.get("primary_category", ""),
                        "confidence": categorization.get("confidence", 0),
                        "is_cached_replay": True
                    })
                    categorize_span.set_attribute("primary_category", categorization.get("primary_category", ""))
                    categorize_span.set_attribute("confidence", categorization.get("confidence", 0))
                    logger.info(f"[REPROCESS {file_id}] Replayed categorize span (cached)")

            # Replay Stage 4: Extract (if we're skipping it)
            if start_index > 2 and extract_result:
                with mlflow.start_span(
                    name="stage_4_extract",
                    span_type="RETRIEVER",
                    attributes={
                        "stage": "extract",
                        "is_cached_replay": True,
                        "original_file_id": file_id
                    }
                ) as extract_span:
                    # Log cached inputs
                    extract_span.set_inputs({
                        "document_text_length": len(categorize_result.get("document_text", "")) if categorize_result else 0,
                        "is_cached_replay": True,
                        "note": "Replayed from previous run"
                    })

                    # Log cached outputs
                    entities_count = extract_result.get("entities_count", 0)
                    extract_span.set_outputs({
                        "status": extract_result.get("status", "success"),
                        "entities_count": entities_count,
                        "is_cached_replay": True
                    })
                    extract_span.set_attribute("entities_count", entities_count)
                    logger.info(f"[REPROCESS {file_id}] Replayed extract span (cached)")

        # Update status to processing with new trace info
        if status_table:
            try:
                # Only reset status for stages we're going to run
                stage_status_updates = {
                    "stage_parse_status": None if start_index <= 0 else "completed",
                    "stage_categorize_status": None if start_index <= 1 else "completed",
                    "stage_extract_status": None if start_index <= 2 else "completed",
                    "stage_deidentify_status": None
                }
                status_table.update_file_status(
                    file_id=file_id,
                    status="processing",
                    current_stage=start_stage,
                    trace_id=combined_trace_ids,  # Store all trace IDs comma-separated
                    experiment_id=exp.experiment_id,
                    log_file_path=log_file_path,
                    error_message=None,  # Clear previous error
                    **stage_status_updates
                )
                if on_stage_update:
                    on_stage_update()
            except Exception as e:
                logger.warning(f"Could not update status: {e}")

        try:
            # Stage 2: Parse - Extract text and structure
            if start_index <= 0:
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
            else:
                logger.info(f"[REPROCESS {file_id}] Stage 2: Parse - SKIPPED (using previous result)")

            # Stage 3: Categorize - Classify document
            if start_index <= 1:
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
            else:
                logger.info(f"[REPROCESS {file_id}] Stage 3: Categorize - SKIPPED (using previous result)")

            # Stage 4: Extract - Extract entities
            if start_index <= 2:
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
            else:
                logger.info(f"[REPROCESS {file_id}] Stage 4: Extract - SKIPPED (using previous result)")

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

            # Determine the failed stage
            completed_stages = list(results.get("stages", {}).keys())
            failed_stage = completed_stages[-1] if completed_stages else "parse"

            # Determine which stages completed successfully
            stages_completed = len(completed_stages)

            # Mark as failed in status table
            if status_table:
                try:
                    status_table.mark_failed(
                        file_id=file_id,
                        error_message=str(e),
                        current_stage=failed_stage
                    )
                except Exception as status_err:
                    logger.warning(f"Could not mark as failed: {status_err}")

            # Set failure attributes on the span for easy filtering/searching in MLflow
            span.set_attribute("pipeline_status", "failed")
            span.set_attribute("failed_at_stage", failed_stage)
            span.set_attribute("stages_completed", stages_completed)
            span.set_attribute("error_message", str(e)[:500])  # Truncate long errors

            # Set outputs for the trace
            span.set_outputs({
                "status": "failed",
                "failed_at_stage": failed_stage,
                "stages_completed": stages_completed,
                "error": str(e),
                "total_time_seconds": results["total_time_seconds"]
            })

        # Flush any pending async trace logging to ensure trace is persisted
        # This is critical for failed traces which may not auto-flush
        try:
            mlflow.flush_trace_async_logging()
            logger.info(f"[REPROCESS {file_id}] Flushed trace to MLflow")
        except Exception as flush_err:
            logger.warning(f"[REPROCESS {file_id}] Could not flush trace: {flush_err}")

        # Clean up pipeline logging
        cleanup_pipeline_logging(uc_log_handler)

        return results


def reset_storage():
    """Clear all storage and logs (for testing)"""
    logger.info("Reset storage requested")
    # Note: Lakebase table data is persistent, this is a no-op
    # In production, you might want to implement archival instead of deletion
    pass


def reset_stuck_processing_files() -> Dict[str, Any]:
    """
    Reset all files stuck in 'processing' status to 'failed'.

    This is useful when the app is redeployed mid-processing and files
    get stuck in an incomplete state.

    Returns:
        Dictionary with count of reset files and any errors
    """
    status_table = get_status_table()

    if not status_table:
        logger.warning("Status table not available")
        return {"error": "Status table not available", "reset_count": 0}

    try:
        # Get all files with status='processing'
        all_files = status_table.get_all_files(limit=1000)
        stuck_files = [f for f in all_files if f.get("status") == "processing"]

        reset_count = 0
        for file_record in stuck_files:
            file_id = file_record.get("file_id")
            current_stage = file_record.get("current_stage", "unknown")

            try:
                status_table.mark_failed(
                    file_id=file_id,
                    error_message=f"Processing interrupted by app restart (was at stage: {current_stage})",
                    current_stage=current_stage
                )
                reset_count += 1
                logger.info(f"Reset stuck file {file_id} (was at {current_stage})")
            except Exception as e:
                logger.warning(f"Could not reset file {file_id}: {e}")

        logger.info(f"Reset {reset_count} stuck processing files")
        return {"reset_count": reset_count, "status": "success"}

    except Exception as e:
        logger.error(f"Error resetting stuck files: {str(e)}", exc_info=True)
        return {"error": str(e), "reset_count": 0}


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


def delete_file_record(file_id: str) -> Dict[str, Any]:
    """
    Delete a file record from the status table.

    Used for cleaning up corrupt/stuck records where file doesn't exist in volume.

    Args:
        file_id: File identifier

    Returns:
        Dictionary with status of deletion
    """
    status_table = get_status_table()

    if not status_table:
        logger.warning("Status table not available")
        return {"error": "Status table not available", "deleted": False}

    try:
        deleted = status_table.delete_file_record(file_id)
        if deleted:
            logger.info(f"Deleted file record: {file_id}")
            return {"deleted": True, "file_id": file_id}
        else:
            return {"error": "File not found", "deleted": False}
    except Exception as e:
        logger.error(f"Error deleting file record: {str(e)}", exc_info=True)
        return {"error": str(e), "deleted": False}
