"""
Stage 1: Ingest - Upload files to Unity Catalog Volume

Handles file upload to UC volumes using the Files API.
Includes MLflow tracing for observability.
"""

import os
import re
import requests
import mlflow
import hashlib
from datetime import datetime
from typing import Dict, Any
import logging
from utils import get_databricks_token

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to avoid issues with special characters

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for UC volume storage
    """
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Remove or replace special characters but keep extension
    name, ext = os.path.splitext(filename)
    # Keep only alphanumeric, underscores, hyphens
    name = re.sub(r'[^\w\-]', '_', name)
    return f"{name}{ext}"


def ingest_file(
    file_bytes: bytes,
    filename: str,
    catalog: str,
    schema: str,
    volume_name: str,
    overwrite: bool = True
) -> Dict[str, Any]:
    """
    Upload file to Unity Catalog volume using App Service Principal authentication.

    Args:
        file_bytes: File content as bytes
        filename: Original filename
        catalog: UC catalog name
        schema: UC schema name
        volume_name: UC volume name
        overwrite: Whether to overwrite existing files

    Returns:
        Dictionary with upload status and metadata
    """
    # Use context manager to avoid logging file_bytes in trace
    file_size_bytes = len(file_bytes)

    with mlflow.start_span(
        name="stage_1_ingest",
        span_type="CHAIN",
        attributes={
            "stage": "ingest",
            "filename": filename,
            "catalog": catalog,
            "schema": schema,
            "volume_name": volume_name,
            "file_size_bytes": file_size_bytes
        }
    ) as span:
        # Set inputs for trace
        span.set_inputs({
            "filename": filename,
            "catalog": catalog,
            "schema": schema,
            "volume_name": volume_name,
            "file_size_bytes": file_size_bytes,
            "overwrite": overwrite
        })

        try:
            # Calculate file hash for lineage tracking
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            logger.info(f"File hash (SHA256): {file_hash}")

            # Sanitize filename
            safe_filename = sanitize_filename(filename)
            logger.info(f"Sanitized filename: {filename} -> {safe_filename}")

            # Construct volume file path
            volume_file_path = f"/Volumes/{catalog}/{schema}/{volume_name}/{safe_filename}"
            logger.info(f"Target volume path: {volume_file_path}")

            # Get App SP OAuth token (retrieved INSIDE traced function to avoid logging in trace)
            token = get_databricks_token()

            # Get workspace URL from environment
            workspace_url = os.environ.get('DATABRICKS_HOST')
            if not workspace_url.startswith('http'):
                workspace_url = f"https://{workspace_url}"

            # Construct Files API URL
            api_url = f"{workspace_url}/api/2.0/fs/files{volume_file_path}"
            logger.info(f"API URL: {api_url}")

            # Upload file using PUT request with App SP token
            response = requests.put(
                api_url,
                data=file_bytes,
                headers={'Authorization': f'Bearer {token}'},
                params={'overwrite': 'true' if overwrite else 'false'}
            )

            logger.info(f"Response status: {response.status_code}")

            # Check for success
            if response.status_code in [200, 201, 204]:
                logger.info(f"Upload successful!")

                result = {
                    "status": "success",
                    "original_filename": filename,
                    "safe_filename": safe_filename,
                    "volume_path": volume_file_path,
                    "size_bytes": len(file_bytes),
                    "file_hash_sha256": file_hash,
                    "timestamp": datetime.now().isoformat(),
                    "catalog": catalog,
                    "schema": schema,
                    "volume": volume_name
                }

                # Set outputs for trace (include file hash for lineage)
                span.set_outputs({
                    "status": "success",
                    "volume_path": volume_file_path,
                    "safe_filename": safe_filename,
                    "size_bytes": len(file_bytes),
                    "file_hash_sha256": file_hash
                })

                return result
            else:
                error_msg = f"Upload failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)

        except Exception as e:
            logger.error(f"Upload failed: {str(e)}", exc_info=True)

            # Set error output for trace
            span.set_outputs({
                "error": str(e)
            })

            return {
                "status": "failed",
                "original_filename": filename,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
