"""
UC Volume Logger

Custom logging handler that writes pipeline logs to UC Volume.
Each pipeline gets its own log file in JSON lines format.

Uses Databricks SDK Files API for volume operations (required for Databricks Apps
since FUSE-mounted volumes are not supported).
"""

import logging
import io
from datetime import datetime
from typing import Optional
from databricks.sdk import WorkspaceClient
from config import LOGS_VOLUME_PATH


class UCVolumeLogHandler(logging.Handler):
    """
    Custom logging handler that writes to UC Volume.
    Creates a separate log file per pipeline_id in JSON lines format.
    Uses Databricks SDK Files API instead of direct file I/O.
    """

    def __init__(self, pipeline_id: str, level=logging.DEBUG):
        """
        Initialize the UC Volume log handler.

        Args:
            pipeline_id: Unique identifier for the pipeline run
            level: Minimum logging level to capture
        """
        super().__init__(level)
        self.pipeline_id = pipeline_id
        self.logs_base_path = LOGS_VOLUME_PATH
        self.log_buffer = []
        self.file_path = None
        self._workspace_client = None
        self._setup_log_file()

    def _get_workspace_client(self) -> WorkspaceClient:
        """Get or create WorkspaceClient instance."""
        if self._workspace_client is None:
            self._workspace_client = WorkspaceClient()
        return self._workspace_client

    def _setup_log_file(self):
        """Set up the log file path and create directories if needed."""
        # Organize by date for easier browsing
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_dir = f"{self.logs_base_path}/{date_str}"

        # Log file path: {base_path}/{date}/{pipeline_id}.log
        self.file_path = f"{log_dir}/{self.pipeline_id}.log"

        # Create directory using SDK Files API
        try:
            w = self._get_workspace_client()
            w.files.create_directory(log_dir)
        except Exception as e:
            # Directory might already exist, or we might not have permissions
            # Log to stderr but continue - we'll handle errors during flush
            import sys
            # Only warn if it's not an "already exists" error
            error_str = str(e).lower()
            if "already exists" not in error_str and "resource already exists" not in error_str:
                print(f"Warning: Could not create log directory {log_dir}: {e}", file=sys.stderr)

    def emit(self, record: logging.LogRecord):
        """
        Emit a log record by adding it to the buffer.

        Args:
            record: The log record to emit
        """
        try:
            # Format the log entry as plain text
            log_line = self.format(record)

            # Add exception info if present
            if record.exc_info:
                log_line += '\n' + self.formatException(record.exc_info)

            # Add to buffer
            self.log_buffer.append(log_line)

        except Exception:
            self.handleError(record)

    def flush(self):
        """Flush buffered logs to UC Volume using SDK Files API."""
        if not self.log_buffer or not self.file_path:
            return

        try:
            # Convert buffer to plain text (one log entry per line)
            content = '\n'.join(self.log_buffer) + '\n'

            # Upload using SDK Files API
            w = self._get_workspace_client()
            w.files.upload(
                file_path=self.file_path,
                contents=io.BytesIO(content.encode('utf-8')),
                overwrite=True  # Overwrite if file exists (shouldn't happen with unique pipeline IDs)
            )

            # Clear buffer after successful upload
            self.log_buffer = []

        except Exception as e:
            import sys
            print(f"Error flushing logs to {self.file_path}: {e}", file=sys.stderr)
            # Clear buffer anyway to avoid memory buildup
            self.log_buffer = []

    def close(self):
        """Close the handler, flushing any remaining logs."""
        self.flush()
        super().close()


class PipelineLogger:
    """
    Context manager for pipeline-specific logging to UC Volume.

    Usage:
        with PipelineLogger(pipeline_id) as logger:
            logger.info("Processing started")
            # ... pipeline code ...
            logger.info("Processing complete")
    """

    def __init__(self, pipeline_id: str, level=logging.DEBUG):
        """
        Initialize pipeline logger.

        Args:
            pipeline_id: Unique identifier for the pipeline run
            level: Minimum logging level to capture
        """
        self.pipeline_id = pipeline_id
        self.level = level
        self.handler = None
        self.loggers = []

    def __enter__(self):
        """Set up logging handler for the pipeline."""
        # Create the UC Volume handler
        self.handler = UCVolumeLogHandler(self.pipeline_id, self.level)

        # Set formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.handler.setFormatter(formatter)

        # Add handler to root logger and key module loggers
        loggers_to_capture = [
            '',  # Root logger
            'backend',
            'stages.ingest',
            'stages.parse',
            'stages.categorize',
            'stages.extract',
            'stages.deidentify',
            'storage.delta_table',
            'storage.results_table'
        ]

        for logger_name in loggers_to_capture:
            logger = logging.getLogger(logger_name)
            logger.addHandler(self.handler)
            self.loggers.append(logger)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up logging handler."""
        # Flush and close the handler
        if self.handler:
            self.handler.close()

            # Remove handler from all loggers
            for logger in self.loggers:
                logger.removeHandler(self.handler)

        return False  # Don't suppress exceptions

    def get_log_path(self) -> Optional[str]:
        """Get the path to the log file."""
        return self.handler.file_path if self.handler else None


def setup_pipeline_logging(pipeline_id: str) -> UCVolumeLogHandler:
    """
    Set up logging for a pipeline run.

    This is a simpler alternative to the context manager for cases
    where you need more control.

    Args:
        pipeline_id: Unique identifier for the pipeline run

    Returns:
        The UCVolumeLogHandler instance (caller must call close() when done)
    """
    handler = UCVolumeLogHandler(pipeline_id)

    # Set formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    # Add handler to root logger only
    # Module loggers will propagate to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    return handler


def cleanup_pipeline_logging(handler: UCVolumeLogHandler):
    """
    Clean up logging for a pipeline run.

    Args:
        handler: The handler returned by setup_pipeline_logging
    """
    if handler:
        handler.close()
        root_logger = logging.getLogger()
        root_logger.removeHandler(handler)
