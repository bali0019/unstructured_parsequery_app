"""
Lakebase PostgreSQL Storage for Processing Status

Manages PostgreSQL table in Lakebase for tracking file processing status.
Uses psycopg2 with parameterized queries for security and performance.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from config import STATUS_TABLE_NAME
from storage.lakebase_connection import get_connection_manager

logger = logging.getLogger(__name__)


class ProcessingStatusTable:
    """
    Manages PostgreSQL table for tracking file processing status using Lakebase
    """

    def __init__(self, table_name: str = None):
        """
        Initialize status table manager

        Args:
            table_name: Table name (default from config)
        """
        self.table_name = table_name or STATUS_TABLE_NAME
        self.conn_manager = get_connection_manager()
        self._ensure_schema_exists()
        self._ensure_table_exists()

    def _ensure_schema_exists(self):
        """Create schema if it doesn't exist (app has CREATE on database)"""
        try:
            # Extract schema name from table_name (e.g., "unstructured_parsequery.status" -> "unstructured_parsequery")
            if '.' in self.table_name:
                schema_name = self.table_name.split('.')[0]
            else:
                schema_name = "unstructured_parsequery"  # default schema

            logger.info(f"Ensuring schema exists: {schema_name}")

            create_schema_sql = f"CREATE SCHEMA IF NOT EXISTS {schema_name}"

            conn = self.conn_manager.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(create_schema_sql)
                conn.commit()
            finally:
                conn.close()

            logger.info(f"Schema created/ensured successfully: {schema_name}")

        except Exception as e:
            logger.error(f"Error ensuring schema exists: {str(e)}", exc_info=True)
            raise

    def _ensure_table_exists(self):
        """Create table if it doesn't exist"""
        try:
            logger.info(f"Ensuring status table exists: {self.table_name}")

            # PostgreSQL schema
            create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    file_id VARCHAR(255) PRIMARY KEY,
                    filename VARCHAR(500),
                    volume_path VARCHAR(1000),
                    status VARCHAR(50),
                    current_stage VARCHAR(50),
                    trace_id VARCHAR(255),
                    experiment_id VARCHAR(255),
                    run_id VARCHAR(255),
                    log_file_path VARCHAR(1000),
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    error_message TEXT,
                    stage_ingest_status VARCHAR(50),
                    stage_parse_status VARCHAR(50),
                    stage_categorize_status VARCHAR(50),
                    stage_extract_status VARCHAR(50),
                    stage_deidentify_status VARCHAR(50),
                    primary_category VARCHAR(255),
                    entities_count INTEGER,
                    pii_items_masked INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """

            conn = self.conn_manager.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(create_table_sql)
                conn.commit()
            finally:
                conn.close()

            logger.info(f"Table created/ensured successfully: {self.table_name}")

        except Exception as e:
            logger.error(f"Error ensuring table exists: {str(e)}", exc_info=True)
            raise

    def insert_file_record(
        self,
        file_id: str,
        filename: str,
        volume_path: str = None
    ) -> None:
        """
        Insert new file record with initial status

        Args:
            file_id: Unique file identifier
            filename: Original filename
            volume_path: Path in UC volume
        """
        try:
            now = datetime.now()

            insert_sql = f"""
                INSERT INTO {self.table_name} (
                    file_id,
                    filename,
                    volume_path,
                    status,
                    current_stage,
                    start_time,
                    created_at,
                    updated_at
                ) VALUES (%s, %s, %s, 'pending', 'ingest', %s, %s, %s)
            """

            conn = self.conn_manager.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(insert_sql, (file_id, filename, volume_path, now, now, now))
                conn.commit()
            finally:
                conn.close()

            logger.info(f"Inserted file record: {file_id}")

        except Exception as e:
            logger.error(f"Error inserting file record: {str(e)}", exc_info=True)
            raise

    def update_file_status(
        self,
        file_id: str,
        status: str = None,
        current_stage: str = None,
        volume_path: str = None,
        trace_id: str = None,
        experiment_id: str = None,
        run_id: str = None,
        error_message: str = None,
        **stage_statuses
    ) -> None:
        """
        Update file processing status

        Args:
            file_id: File identifier
            status: Overall status (pending, processing, completed, failed)
            current_stage: Current pipeline stage
            volume_path: UC volume path
            trace_id: MLflow trace ID
            experiment_id: MLflow experiment ID
            run_id: MLflow run ID
            error_message: Error message if failed
            **stage_statuses: Individual stage statuses and other fields
        """
        try:
            # Build dynamic UPDATE query
            set_clauses = ["updated_at = %s"]
            params = [datetime.now()]

            if status:
                set_clauses.append("status = %s")
                params.append(status)
            if current_stage:
                set_clauses.append("current_stage = %s")
                params.append(current_stage)
            if volume_path:
                set_clauses.append("volume_path = %s")
                params.append(volume_path)
            if trace_id:
                logger.info(f"Adding trace_id to update: {trace_id}")
                set_clauses.append("trace_id = %s")
                params.append(trace_id)
            if experiment_id:
                logger.info(f"Adding experiment_id to update: {experiment_id}")
                set_clauses.append("experiment_id = %s")
                params.append(experiment_id)
            if run_id:
                set_clauses.append("run_id = %s")
                params.append(run_id)
            if error_message:
                set_clauses.append("error_message = %s")
                params.append(error_message)

            # Handle additional fields from stage_statuses
            for key, value in stage_statuses.items():
                set_clauses.append(f"{key} = %s")
                params.append(value)

            if len(set_clauses) == 1:  # Only updated_at
                logger.warning(f"No updates provided for file_id: {file_id}")
                return

            # Add file_id to params for WHERE clause
            params.append(file_id)

            set_clause = ", ".join(set_clauses)
            update_sql = f"""
                UPDATE {self.table_name}
                SET {set_clause}
                WHERE file_id = %s
            """

            conn = self.conn_manager.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(update_sql, params)
                conn.commit()
            finally:
                conn.close()

            logger.info(f"Updated file status: {file_id}")

        except Exception as e:
            logger.error(f"Error updating file status: {str(e)}", exc_info=True)
            raise

    def get_file_status(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status for a specific file

        Args:
            file_id: File identifier

        Returns:
            Dictionary with file status or None if not found
        """
        try:
            select_sql = f"""
                SELECT * FROM {self.table_name}
                WHERE file_id = %s
            """

            conn = self.conn_manager.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(select_sql, (file_id,))
                    row = cur.fetchone()

                    if row:
                        # Get column names from cursor description
                        cols = [desc[0] for desc in cur.description]
                        return dict(zip(cols, row))
            finally:
                conn.close()

            return None

        except Exception as e:
            logger.error(f"Error getting file status: {str(e)}", exc_info=True)
            return None

    def get_all_files(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get status for all files

        Args:
            limit: Maximum number of records to return

        Returns:
            List of file status dictionaries
        """
        try:
            select_sql = f"""
                SELECT * FROM {self.table_name}
                ORDER BY created_at DESC
                LIMIT %s
            """

            logger.info(f"Executing get_all_files query with limit: {limit}")

            conn = self.conn_manager.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(select_sql, (limit,))
                    rows = cur.fetchall()

                    # Get column names from cursor description
                    cols = [desc[0] for desc in cur.description]

                    # Convert rows to list of dicts
                    results = [dict(zip(cols, row)) for row in rows]
                    logger.info(f"Returning {len(results)} file records")
                    return results
            finally:
                conn.close()

        except Exception as e:
            logger.error(f"Error getting all files: {str(e)}", exc_info=True)
            return []

    def mark_completed(
        self,
        file_id: str,
        primary_category: str = None,
        entities_count: int = None,
        pii_items_masked: int = None
    ) -> None:
        """
        Mark file as completed with final statistics

        Args:
            file_id: File identifier
            primary_category: Primary document category
            entities_count: Number of entities extracted
            pii_items_masked: Number of PII items masked
        """
        try:
            now = datetime.now()

            updates = {
                "status": "completed",
                "current_stage": "deidentify",
                "end_time": now,
                "stage_ingest_status": "completed",
                "stage_parse_status": "completed",
                "stage_categorize_status": "completed",
                "stage_extract_status": "completed",
                "stage_deidentify_status": "completed"
            }

            if primary_category:
                updates["primary_category"] = primary_category
            if entities_count is not None:
                updates["entities_count"] = entities_count
            if pii_items_masked is not None:
                updates["pii_items_masked"] = pii_items_masked

            self.update_file_status(file_id, **updates)

        except Exception as e:
            logger.error(f"Error marking file as completed: {str(e)}", exc_info=True)
            raise

    def mark_failed(self, file_id: str, error_message: str, current_stage: str = None) -> None:
        """
        Mark file as failed

        Args:
            file_id: File identifier
            error_message: Error message
            current_stage: Stage where failure occurred
        """
        try:
            now = datetime.now()

            updates = {
                "status": "failed",
                "error_message": error_message,
                "end_time": now
            }

            if current_stage:
                updates["current_stage"] = current_stage
                updates[f"stage_{current_stage}_status"] = "failed"

            self.update_file_status(file_id, **updates)

        except Exception as e:
            logger.error(f"Error marking file as failed: {str(e)}", exc_info=True)
            raise

    def delete_file_record(self, file_id: str) -> bool:
        """
        Delete a file record from the status table

        Args:
            file_id: File identifier

        Returns:
            True if deleted, False if not found
        """
        try:
            delete_sql = f"""
                DELETE FROM {self.table_name}
                WHERE file_id = %s
            """

            conn = self.conn_manager.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(delete_sql, (file_id,))
                    deleted = cur.rowcount > 0
                conn.commit()
            finally:
                conn.close()

            if deleted:
                logger.info(f"Deleted file record: {file_id}")
            else:
                logger.warning(f"File record not found for deletion: {file_id}")

            return deleted

        except Exception as e:
            logger.error(f"Error deleting file record: {str(e)}", exc_info=True)
            raise
