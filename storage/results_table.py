"""
Lakebase PostgreSQL Storage for Pipeline Results

Stores pipeline stage results in PostgreSQL table in Lakebase.
Each stage result is stored as a JSON string in its own column.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from config import RESULTS_TABLE_NAME
from storage.lakebase_connection import get_connection_manager

logger = logging.getLogger(__name__)


class ResultsTable:
    """
    Manages storage of pipeline results in Lakebase PostgreSQL table
    """

    def __init__(self):
        """Initialize results table manager"""
        self.table_name = RESULTS_TABLE_NAME
        self.conn_manager = get_connection_manager()
        self._ensure_schema_exists()
        self._ensure_table_exists()
        logger.info(f"ResultsTable initialized: {self.table_name}")

    def _ensure_schema_exists(self):
        """Create schema if it doesn't exist (app has CREATE on database)"""
        try:
            # Extract schema name from table_name (e.g., "unstructured_parsequery.results" -> "unstructured_parsequery")
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
        """Create results table if it doesn't exist"""
        try:
            logger.info(f"Ensuring results table exists: {self.table_name}")

            # PostgreSQL schema
            create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    file_id VARCHAR(255) PRIMARY KEY,
                    trace_id VARCHAR(255),
                    experiment_id VARCHAR(255),
                    source_volume_path VARCHAR(1000),
                    parse_result TEXT,
                    categorize_result TEXT,
                    extract_result TEXT,
                    deidentify_result TEXT,
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

            logger.info(f"Results table created/ensured successfully: {self.table_name}")

        except Exception as e:
            logger.error(f"Failed to create results table: {str(e)}", exc_info=True)
            raise

    def create_result_record(
        self,
        file_id: str,
        trace_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        source_volume_path: Optional[str] = None
    ) -> bool:
        """
        Create initial result record for a pipeline run

        Args:
            file_id: Pipeline/file ID
            trace_id: MLflow trace ID
            experiment_id: MLflow experiment ID
            source_volume_path: Path to source file in UC volume

        Returns:
            True if successful, False otherwise
        """
        try:
            now = datetime.now()

            insert_sql = f"""
                INSERT INTO {self.table_name}
                (file_id, trace_id, experiment_id, source_volume_path, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """

            conn = self.conn_manager.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(insert_sql, (file_id, trace_id, experiment_id, source_volume_path, now, now))
                conn.commit()
            finally:
                conn.close()

            logger.info(f"Created result record for file_id: {file_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create result record: {str(e)}", exc_info=True)
            return False

    def update_stage_result(
        self,
        file_id: str,
        stage_name: str,
        result_data: Dict[str, Any]
    ) -> bool:
        """
        Update result for a specific stage

        Args:
            file_id: Pipeline/file ID
            stage_name: Stage name (parse, categorize, extract, deidentify)
            result_data: Result data to store as JSON string

        Returns:
            True if successful, False otherwise
        """
        try:
            # Map stage name to column
            column_map = {
                "parse": "parse_result",
                "categorize": "categorize_result",
                "extract": "extract_result",
                "deidentify": "deidentify_result"
            }

            if stage_name not in column_map:
                logger.warning(f"Unknown stage name: {stage_name}, skipping result storage")
                return False

            column_name = column_map[stage_name]

            # Convert result to JSON string
            result_json = json.dumps(result_data)

            # Update the specific column
            update_sql = f"""
                UPDATE {self.table_name}
                SET {column_name} = %s,
                    updated_at = %s
                WHERE file_id = %s
            """

            conn = self.conn_manager.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(update_sql, (result_json, datetime.now(), file_id))
                conn.commit()
            finally:
                conn.close()

            logger.info(f"Updated {stage_name} result for file_id: {file_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update stage result: {str(e)}", exc_info=True)
            return False

    def get_results(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Get all results for a file

        Args:
            file_id: Pipeline/file ID

        Returns:
            Result record or None
        """
        try:
            query = f"""
                SELECT
                    file_id,
                    trace_id,
                    experiment_id,
                    source_volume_path,
                    parse_result,
                    categorize_result,
                    extract_result,
                    deidentify_result,
                    created_at,
                    updated_at
                FROM {self.table_name}
                WHERE file_id = %s
            """

            conn = self.conn_manager.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(query, (file_id,))
                    row = cur.fetchone()

                    if row:
                        # Get column names from cursor description
                        cols = [desc[0] for desc in cur.description]
                        return dict(zip(cols, row))
            finally:
                conn.close()

            return None

        except Exception as e:
            logger.error(f"Error getting results: {str(e)}", exc_info=True)
            return None
