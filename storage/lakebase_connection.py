"""
Lakebase Connection Manager

Manages PostgreSQL connections to Lakebase using psycopg with OAuth token.
Uses environment variables automatically injected by Databricks Apps:
- PGHOST, PGPORT, PGUSER, PGDATABASE, PGSSLMODE
"""

import os
import logging
import psycopg2
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

# Singleton connection manager
_connection_manager = None


class LakebaseConnectionManager:
    """
    Manages psycopg2 connections to Lakebase with OAuth token authentication
    """

    def __init__(self):
        """
        Initialize connection manager using auto-injected PG environment variables
        """
        # Get connection info from environment (auto-injected by Databricks Apps)
        self.host = os.environ.get("PGHOST")
        self.port = os.environ.get("PGPORT", "5432")
        self.user = os.environ.get("PGUSER")
        self.database = os.environ.get("PGDATABASE", "databricks_postgres")
        self.sslmode = os.environ.get("PGSSLMODE", "require")

        if not self.host or not self.user:
            raise ValueError(
                "Lakebase connection requires PGHOST and PGUSER environment variables. "
                "Ensure the Lakebase database resource is added to your Databricks App."
            )

        # Initialize Databricks SDK for token generation
        self.w = WorkspaceClient()

        logger.info(f"LakebaseConnectionManager initialized: {self.host}:{self.port}/{self.database}")

    def _get_token(self):
        """
        Get OAuth token from WorkspaceClient config
        """
        return self.w.config.oauth_token().access_token

    def get_connection(self):
        """
        Get a new connection to the Lakebase database using OAuth token.

        Returns:
            psycopg2 connection
        """
        token = self._get_token()

        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.database,
            user=self.user,
            password=token,
            sslmode=self.sslmode,
        )
        return conn

    def dispose(self):
        """
        Cleanup (no-op for psycopg2 without connection pool)
        """
        logger.info("LakebaseConnectionManager disposed")


def get_connection_manager():
    """
    Get or create singleton connection manager

    Returns:
        LakebaseConnectionManager instance
    """
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = LakebaseConnectionManager()
    return _connection_manager
