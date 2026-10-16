"""
Database manager for AI Traffic Monitoring System.

Handles SQLite connection lifecycle, parameterized queries, session persistence,
traffic snapshots, and analytical aggregation.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from config.settings import get_config
from utils.logger import setup_logger

logger = setup_logger("db_manager")


class DatabaseManager:
    """Manages SQLite operations for sessions and traffic snapshots."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize DatabaseManager.

        Args:
            db_path: Path to SQLite database file. If None, uses configuration path.
        """
        config = get_config()
        self.db_path = Path(db_path) if db_path else config.db_path
        self.schema_path = Path(__file__).resolve().parent / "schema.sql"
        self._ensure_initialized()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a configured SQLite connection with row factories."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _ensure_initialized(self) -> None:
        """Initialize database schema if tables do not exist."""
        try:
            with self._get_connection() as conn:
                if self.schema_path.exists():
                    with open(self.schema_path, "r", encoding="utf-8") as f:
                        conn.executescript(f.read())
                    logger.debug("Database initialized using schema: %s", self.schema_path)
                else:
                    logger.error("Database schema file not found at: %s", self.schema_path)
        except Exception as e:
            logger.error("Failed to initialize database: %s", e)
            raise

    def create_session(
        self,
        session_id: str,
        source_type: str,
        source_name: str,
        start_time: Optional[datetime] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Create a new traffic monitoring session record.

        Args:
            session_id: Unique string identifier.
            source_type: 'video_file', 'webcam', or 'ip_camera'.
            source_name: Sanitized file name or sanitized stream identifier.
            start_time: Start timestamp (defaults to current UTC time).
            notes: Optional user or system notes.

        Returns:
            True if inserted successfully, False otherwise.
        """
        st = (start_time or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        sql = """
        INSERT INTO sessions (session_id, source_type, source_name, start_time, notes)
        VALUES (?, ?, ?, ?, ?)
        """
        try:
            with self._get_connection() as conn:
                conn.execute(sql, (session_id, source_type, source_name, st, notes))
                conn.commit()
            logger.info("Created session '%s' [%s: %s]", session_id, source_type, source_name)
            return True
        except sqlite3.IntegrityError:
            logger.warning("Session ID '%s' already exists. Skipping duplicate insert.", session_id)
            return False
        except Exception as e:
            logger.error("Error creating session '%s': %s", session_id, e)
            return False

    def save_snapshot(
        self,
        session_id: str,
        frame_number: int,
        active_vehicles: int,
        congestion_level: str = "LOW",
        cumulative_inbound: int = 0,
        cumulative_outbound: int = 0,
        current_fps: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Save a periodic traffic snapshot for a running session.

        Args:
            session_id: Session identifier.
            frame_number: Current frame index.
            active_vehicles: Number of vehicles visible in current frame.
            congestion_level: 'LOW', 'MODERATE', or 'HIGH'.
            cumulative_inbound: Cumulative inbound crossings so far.
            cumulative_outbound: Cumulative outbound crossings so far.
            current_fps: Measured processing FPS.
            timestamp: Snapshot timestamp.

        Returns:
            True if saved, False otherwise.
        """
        ts = (timestamp or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        sql = """
        INSERT INTO traffic_snapshots (
            session_id, timestamp, frame_number, active_vehicles,
            congestion_level, cumulative_inbound, cumulative_outbound, current_fps
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            with self._get_connection() as conn:
                conn.execute(
                    sql,
                    (
                        session_id,
                        ts,
                        frame_number,
                        active_vehicles,
                        congestion_level,
                        cumulative_inbound,
                        cumulative_outbound,
                        round(current_fps, 2),
                    ),
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error("Error saving snapshot for session '%s': %s", session_id, e)
            return False

    def finalize_session(
        self,
        session_id: str,
        end_time: Optional[datetime] = None,
        duration_sec: float = 0.0,
        total_vehicles: int = 0,
        cars: int = 0,
        buses: int = 0,
        trucks: int = 0,
        motorcycles: int = 0,
        pedestrians: int = 0,
        inbound_count: int = 0,
        outbound_count: int = 0,
        avg_fps: float = 0.0,
        output_video_path: Optional[str] = None,
    ) -> bool:
        """
        Finalize a session with final counts and metrics.

        Returns:
            True if updated successfully, False otherwise.
        """
        et = (end_time or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        sql = """
        UPDATE sessions SET
            end_time = ?,
            duration_sec = ?,
            total_vehicles = ?,
            cars = ?,
            buses = ?,
            trucks = ?,
            motorcycles = ?,
            pedestrians = ?,
            inbound_count = ?,
            outbound_count = ?,
            avg_fps = ?,
            output_video_path = ?
        WHERE session_id = ?
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    sql,
                    (
                        et,
                        round(duration_sec, 2),
                        total_vehicles,
                        cars,
                        buses,
                        trucks,
                        motorcycles,
                        pedestrians,
                        inbound_count,
                        outbound_count,
                        round(avg_fps, 2),
                        output_video_path,
                        session_id,
                    ),
                )
                conn.commit()
                if cursor.rowcount > 0:
                    logger.info("Finalized session '%s' successfully.", session_id)
                    return True
                logger.warning("Session '%s' not found for finalization.", session_id)
                return False
        except Exception as e:
            logger.error("Error finalizing session '%s': %s", session_id, e)
            return False

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single session by ID."""
        sql = "SELECT * FROM sessions WHERE session_id = ?"
        try:
            with self._get_connection() as conn:
                row = conn.execute(sql, (session_id,)).fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error("Error retrieving session '%s': %s", session_id, e)
            return None

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve most recent processing sessions."""
        sql = "SELECT * FROM sessions ORDER BY start_time DESC LIMIT ?"
        try:
            with self._get_connection() as conn:
                rows = conn.execute(sql, (limit,)).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error("Error retrieving recent sessions: %s", e)
            return []

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Retrieve all recorded sessions ordered by start_time descending."""
        sql = "SELECT * FROM sessions ORDER BY start_time DESC"
        try:
            with self._get_connection() as conn:
                rows = conn.execute(sql).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error("Error retrieving all sessions: %s", e)
            return []

    def filter_sessions(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter recorded sessions by date range and source type.

        Args:
            start_date: Start date string 'YYYY-MM-DD'.
            end_date: End date string 'YYYY-MM-DD'.
            source_type: Optional source filter ('video_file', 'webcam', 'ip_camera').

        Returns:
            List of matching session dictionaries.
        """
        conditions = []
        params: List[Any] = []

        if start_date:
            conditions.append("DATE(start_time) >= DATE(?)")
            params.append(start_date)
        if end_date:
            conditions.append("DATE(start_time) <= DATE(?)")
            params.append(end_date)
        if source_type and source_type != "All":
            conditions.append("source_type = ?")
            params.append(source_type)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM sessions {where_clause} ORDER BY start_time DESC"

        try:
            with self._get_connection() as conn:
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error("Error filtering sessions: %s", e)
            return []

    def get_snapshots_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all snapshots for a given session."""
        sql = "SELECT * FROM traffic_snapshots WHERE session_id = ? ORDER BY frame_number ASC"
        try:
            with self._get_connection() as conn:
                rows = conn.execute(sql, (session_id,)).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error("Error retrieving snapshots for session '%s': %s", session_id, e)
            return []

    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all recorded sessions."""
        sql = """
        SELECT
            COUNT(*) as total_sessions,
            COALESCE(SUM(total_vehicles), 0) as total_vehicles,
            COALESCE(SUM(cars), 0) as total_cars,
            COALESCE(SUM(buses), 0) as total_buses,
            COALESCE(SUM(trucks), 0) as total_trucks,
            COALESCE(SUM(motorcycles), 0) as total_motorcycles,
            COALESCE(SUM(pedestrians), 0) as total_pedestrians,
            COALESCE(SUM(inbound_count), 0) as total_inbound,
            COALESCE(SUM(outbound_count), 0) as total_outbound,
            COALESCE(AVG(avg_fps), 0.0) as overall_avg_fps,
            COALESCE(SUM(duration_sec), 0.0) as total_monitoring_seconds
        FROM sessions
        """
        try:
            with self._get_connection() as conn:
                row = conn.execute(sql).fetchone()
                if row:
                    res = dict(row)
                    res["overall_avg_fps"] = round(res.get("overall_avg_fps", 0.0), 2)
                    res["total_monitoring_seconds"] = round(res.get("total_monitoring_seconds", 0.0), 2)
                    return res
                return {}
        except Exception as e:
            logger.error("Error calculating aggregate metrics: %s", e)
            return {}

    def export_sessions_dataframe(self) -> pd.DataFrame:
        """Export all session records as a Pandas DataFrame for CSV download."""
        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query("SELECT * FROM sessions ORDER BY start_time DESC", conn)
                return df
        except Exception as e:
            logger.error("Error exporting sessions DataFrame: %s", e)
            return pd.DataFrame()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and associated snapshots by session ID."""
        sql = "DELETE FROM sessions WHERE session_id = ?"
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(sql, (session_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error("Error deleting session '%s': %s", session_id, e)
            return False
