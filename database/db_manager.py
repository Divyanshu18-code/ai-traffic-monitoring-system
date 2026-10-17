"""
Database manager for AI Traffic Monitoring System.

Handles SQLite connection lifecycle, parameterized queries, non-destructive migrations,
session persistence, traffic snapshots, analytical aggregation, and data validation.
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

VALID_STATUSES = {"started", "in_progress", "completed", "failed", "cancelled"}
VALID_SOURCE_TYPES = {"video_file", "webcam", "ip_camera", "stream", "image"}


def _format_timestamp(ts: Optional[Union[datetime, str]]) -> str:
    """Format and validate a datetime or timestamp string to 'YYYY-MM-DD HH:MM:SS'."""
    if ts is None:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(ts, str):
        cleaned = ts.strip()
        # Accept standard formats
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(cleaned, fmt)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
        try:
            dt = datetime.fromisoformat(cleaned)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise ValueError(f"Invalid timestamp format: '{ts}'. Expected ISO or 'YYYY-MM-DD HH:MM:SS'.")
    raise ValueError(f"Invalid timestamp type: {type(ts)}. Expected datetime or str.")


def _validate_non_negative_int(name: str, val: Optional[int]) -> int:
    """Ensure integer is non-negative."""
    if val is None:
        return 0
    if not isinstance(val, (int, float)) or int(val) < 0:
        raise ValueError(f"Field '{name}' must be a non-negative integer, got {val}")
    return int(val)


def _validate_non_negative_float(name: str, val: Optional[float]) -> float:
    """Ensure float is non-negative."""
    if val is None:
        return 0.0
    if not isinstance(val, (int, float)) or float(val) < 0:
        raise ValueError(f"Field '{name}' must be a non-negative float, got {val}")
    return float(val)


class DatabaseManager:
    """
    Manages SQLite database lifecycle, schema initialization, session records,
    traffic snapshots, and analytical reporting.
    """

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
        """Create and return a configured SQLite connection with foreign keys enabled."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 10000;")
        return conn

    def _ensure_initialized(self) -> None:
        """Initialize database schema and apply non-destructive migrations."""
        try:
            with self._get_connection() as conn:
                if self.schema_path.exists():
                    with open(self.schema_path, "r", encoding="utf-8") as f:
                        schema_sql = f.read()

                    # Separate table definitions from index definitions
                    table_ddl_parts = []
                    for stmt in schema_sql.split(";"):
                        cleaned = stmt.strip()
                        if cleaned and not cleaned.upper().startswith("CREATE INDEX"):
                            table_ddl_parts.append(cleaned)

                    if table_ddl_parts:
                        conn.executescript("; ".join(table_ddl_parts) + ";")

                # Apply non-destructive column migrations on existing tables
                self._apply_migrations(conn)

                # Safely create indexes now that all columns are guaranteed to exist
                conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON sessions(start_time);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_session_id ON traffic_snapshots(session_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON traffic_snapshots(timestamp);")
                conn.commit()
                logger.debug("Database initialized using schema: %s", self.schema_path)
        except Exception as e:
            logger.error("Failed to initialize database: %s", e)
            raise

    def _apply_migrations(self, conn: sqlite3.Connection) -> None:
        """Apply safe non-destructive ALTER TABLE migrations if columns are missing."""
        try:
            # Check sessions table
            session_cols = {col[1] for col in conn.execute("PRAGMA table_info(sessions)").fetchall()}
            if session_cols:
                if "total_crossings" not in session_cols:
                    conn.execute("ALTER TABLE sessions ADD COLUMN total_crossings INTEGER DEFAULT 0;")
                if "status" not in session_cols:
                    conn.execute("ALTER TABLE sessions ADD COLUMN status TEXT DEFAULT 'completed';")
                if "error_summary" not in session_cols:
                    conn.execute("ALTER TABLE sessions ADD COLUMN error_summary TEXT;")

            # Check traffic_snapshots table
            snap_cols = {col[1] for col in conn.execute("PRAGMA table_info(traffic_snapshots)").fetchall()}
            if snap_cols:
                if "cars" not in snap_cols:
                    conn.execute("ALTER TABLE traffic_snapshots ADD COLUMN cars INTEGER DEFAULT 0;")
                if "buses" not in snap_cols:
                    conn.execute("ALTER TABLE traffic_snapshots ADD COLUMN buses INTEGER DEFAULT 0;")
                if "trucks" not in snap_cols:
                    conn.execute("ALTER TABLE traffic_snapshots ADD COLUMN trucks INTEGER DEFAULT 0;")
                if "motorcycles" not in snap_cols:
                    conn.execute("ALTER TABLE traffic_snapshots ADD COLUMN motorcycles INTEGER DEFAULT 0;")
                if "total_crossings" not in snap_cols:
                    conn.execute("ALTER TABLE traffic_snapshots ADD COLUMN total_crossings INTEGER DEFAULT 0;")
            conn.commit()
        except Exception as e:
            logger.warning("Database migration note: %s", e)

    def create_session(
        self,
        session_id: str,
        source_type: str,
        source_name: str,
        start_time: Optional[Union[datetime, str]] = None,
        status: str = "started",
        notes: Optional[str] = None,
    ) -> bool:
        """
        Create a new traffic monitoring session record.

        Args:
            session_id: Unique string identifier.
            source_type: 'video_file', 'webcam', 'ip_camera', or 'stream'.
            source_name: Sanitized file name or stream identifier.
            start_time: Start timestamp (datetime or string).
            status: Initial session status (default 'started').
            notes: Optional user or system notes.

        Returns:
            True if inserted successfully, False otherwise.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")

        st_clean = source_type.strip().lower() if isinstance(source_type, str) else ""
        if st_clean not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"Invalid source_type '{source_type}'. Expected one of {sorted(VALID_SOURCE_TYPES)}"
            )

        stat_clean = status.strip().lower() if isinstance(status, str) else "started"
        if stat_clean not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Expected one of {sorted(VALID_STATUSES)}")

        st_time_str = _format_timestamp(start_time)

        sql = """
        INSERT INTO sessions (session_id, source_type, source_name, start_time, status, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        try:
            with self._get_connection() as conn:
                conn.execute(sql, (session_id.strip(), st_clean, str(source_name).strip(), st_time_str, stat_clean, notes))
                conn.commit()
            logger.info("Created session '%s' [%s: %s, status=%s]", session_id, st_clean, source_name, stat_clean)
            return True
        except sqlite3.IntegrityError:
            logger.warning("Session ID '%s' already exists. Skipping duplicate insert.", session_id)
            return False
        except Exception as e:
            logger.error("Error creating session '%s': %s", session_id, e)
            return False

    def update_session_status(
        self,
        session_id: str,
        status: str,
        error_summary: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Update status and optional error summary for a session.

        Args:
            session_id: Session identifier.
            status: New status ('started', 'in_progress', 'completed', 'failed', 'cancelled').
            error_summary: Optional error message or details.

        Returns:
            True if updated, False otherwise.
        """
        stat_clean = status.strip().lower() if isinstance(status, str) else ""
        if stat_clean not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Expected one of {sorted(VALID_STATUSES)}")

        sql = "UPDATE sessions SET status = ?, error_summary = ? WHERE session_id = ?"
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(sql, (stat_clean, error_summary, session_id))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error("Error updating status for session '%s': %s", session_id, e)
            return False

    def update_session_statistics(
        self,
        session_id: str,
        total_vehicles: Optional[int] = None,
        cars: Optional[int] = None,
        buses: Optional[int] = None,
        trucks: Optional[int] = None,
        motorcycles: Optional[int] = None,
        pedestrians: Optional[int] = None,
        total_crossings: Optional[int] = None,
        inbound_count: Optional[int] = None,
        outbound_count: Optional[int] = None,
        avg_fps: Optional[float] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Update calculated statistics on an active session.

        Returns:
            True if updated, False otherwise.
        """
        updates = []
        params = []

        if total_vehicles is not None:
            updates.append("total_vehicles = ?")
            params.append(_validate_non_negative_int("total_vehicles", total_vehicles))
        if cars is not None:
            updates.append("cars = ?")
            params.append(_validate_non_negative_int("cars", cars))
        if buses is not None:
            updates.append("buses = ?")
            params.append(_validate_non_negative_int("buses", buses))
        if trucks is not None:
            updates.append("trucks = ?")
            params.append(_validate_non_negative_int("trucks", trucks))
        if motorcycles is not None:
            updates.append("motorcycles = ?")
            params.append(_validate_non_negative_int("motorcycles", motorcycles))
        if pedestrians is not None:
            updates.append("pedestrians = ?")
            params.append(_validate_non_negative_int("pedestrians", pedestrians))
        if total_crossings is not None:
            updates.append("total_crossings = ?")
            params.append(_validate_non_negative_int("total_crossings", total_crossings))
        if inbound_count is not None:
            updates.append("inbound_count = ?")
            params.append(_validate_non_negative_int("inbound_count", inbound_count))
        if outbound_count is not None:
            updates.append("outbound_count = ?")
            params.append(_validate_non_negative_int("outbound_count", outbound_count))
        if avg_fps is not None:
            updates.append("avg_fps = ?")
            params.append(round(_validate_non_negative_float("avg_fps", avg_fps), 2))

        if not updates:
            return False

        params.append(session_id)
        sql = f"UPDATE sessions SET {', '.join(updates)} WHERE session_id = ?"
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(sql, params)
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error("Error updating statistics for session '%s': %s", session_id, e)
            return False

    def finalize_session(
        self,
        session_id: str,
        end_time: Optional[Union[datetime, str]] = None,
        duration_sec: float = 0.0,
        total_vehicles: int = 0,
        cars: int = 0,
        buses: int = 0,
        trucks: int = 0,
        motorcycles: int = 0,
        pedestrians: int = 0,
        total_crossings: Optional[int] = None,
        inbound_count: int = 0,
        outbound_count: int = 0,
        avg_fps: float = 0.0,
        output_video_path: Optional[str] = None,
        status: str = "completed",
        error_summary: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Finalize a session with end timestamp, metrics, and final status.

        Returns:
            True if updated successfully, False otherwise.
        """
        et_str = _format_timestamp(end_time)
        dur = round(_validate_non_negative_float("duration_sec", duration_sec), 2)
        fps = round(_validate_non_negative_float("avg_fps", avg_fps), 2)

        tot_veh = _validate_non_negative_int("total_vehicles", total_vehicles)
        c_cars = _validate_non_negative_int("cars", cars)
        c_buses = _validate_non_negative_int("buses", buses)
        c_trucks = _validate_non_negative_int("trucks", trucks)
        c_motos = _validate_non_negative_int("motorcycles", motorcycles)
        c_peds = _validate_non_negative_int("pedestrians", pedestrians)
        c_in = _validate_non_negative_int("inbound_count", inbound_count)
        c_out = _validate_non_negative_int("outbound_count", outbound_count)

        if total_crossings is not None:
            c_cross = _validate_non_negative_int("total_crossings", total_crossings)
        else:
            c_cross = c_in + c_out

        stat_clean = status.strip().lower() if isinstance(status, str) else "completed"
        if stat_clean not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Expected one of {sorted(VALID_STATUSES)}")

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
            total_crossings = ?,
            inbound_count = ?,
            outbound_count = ?,
            avg_fps = ?,
            output_video_path = ?,
            status = ?,
            error_summary = ?
        WHERE session_id = ?
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    sql,
                    (
                        et_str,
                        dur,
                        tot_veh,
                        c_cars,
                        c_buses,
                        c_trucks,
                        c_motos,
                        c_peds,
                        c_cross,
                        c_in,
                        c_out,
                        fps,
                        output_video_path,
                        stat_clean,
                        error_summary,
                        session_id,
                    ),
                )
                conn.commit()
                if cursor.rowcount > 0:
                    logger.info("Finalized session '%s' [%s].", session_id, stat_clean)
                    return True
                logger.warning("Session '%s' not found for finalization.", session_id)
                return False
        except Exception as e:
            logger.error("Error finalizing session '%s': %s", session_id, e)
            return False

    def insert_traffic_snapshot(
        self,
        session_id: str,
        frame_number: int,
        active_vehicles: int,
        congestion_level: str = "LOW",
        cumulative_inbound: int = 0,
        cumulative_outbound: int = 0,
        current_fps: float = 0.0,
        timestamp: Optional[Union[datetime, str]] = None,
        cars: int = 0,
        buses: int = 0,
        trucks: int = 0,
        motorcycles: int = 0,
        total_crossings: Optional[int] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Insert a periodic traffic snapshot for a running session.

        Returns:
            True if inserted successfully, False otherwise.
        """
        f_num = _validate_non_negative_int("frame_number", frame_number)
        act_veh = _validate_non_negative_int("active_vehicles", active_vehicles)
        c_in = _validate_non_negative_int("cumulative_inbound", cumulative_inbound)
        c_out = _validate_non_negative_int("cumulative_outbound", cumulative_outbound)
        fps = round(_validate_non_negative_float("current_fps", current_fps), 2)
        c_cars = _validate_non_negative_int("cars", cars)
        c_buses = _validate_non_negative_int("buses", buses)
        c_trucks = _validate_non_negative_int("trucks", trucks)
        c_motos = _validate_non_negative_int("motorcycles", motorcycles)
        c_tot = (
            _validate_non_negative_int("total_crossings", total_crossings)
            if total_crossings is not None
            else (c_in + c_out)
        )

        ts_str = _format_timestamp(timestamp)
        cong = congestion_level.strip().upper() if isinstance(congestion_level, str) else "LOW"

        sql = """
        INSERT INTO traffic_snapshots (
            session_id, timestamp, frame_number, active_vehicles,
            cars, buses, trucks, motorcycles, total_crossings,
            congestion_level, cumulative_inbound, cumulative_outbound, current_fps
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            with self._get_connection() as conn:
                conn.execute(
                    sql,
                    (
                        session_id,
                        ts_str,
                        f_num,
                        act_veh,
                        c_cars,
                        c_buses,
                        c_trucks,
                        c_motos,
                        c_tot,
                        cong,
                        c_in,
                        c_out,
                        fps,
                    ),
                )
                conn.commit()
            return True
        except sqlite3.IntegrityError as e:
            logger.error("Foreign key or integrity violation for snapshot on session '%s': %s", session_id, e)
            return False
        except Exception as e:
            logger.error("Error inserting snapshot for session '%s': %s", session_id, e)
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
        timestamp: Optional[Union[datetime, str]] = None,
        cars: int = 0,
        buses: int = 0,
        trucks: int = 0,
        motorcycles: int = 0,
        total_crossings: Optional[int] = None,
        **kwargs: Any,
    ) -> bool:
        """Backwards-compatible alias for insert_traffic_snapshot."""
        return self.insert_traffic_snapshot(
            session_id=session_id,
            frame_number=frame_number,
            active_vehicles=active_vehicles,
            congestion_level=congestion_level,
            cumulative_inbound=cumulative_inbound,
            cumulative_outbound=cumulative_outbound,
            current_fps=current_fps,
            timestamp=timestamp,
            cars=cars,
            buses=buses,
            trucks=trucks,
            motorcycles=motorcycles,
            total_crossings=total_crossings,
            **kwargs,
        )

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single session by its unique ID."""
        sql = "SELECT * FROM sessions WHERE session_id = ?"
        try:
            with self._get_connection() as conn:
                row = conn.execute(sql, (session_id,)).fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error("Error retrieving session '%s': %s", session_id, e)
            return None

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

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve most recent processing sessions."""
        lim = max(1, int(limit))
        sql = "SELECT * FROM sessions ORDER BY start_time DESC LIMIT ?"
        try:
            with self._get_connection() as conn:
                rows = conn.execute(sql, (lim,)).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error("Error retrieving recent sessions: %s", e)
            return []

    def filter_sessions(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        source_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter recorded sessions by date range, source type, and status using parameterized SQL.

        Args:
            start_date: Start date string 'YYYY-MM-DD'.
            end_date: End date string 'YYYY-MM-DD'.
            source_type: Optional source filter ('video_file', 'webcam', 'ip_camera').
            status: Optional status filter ('started', 'completed', 'failed', 'cancelled').

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
            params.append(source_type.strip().lower())
        if status and status != "All":
            conditions.append("status = ?")
            params.append(status.strip().lower())

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

    def get_snapshots(self, session_id: str) -> List[Dict[str, Any]]:
        """Convenience alias for get_snapshots_for_session."""
        return self.get_snapshots_for_session(session_id)

    def get_historical_statistics(self, days: Optional[int] = None) -> Dict[str, Any]:
        """
        Compute historical aggregate statistics across sessions, optionally filtered by past N days.

        Returns:
            Dictionary with comprehensive metrics.
        """
        conditions = []
        params: List[Any] = []

        if days is not None and days > 0:
            conditions.append("start_time >= datetime('now', ?)")
            params.append(f"-{int(days)} days")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
        SELECT
            COUNT(*) as total_sessions,
            COALESCE(SUM(total_vehicles), 0) as total_vehicles,
            COALESCE(SUM(cars), 0) as total_cars,
            COALESCE(SUM(buses), 0) as total_buses,
            COALESCE(SUM(trucks), 0) as total_trucks,
            COALESCE(SUM(motorcycles), 0) as total_motorcycles,
            COALESCE(SUM(pedestrians), 0) as total_pedestrians,
            COALESCE(SUM(total_crossings), 0) as total_crossings,
            COALESCE(SUM(inbound_count), 0) as total_inbound,
            COALESCE(SUM(outbound_count), 0) as total_outbound,
            COALESCE(AVG(avg_fps), 0.0) as overall_avg_fps,
            COALESCE(SUM(duration_sec), 0.0) as total_monitoring_seconds
        FROM sessions {where_clause}
        """
        try:
            with self._get_connection() as conn:
                row = conn.execute(sql, params).fetchone()
                if row:
                    res = dict(row)
                    res["overall_avg_fps"] = round(res.get("overall_avg_fps", 0.0), 2)
                    res["total_monitoring_seconds"] = round(res.get("total_monitoring_seconds", 0.0), 2)
                    return res
                return {}
        except Exception as e:
            logger.error("Error calculating historical statistics: %s", e)
            return {}

    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all recorded sessions."""
        return self.get_historical_statistics(days=None)

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
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM traffic_snapshots WHERE session_id = ?", (session_id,))
                cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error("Error deleting session '%s': %s", session_id, e)
            return False

    def clear_all_history(self) -> int:
        """Delete all recorded sessions and snapshots from database. Returns count of deleted sessions."""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM traffic_snapshots")
                cur = conn.execute("DELETE FROM sessions")
                conn.commit()
                return cur.rowcount
        except Exception as e:
            logger.error("Error clearing all history: %s", e)
            return 0

    def purge_old_sessions(self, keep_latest: int = 20) -> int:
        """Keep only the latest N sessions and purge older ones and their snapshots."""
        try:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT session_id FROM sessions ORDER BY start_time DESC LIMIT -1 OFFSET ?",
                    (max(0, keep_latest),),
                ).fetchall()
                sids_to_del = [r[0] for r in rows]
                if not sids_to_del:
                    return 0

                placeholders = ",".join("?" for _ in sids_to_del)
                conn.execute(f"DELETE FROM traffic_snapshots WHERE session_id IN ({placeholders})", sids_to_del)
                cur = conn.execute(f"DELETE FROM sessions WHERE session_id IN ({placeholders})", sids_to_del)
                conn.commit()
                return cur.rowcount
        except Exception as e:
            logger.error("Error purging old sessions: %s", e)
            return 0

    def __enter__(self) -> DatabaseManager:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass
