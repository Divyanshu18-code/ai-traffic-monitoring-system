-- Schema for AI-Powered Traffic Monitoring & Vehicle Analytics System
-- SQLite3 Engine

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,                -- 'video_file', 'webcam', 'ip_camera'
    source_name TEXT NOT NULL,                -- Sanitized filename or sanitized stream label
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    duration_sec REAL DEFAULT 0.0,
    total_vehicles INTEGER DEFAULT 0,
    cars INTEGER DEFAULT 0,
    buses INTEGER DEFAULT 0,
    trucks INTEGER DEFAULT 0,
    motorcycles INTEGER DEFAULT 0,
    pedestrians INTEGER DEFAULT 0,
    inbound_count INTEGER DEFAULT 0,
    outbound_count INTEGER DEFAULT 0,
    avg_fps REAL DEFAULT 0.0,
    output_video_path TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS traffic_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    frame_number INTEGER NOT NULL,
    active_vehicles INTEGER DEFAULT 0,
    congestion_level TEXT DEFAULT 'LOW',     -- 'LOW', 'MODERATE', 'HIGH'
    cumulative_inbound INTEGER DEFAULT 0,
    cumulative_outbound INTEGER DEFAULT 0,
    current_fps REAL DEFAULT 0.0,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS system_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON sessions(start_time);
CREATE INDEX IF NOT EXISTS idx_snapshots_session_id ON traffic_snapshots(session_id);
