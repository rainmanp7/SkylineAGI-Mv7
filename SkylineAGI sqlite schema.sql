-- TABLE
CREATE TABLE context_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            context_key TEXT NOT NULL,
            context_value TEXT NOT NULL,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        );
CREATE TABLE data_connections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            connection_name TEXT NOT NULL,
            connection_type TEXT NOT NULL,
            connection_details TEXT NOT NULL,
            last_connected DATETIME DEFAULT CURRENT_TIMESTAMP
        );
CREATE TABLE knowledge_base (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            complexity_score REAL NOT NULL,
            metadata TEXT,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
            complexity_range TEXT NOT NULL
        );
CREATE TABLE memory_retention (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_key TEXT NOT NULL,
            data_value TEXT NOT NULL,
            retention_period INTEGER NOT NULL,  -- Retention period in days
            last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP
        );
CREATE TABLE query_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            llm_query TEXT NOT NULL,
            agi_response TEXT NOT NULL,
            context_id INTEGER,
            complexity_range TEXT NOT NULL,
            FOREIGN KEY (context_id) REFERENCES context_memory(id)
        );
CREATE TABLE tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            assigned_by TEXT NOT NULL,
            status TEXT NOT NULL,
            complexity_factor REAL NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            completed_at DATETIME,
            complexity_range TEXT NOT NULL
        );
 
-- INDEX
 
-- TRIGGER
 
-- VIEW
 
