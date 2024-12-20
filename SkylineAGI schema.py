import sqlite3
import os

def get_table_info(cursor, table_name):
    """Retrieve detailed information about a table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    return cursor.fetchall()

def get_indexes(cursor, table_name):
    """Retrieve index information for a table."""
    cursor.execute(f"PRAGMA index_list({table_name})")
    indexes = cursor.fetchall()
    index_details = []
    for index in indexes:
        cursor.execute(f"PRAGMA index_info({index[1]})")
        index_details.append((index, cursor.fetchall()))
    return index_details

def get_foreign_keys(cursor, table_name):
    """Retrieve foreign key information for a table."""
    cursor.execute(f"PRAGMA foreign_key_list({table_name})")
    return cursor.fetchall()

def get_primary_key_info(cursor, table_name):
    """Retrieve primary key information for a table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    primary_keys = [column for column in columns if column[5] > 0]
    return primary_keys

def get_table_sql(cursor, table_name):
    """Retrieve the CREATE TABLE SQL for a table."""
    cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
    return cursor.fetchone()[0]

def get_view_sql(cursor, view_name):
    """Retrieve the CREATE VIEW SQL for a view."""
    cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='view' AND name='{view_name}'")
    return cursor.fetchone()[0]

def get_trigger_sql(cursor, trigger_name):
    """Retrieve the CREATE TRIGGER SQL for a trigger."""
    cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='trigger' AND name='{trigger_name}'")
    return cursor.fetchone()[0]

def get_trigger_info(cursor, trigger_name):
    """Retrieve detailed information about a trigger."""
    cursor.execute(f"SELECT * FROM sqlite_master WHERE type='trigger' AND name='{trigger_name}'")
    return cursor.fetchone()

def get_index_sql(cursor, index_name):
    """Retrieve the CREATE INDEX SQL for an index."""
    cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='index' AND name='{index_name}'")
    return cursor.fetchone()[0]

def get_table_constraints(cursor, table_name):
    """Retrieve constraints for a table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    constraints = []
    for column in columns:
        if column[3] == 0:  # Not nullable
            constraints.append(f"# - {column[1]}: NOT NULL")
        if column[4]:  # Default value
            constraints.append(f"# - {column[1]}: DEFAULT {column[4]}")
    return constraints

def generate_schema(db_path, output_file):
    """Generate a detailed schema from an SQLite database and write it to a file."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"The database file {db_path} does not exist.")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        with open(output_file, 'w', encoding='utf-8') as f:
            # Write all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            for table in tables:
                table_name = table[0]
                f.write(f"# ==============================================\n")
                f.write(f"# **Table:** {table_name}\n")
                f.write(f"# ---------------------------------------------\n")
                f.write(f"{get_table_sql(cursor, table_name)};\n\n")
                
                # Table Info
                f.write(f"# **Table Columns:**\n")
                for column in get_table_info(cursor, table_name):
                    f.write(f"# - {column[1]} ({column[2]}) - Nullable: {bool(column[3])} - Default: {column[4]} - PK: {column[5]}\n")
                primary_keys = get_primary_key_info(cursor, table_name)
                if primary_keys:
                    f.write(f"# **Primary Key(s):**\n")
                    for pk in primary_keys:
                        f.write(f"# - {pk[1]} (Position: {pk[5]})\n")
                
                # Constraints
                constraints = get_table_constraints(cursor, table_name)
                if constraints:
                    f.write(f"# **Constraints:**\n")
                    for constraint in constraints:
                        f.write(f"# {constraint}\n")
                
                # Indexes
                f.write(f"# **Indexes on {table_name}:**\n")
                for index, details in get_indexes(cursor, table_name):
                    f.write(f"# - **Index Name:** {index[1]} - Unique: {bool(index[2])}\n")
                    for detail in details:
                        f.write(f"#   - Column: {detail[2]} - Desc: {bool(detail[3]) == False}\n")
                
                # Foreign Keys
                fks = get_foreign_keys(cursor, table_name)
                if fks:
                    f.write(f"# **Foreign Keys:**\n")
                    for fk in fks:
                        f.write(f"# - Column: {fk[3]} -> Table: {fk[2]}, Column: {fk[4]}\n")
                
                f.write(f"\n\n")
            
            # Write all views
            cursor.execute("SELECT name FROM sqlite_master WHERE type='view';")
            views = cursor.fetchall()
            for view in views:
                view_name = view[0]
                f.write(f"# ==============================================\n")
                f.write(f"# **View:** {view_name}\n")
                f.write(f"# ---------------------------------------------\n")
                f.write(f"{get_view_sql(cursor, view_name)};\n\n")
                f.write(f"\n\n")
            
            # Write all triggers
            cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger';")
            triggers = cursor.fetchall()
            for trigger in triggers:
                trigger_name = trigger[0]
                trigger_sql = get_trigger_sql(cursor, trigger_name)
                trigger_info = get_trigger_info(cursor, trigger_name)
                f.write(f"# ==============================================\n")
                f.write(f"# **Trigger:** {trigger_name}\n")
                f.write(f"# **Type:** {trigger_info[3]} - **Event:** {trigger_info[4]}\n")
                f.write(f"# **Table:** {trigger_info[5]}\n")
                f.write(f"# **Condition:** {trigger_info[6]}\n")
                f.write(f"# ---------------------------------------------\n")
                f.write(f"{trigger_sql};\n\n")
                f.write(f"\n\n")
            
            # Write all indexes
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index';")
            indexes = cursor.fetchall()
            for index in indexes:
                index_name = index[0]
                f.write(f"# ==============================================\n")
                f.write(f"# **Index:** {index_name}\n")
                f.write(f"# ---------------------------------------------\n")
                f.write(f"{get_index_sql(cursor, index_name)};\n\n")
                f.write(f"\n\n")

if __name__ == "__main__":
    db_path = 'skyline_agi.db'
    output_file = 'schema_details.sql'
    generate_schema(db_path, output_file)
    print(f"Schema details have been written to {output_file}")
