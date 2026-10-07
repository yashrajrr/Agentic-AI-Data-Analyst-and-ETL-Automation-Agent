import psycopg2
from dotenv import load_dotenv
import os
import datetime
import decimal
load_dotenv()


def json_safe(value):
    """Convert psycopg2 result values into JSON-serializable primitives."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, datetime.timedelta):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return bytes(value).decode("utf-8", errors="replace")
        except Exception:
            return repr(bytes(value))
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return str(value)


class DatabaseUtil:

    def __init__(self, db_config):
        self.db_config = db_config

        try: 
            self.connection = psycopg2.connect(**db_config) 

        except Exception as e:
            print(f"Error connecting to the database: {e}")
            self.connection = None

    def schema_details(self,schema_name):

        schema_info_context = ""
        
        connection = self.connection
        if connection is None:
            print("Error creating cursor: database connection is not available")
            return None

        try:
            cursor = connection.cursor()
        except Exception as e:
            print(f"Error creating cursor: {e}")
            return None

        schema_info_context = f"Database Schema: {schema_name}\n"

        try: 

            cursor.execute("SELECT table_name from information_schema.tables where table_schema = %s;", (schema_name,))
            tables_list = cursor.fetchall()

            for table in tables_list:
                table_name = table[0]
                schema_info_context = f"{schema_info_context}\nTable: {table_name}\n"

                # Adding Columns & Data Types
                cursor.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s;", (table_name,))
                columns_list = cursor.fetchall()

                for column in columns_list:
                    column_name = column[0]
                    data_type = column[1]
                    schema_info_context = f"{schema_info_context}  Column: {column_name}, Data Type: {data_type}\n"

                # Adding Sample Data
                cursor.execute(f"SELECT * FROM {schema_name}.{table_name} LIMIT 5;")
                sample_data = cursor.fetchall()
                schema_info_context = f"{schema_info_context}  Sample Data:\n"
                for row in sample_data:
                    schema_info_context = f"{schema_info_context}    {row}\n"

        except Exception as e:
            print(f"Error fetching schema details: {e}")
            schema_info_context = f"Error fetching schema details: {e}"

        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
        
        return schema_info_context

    def execute_sql(self, query):
        connection = self.connection
        cursor = None

        if connection is None:
            print("Error executing query: database connection is not available")
            return None

        try:
            cursor = connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            connection.commit()
            return str(result)
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    def execute_sql_structured(self, query):
        """Execute a read-only query and return columns + JSON-safe rows.

        Unlike ``execute_sql`` (which returns a bare ``str(list_of_tuples)`` for the
        LLM to read), this returns a tabular shape the UI can render directly.
        """
        empty = {"columns": [], "rows": [], "row_count": 0, "error": None}

        connection = self.connection
        if connection is None:
            return {**empty, "error": "database connection is not available"}

        cursor = None
        try:
            cursor = connection.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            raw_rows = cursor.fetchall()
            rows = [[json_safe(value) for value in row] for row in raw_rows]
            return {
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "error": None,
            }
        except Exception as e:
            print(f"Error executing query: {e}")
            return {**empty, "error": str(e)}
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

if __name__ == "__main__":
    if 'port' not in os.environ:
        os.environ['port'] = '5432'

    obj = DatabaseUtil({
        "host": os.environ['host'],
        "port": int(os.environ['port']),
        "database": os.environ['database'],
        "user": os.environ['user'],
        "password": os.environ['password'],
    })

    result = obj.schema_details("public")

    with open("test_schema_details.txt", "w") as f:
        f.write(result) # pyright: ignore[reportArgumentType]
