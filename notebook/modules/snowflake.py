import os
from snowflake.snowpark.session import Session

class Snowflake:
    def __init__(self):
        connection_parameters = {
            "account": os.environ['SNOWFLAKE_ACCOUNT'],
            "user": os.environ['SNOWFLAKE_USER'],
            "password": os.environ['SNOWFLAKE_PASSWORD'],
            "role": os.environ['SNOWFLAKE_ROLE'],
            "warehouse": os.environ['SNOWFLAKE_WAREHOUSE'],
            "database": os.environ['SNOWFLAKE_DATABASE'],
            "schema": os.environ['SNOWFLAKE_SCHEMA']
        }
        self.session = Session.builder.configs(connection_parameters).create()
        self.name = 'test'

    def exec_statement(self, statement):
        return self.session.sql(statement)
    
    def fetch_dataframe(self, tablename):
        return self.session.table(tablename)
