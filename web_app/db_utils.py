import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, '.env'))

# Cloud MySQL Configuration
# Use environment variables for sensitive info
DB_HOST = os.getenv('DB_HOST', '114.132.63.39')
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', '') # Set this in .env
DB_NAME = os.getenv('DB_NAME', 'jiuzhaigou_fyp')
DB_PORT = os.getenv('DB_PORT', '3306')

# Local SQLite Configuration (Fallback/Dev)
SQLITE_PATH = os.path.join(base_dir, 'jiuzhaigou_local.db')

def get_db_engine(use_cloud=False):
    """
    Get SQLAlchemy engine.
    :param use_cloud: If True, connects to Cloud MySQL. If False, connects to local SQLite.
    """
    if use_cloud:
        # Construct MySQL connection string
        # PyMySQL is required: pip install pymysql
        db_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        print(f"Connecting to Cloud Database: {DB_HOST}...")
    else:
        # Construct SQLite connection string
        db_url = f"sqlite:///{SQLITE_PATH}"
        print(f"Connecting to Local Database: {SQLITE_PATH}...")
    
    engine = create_engine(db_url)
    return engine
