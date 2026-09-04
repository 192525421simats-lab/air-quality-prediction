import os
import pymysql
from dotenv import load_dotenv

# Load .env explicitly with override=True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, '.env')
load_dotenv(DOTENV_PATH, override=True)

from app import app, db
from models import PollutionData, Prediction

DB_HOST = os.getenv('DB_HOST', 'localhost').strip()
DB_PORT = int(os.getenv('DB_PORT', '3306').strip() if os.getenv('DB_PORT') else 3306)
DB_USER = os.getenv('DB_USER', 'root').strip()
DB_PASSWORD = os.getenv('DB_PASSWORD', '')
DB_NAME = os.getenv('DB_NAME', 'air_quality_db').strip()


def create_database():
    """Create the MySQL database if it does not already exist."""
    print(f"[*] Connecting to MySQL server at {DB_HOST}:{DB_PORT} as user '{DB_USER}'...")
    connection = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        autocommit=True
    )
    with connection.cursor() as cursor:
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
        )
        print(f"[OK] Database '{DB_NAME}' checked/created successfully.")
    connection.close()


def create_tables():
    """Create all SQLAlchemy tables if they do not exist."""
    with app.app_context():
        db.create_all()
        print("[OK] Tables 'pollution_data' and 'predictions' checked/created successfully.")


if __name__ == '__main__':
    try:
        create_database()
        create_tables()
        print("\n[SUCCESS] Database initialization completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Database initialization failed: {e}")
        print("Please verify the database configuration in .env and ensure MySQL Server is running.")
