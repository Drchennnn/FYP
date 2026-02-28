import os
import glob
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables from project root
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, '.env'))

# Database Configuration - SQLite
DB_PATH = os.path.join(base_dir, 'jiuzhaigou_fyp.db')
db_url = f"sqlite:///{DB_PATH}"

OUTPUT_RUNS_DIR = os.path.join(base_dir, 'output', 'runs')
PROCESSED_DATA_DIR = os.path.join(base_dir, 'data', 'processed')

def get_latest_processed_csv():
    """Get the latest processed data file (real historical data)"""
    files = list(glob.glob(os.path.join(PROCESSED_DATA_DIR, 'jiuzhaigou_daily_features_*.csv')))
    # Filter out 'latest' if timestamped ones exist, similar to logic elsewhere
    timestamped_files = [f for f in files if 'latest' not in f]
    
    if timestamped_files:
        return max(timestamped_files, key=os.path.getmtime)
    
    # If no timestamped files, use whatever is there (even if it has 'latest' in name)
    # This ensures we don't fail if only the latest file exists
    if files:
        return max(files, key=os.path.getmtime)
        
    raise FileNotFoundError(f"No processed CSV files found in {PROCESSED_DATA_DIR}")

def get_latest_prediction_csv():
    """Find the latest prediction CSV in output/runs"""
    if not os.path.exists(OUTPUT_RUNS_DIR):
        return None
    
    # Get all run directories
    run_dirs = glob.glob(os.path.join(OUTPUT_RUNS_DIR, 'run_*'))
    if not run_dirs:
        return None
        
    # Find latest run dir
    latest_run_dir = max(run_dirs, key=os.path.getmtime)
    csv_path = os.path.join(latest_run_dir, 'lstm_test_predictions.csv')
    
    return csv_path if os.path.exists(csv_path) else None

def sync_data():
    try:
        engine = create_engine(db_url)
        
        with engine.connect() as conn:
            # Drop old table to ensure schema consistency
            conn.execute(text("DROP TABLE IF EXISTS traffic_records"))
            
            # Create Table matching models.py definition exactly
            # models.py: id (PK), record_date, actual_visitor, predicted_visitor, is_forecast
            conn.execute(text("""
                CREATE TABLE traffic_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_date DATE NOT NULL UNIQUE,
                    actual_visitor INTEGER,
                    predicted_visitor INTEGER,
                    is_forecast BOOLEAN DEFAULT 0
                )
            """))
            
            # 1. Sync Real Historical Data
            try:
                processed_csv = get_latest_processed_csv()
                print(f"Found latest processed CSV: {processed_csv}")
                print(f"Syncing REAL data to DB: {DB_PATH}")
                
                df_real = pd.read_csv(processed_csv)
                df_real['date'] = pd.to_datetime(df_real['date']).dt.strftime('%Y-%m-%d')
                df_real['tourism_num'] = pd.to_numeric(df_real['tourism_num'], errors='coerce').fillna(0).astype(int)
                
                # Insert data mapping to correct column names (actual_visitor)
                upsert_real_sql = text("""
                    INSERT INTO traffic_records (record_date, actual_visitor, is_forecast)
                    VALUES (:record_date, :actual_visitor, 0)
                """)
                
                for _, row in df_real.iterrows():
                    conn.execute(upsert_real_sql, {
                        "record_date": row['date'],
                        "actual_visitor": row['tourism_num']
                    })
                print(f"Synced {len(df_real)} real records.")
                
            except FileNotFoundError:
                print("Warning: No processed data found to sync.")
            except Exception as e:
                print(f"Error syncing real data: {e}")

            # 2. Sync Prediction Data (Test Set Predictions)
            try:
                pred_csv = get_latest_prediction_csv()
                if pred_csv:
                    print(f"Found latest prediction CSV: {pred_csv}")
                    df_pred = pd.read_csv(pred_csv)
                    df_pred['date'] = pd.to_datetime(df_pred['date']).dt.strftime('%Y-%m-%d')
                    
                    # Update predicted_visitor for existing records
                    # Note: We only update predicted_visitor, we don't insert new rows for predictions 
                    # because test predictions should correspond to existing dates (mostly)
                    # or if they are future, we insert them.
                    
                    # Upsert logic for predictions
                    upsert_pred_sql = text("""
                        INSERT INTO traffic_records (record_date, predicted_visitor, is_forecast)
                        VALUES (:record_date, :predicted_visitor, 0)
                        ON CONFLICT(record_date) DO UPDATE SET
                        predicted_visitor = excluded.predicted_visitor
                    """)
                    
                    count = 0
                    for _, row in df_pred.iterrows():
                        # Handle potential NaN
                        if pd.isna(row['y_pred']): continue
                        
                        conn.execute(upsert_pred_sql, {
                            "record_date": row['date'],
                            "predicted_visitor": int(row['y_pred'])
                        })
                        count += 1
                    print(f"Synced {count} prediction records.")
                else:
                    print("No prediction CSV found. Skipping prediction sync.")
                    
            except Exception as e:
                print(f"Error syncing prediction data: {e}")
            
            conn.commit()
            print("Database synchronization completed.")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during synchronization: {e}")

if __name__ == "__main__":
    sync_data()
