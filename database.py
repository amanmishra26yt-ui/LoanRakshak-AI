from pathlib import Path
import sqlite3
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "loanrakshak_history.db"


def get_connection():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                applicant_name TEXT NOT NULL,
                annual_revenue REAL NOT NULL,
                annual_profit REAL NOT NULL,
                loan_amount REAL NOT NULL,
                credit_score INTEGER NOT NULL,
                business_age_years INTEGER NOT NULL,
                existing_debt REAL NOT NULL,
                late_payments INTEGER NOT NULL,
                default_probability REAL NOT NULL,
                risk_level TEXT NOT NULL
            )
            """
        )


def save_prediction(
    applicant_name,
    annual_revenue,
    annual_profit,
    loan_amount,
    credit_score,
    business_age_years,
    existing_debt,
    late_payments,
    default_probability,
    risk_level,
):
    initialize_database()

    created_at = datetime.now().astimezone().isoformat(timespec="seconds")

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO prediction_history (
                created_at,
                applicant_name,
                annual_revenue,
                annual_profit,
                loan_amount,
                credit_score,
                business_age_years,
                existing_debt,
                late_payments,
                default_probability,
                risk_level
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                applicant_name,
                annual_revenue,
                annual_profit,
                loan_amount,
                credit_score,
                business_age_years,
                existing_debt,
                late_payments,
                default_probability,
                risk_level,
            ),
        )


def get_prediction_history(limit=500):
    initialize_database()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM prediction_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()

    return [dict(row) for row in rows]


def clear_prediction_history():
    """Delete saved prediction records."""
    initialize_database()

    with get_connection() as connection:
        connection.execute("DELETE FROM prediction_history")