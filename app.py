import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import logging
import sqlite3
from sqlite3 import Error

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)


# Database connection setup
def create_connection(db_file):
    """Create a database connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        logging.info(f"SQLite DB connected to {db_file}")
    except Error as e:
        logging.error(f"Error connecting to database: {e}")
    return conn


# Initialize the database and create table if it doesn't exist
def init_db():
    conn = create_connection("predictions.db")
    try:
        with conn:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                features TEXT NOT NULL,
                prediction REAL NOT NULL,
                risk_category TEXT NOT NULL,
                advice TEXT NOT NULL
            );
            """
            conn.execute(create_table_sql)
            logging.info("Table `predictions` is ready.")
    except Error as e:
        logging.error(f"Error initializing database: {e}")
    finally:
        if conn:
            conn.close()


init_db()


# Function to save prediction to database
def save_prediction(features, prediction, risk_category, advice):
    conn = create_connection("predictions.db")
    try:
        with conn:
            sql = """
            INSERT INTO predictions (features, prediction, risk_category, advice)
            VALUES (?, ?, ?, ?);
            """
            conn.execute(sql, (str(features), prediction, risk_category, advice))
            conn.commit()
            logging.info("Prediction saved to database.")

            # Print the database table
            cur = conn.cursor()
            cur.execute("SELECT * FROM predictions")
            rows = cur.fetchall()
            logging.info("Database content:")
            for row in rows:
                logging.info(row)
    except Error as e:
        logging.error(f"Error saving prediction to database: {e}")
    finally:
        if conn:
            conn.close()


try:
    model = pickle.load(open("model.pkl", "rb"))
except (FileNotFoundError, IOError) as e:
    logging.error("Error loading model: %s", e)
    model = None


def classify_risk(pm25):
    try:
        if pm25 <= 12:
            return "Good"
        elif pm25 <= 35.4:
            return "Moderate"
        elif pm25 <= 55.4:
            return "Unhealthy for Sensitive Groups"
        elif pm25 <= 150.4:
            return "Unhealthy"
        elif pm25 <= 250.4:
            return "Very Unhealthy"
        else:
            return "Hazardous"
    except Exception as e:
        logging.error("Error classifying risk: %s", e)
        return "Unknown"


def get_advice_for_heart_patients(risk_category):
    try:
        advice = {
            "Good": "Air quality is good. No specific precautions needed.",
            "Moderate": "Air quality is acceptable. Heart patients should consider limiting prolonged exertion.",
            "Unhealthy for Sensitive Groups": "Members of sensitive groups, including heart patients, may experience health effects. It's advisable to limit outdoor exertion.",
            "Unhealthy": "Everyone may begin to experience health effects. Heart patients should avoid outdoor exertion.",
            "Very Unhealthy": "Health alert: everyone may experience more serious health effects. Heart patients should stay indoors and limit physical activity.",
            "Hazardous": "Health warning of emergency conditions. The entire population is likely to be affected. Heart patients should stay indoors and avoid physical activities.",
        }
        return advice[risk_category]
    except KeyError as e:
        logging.error("Invalid risk category: %s", e)
        return "No advice available."


@app.route("/")
def home():
    try:
        return render_template("index.html")
    except Exception as e:
        logging.error("Error rendering home page: %s", e)
        return "An error occurred while loading the home page."


@app.route("/predict", methods=["POST"])
def predict():
    try:
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]

        if model:
            prediction = model.predict(final_features)
        else:
            raise ValueError("Model not loaded.")

        pm25 = round(prediction[0], 2)
        risk_category = classify_risk(pm25)
        advice = get_advice_for_heart_patients(risk_category)

        # Save the prediction to the database
        save_prediction(int_features, pm25, risk_category, advice)

        return render_template(
            "index.html", pm25=pm25, risk_category=risk_category, advice=advice
        )
    except ValueError as e:
        logging.error("Error in prediction: %s", e)
        return "An error occurred during prediction."
    except Exception as e:
        logging.error("Unexpected error: %s", e)
        return "An unexpected error occurred."


@app.route("/information")
def information():
    try:
        return render_template("information.html")
    except Exception as e:
        logging.error("Error rendering information page: %s", e)
        return "An error occurred while loading the information page."


@app.route("/predict_api", methods=["POST"])
def predict_api():
    try:
        data = request.get_json(force=True)
        final_features = [np.array(list(data.values()))]

        if model:
            prediction = model.predict(final_features)
        else:
            raise ValueError("Model not loaded.")

        pm25 = prediction[0]
        risk_category = classify_risk(pm25)
        advice = get_advice_for_heart_patients(risk_category)

        # Save the prediction to the database
        save_prediction(data.values(), pm25, risk_category, advice)

        return jsonify({"pm25": pm25, "risk_category": risk_category, "advice": advice})
    except ValueError as e:
        logging.error("Error in API prediction: %s", e)
        return jsonify({"error": str(e)})
    except Exception as e:
        logging.error("Unexpected error in API prediction: %s", e)
        return jsonify({"error": "An unexpected error occurred."})


if __name__ == "__main__":
    app.run(debug=True)
