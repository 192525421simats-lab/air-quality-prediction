import os
import json
import urllib.parse
from datetime import datetime, timezone
from flask import Flask, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv
from sqlalchemy import text, inspect
import pymysql

# Explicitly load project .env file with override enabled
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, '.env')
load_dotenv(DOTENV_PATH, override=True)

# Import database instance and models
from models import db, PollutionData, Prediction

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-dev-secret-key')

# MySQL Database Configuration from .env
DB_HOST = os.getenv('DB_HOST', 'localhost').strip()
DB_PORT = int(os.getenv('DB_PORT', '3306').strip() if os.getenv('DB_PORT') else 3306)
DB_USER = os.getenv('DB_USER', 'root').strip()
DB_PASSWORD = os.getenv('DB_PASSWORD', '')
DB_NAME = os.getenv('DB_NAME', 'air_quality_db').strip()

# Safely encode user and password for SQLAlchemy URI
encoded_user = urllib.parse.quote_plus(DB_USER)
encoded_password = urllib.parse.quote_plus(DB_PASSWORD)

# Construct SQLAlchemy Database URI dynamically from .env
app.config['SQLALCHEMY_DATABASE_URI'] = (
    f"mysql+pymysql://{encoded_user}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Bind SQLAlchemy to Flask app
db.init_app(app)


def init_database():
    """
    Safely creates the MySQL database (if it doesn't exist)
    and initializes all SQLAlchemy tables (pollution_data, predictions).
    """
    try:
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
        connection.close()

        with app.app_context():
            db.create_all()
        return True, "Database initialized successfully."
    except Exception as e:
        error_msg = str(e)
        return False, error_msg


# Initialize tables when starting application
init_database()


# ---------------------------------------------------------
# Web Routes
# ---------------------------------------------------------

@app.route('/')
def home():
    """Home / Landing page."""
    return render_template('index.html')


@app.route('/pollution-data')
def pollution_data_redirect():
    """Redirect legacy link to the data collection page."""
    return redirect(url_for('data_collection'))


@app.route('/data', methods=['GET', 'POST'])
def data_collection():
    """
    Module 1: Pollution Data Collection
    - GET: Displays the collection form and all previously collected records.
    - POST: Validates and saves new pollution records to the database.
    """
    if request.method == 'POST':
        date_str = request.form.get('date', '').strip()
        time_str = request.form.get('time', '').strip()
        location = request.form.get('location', '').strip()
        pm25_str = request.form.get('pm25', '').strip()
        pm10_str = request.form.get('pm10', '').strip()
        co_str = request.form.get('co', '').strip()
        no2_str = request.form.get('no2', '').strip()
        so2_str = request.form.get('so2', '').strip()
        o3_str = request.form.get('o3', '').strip()
        temp_str = request.form.get('temperature', '').strip()
        humidity_str = request.form.get('humidity', '').strip()

        # 1. Required fields check
        if not all([date_str, time_str, location, pm25_str, pm10_str, co_str, no2_str, so2_str, o3_str, temp_str, humidity_str]):
            flash("All fields are required. Please fill out the entire form.", "danger")
            return redirect(url_for('data_collection'))

        # 2. Parse and validate numeric values
        try:
            pm25 = float(pm25_str)
            pm10 = float(pm10_str)
            co = float(co_str)
            no2 = float(no2_str)
            so2 = float(so2_str)
            o3 = float(o3_str)
            temperature = float(temp_str)
            humidity = float(humidity_str)
        except ValueError:
            flash("Invalid numeric input. Please enter valid numbers for pollutant and weather metrics.", "danger")
            return redirect(url_for('data_collection'))

        # 3. Value range validation
        if any(val < 0 for val in [pm25, pm10, co, no2, so2, o3]):
            flash("Pollutant values (PM2.5, PM10, CO, NO2, SO2, O3) cannot be negative.", "danger")
            return redirect(url_for('data_collection'))

        if not (0.0 <= humidity <= 100.0):
            flash("Humidity percentage must be between 0 and 100.", "danger")
            return redirect(url_for('data_collection'))

        # 4. Parse Date and Time objects
        try:
            parsed_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            flash("Invalid Date format. Please use YYYY-MM-DD.", "danger")
            return redirect(url_for('data_collection'))

        try:
            if len(time_str) == 5:
                parsed_time = datetime.strptime(time_str, '%H:%M').time()
            else:
                parsed_time = datetime.strptime(time_str, '%H:%M:%S').time()
        except ValueError:
            flash("Invalid Time format.", "danger")
            return redirect(url_for('data_collection'))

        # 5. Save to MySQL database
        try:
            init_database()
            new_record = PollutionData(
                date=parsed_date,
                time=parsed_time,
                location=location,
                pm25=pm25,
                pm10=pm10,
                co=co,
                no2=no2,
                so2=so2,
                o3=o3,
                temperature=temperature,
                humidity=humidity,
                aqi=None  # Left empty/NULL; handled in subsequent module
            )
            db.session.add(new_record)
            db.session.commit()
            flash(f"Pollution record for '{location}' saved successfully!", "success")
        except Exception as e:
            db.session.rollback()
            flash(f"Failed to save record to database: {str(e)}", "danger")

        return redirect(url_for('data_collection'))

    # GET Request: Fetch records newest first
    records = []
    try:
        init_database()
        records = PollutionData.query.order_by(PollutionData.id.desc()).all()
    except Exception as e:
        flash(f"Notice: Database connection issue ({str(e)}). Please verify database configuration in .env.", "warning")

    today_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M')

    return render_template(
        'data.html',
        records=records,
        today_date=today_date,
        current_time=current_time
    )


@app.route('/analysis')
def analysis():
    """
    Module 2: Air Quality Analysis
    - Retrieves collected records from MySQL.
    - Computes summary statistics (mean, min, max).
    - Determines standard AQI air quality category.
    - Serializes structured time-series and location data for Chart.js.
    """
    records = []
    try:
        init_database()
        # Records ordered reverse chronological for table display
        records = PollutionData.query.order_by(PollutionData.id.desc()).all()
    except Exception as e:
        flash(f"Notice: Could not load data for analysis ({str(e)}).", "warning")

    # Chronological records for time-series charts
    chronological_records = sorted(records, key=lambda r: (r.date if r.date else datetime.min.date(), r.time if r.time else datetime.min.time(), r.id))

    # Initialize statistics dictionary
    stats = {
        'count': len(records),
        'avg_pm25': 0.0, 'min_pm25': 0.0, 'max_pm25': 0.0,
        'avg_pm10': 0.0, 'min_pm10': 0.0, 'max_pm10': 0.0,
        'avg_no2': 0.0, 'min_no2': 0.0, 'max_no2': 0.0,
        'avg_so2': 0.0, 'min_so2': 0.0, 'max_so2': 0.0,
        'avg_co': 0.0, 'min_co': 0.0, 'max_co': 0.0,
        'avg_o3': 0.0, 'min_o3': 0.0, 'max_o3': 0.0,
        'avg_temp': 0.0, 'min_temp': 0.0, 'max_temp': 0.0,
        'avg_humidity': 0.0, 'min_humidity': 0.0, 'max_humidity': 0.0,
    }

    # Air quality status dictionary
    aqi_status = {
        'category_name': 'No Data',
        'category_class': 'moderate',
        'icon': '🌿',
        'description': 'Awaiting environmental data collection entries.',
        'dominant_pollutant': 'None'
    }

    # Chart datasets
    timeline_labels = []
    pm25_series = []
    pm10_series = []
    no2_series = []
    so2_series = []
    o3_series = []
    co_series = []
    temp_series = []
    humidity_series = []
    location_groups = {}

    if records:
        n = len(records)
        stats['avg_pm25'] = round(sum(r.pm25 for r in records if r.pm25 is not None) / n, 2)
        stats['min_pm25'] = round(min(r.pm25 for r in records if r.pm25 is not None), 2)
        stats['max_pm25'] = round(max(r.pm25 for r in records if r.pm25 is not None), 2)

        stats['avg_pm10'] = round(sum(r.pm10 for r in records if r.pm10 is not None) / n, 2)
        stats['min_pm10'] = round(min(r.pm10 for r in records if r.pm10 is not None), 2)
        stats['max_pm10'] = round(max(r.pm10 for r in records if r.pm10 is not None), 2)

        stats['avg_no2'] = round(sum(r.no2 for r in records if r.no2 is not None) / n, 2)
        stats['min_no2'] = round(min(r.no2 for r in records if r.no2 is not None), 2)
        stats['max_no2'] = round(max(r.no2 for r in records if r.no2 is not None), 2)

        stats['avg_so2'] = round(sum(r.so2 for r in records if r.so2 is not None) / n, 2)
        stats['min_so2'] = round(min(r.so2 for r in records if r.so2 is not None), 2)
        stats['max_so2'] = round(max(r.so2 for r in records if r.so2 is not None), 2)

        stats['avg_co'] = round(sum(r.co for r in records if r.co is not None) / n, 2)
        stats['min_co'] = round(min(r.co for r in records if r.co is not None), 2)
        stats['max_co'] = round(max(r.co for r in records if r.co is not None), 2)

        stats['avg_o3'] = round(sum(r.o3 for r in records if r.o3 is not None) / n, 2)
        stats['min_o3'] = round(min(r.o3 for r in records if r.o3 is not None), 2)
        stats['max_o3'] = round(max(r.o3 for r in records if r.o3 is not None), 2)

        stats['avg_temp'] = round(sum(r.temperature for r in records if r.temperature is not None) / n, 1)
        stats['min_temp'] = round(min(r.temperature for r in records if r.temperature is not None), 1)
        stats['max_temp'] = round(max(r.temperature for r in records if r.temperature is not None), 1)

        stats['avg_humidity'] = round(sum(r.humidity for r in records if r.humidity is not None) / n, 1)
        stats['min_humidity'] = round(min(r.humidity for r in records if r.humidity is not None), 1)
        stats['max_humidity'] = round(max(r.humidity for r in records if r.humidity is not None), 1)

        # Categorize overall atmospheric health using standard particulate breakpoints
        avg_pm25 = stats['avg_pm25']
        avg_pm10 = stats['avg_pm10']

        if avg_pm25 <= 30 and avg_pm10 <= 50:
            aqi_status['category_name'] = 'Good'
            aqi_status['category_class'] = 'good'
            aqi_status['icon'] = '🌱'
            aqi_status['description'] = 'Air quality is considered satisfactory, and air pollution poses little or no risk.'
        elif avg_pm25 <= 60 and avg_pm10 <= 100:
            aqi_status['category_name'] = 'Moderate / Satisfactory'
            aqi_status['category_class'] = 'moderate'
            aqi_status['icon'] = '🌤️'
            aqi_status['description'] = 'Air quality is acceptable; minor breathing discomfort may occur for unusually sensitive individuals.'
        elif avg_pm25 <= 90 and avg_pm10 <= 250:
            aqi_status['category_name'] = 'Poor'
            aqi_status['category_class'] = 'poor'
            aqi_status['icon'] = '🌫️'
            aqi_status['description'] = 'May cause breathing discomfort to people with lung/heart diseases, children and older adults.'
        elif avg_pm25 <= 120 and avg_pm10 <= 350:
            aqi_status['category_name'] = 'Very Poor'
            aqi_status['category_class'] = 'very-poor'
            aqi_status['icon'] = '⚠️'
            aqi_status['description'] = 'Prolonged exposure may lead to respiratory illness. Sensitive groups should avoid outdoor activities.'
        else:
            aqi_status['category_name'] = 'Severe'
            aqi_status['category_class'] = 'severe'
            aqi_status['icon'] = '🚨'
            aqi_status['description'] = 'Emergency health warning: Everyone may experience more serious health effects.'

        aqi_status['dominant_pollutant'] = 'PM2.5 & PM10' if avg_pm25 > 40 or avg_pm10 > 80 else 'Particulate Matter'

        # Build chronological series for charts
        for r in chronological_records:
            label = f"{r.date.strftime('%b %d') if r.date else 'Obs'} {r.time.strftime('%H:%M') if r.time else ''}"
            timeline_labels.append(label)
            pm25_series.append(r.pm25 or 0)
            pm10_series.append(r.pm10 or 0)
            no2_series.append(r.no2 or 0)
            so2_series.append(r.so2 or 0)
            o3_series.append(r.o3 or 0)
            co_series.append(r.co or 0)
            temp_series.append(r.temperature or 0)
            humidity_series.append(r.humidity or 0)

            loc = r.location or 'General Station'
            if loc not in location_groups:
                location_groups[loc] = {'pm25': [], 'pm10': []}
            if r.pm25 is not None:
                location_groups[loc]['pm25'].append(r.pm25)
            if r.pm10 is not None:
                location_groups[loc]['pm10'].append(r.pm10)

    # Location averages
    location_labels = list(location_groups.keys())
    location_pm25 = [round(sum(location_groups[l]['pm25'])/len(location_groups[l]['pm25']), 2) if location_groups[l]['pm25'] else 0 for l in location_labels]
    location_pm10 = [round(sum(location_groups[l]['pm10'])/len(location_groups[l]['pm10']), 2) if location_groups[l]['pm10'] else 0 for l in location_labels]

    chart_dict = {
        'timeline_labels': timeline_labels,
        'pm25_series': pm25_series,
        'pm10_series': pm10_series,
        'no2_series': no2_series,
        'so2_series': so2_series,
        'o3_series': o3_series,
        'co_series': co_series,
        'temp_series': temp_series,
        'humidity_series': humidity_series,
        'location_labels': location_labels,
        'location_pm25': location_pm25,
        'location_pm10': location_pm10,
        'avg_pm25': stats['avg_pm25'],
        'avg_pm10': stats['avg_pm10'],
        'avg_no2': stats['avg_no2'],
        'avg_so2': stats['avg_so2'],
        'avg_o3': stats['avg_o3'],
        'avg_co': stats['avg_co']
    }

    return render_template(
        'analysis.html',
        records=records,
        stats=stats,
        aqi_status=aqi_status,
        chart_json=json.dumps(chart_dict)
    )


# Import ML predictor engine
from ml_model import predictor


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    """
    Module 3: Machine Learning AQI Prediction
    - Trains Random Forest regression model on stored MySQL pollution records.
    - Handles real-time user inputs for pollutant and weather metrics.
    - Computes predicted AQI, category classification, and health advisories.
    - Records prediction history to MySQL database.
    - Displays model diagnostics and performance metrics.
    """
    init_database()
    
    # 1. Fetch records and train / update ML model
    records = []
    try:
        records = PollutionData.query.all()
    except Exception as e:
        flash(f"Notice: Could not load training records from database ({str(e)}).", "warning")

    train_success, train_msg = predictor.train(records)

    model_info = {
        'algorithm': predictor.algorithm_name,
        'records_count': predictor.training_records_count,
        'is_trained': predictor.is_trained,
        'r2_score': predictor.metrics.get('r2_score', 0.0),
        'mae': predictor.metrics.get('mae', 0.0),
        'rmse': predictor.metrics.get('rmse', 0.0),
        'feature_importances': predictor.feature_importances,
        'status_message': train_msg
    }

    prediction_result = None
    form_data = {
        'pm25': '',
        'pm10': '',
        'no2': '',
        'so2': '',
        'co': '',
        'o3': '',
        'temperature': '',
        'humidity': ''
    }

    if request.method == 'POST':
        pm25_str = request.form.get('pm25', '').strip()
        pm10_str = request.form.get('pm10', '').strip()
        no2_str = request.form.get('no2', '').strip()
        so2_str = request.form.get('so2', '').strip()
        co_str = request.form.get('co', '').strip()
        o3_str = request.form.get('o3', '').strip()
        temp_str = request.form.get('temperature', '').strip()
        humidity_str = request.form.get('humidity', '').strip()

        form_data.update({
            'pm25': pm25_str,
            'pm10': pm10_str,
            'no2': no2_str,
            'so2': so2_str,
            'co': co_str,
            'o3': o3_str,
            'temperature': temp_str,
            'humidity': humidity_str
        })

        # Required fields validation
        if not all([pm25_str, pm10_str, no2_str, so2_str, co_str, o3_str]):
            flash("Please enter all required pollutant values to predict AQI.", "danger")
        else:
            try:
                pm25 = float(pm25_str)
                pm10 = float(pm10_str)
                no2 = float(no2_str)
                so2 = float(so2_str)
                co = float(co_str)
                o3 = float(o3_str)
                temp = float(temp_str) if temp_str else 25.0
                hum = float(humidity_str) if humidity_str else 50.0

                if any(val < 0 for val in [pm25, pm10, no2, so2, co, o3]):
                    flash("Pollutant values cannot be negative numbers.", "danger")
                elif not (0.0 <= hum <= 100.0):
                    flash("Humidity must be between 0% and 100%.", "danger")
                else:
                    input_dict = {
                        'pm25': pm25,
                        'pm10': pm10,
                        'no2': no2,
                        'so2': so2,
                        'co': co,
                        'o3': o3,
                        'temperature': temp,
                        'humidity': hum
                    }

                    # Generate ML prediction
                    prediction_result = predictor.predict(input_dict)

                    # Persist prediction entry in MySQL database
                    try:
                        new_pred = Prediction(
                            pm25=pm25,
                            pm10=pm10,
                            no2=no2,
                            so2=so2,
                            co=co,
                            o3=o3,
                            temperature=temp,
                            humidity=hum,
                            predicted_aqi=prediction_result['predicted_aqi'],
                            category=prediction_result['category']
                        )
                        db.session.add(new_pred)
                        db.session.commit()
                    except Exception as e:
                        db.session.rollback()
                        # Non-fatal log: prediction is still shown to user
                        print(f"Prediction DB log notice: {e}")

                    flash("AQI prediction generated successfully!", "success")

            except ValueError:
                flash("Invalid numeric format. Please provide valid numeric values.", "danger")

    # Fetch recent predictions for display
    recent_predictions = []
    try:
        recent_predictions = Prediction.query.order_by(Prediction.id.desc()).limit(8).all()
    except Exception as e:
        print(f"Could not load recent predictions: {e}")

    return render_template(
        'prediction.html',
        model_info=model_info,
        prediction_result=prediction_result,
        form_data=form_data,
        recent_predictions=recent_predictions
    )



@app.route('/db-test')
def db_test():
    """Tests the MySQL database connection and table availability."""
    try:
        init_ok, init_err = init_database()
        db.session.execute(text("SELECT 1"))
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()

        return render_template(
            'db_test.html',
            success=True,
            message="Database connection successful!",
            database=DB_NAME,
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            tables=tables
        )
    except Exception as e:
        error_details = str(e)
        if DB_PASSWORD and DB_PASSWORD in error_details:
            error_details = error_details.replace(DB_PASSWORD, "******")

        return render_template(
            'db_test.html',
            success=False,
            message="Database connection failed!",
            error_details=error_details,
            database=DB_NAME,
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            tables=[]
        ), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    app.run(debug=True, port=port)
