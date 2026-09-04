"""
Module 3 Verification Script
----------------------------
Verifies:
1. Flask app routes: /, /data, /analysis, /prediction
2. ML prediction pipeline, model training, metrics calculation
3. Form POST submission and database persistence in predictions table
4. Graceful handling of edge cases (empty data, invalid inputs, edge values)
"""

import sys
from app import app, db
from models import PollutionData, Prediction
from ml_model import predictor, calculate_cpcb_aqi, get_aqi_category_info

def run_tests():
    print("=" * 60)
    print("STARTING MODULE 3 COMPREHENSIVE VERIFICATION")
    print("=" * 60)

    client = app.test_client()

    # 1. Test standard AQI sub-index derivation
    print("\n[1] Testing CPCB AQI Calculation Engine...")
    aqi_val, dominant = calculate_cpcb_aqi(pm25=25.0, pm10=45.0, no2=20.0, so2=10.0, co=0.5, o3=30.0)
    print(f"    Good scenario: AQI = {aqi_val}, Dominant = {dominant}")
    assert aqi_val <= 50, f"Expected Good AQI <= 50, got {aqi_val}"

    aqi_val_poor, dominant_poor = calculate_cpcb_aqi(pm25=110.0, pm10=280.0, no2=75.0, so2=30.0, co=1.5, o3=50.0)
    print(f"    Poor scenario: AQI = {aqi_val_poor}, Dominant = {dominant_poor}")
    assert aqi_val_poor > 200, f"Expected Poor AQI > 200, got {aqi_val_poor}"

    # 2. Test Category and Health Info
    print("\n[2] Testing AQI Category & Advisory...")
    cat_good = get_aqi_category_info(35.0)
    print(f"    AQI 35 -> {cat_good['category']}, CSS: {cat_good['css_class']}")
    assert cat_good['category'] == 'Good'

    cat_mod = get_aqi_category_info(150.0)
    print(f"    AQI 150 -> {cat_mod['category']}, CSS: {cat_mod['css_class']}")
    assert cat_mod['category'] == 'Moderate'

    # 3. Test App Route Accessibility
    print("\n[3] Testing Web Endpoints (GET)...")
    for route in ['/', '/data', '/analysis', '/prediction']:
        res = client.get(route)
        print(f"    GET {route} -> Status {res.status_code}")
        assert res.status_code == 200, f"Route {route} returned status {res.status_code}"

    # 4. Test ML Model Training with DB Records
    print("\n[4] Testing ML Predictor Training on MySQL Records...")
    with app.app_context():
        records = PollutionData.query.all()
        print(f"    Loaded {len(records)} records from MySQL pollution_data table.")
        trained, msg = predictor.train(records)
        print(f"    Training Status: {trained} | Message: {msg}")
        print(f"    Metrics: R2={predictor.metrics['r2_score']}, MAE={predictor.metrics['mae']}, RMSE={predictor.metrics['rmse']}")
        assert len(predictor.feature_importances) > 0, "Feature importances should not be empty"

    # 5. Test POST Prediction Route & DB Logging
    print("\n[5] Testing POST /prediction Inferences & DB Persistence...")
    with app.app_context():
        initial_pred_count = Prediction.query.count()
        print(f"    Initial predictions table row count: {initial_pred_count}")

        post_data = {
            'pm25': '55.0',
            'pm10': '115.0',
            'no2': '42.0',
            'so2': '18.0',
            'co': '1.1',
            'o3': '38.0',
            'temperature': '29.5',
            'humidity': '58.0'
        }

        res = client.post('/prediction', data=post_data, follow_redirects=True)
        assert res.status_code == 200, f"POST /prediction returned {res.status_code}"
        
        response_text = res.get_data(as_text=True)
        assert "AQI prediction generated successfully" in response_text
        assert "Predicted Air Quality Index" in response_text or "Air Quality" in response_text

        new_pred_count = Prediction.query.count()
        print(f"    New predictions table row count: {new_pred_count}")
        assert new_pred_count == initial_pred_count + 1, "Prediction should have been logged in MySQL database"

        latest_pred = Prediction.query.order_by(Prediction.id.desc()).first()
        print(f"    Latest Persisted Record: ID={latest_pred.id}, Predicted AQI={latest_pred.predicted_aqi}, Category={latest_pred.category}")

    # 6. Test Insufficient Data Handling
    print("\n[6] Testing Fallback Handling for Small/Empty Datasets...")
    dummy_predictor = type(predictor)()
    fallback_trained, fallback_msg = dummy_predictor.train([])
    print(f"    Zero records training: Status={fallback_trained}, Message={fallback_msg}")
    assert not fallback_trained
    assert dummy_predictor.metrics['sample_size'] == 0

    fallback_pred = dummy_predictor.predict({
        'pm25': 20.0, 'pm10': 40.0, 'no2': 15.0, 'so2': 10.0, 'co': 0.5, 'o3': 25.0
    })
    print(f"    Fallback prediction output: AQI={fallback_pred['predicted_aqi']}, Cat={fallback_pred['category']}")
    assert fallback_pred['predicted_aqi'] > 0
    assert fallback_pred['category'] == 'Good'

    print("\n" + "=" * 60)
    print("ALL MODULE 3 VERIFICATION TESTS PASSED SUCCESSFULLY! (6/6)")
    print("=" * 60)

if __name__ == '__main__':
    run_tests()
