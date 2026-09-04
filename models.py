from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone

# Initialize SQLAlchemy extension instance
db = SQLAlchemy()

class PollutionData(db.Model):
    """Model representing atmospheric pollution records."""
    __tablename__ = 'pollution_data'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    date = db.Column(db.Date, nullable=True)
    time = db.Column(db.Time, nullable=True)
    location = db.Column(db.String(100), nullable=True)
    pm25 = db.Column(db.Float, nullable=True)
    pm10 = db.Column(db.Float, nullable=True)
    co = db.Column(db.Float, nullable=True)
    no2 = db.Column(db.Float, nullable=True)
    so2 = db.Column(db.Float, nullable=True)
    o3 = db.Column(db.Float, nullable=True)
    temperature = db.Column(db.Float, nullable=True)
    humidity = db.Column(db.Float, nullable=True)
    aqi = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<PollutionData id={self.id} location={self.location} aqi={self.aqi}>"


class Prediction(db.Model):
    """Model representing AQI prediction outputs."""
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    pm25 = db.Column(db.Float, nullable=True)
    pm10 = db.Column(db.Float, nullable=True)
    co = db.Column(db.Float, nullable=True)
    no2 = db.Column(db.Float, nullable=True)
    so2 = db.Column(db.Float, nullable=True)
    o3 = db.Column(db.Float, nullable=True)
    temperature = db.Column(db.Float, nullable=True)
    humidity = db.Column(db.Float, nullable=True)
    predicted_aqi = db.Column(db.Float, nullable=True)
    category = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<Prediction id={self.id} predicted_aqi={self.predicted_aqi} category={self.category}>"
