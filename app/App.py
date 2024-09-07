from flask import Flask
from nyc_trip_duration import nyc_trip_duration_bp
from air_quality import air_quality_bp
from chicago_crimes import chicago_crimes_bp
from world_bank_infra import world_bank_infra_bp

app = Flask(__name__)

# Register Blueprints
app.register_blueprint(nyc_trip_duration_bp, url_prefix='/nyc')
app.register_blueprint(air_quality_bp, url_prefix='/air-quality')
app.register_blueprint(chicago_crimes_bp, url_prefix='/crimes')
app.register_blueprint(world_bank_infra_bp, url_prefix='/infra')

if __name__ == '__main__':
    app.run(debug=True)
