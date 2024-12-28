from flask import Flask, request, render_template
import joblib
import numpy as np

# Inisialisasi Flask
app = Flask(__name__)

# Load model
try:
    model = joblib.load('house_price_model.pkl')
except FileNotFoundError:
    raise FileNotFoundError("Model file 'house_price_model.pkl' not found. Ensure the file exists in the correct directory.")

# Fungsi untuk validasi dan preprocessing input
def preprocess_input(data):
    # Map untuk encoding kategori
    encoding_map = {
    'Neighborhood': {
        'CollgCr': 0, 'Veenker': 1, 'Crawfor': 2, 'NAmes': 3, 'Gilbert': 4, 
        'StoneBr': 5, 'BrDale': 6, 'NPkVill': 7, 'NridgHt': 8, 'Blmngtn': 9, 
        'NoRidge': 10, 'Somerst': 11, 'SawyerW': 12, 'Sawyer': 13, 
        'OldTown': 14, 'BrkSide': 15, 'ClearCr': 16, 'SWISU': 17, 
        'Edwards': 18, 'Blueste': 19, 'IDOTRR': 20, 'Mitchel': 21, 
        'Timber': 22, 'MeadowV': 23
    },

    'Exterior1st': {
    'VinylSd': 0, 'Wd Sdng': 1, 'HdBoard': 2, 'MetalSd': 3, 'Plywood': 4, 
    'CemntBd': 5, 'WdShing': 6, 'BrkFace': 7, 'AsbShng': 8
    },

    'RoofMatl': {'CompShg': 0, 'Metal': 1, 'WdShake': 2},
    'BsmtQual': {'Ex': 0, 'Gd': 1, 'TA': 2},
    'CentralAir': {'Yes': 1, 'No': 0},
    'KitchenQual': {'Ex': 0, 'Gd': 1, 'TA': 2}
}

    # Validasi dan konversi input
    try:
        processed = [
            encoding_map['Neighborhood'][data['Neighborhood']],
            int(data['OverallQual']),
            encoding_map['RoofMatl'][data['RoofMatl']],
            encoding_map['BsmtQual'][data['BsmtQual']],
            encoding_map['CentralAir'][data['CentralAir']],
            float(data['1stFlrSF']),
            float(data['GrLivArea']),
            int(data['FullBath']),
            encoding_map['KitchenQual'][data['KitchenQual']],
            int(data['GarageCars'])
        ]
        return np.array(processed).reshape(1, -1)
    except KeyError as e:
        raise ValueError(f"Invalid or missing value for {e.args[0]}.")
    except ValueError as e:
        raise ValueError(f"Invalid value type: {e}")

# Halaman utama untuk frontend
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari form (POST request)
        data = request.form.to_dict()
        
        # Preprocess data input
        input_array = preprocess_input(data)
        
        # Prediksi harga
        prediction = model.predict(input_array)[0]
        
        # Format prediksi ke mata uang
        formatted_prediction = f"${prediction:,.2f}"
        
        # Render hasil ke halaman
        return render_template('index.html', prediction_text=f"Predicted House Price: {formatted_prediction}")
    except ValueError as ve:
        # Handle error validasi
        return render_template('index.html', error_text=f"Input Error: {ve}")
    except Exception as e:
        # Handle error tidak terduga
        return render_template('index.html', error_text=f"Unexpected Error: {e}")

# Jalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)