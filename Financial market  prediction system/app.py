from flask import Flask, request, render_template, jsonify
import yfinance as yf
import pandas as pd
from prophet import Prophet

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_data', methods=['POST'])
def get_data():
    try:
        ticker = request.form['ticker'].upper()
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        # Fetch stock data
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            return jsonify({'error': 'No data found. Check the ticker and dates.'})

        # Convert DataFrame to JSON
        data_json = data[['Open', 'High', 'Low', 'Close', 'Volume']].to_dict(orient='records')

        return jsonify({'ticker': ticker, 'historical_data': data_json})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        ticker = request.form['ticker'].upper()
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        # Fetch stock data
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            return jsonify({'error': 'No data found. Check the ticker and dates.'})

        # Prepare data for Prophet
        df = data[['Open']].reset_index()
        df.columns = ['ds', 'y']

        # Train Prophet model
        model = Prophet()
        model.fit(df)

        # Make future predictions (30 days ahead)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Get prediction results
        last_actual_price = df['y'].iloc[-1]
        predicted_price = forecast['yhat'].iloc[-1]
        trend = "UP" if predicted_price > last_actual_price else "DOWN"

        # Identify currency symbol (₹ for Indian stocks, $ otherwise)
        currency_symbol = "₹" if ticker.endswith(".NS") or ticker.endswith(".BO") else "$"

        return jsonify({
            'ticker': ticker,
            'current_price': f"{currency_symbol}{round(last_actual_price, 2)}",
            'predicted_price': f"{currency_symbol}{round(predicted_price, 2)}",
            'trend': trend
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
