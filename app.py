from flask import Flask, render_template, request
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from product_recognition import recognize_product_from_image  
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import io
import base64
from Product_recommendation import generate_product_variants
# Use the 'Agg' backend for Matplotlib
plt.switch_backend('Agg')

app = Flask(__name__)

# Initialize the API wrapper
api_wrapper = TavilySearchAPIWrapper(tavily_api_key="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")
tool = TavilySearchResults(api_wrapper=api_wrapper)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    product_name = request.form.get('product_name', '')
    image = request.files.get('product_image')

    if product_name:  # Text-based search
        results = tool.invoke(f"""Search for "{product_name}" across the following websites: Flipkart, Amazon, Myntra, Reliance Digital, Snapdeal, and Croma. For each result, return:
        The URL of the product page. Sort the results by price in ascending order. Ensure the URLs are valid and the prices are accurate for the specified product.
        """)
        print("Search Results for text:", results)
        return render_template('searchresults.html', results=results, query=product_name)

    elif image:  # Image-based search
        if image.filename == '':
            return "No selected file"
        
        image_path = os.path.join('static', 'uploads', image.filename)
        image.save(image_path)  # Save uploaded image
        
        recognised_product_name = recognize_product_from_image(image_path)
        print("Recognized Product:", recognised_product_name)

        results = tool.invoke(f"""Search for "{recognised_product_name}" across the following websites: Flipkart, Amazon, Myntra, Reliance Digital, Snapdeal, and Croma. For each result, return:
        The URL of the product page. Sort the results by price in ascending order. Ensure the URLs are valid and the prices are accurate for the specified product.
        """)
        print("Search Results for image:", results)
        return render_template('searchresults.html', results=results, query=recognised_product_name)

    else:
        return "No product name or image provided"

@app.route('/predict', methods=['POST'])
def predict():
        product_name = request.form.get('product_name', '')
        def generate_sample_data(product_name):
            if 'apple' in product_name.lower() or "iphone" in product_name.lower():
                price_range = (50000, 150000)
            elif 'laptop' in product_name.lower():
                price_range = (30000, 100000)
            elif 'smartphone' in product_name.lower():
                price_range = (10000, 80000)
            elif 'tv' in product_name.lower():
                price_range = (15000, 120000)
            elif "electric fan" in product_name.lower() or "fan" in product_name.lower():
                price_range = (1000, 5000)
            else:
                price_range = (100, 5000)            
            
            data = {
                'date': pd.date_range(start='2025-01-01', periods=30, freq='D'),
                'product_name': [product_name] * 100,
                'price': np.random.uniform(price_range[0], price_range[1], 100)
            }
            
            return pd.DataFrame(data)

        def predict_prices(product_name, days=7):
            df = generate_sample_data(product_name)
            df.set_index(pd.DatetimeIndex(df['date'], freq='D'), inplace=True)
            
            model = ARIMA(df['price'], order=(5, 1, 0))
            model_fit = model.fit()
            
            forecast = model_fit.forecast(steps=days)
            
            forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days, freq='D')
            forecast_df = pd.DataFrame({'price': forecast}, index=forecast_dates)
            
            plt.figure(figsize=(12, 6))
            plt.plot(df.index[-7:], df['price'][-7:], label='Historical Prices', color='blue', marker='o')
            plt.plot(forecast_df.index, forecast_df['price'], label='Predicted Prices', color='red', linestyle='--', marker='o')
            
            plt.title(f'Price Prediction for {product_name}')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            
            # Save plot to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            # Encode plot to base64 string
            plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            return plot_base64
        if product_name:
            plot_base64 = predict_prices(product_name)
        return render_template('predictresults.html', plot_base64=plot_base64, query=product_name)
        


@app.route('/recommend', methods=['POST'])

def recommend():
    product_name = request.form.get('product_name', '')
    variants,links=generate_product_variants(product_name)
    return render_template('recommendations.html', variants=variants,links=links, query=product_name)

if __name__ == '__main__':
    app.run(debug=True)
