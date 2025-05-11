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
        search_query = f"""
        Find product "{product_name}" on Amazon.in, Flipkart.com, or Reliancedigital.in
        """
        
        try:
            # Print raw results for debugging
            results = tool.invoke(search_query)
            print("Raw API Response:", results)  # Add this to see what the API returns
            
            # If results is empty or None, return a specific message
            if not results:
                return render_template('searchresults.html', 
                                    results=[], 
                                    query=product_name, 
                                    error="API returned no results. Please check your API key.")
            
            filtered_results = []
            allowed_domains = ['amazon.in', 'flipkart.com', 'reliancedigital.in']
            
            # Handle results as a list of dictionaries
            if isinstance(results, list):
                for result in results:
                    if isinstance(result, dict):
                        url = result.get('url', '')
                        if any(domain in url.lower() for domain in allowed_domains):
                            filtered_results.append({
                                'url': url,
                                'title': result.get('title', f'{product_name} on {url.split("/")[2]}'),
                                
                            })
            
            print("Filtered Results:", filtered_results)  # Debug print
            
            return render_template('searchresults.html', 
                                results=filtered_results, 
                                query=product_name)
                                
        except Exception as e:
            print(f"Search Error: {str(e)}")  # Debug print
            return render_template('searchresults.html', 
                                results=[], 
                                query=product_name, 
                                error=f"Search error: {str(e)}")

    elif image:  # Image-based search
        if image.filename == '':
            return "No selected file"
        
        # Ensure the 'static/uploads' directory exists
        upload_dir = os.path.join('static', 'uploads')
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        
        image_path = os.path.join(upload_dir, image.filename)
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

def generate_sample_data(product_name):
    # Get current date and calculate start date (60 days ago)
    end_date = pd.Timestamp.now().normalize()
    start_date = end_date - pd.Timedelta(days=60)
    
    # Set initial price based on product
    if 'apple' in product_name.lower() or "iphone" in product_name.lower():
        base_price = 120000
        percent_variation = 0.03  # 3% variation for prices > 100000
        seasonality = 'technology'  # Technology products have their own pricing patterns
    elif 'laptop' in product_name.lower():
        base_price = 75000
        percent_variation = 0.04  # 4% variation for prices 50000-90000
        seasonality = 'technology'
    elif 'smartphone' in product_name.lower():
        base_price = 35000
        percent_variation = 0.05  # 5% variation for prices 10000-90000
        seasonality = 'technology'
    elif 'tv' in product_name.lower():
        base_price = 45000
        percent_variation = 0.05  # 5% variation
        seasonality = 'technology'
    elif "electric fan" in product_name.lower() or "fan" in product_name.lower():
        base_price = 5000
        percent_variation = 0.15  # 15% variation for prices 1000-9000
        seasonality = 'seasonal'  # Seasonal products have different price patterns
    elif "bag" in product_name.lower() or "school bag" in product_name.lower():
        base_price = 1500
        percent_variation = 0.15
        seasonality = 'consumer'
    elif "bottle" in product_name.lower() or "water bottle" in product_name.lower():
        base_price = 500
        percent_variation = 0.15
        seasonality = 'consumer'
    elif "shoes" in product_name.lower() or "slippers" in product_name.lower():
        base_price = 2000
        percent_variation = 0.15
        seasonality = 'fashion'  # Fashion products have different pricing patterns
    else:
        base_price = 5000
        percent_variation = 0.15
        seasonality = 'general'

    # Generate price variations based on product type
    num_days = 61  # 60 days + today
    prices = []
    
    # Add product-specific trends
    if seasonality == 'technology':
        # Technology products typically show price decay over time with occasional drops
        trend_component = np.linspace(0, -0.08, num_days)  # Overall 8% decay
        
        # Add occasional price drops (like sales or new model releases)
        sales_events = np.zeros(num_days)
        sale_indices = np.random.choice(range(10, num_days), size=3, replace=False)
        for idx in sale_indices:
            sales_events[idx] = -0.05  # 5% drop
            # Recovery period after sale
            recovery_length = min(7, num_days-idx)
            sales_events[idx:idx+recovery_length] = np.linspace(-0.05, 0, recovery_length)
            
    elif seasonality == 'seasonal':
        # Seasonal products have wave patterns
        days = np.arange(num_days)
        trend_component = 0.05 * np.sin(2 * np.pi * days / 30)  # Monthly cycle
        sales_events = np.zeros(num_days)
        
    elif seasonality == 'fashion':
        # Fashion products have more volatility and seasonal trends
        days = np.arange(num_days)
        trend_component = 0.03 * np.sin(2 * np.pi * days / 60)  # Bi-monthly trend
        
        # Add fashion sale events (more frequent)
        sales_events = np.zeros(num_days)
        sale_indices = np.random.choice(range(5, num_days), size=5, replace=False)
        for idx in sale_indices:
            sales_events[idx] = -0.08  # 8% drop during sales
            # Quick recovery for fashion items
            recovery_length = min(5, num_days-idx)
            sales_events[idx:idx+recovery_length] = np.linspace(-0.08, 0, recovery_length)
            
    else:  # consumer and general products
        # General consumer goods have gradual inflation
        trend_component = np.linspace(0, 0.03, num_days)  # 3% gradual increase
        sales_events = np.zeros(num_days)
        sale_indices = np.random.choice(range(10, num_days), size=2, replace=False)
        for idx in sale_indices:
            sales_events[idx] = -0.07  # 7% drop
            recovery_length = min(10, num_days-idx)
            sales_events[idx:idx+recovery_length] = np.linspace(-0.07, 0, recovery_length)
    
    # Combine base price with trends and random variations
    for i in range(num_days):
        # Apply trend component
        trend_factor = 1 + trend_component[i]
        
        # Apply sales events
        sales_factor = 1 + sales_events[i]
        
        # Calculate current price
        current_price = base_price * trend_factor * sales_factor
        
        # Add random noise based on price range
        if current_price > 100000:
            noise_factor = np.random.uniform(-0.01, 0.01)  # 1% noise for high-end products
        elif current_price > 50000:
            noise_factor = np.random.uniform(-0.015, 0.015)  # 1.5% noise
        elif current_price > 10000:
            noise_factor = np.random.uniform(-0.02, 0.02)  # 2% noise
        else:
            noise_factor = np.random.uniform(-0.03, 0.03)  # 3% noise for lower-priced products
            
        # Apply noise
        current_price *= (1 + noise_factor)
        
        # Add small day-to-day correlation
        if prices and i > 0:
            previous_price = prices[-1]
            correlation_factor = 0.3  # 30% correlation with previous day
            current_price = previous_price * correlation_factor + current_price * (1 - correlation_factor)
        
        prices.append(current_price)

    # Create DataFrame
    data = {
        'date': pd.date_range(start=start_date, end=end_date, freq='D'),
        'product_name': [product_name] * num_days,
        'price': prices
    }

    return pd.DataFrame(data)

@app.route('/predict', methods=['POST'])
def predict_prices():
    product_name = request.form.get('product_name', '')
    
    # Generate and prepare historical data
    df = generate_sample_data(product_name)
    df.set_index(pd.DatetimeIndex(df['date'], freq='D'), inplace=True)

    # Determine product category for prediction model parameters
    if 'apple' in product_name.lower() or 'iphone' in product_name.lower() or 'laptop' in product_name.lower():
        # Premium technology products - less volatile
        arima_order = (3, 1, 1)
        confidence_range = 0.03
        prediction_trend = -0.01  # Slight downward trend for tech
    elif 'smartphone' in product_name.lower() or 'tv' in product_name.lower():
        # Regular technology products
        arima_order = (2, 1, 2)
        confidence_range = 0.04
        prediction_trend = -0.005  # Very small downward trend
    elif 'fan' in product_name.lower() or 'seasonal' in product_name.lower():
        # Seasonal products - more cyclical
        arima_order = (5, 1, 0)  # Higher AR component for seasonal patterns
        confidence_range = 0.07
        prediction_trend = 0.02  # Upward trend (assuming approaching summer)
    elif 'fashion' in product_name.lower() or 'shoes' in product_name.lower():
        # Fashion items - more volatile
        arima_order = (2, 1, 3)  # Higher MA component for rapid changes
        confidence_range = 0.08
        prediction_trend = 0.01
    else:
        # General consumer products
        arima_order = (3, 1, 2)
        confidence_range = 0.06
        prediction_trend = 0.005  # Slight inflation
    
    # Fit ARIMA model with product-specific parameters
    model = ARIMA(df['price'], order=arima_order)
    model_fit = model.fit()

    # Generate initial forecast
    tomorrow = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    forecast_days = 10  # Extend to 10 days for more interesting patterns
    forecast_dates = pd.date_range(start=tomorrow, periods=forecast_days, freq='D')
    raw_forecast = model_fit.forecast(steps=forecast_days)
    
    # Apply dynamic, product-specific variations to predictions
    last_historical_price = df['price'].iloc[-1]
    predicted_prices = [last_historical_price]
    
    # Add market events for the future (like product launches, sales events)
    sale_probability = 0.2  # 20% chance of a sale in the forecast period
    will_have_sale = np.random.random() < sale_probability
    
    sale_start = None
    if will_have_sale:
        sale_start = np.random.randint(2, forecast_days - 2)  # Sale starts between day 2 and forecast_days-2
    
    for i, pred in enumerate(raw_forecast):
        last_price = predicted_prices[-1]
        
        # Apply product trend
        trend_adjustment = last_price * prediction_trend * (i+1)/forecast_days
        
        # Apply sales events if applicable
        sale_effect = 0
        if will_have_sale:
            if i == sale_start - 1:  # Day before sale
                sale_effect = last_price * 0.01  # 1% increase before sale (anticipation)
            elif i == sale_start:  # First day of sale
                sale_effect = -last_price * 0.08  # 8% drop for sale
            elif i == sale_start + 1:  # Second day of sale
                sale_effect = -last_price * 0.06  # Slightly smaller drop
        
        # Get ARIMA model prediction
        model_prediction = pred - last_price
        
        # Calculate new price with all factors
        new_price = last_price + model_prediction * 0.7 + trend_adjustment + sale_effect
        
        # Add day-specific noise (higher volatility for weekends)
        day_of_week = (forecast_dates[i].weekday() + 1) % 7  # 0=Monday, 6=Sunday
        weekend_factor = 1.2 if day_of_week >= 5 else 1.0  # More volatile on weekends
        
        # Calculate appropriate noise level
        if last_price > 100000:
            noise_level = 0.005 * weekend_factor
        elif last_price > 50000:
            noise_level = 0.008 * weekend_factor
        elif last_price > 10000:
            noise_level = 0.012 * weekend_factor
        else:
            noise_level = 0.02 * weekend_factor
            
        # Apply noise
        noise = np.random.uniform(-noise_level, noise_level) * last_price
        new_price += noise
        
        # Ensure reasonable bounds
        predicted_prices.append(new_price)
    
    forecast_df = pd.DataFrame({'price': predicted_prices[1:]}, index=forecast_dates)
    
    # Create a better visualization
    plt.figure(figsize=(12, 6))
    
    # Plot historical prices (last 30 days for better visualization)
    last_30_days = df.index[-30:]
    plt.plot(last_30_days, df.loc[last_30_days, 'price'], 
             label='Historical Prices', color='blue', marker='o')
    
    # Plot predicted prices
    plt.plot(forecast_df.index, forecast_df['price'], 
             label='Predicted Prices', color='red', linestyle='--', marker='o')
    
    # Add confidence interval with product-specific variation
    days_ahead = np.arange(len(forecast_df))
    
    # Calculate confidence interval that widens over time
    interval_width = confidence_range * (1 + 0.2 * days_ahead/len(days_ahead))
    
    # Determine appropriate confidence interval colors based on product type
    if 'apple' in product_name.lower() or 'iphone' in product_name.lower():
        ci_color = 'rgba(255, 102, 102, 0.2)'  # Light red for Apple products
    elif 'laptop' in product_name.lower() or 'tv' in product_name.lower():
        ci_color = 'rgba(102, 178, 255, 0.2)'  # Light blue for other electronics
    elif 'fashion' in product_name.lower() or 'shoes' in product_name.lower():
        ci_color = 'rgba(255, 153, 51, 0.2)'  # Orange for fashion
    else:
        ci_color = 'rgba(153, 204, 153, 0.2)'  # Light green for other items
    
    # Add shaded confidence interval
    plt.fill_between(
        forecast_df.index,
        forecast_df['price'] * (1 - interval_width),
        forecast_df['price'] * (1 + interval_width),
        color='red', alpha=0.2
    )
    
    # Improve plot formatting
    plt.title(f'Price Prediction for {product_name}')
    plt.xlabel('Date')
    plt.ylabel('Price (₹)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    
    # Format y-axis to show comma-separated thousands
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # Ensure dates are formatted nicely
    plt.gcf().autofmt_xdate()
    
    # Add analysis text to the chart
    current_price = df['price'].iloc[-1]
    average_future_price = forecast_df['price'].mean()
    price_change = ((average_future_price / current_price) - 1) * 100
    
    if price_change > 1:
        price_text = f"Prices expected to rise by {price_change:.1f}%"
        plt.figtext(0.15, 0.02, price_text, color='red')
    elif price_change < -1:
        price_text = f"Prices expected to fall by {abs(price_change):.1f}%"
        plt.figtext(0.15, 0.02, price_text, color='green')
    else:
        price_text = "Prices expected to remain stable"
        plt.figtext(0.15, 0.02, price_text, color='blue')

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()

    # Encode plot to base64 string
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return render_template('predictresults.html', plot_base64=plot_base64, query=product_name)

@app.route('/recommend', methods=['POST'])
def recommend():
    product_name = request.form.get('product_name', '')
    variants, links = generate_product_variants(product_name)
    return render_template('recommendations.html', variants=variants, links=links, query=product_name)

if __name__ == '__main__':
    app.run(debug=True)