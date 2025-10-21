import os
import pandas as pd
import numpy as np
import re
from dateutil import parser
from collections import defaultdict
import glob
import matplotlib.pyplot as plt
import psycopg2
import seaborn as sns

DATA_DIR = os.path.dirname(__file__)

# File pattern for all years
file_pattern = 'amazon_india_20*.csv'

# City standardization map
CITY_MAP = {
    # Major metros and their variants
    'bangalore': 'Bengaluru', 'bengaluru': 'Bengaluru', 'bangalore/bengaluru': 'Bengaluru',
    'bengalore': 'Bengaluru', 'bengaluru urban': 'Bengaluru', 'bengaluru rural': 'Bengaluru',
    'mumbai': 'Mumbai', 'bombay': 'Mumbai', 'mumbai/bombay': 'Mumbai', 'bambai': 'Mumbai',
    'delhi': 'Delhi', 'new delhi': 'Delhi', 'delhi/new delhi': 'Delhi', 'dilli': 'Delhi',
    'chennai': 'Chennai', 'madras': 'Chennai', 'chennai/madras': 'Chennai',
    'kolkata': 'Kolkata', 'calcutta': 'Kolkata', 'kolkata/calcutta': 'Kolkata',
    'hyderabad': 'Hyderabad', 'secunderabad': 'Hyderabad', 'hyderabad/secunderabad': 'Hyderabad',
    'pune': 'Pune', 'poona': 'Pune', 'pune/poona': 'Pune',
    'ahmedabad': 'Ahmedabad', 'amdavad': 'Ahmedabad',
    'coimbatore': 'Coimbatore', 'kovai': 'Coimbatore',
    'kochi': 'Kochi', 'cochin': 'Kochi', 'kochi/cochin': 'Kochi',
    'thiruvananthapuram': 'Thiruvananthapuram', 'trivandrum': 'Thiruvananthapuram',
    'visakhapatnam': 'Visakhapatnam', 'vizag': 'Visakhapatnam',
    'vadodara': 'Vadodara', 'baroda': 'Vadodara',
    'gurgaon': 'Gurugram', 'gurugram': 'Gurugram', 'gurgaon/gurugram': 'Gurugram',
    'mysore': 'Mysuru', 'mysuru': 'Mysuru',
    'varanasi': 'Varanasi', 'benaras': 'Varanasi', 'kashi': 'Varanasi',
    'kanpur': 'Kanpur', 'cawnpore': 'Kanpur',
    'allahabad': 'Prayagraj', 'prayagraj': 'Prayagraj',
    'panaji': 'Panaji', 'panjim': 'Panaji',
    'tiruchirappalli': 'Tiruchirappalli', 'trichy': 'Tiruchirappalli',
    'madurai': 'Madurai', 'madura': 'Madurai',
    'jaipur': 'Jaipur', 'pink city': 'Jaipur',
    'bhubaneswar': 'Bhubaneswar', 'cuttack': 'Cuttack',
    'patna': 'Patna', 'pataliputra': 'Patna',
    'lucknow': 'Lucknow', 'lakhnau': 'Lucknow',
    'indore': 'Indore', 'indur': 'Indore',
    'nagpur': 'Nagpur', 'nagpore': 'Nagpur',
    'chandigarh': 'Chandigarh',
    'goa': 'Goa',
    # Add more as needed
}

# Category standardization map
CATEGORY_MAP = {
    'electronics': 'Electronics', 'electronic': 'Electronics', 'electronics & accessories': 'Electronics', 'electronics&accessories': 'Electronics', 'electronicss': 'Electronics', 'ELECTRONICS': 'Electronics',
    'mobile': 'Electronics', 'mobiles': 'Electronics', 'smartphone': 'Electronics', 
    'Electronics&Accessories': 'Electronics', 'Electronicss': 'Electronics',
    'smartphones': 'Electronics', 'Electronic': 'Electronics',
    'laptop': 'Electronics', 'laptops': 'Electronics', 'computer': 'Electronics', 'computers': 'Electronics',
    'home appliances': 'Home Appliances', 'appliances': 'Home Appliances', 'home appliance': 'Home Appliances',
    'fashion': 'Fashion', 'clothing': 'Fashion', 'apparel': 'Fashion', 'garments': 'Fashion',
    'books': 'Books', 'book': 'Books', 'literature': 'Books',
    'toys': 'Toys', 'games': 'Toys', 'toy': 'Toys',
    'beauty': 'Beauty', 'personal care': 'Beauty', 'cosmetics': 'Beauty',
    'sports': 'Sports', 'sporting goods': 'Sports', 'sports equipment': 'Sports',
    'grocery': 'Grocery', 'groceries': 'Grocery', 'food': 'Grocery',
    'automotive': 'Automotive', 'car accessories': 'Automotive', 'vehicle': 'Automotive',
    'furniture': 'Furniture', 'home decor': 'Furniture', 'decor': 'Furniture',
    'stationery': 'Stationery', 'office supplies': 'Stationery',
    'health': 'Health', 'healthcare': 'Health', 'medical': 'Health',
    'jewellery': 'Jewellery', 'jewelry': 'Jewellery', 'ornaments': 'Jewellery',
    'footwear': 'Footwear', 'shoes': 'Footwear', 'sandals': 'Footwear',
    'watches': 'Watches', 'watch': 'Watches',
    # Add more as needed
}

# Payment method standardization map
PAYMENT_MAP = {
    'upi': 'UPI', 'phonepe': 'UPI', 'googlepay': 'UPI', 'gpay': 'UPI', 'paytm': 'UPI', 'mobikwik': 'UPI',
    'credit card': 'Credit Card', 'credit_card': 'Credit Card', 'cc': 'Credit Card', 'visa': 'Credit Card', 'mastercard': 'Credit Card', 'amex': 'Credit Card',
    'cash on delivery': 'COD', 'cod': 'COD', 'c.o.d': 'COD', 'cashondelivery': 'COD', 'cash': 'COD',
    'debit card': 'Debit Card', 'debit_card': 'Debit Card', 'dc': 'Debit Card',
    'net banking': 'Net Banking', 'netbanking': 'Net Banking', 'online banking': 'Net Banking', 'internet banking': 'Net Banking',
    'wallet': 'Wallet', 'amazon pay': 'Wallet', 'amazonpay': 'Wallet', 'wallets': 'Wallet',
    'emi': 'EMI', 'installment': 'EMI', 'emi option': 'EMI',
    'gift card': 'Gift Card', 'giftcard': 'Gift Card', 'voucher': 'Gift Card',
    'cheque': 'Cheque', 'check': 'Cheque',
    'rtgs': 'Bank Transfer', 'neft': 'Bank Transfer', 'imps': 'Bank Transfer', 'bank transfer': 'Bank Transfer',
    'paypal': 'PayPal', 'pay pal': 'PayPal',
    'BNPL': 'BNPL', 'buy now pay later': 'BNPL', 'buy now, pay later': 'BNPL', 'Bnpl': 'BNPL','bnpl': 'BNPL',
    # Add more as needed
}

# Boolean standardization function
BOOL_MAP = {'true': True, 'yes': True, '1': True, 'y': True, 'false': False, 'no': False, '0': False, 'n': False}

def get_postgres_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="data_analytics_dashboard",
        user="postgres",
        password="postgres"  # Replace with your actual password
    )

def create_time_dimension_table(start_date='2015-01-01', end_date='2025-12-31'):
    """
    Creates and populates time_dimension table in Postgres for the given date range.
    """

    import pandas as pd
    from datetime import date
    if end_date is None or end_date == '' or end_date == 'today':
        end_date = date.today().strftime('%Y-%m-%d')
    conn = get_postgres_connection()
    cur = conn.cursor()
    # Drop if exists
    cur.execute('DROP TABLE IF EXISTS time_dimension CASCADE;')
    cur.execute('''
        CREATE TABLE time_dimension (
            date DATE PRIMARY KEY,
            year INTEGER,
            quarter INTEGER,
            month INTEGER,
            month_name TEXT,
            week INTEGER,
            day INTEGER,
            day_of_week INTEGER,
            day_name TEXT,
            is_weekend BOOLEAN,
            is_holiday BOOLEAN DEFAULT FALSE
        );
    ''')
    conn.commit()
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date)
    time_dim = pd.DataFrame({'date': dates})
    time_dim['year'] = time_dim['date'].dt.year
    time_dim['quarter'] = time_dim['date'].dt.quarter
    time_dim['month'] = time_dim['date'].dt.month
    time_dim['month_name'] = time_dim['date'].dt.strftime('%B')
    time_dim['week'] = time_dim['date'].dt.isocalendar().week
    time_dim['day'] = time_dim['date'].dt.day
    time_dim['day_of_week'] = time_dim['date'].dt.weekday + 1
    time_dim['day_name'] = time_dim['date'].dt.strftime('%A')
    time_dim['is_weekend'] = time_dim['day_of_week'].isin([6,7])
    time_dim['is_holiday'] = False  # You can update holidays later if needed
    # Insert into DB
    for _, row in time_dim.iterrows():
        cur.execute('''
            INSERT INTO time_dimension (date, year, quarter, month, month_name, week, day, day_of_week, day_name, is_weekend, is_holiday)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            row['date'].date(), row['year'], row['quarter'], row['month'], row['month_name'], int(row['week']), row['day'], row['day_of_week'], row['day_name'], row['is_weekend'], row['is_holiday']
        ))
    conn.commit()
    cur.close()
    conn.close()
    print("time_dimension table created and populated.")
def create_postgres_tables():
    """
    Creates orders, customers, and products tables in Postgres DB.
    """
    conn = get_postgres_connection()
    cur = conn.cursor()
    # Drop tables if exist
    cur.execute('DROP TABLE IF EXISTS orders CASCADE;')
    cur.execute('DROP TABLE IF EXISTS customers CASCADE;')
    cur.execute('DROP TABLE IF EXISTS products CASCADE;')
    # Orders table
    cur.execute('''
        CREATE TABLE orders (
            order_id VARCHAR PRIMARY KEY,
            order_date DATE,
            customer_id VARCHAR,
            product_id VARCHAR,
            product_name TEXT,
            category TEXT,
            subcategory TEXT,
            brand TEXT,
            quantity INTEGER,
            price NUMERIC,
            final_amount_inr NUMERIC,
            payment_method TEXT,
            delivery_days INTEGER,
            return_status TEXT,
            customer_rating NUMERIC,
            customer_city TEXT,
            customer_state TEXT,
            customer_age_group TEXT,
            is_prime_member BOOLEAN,
            is_festival_sale BOOLEAN,
            festival_name TEXT,
            discount_percent NUMERIC
        );
    ''')
    # Customers table
    cur.execute('''
        CREATE TABLE customers (
            customer_id VARCHAR PRIMARY KEY,
            customer_name TEXT,
            customer_age_group TEXT,
            customer_city TEXT,
            customer_state TEXT,
            is_prime_member BOOLEAN,
            acquisition_date DATE,
            customer_spending_tier TEXT
        );
    ''')
    # Products table
    cur.execute('''
        CREATE TABLE products (
            product_id VARCHAR PRIMARY KEY,
            product_name TEXT,
            category TEXT,
            subcategory TEXT,
            brand TEXT,
            product_rating NUMERIC
        );
    ''')
    # Indexes for performance
    cur.execute('CREATE INDEX idx_orders_order_date ON orders(order_date);')
    cur.execute('CREATE INDEX idx_orders_customer_id ON orders(customer_id);')
    cur.execute('CREATE INDEX idx_orders_product_id ON orders(product_id);')
    cur.execute('CREATE INDEX idx_orders_category ON orders(category);')
    cur.execute('CREATE INDEX idx_orders_subcategory ON orders(subcategory);')
    cur.execute('CREATE INDEX idx_orders_brand ON orders(brand);')
    cur.execute('CREATE INDEX idx_orders_customer_city ON orders(customer_city);')
    cur.execute('CREATE INDEX idx_orders_customer_state ON orders(customer_state);')
    # Indexes for products table
    cur.execute('CREATE INDEX idx_products_category ON products(category);')
    cur.execute('CREATE INDEX idx_products_subcategory ON products(subcategory);')
    cur.execute('CREATE INDEX idx_products_brand ON products(brand);')
    # Indexes for customers table
    cur.execute('CREATE INDEX idx_customers_city ON customers(customer_city);')
    cur.execute('CREATE INDEX idx_customers_state ON customers(customer_state);')
    cur.execute('CREATE INDEX idx_customers_age_group ON customers(customer_age_group);')
    conn.commit()
    cur.close()
    conn.close()
    print("Postgres tables dropped, created, and indexed.")

def load_orders_to_postgres():
    """
    Loads all cleaned CSVs into orders table in Postgres.
    """
    import glob
    conn = get_postgres_connection()
    cur = conn.cursor()
    files = glob.glob(os.path.join(DATA_DIR, 'amazon_india_*_cleaned.csv'))
    for file in files:
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            order_id = row.get('transaction_id')
            if pd.isnull(order_id):
                continue  # Skip rows with missing transaction_id
            def safe_int(val):
                try:
                    return int(val)
                except:
                    return None
            def safe_float(val):
                try:
                    return float(val)
                except:
                    return None
            quantity = safe_int(row.get('quantity'))
            price = safe_float(row.get('original_price_inr'))
            final_amount_inr = safe_float(row.get('final_amount_inr'))
            delivery_days = safe_int(row.get('delivery_days'))
            customer_rating = safe_float(row.get('customer_rating'))
            discount_percent = safe_float(row.get('discount_percent'))
            cur.execute('''
                INSERT INTO orders (
                    order_id, order_date, customer_id, product_id, product_name, category, subcategory, brand, quantity, price, final_amount_inr, payment_method, delivery_days, return_status, customer_rating, customer_city, customer_state, customer_age_group, is_prime_member, is_festival_sale, festival_name, discount_percent
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                order_id, row.get('order_date'), row.get('customer_id'), row.get('product_id'), row.get('product_name'), row.get('category'), row.get('subcategory'), row.get('brand'), quantity, price, final_amount_inr, row.get('payment_method'), delivery_days, row.get('return_status'), customer_rating, row.get('customer_city'), row.get('customer_state'), row.get('customer_age_group'), row.get('is_prime_member'), row.get('is_festival_sale'), row.get('festival_name'), discount_percent
            ))
    conn.commit()
    cur.close()
    conn.close()
    print("Orders loaded to Postgres.")

def load_customers_to_postgres():
    """
    Loads unique customers from orders table into customers table.
    """
    import glob
    conn = get_postgres_connection()
    cur = conn.cursor()
    seen_customers = set()
    cur.execute('SELECT customer_id FROM customers;')
    for row in cur.fetchall():
        seen_customers.add(row[0])
    files = glob.glob(os.path.join(DATA_DIR, 'amazon_india_*_cleaned.csv'))
    for file in files:
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            customer_id = row.get('customer_id')
            if pd.isnull(customer_id) or customer_id in seen_customers:
                continue
            cur.execute('INSERT INTO customers (customer_id, customer_name, customer_age_group, customer_city, customer_state, is_prime_member, acquisition_date, customer_spending_tier) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)', (
                customer_id,
                None,
                row.get('customer_age_group'),
                row.get('customer_city'),
                row.get('customer_state'),
                row.get('is_prime_member'),
                row.get('order_date'),
                row.get('customer_spending_tier')
            ))
            seen_customers.add(customer_id)
    conn.commit()
    cur.close()
    conn.close()
    print("Customers loaded to Postgres.")

def load_products_to_postgres():
    """
    Loads unique products from orders table into products table.
    """
    import glob
    conn = get_postgres_connection()
    cur = conn.cursor()
    seen_products = set()
    cur.execute('SELECT product_id FROM products;')
    for row in cur.fetchall():
        seen_products.add(row[0])
    files = glob.glob(os.path.join(DATA_DIR, 'amazon_india_*_cleaned.csv'))
    product_ratings = {}
    for file in files:
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            product_id = row.get('product_id')
            if pd.isnull(product_id) or product_id in seen_products:
                continue
            # Collect ratings for averaging
            rating = row.get('product_rating')
            if product_id not in product_ratings:
                product_ratings[product_id] = []
            if not pd.isnull(rating):
                product_ratings[product_id].append(rating)
            cur.execute('INSERT INTO products (product_id, product_name, category, subcategory, brand, product_rating) VALUES (%s, %s, %s, %s, %s, %s)', (
                product_id,
                row.get('product_name'),
                row.get('category'),
                row.get('subcategory'),
                row.get('brand'),
                None  # Placeholder, will update with average below
            ))
            seen_products.add(product_id)
    # Update product_rating with average
    for product_id, ratings in product_ratings.items():
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            cur.execute('UPDATE products SET product_rating = %s WHERE product_id = %s', (avg_rating, product_id))
    conn.commit()
    cur.close()
    conn.close()
    print("Products loaded to Postgres.")

def clean_order_date(val):
    try:
        dt = parser.parse(str(val), dayfirst=True, yearfirst=False, fuzzy=True)
        return dt.strftime('%Y-%m-%d')
    except Exception:
        return np.nan

def clean_price(val):
    if pd.isnull(val): return np.nan
    s = str(val).strip()
    # Remove currency symbols and words
    s = re.sub(r'(?i)(rs\.?|r\.?|₹|inr|rupees|price on request)', '', s)
    # Remove commas and spaces
    s = s.replace(',', '').replace(' ', '')
    # If value is still not a valid float, try to extract numbers
    if s == '': return np.nan
    # Sometimes value is like 'Rs 43,107' or '₹191,519.73'
    match = re.search(r'(\d+[.]?\d*)', s)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def clean_rating(val):
    if pd.isnull(val): return np.nan
    s = str(val).lower().replace('stars', '').replace('star', '').strip()
    if '/' in s:
        parts = re.split(r'/', s)
        try:
            num = float(parts[0])
            denom = float(parts[1])
            if denom > 0: return round(num / denom * 5, 2)
        except: return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def clean_city(val):
    if pd.isnull(val): return np.nan
    s = str(val).lower().replace(' ', '')
    return CITY_MAP.get(s, val.title())

def clean_bool(val):
    if pd.isnull(val): return False
    s = str(val).lower().strip()
    return BOOL_MAP.get(s, False)

def clean_category(val):
    if pd.isnull(val): return np.nan
    s = str(val).lower().replace(' ', '')
    return CATEGORY_MAP.get(s, val.title())

def clean_delivery_days(val):
    if pd.isnull(val): return np.nan
    s = str(val).lower().strip()
    if s in ['same day', 'sameday']: return 0
    if '-' in s:
        try:
            parts = s.split('-')
            nums = [int(p) for p in parts if p.isdigit()]
            if nums: return int(np.mean(nums))
        except: return np.nan
    try:
        days = int(s)
        if 0 <= days <= 30: return days
        else: return np.nan
    except: return np.nan

def clean_payment(val):
    if pd.isnull(val): return np.nan
    s = str(val).lower().replace(' ', '').replace('_', '').replace('.', '')
    return PAYMENT_MAP.get(s, val.title())

def fix_outlier_prices(df):
    # Use IQR to detect outliers, fix if 100x above Q3
    q3 = df['original_price_inr'].quantile(0.75)
    outlier_mask = df['original_price_inr'] > (q3 * 100)
    df.loc[outlier_mask, 'original_price_inr'] = df.loc[outlier_mask, 'original_price_inr'] / 100
    return df

def handle_duplicates(df):
    # Mark duplicates by transaction_id, customer_id, product_id, order_date, final_amount_inr
    subset = ['transaction_id', 'customer_id', 'product_id', 'order_date', 'final_amount_inr']
    df['is_duplicate'] = df.duplicated(subset=subset, keep=False)
    # Keep all if quantity > 1, else drop
    df = df[~((df['is_duplicate']) & (df['quantity'] == 1))].drop(columns=['is_duplicate'])
    return df

def clean_file(file_path):
    df = pd.read_csv(file_path)
    # Q1: Clean order_date
    df['order_date'] = df['order_date'].apply(clean_order_date)
    # Q2: Clean original_price_inr
    df['original_price_inr'] = df['original_price_inr'].apply(clean_price)
    # Q3: Clean customer_rating
    df['customer_rating'] = df['customer_rating'].apply(clean_rating)
    # Q4: Clean customer_city
    df['customer_city'] = df['customer_city'].apply(clean_city)
    # Q5: Clean boolean columns
    for col in ['is_prime_member', 'is_prime_eligible', 'is_festival_sale']:
        if col in df.columns:
            df[col] = df[col].apply(clean_bool)
    # Q6: Clean product category
    df['category'] = df['category'].apply(clean_category)
    # Q7: Clean delivery_days
    df['delivery_days'] = df['delivery_days'].apply(clean_delivery_days)
    # Q8: Handle duplicates
    df = handle_duplicates(df)
    # Q9: Fix outlier prices
    df = fix_outlier_prices(df)
    # Q10: Clean payment method
    df['payment_method'] = df['payment_method'].apply(clean_payment)
    # Remove any rows with invalid dates or prices
    df = df.dropna(subset=['order_date', 'original_price_inr'])
    # Remove duplicates
    df = df.drop_duplicates()
    return df

def scan_and_suggest_map_values(data_dir):
    category_set = set()
    payment_set = set()
    city_set = set()
    for fname in glob.glob(os.path.join(data_dir, 'amazon_india_*.csv')):
        try:
            df = pd.read_csv(fname, nrows=10000)  # scan first 10k rows for speed
            if 'category' in df.columns:
                category_set.update(df['category'].dropna().unique())
            if 'payment_method' in df.columns:
                payment_set.update(df['payment_method'].dropna().unique())
            if 'customer_city' in df.columns:
                city_set.update(df['customer_city'].dropna().unique())
        except Exception as e:
            print(f"Error reading {fname}: {e}")
    print("\nUnique category values:")
    print(sorted(category_set))
    print("\nUnique payment_method values:")
    print(sorted(payment_set))
    print("\nUnique customer_city values:")
    print(sorted(city_set))

def scan_price_patterns(data_dir):
    import collections
    import glob
    import re
    price_patterns = collections.Counter()
    sample_values = set()
    for fname in glob.glob(os.path.join(data_dir, 'amazon_india_*.csv')):
        try:
            df = pd.read_csv(fname, usecols=['original_price_inr'], nrows=10000)
            for val in df['original_price_inr'].dropna():
                s = str(val).strip()
                # Record raw value
                sample_values.add(s)
                # Find pattern
                if re.match(r'^\d+[.,]?\d*$', s):
                    price_patterns['numeric'] += 1
                elif re.search(r'₹|rs\.?|r\.?|inr|rupees', s, re.IGNORECASE):
                    price_patterns['currency_symbol'] += 1
                elif ',' in s:
                    price_patterns['comma'] += 1
                elif s.lower() == 'price on request':
                    price_patterns['price_on_request'] += 1
                elif s == '' or s.lower() == 'nan':
                    price_patterns['missing'] += 1
                else:
                    price_patterns['other'] += 1
        except Exception as e:
            print(f"Error reading {fname}: {e}")
    print("\nPrice value patterns:")
    for k, v in price_patterns.items():
        print(f"{k}: {v}")
    print("\nSample price values:")
    for v in list(sample_values)[:30]:
        print(v)

def revenue_trend_analysis(df):
    """
    Q1: Revenue trend analysis (2015-2025): yearly revenue, % growth, trend lines, annotate key periods.
    All charts are shown in a single window using subplots.
    """

    # Use final_amount_inr if present, else original_price_inr
    amount_col = 'final_amount_inr' if 'final_amount_inr' in df.columns else 'original_price_inr'
    df['year'] = df['order_date'].dt.year
    yearly = df.groupby('year')[amount_col].sum().reset_index(name='revenue')
    yearly = yearly[(yearly['year'] >= 2015) & (yearly['year'] <= 2025)]
    yearly['pct_growth'] = yearly['revenue'].pct_change() * 100
    # Set up subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
    # Revenue line and trend
    sns.lineplot(data=yearly, x='year', y='revenue', marker='o', label='Revenue', ax=ax1, color='tab:blue')
    z = np.polyfit(yearly['year'], yearly['revenue'], 1)
    p = np.poly1d(z)
    ax1.plot(yearly['year'], p(yearly['year']), '--', color='gray', label='Trend line')
    top_growth = yearly.nlargest(2, 'pct_growth')
    for _, row in top_growth.iterrows():
        ax1.annotate(f"+{row['pct_growth']:.1f}%", (row['year'], row['revenue']),
                     textcoords="offset points", xytext=(0,10), ha='center', color='green', fontsize=10, fontweight='bold')
    ax1.set_title('Amazon India Yearly Revenue Trend (2015-2025)')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Total Revenue (INR)')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)
    # Percentage growth bar
    sns.barplot(data=yearly, x='year', y='pct_growth', alpha=0.3, ax=ax2, color='orange', label='% Growth')
    ax2.set_title('% Revenue Growth by Year')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('% Growth')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    print("[Q1] Revenue trend analysis complete. See plot for details.")

def seasonal_patterns_analysis(df):
    """
    Q2: Analyze seasonal patterns: monthly sales heatmaps, peak months, compare years/categories.
    All charts are shown in a single window using subplots.
    """

    amount_col = 'final_amount_inr' if 'final_amount_inr' in df.columns else 'original_price_inr'
    df['year'] = df['order_date'].dt.year
    df['month'] = df['order_date'].dt.month
    # Prepare data
    monthly = df.groupby(['year', 'month'])[amount_col].sum().reset_index()
    pivot = monthly.pivot(index='year', columns='month', values=amount_col) / 1e9  # Convert to billions
    month_total = df.groupby('month')[amount_col].sum().reset_index()
    peak_month = month_total.loc[month_total[amount_col].idxmax(), 'month']
    # For multi-year line
    yearly_groups = list(monthly.groupby('year'))
    # For category trends
    top_cats = None
    cat_month = None
    if 'category' in df.columns:
        top_cats = df.groupby('category')[amount_col].sum().nlargest(5).index
        df_top = df[df['category'].isin(top_cats)]
        cat_month = df_top.groupby(['category', 'month'])[amount_col].sum().reset_index()
    # Set up subplots
    nrows = 2 if cat_month is not None else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(18, 6*nrows))
    # Heatmap (in billions)
    ax = axes[0,0] if nrows > 1 else axes[0]
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax)
    ax.set_title('Monthly Sales Heatmap (Revenue, Billions INR)')
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    # Multi-year monthly trend
    ax = axes[0,1] if nrows > 1 else axes[1]
    for y, group in yearly_groups:
        ax.plot(group['month'], group[amount_col], marker='o', label=str(int(y)))
    ax.set_title('Monthly Revenue Trend by Year')
    ax.set_xlabel('Month')
    ax.set_ylabel('Revenue (INR)')
    ax.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    # Category trends (if available)
    if cat_month is not None:
        ax = axes[1,0]
        sns.lineplot(data=cat_month, x='month', y=amount_col, hue='category', marker='o', ax=ax)
        ax.set_title('Monthly Revenue Trend by Top 5 Categories (All Years)')
        ax.set_xlabel('Month')
        ax.set_ylabel('Revenue (INR)')
        ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.5)
        # Hide unused subplot if 2x2
        if nrows == 2:
            axes[1,1].axis('off')
    plt.tight_layout()
    plt.show()
    print(f"[Q2] Peak selling month (all years combined): {peak_month}")
    print("[Q2] Seasonal patterns analysis complete. See heatmap and trend plots for details.")

def rfm_customer_segmentation(df):
    """
    Q3: RFM segmentation: scatter plots, segment customers, actionable insights.
    All charts are shown in a single window using subplots. Optimized for speed by sampling customers and removing outliers for plotting.
    """
    import datetime
    amount_col = 'final_amount_inr' if 'final_amount_inr' in df.columns else 'original_price_inr'
    # Filter for valid customer_id and order_date
    df = df.dropna(subset=['customer_id', 'order_date'])
    # RFM calculation
    snapshot_date = df['order_date'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('customer_id').agg({
        'order_date': lambda x: (snapshot_date - x.max()).days,
        'customer_id': 'count',
        amount_col: 'sum'
    }).rename(columns={'order_date': 'Recency', 'customer_id': 'Frequency', amount_col: 'Monetary'}).reset_index()
    # RFM scoring (quintiles)
    rfm['R'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1]).astype(int)
    rfm['F'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    rfm['M'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5]).astype(int)
    rfm['RFM_Score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)
    # Segment customers
    def segment(row):
        if row['R'] >= 4 and row['F'] >= 4 and row['M'] >= 4:
            return 'Champions'
        elif row['R'] >= 3 and row['F'] >= 3 and row['M'] >= 3:
            return 'Loyal'
        elif row['R'] >= 4:
            return 'Recent'
        elif row['F'] >= 4:
            return 'Frequent'
        elif row['M'] >= 4:
            return 'Big Spenders'
        else:
            return 'Others'
    rfm['Segment'] = rfm.apply(segment, axis=1)
    # Sample for plotting
    plot_sample = rfm.sample(n=min(5000, len(rfm)), random_state=42)
    # Remove outliers (clip to 1st-99th percentiles)
    for col in ['Recency', 'Frequency', 'Monetary']:
        lower, upper = plot_sample[col].quantile(0.01), plot_sample[col].quantile(0.99)
        plot_sample = plot_sample[(plot_sample[col] >= lower) & (plot_sample[col] <= upper)]
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    # Recency vs Frequency
    scatter1 = axes[0].scatter(plot_sample['Recency'], plot_sample['Frequency'], c=plot_sample['M'], cmap='viridis', alpha=0.7)
    axes[0].set_title('Recency vs Frequency (Color: Monetary)')
    axes[0].set_xlabel('Recency (days since last order)')
    axes[0].set_ylabel('Frequency (orders)')
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('Monetary (INR)')
    # Frequency vs Monetary by Segment
    for seg, group in plot_sample.groupby('Segment'):
        axes[1].scatter(group['Frequency'], group['Monetary'], label=seg, alpha=0.7)
    axes[1].set_title('Frequency vs Monetary by Segment')
    axes[1].set_xlabel('Frequency (orders)')
    axes[1].set_ylabel('Monetary (INR)')
    axes[1].legend()
    plt.tight_layout()
    plt.show()
    # Actionable insights
    seg_counts = rfm['Segment'].value_counts()
    print("[Q3] RFM Segmentation complete. Segment counts:")
    print(seg_counts)
    print("[Q3] Champions: High value, frequent, recent customers. Target with loyalty rewards.")
    print("[Q3] Loyal: Consistent, valuable customers. Maintain engagement.")
    print("[Q3] Recent: New or reactivated customers. Encourage repeat purchases.")
    print("[Q3] Frequent: Frequent buyers, may not spend much. Upsell/cross-sell.")
    print("[Q3] Big Spenders: High spend, but not frequent/recent. Win back with offers.")
    print("[Q3] Others: Low engagement. Consider reactivation campaigns.")

def payment_method_evolution(df):
        """
        Q4: Payment method evolution: stacked area charts, UPI rise, COD decline, market share over time.
        All charts are shown in a single window using subplots.
        """

        amount_col = 'final_amount_inr' if 'final_amount_inr' in df.columns else 'original_price_inr'
        df['year'] = df['order_date'].dt.year
        # Filter for years 2015-2025 and valid payment_method
        df = df[(df['year'] >= 2015) & (df['year'] <= 2025) & df['payment_method'].notnull()]
        # Aggregate revenue by year and payment method
        pm_year = df.groupby(['year', 'payment_method'])[amount_col].sum().reset_index()
        # Pivot for stacked area chart
        pm_pivot = pm_year.pivot(index='year', columns='payment_method', values=amount_col).fillna(0)
        # Calculate market share
        pm_share = pm_pivot.div(pm_pivot.sum(axis=1), axis=0)
        # Select top payment methods (by total revenue)
        top_methods = pm_pivot.sum().nlargest(6).index.tolist()
        pm_share_top = pm_share[top_methods]
        # Set up subplots
        fig, ax = plt.subplots(figsize=(14, 7))
        pm_share_top.plot.area(ax=ax, cmap='tab20', alpha=0.8)
        ax.set_title('Payment Method Market Share Evolution (2015-2025)')
        ax.set_xlabel('Year')
        ax.set_ylabel('Market Share')
        ax.legend(title='Payment Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
        # Highlight UPI rise and COD decline
        if 'UPI' in pm_share_top.columns and 'COD' in pm_share_top.columns:
            upi_trend = pm_share_top['UPI']
            cod_trend = pm_share_top['COD']
            print(f"[Q4] UPI market share: {upi_trend.iloc[0]:.2%} in {upi_trend.index[0]}, {upi_trend.iloc[-1]:.2%} in {upi_trend.index[-1]}")
            print(f"[Q4] COD market share: {cod_trend.iloc[0]:.2%} in {cod_trend.index[0]}, {cod_trend.iloc[-1]:.2%} in {cod_trend.index[-1]}")
        print("[Q4] Payment method evolution analysis complete. See stacked area chart for details.")

def category_performance_analysis(df):
    """
    Q5: Category-wise performance: treemaps, bar charts, pie charts for revenue, growth, and market share.
    All charts are shown in a single window using subplots.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    try:
        import squarify
    except ImportError:
        print("Installing squarify for treemap plots...")
        import subprocess
        subprocess.check_call(["pip", "install", "squarify"])
        import squarify

    amount_col = 'final_amount_inr' if 'final_amount_inr' in df.columns else 'original_price_inr'
    # Use subcategory instead of category
    if 'subcategory' not in df.columns:
        print("[Q5] No 'subcategory' column found in data. Cannot perform subcategory analysis.")
        return
    df = df[df['subcategory'].notnull()]
    # Revenue by subcategory
    subcat_rev = df.groupby('subcategory')[amount_col].sum().sort_values(ascending=False)
    # Market share
    subcat_share = subcat_rev / subcat_rev.sum()
    # Growth rates (2015-2025)
    df['year'] = df['order_date'].dt.year
    yearly_subcat = df.groupby(['year', 'subcategory'])[amount_col].sum().reset_index()
    # Calculate growth for each subcategory (last vs first year)
    growth = yearly_subcat.groupby('subcategory').apply(lambda x: (x.loc[x['year']==x['year'].max(), amount_col].sum() - x.loc[x['year']==x['year'].min(), amount_col].sum()) / max(x.loc[x['year']==x['year'].min(), amount_col].sum(), 1)).sort_values(ascending=False)

    # Top 10 subcategories for clarity
    top_subcats = subcat_rev.head(10).index.tolist()
    # Set up subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Treemap: Revenue contribution
    sizes = subcat_rev[top_subcats].values
    labels = [f"{subcat}\n₹{subcat_rev[subcat]/1e9:.2f}B" for subcat in top_subcats]
    squarify.plot(sizes=sizes, label=labels, color=sns.color_palette('tab20', len(top_subcats)), alpha=0.8, ax=axes[0,0])
    axes[0,0].set_title('Subcategory Revenue Contribution (Treemap)')
    axes[0,0].axis('off')

    # Bar chart: Market share
    sns.barplot(x=subcat_share[top_subcats].values, y=top_subcats, orient='h', ax=axes[0,1], palette='tab20')
    axes[0,1].set_title('Subcategory Market Share (Bar Chart)')
    axes[0,1].set_xlabel('Market Share')
    axes[0,1].set_ylabel('Subcategory')

    # Pie chart: Revenue share
    axes[1,0].pie(subcat_share[top_subcats].values, labels=top_subcats, autopct='%1.1f%%', colors=sns.color_palette('tab20', len(top_subcats)), startangle=140)
    axes[1,0].set_title('Subcategory Revenue Share (Pie Chart)')

    # Bar chart: Growth rates
    growth_top = growth[top_subcats]
    sns.barplot(x=growth_top.values, y=top_subcats, orient='h', ax=axes[1,1], palette='coolwarm')
    axes[1,1].set_title('Subcategory Growth Rate (2015-2025)')
    axes[1,1].set_xlabel('Growth Rate (last vs first year)')
    axes[1,1].set_ylabel('Subcategory')

    plt.tight_layout()
    plt.show()

    print("[Q5] Subcategory-wise performance analysis complete.")
    print("Top subcategories by revenue:")
    print(subcat_rev.head(10))
    print("Top subcategories by growth rate:")
    print(growth.head(10))

def prime_membership_impact_analysis(df):
    """
    Q6: Analyze Prime membership impact on customer behavior.
    Compare average order values, order frequency, and category preferences between Prime and non-Prime customers.
    For Prime members, compare their behavior before and after becoming Prime.
    Multiple visualization types: boxplots, bar charts, line charts, stacked bars.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    amount_col = 'final_amount_inr' if 'final_amount_inr' in df.columns else 'original_price_inr'
    # Ensure order_date is datetime
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    # Only keep rows with customer_id and order_date
    df = df.dropna(subset=['customer_id', 'order_date'])
    # Prime status
    if 'is_prime_member' not in df.columns:
        print("[Q6] No 'is_prime_member' column found in data. Cannot analyze Prime impact.")
        return
    # For each customer, find first order as Prime
    prime_orders = df[df['is_prime_member']].sort_values(['customer_id', 'order_date'])
    first_prime = prime_orders.groupby('customer_id')['order_date'].min().reset_index().rename(columns={'order_date': 'prime_start_date'})
    # Merge prime_start_date to all orders
    df = df.merge(first_prime, on='customer_id', how='left')
    # Define status: 'Non-Prime', 'Pre-Prime', 'Post-Prime'
    def prime_status(row):
        if not row['is_prime_member']:
            return 'Non-Prime'
        elif pd.notnull(row['prime_start_date']) and row['order_date'] < row['prime_start_date']:
            return 'Pre-Prime'
        elif pd.notnull(row['prime_start_date']) and row['order_date'] >= row['prime_start_date']:
            return 'Post-Prime'
        else:
            return 'Non-Prime'
    df['prime_status'] = df.apply(prime_status, axis=1)
    # Average order value by status
    aov = df.groupby('prime_status')[amount_col].mean().reset_index(name='avg_order_value')
    # Order frequency by status
    freq = df.groupby(['customer_id', 'prime_status']).size().reset_index(name='order_count')
    freq_summary = freq.groupby('prime_status')['order_count'].mean().reset_index(name='avg_order_count')
    # Category preferences
    if 'subcategory' in df.columns:
        cat_pref = df.groupby(['prime_status', 'subcategory']).size().reset_index(name='count')
        cat_pivot = cat_pref.pivot(index='subcategory', columns='prime_status', values='count').fillna(0)
    else:
        cat_pref = df.groupby(['prime_status', 'category']).size().reset_index(name='count')
        cat_pivot = cat_pref.pivot(index='category', columns='prime_status', values='count').fillna(0)
    # Set up subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    # Boxplot: Average order value
    sns.boxplot(x='prime_status', y=amount_col, data=df, ax=axes[0,0], hue='prime_status', palette='Set2', legend=False)
    axes[0,0].set_title('Order Value Distribution by Prime Status')
    axes[0,0].set_xlabel('Prime Status')
    axes[0,0].set_ylabel('Order Value (INR)')
    # Bar chart: Average order frequency
    sns.barplot(x='prime_status', y='avg_order_count', data=freq_summary, ax=axes[0,1], hue='prime_status', palette='Set2', legend=False)
    axes[0,1].set_title('Average Order Frequency by Prime Status')
    axes[0,1].set_xlabel('Prime Status')
    axes[0,1].set_ylabel('Avg Orders per Customer')
    # Stacked bar: Category preferences
    cat_pivot.plot(kind='bar', stacked=True, ax=axes[1,0], colormap='tab20')
    axes[1,0].set_title('Category Preferences by Prime Status')
    axes[1,0].set_xlabel('Category/Subcategory')
    axes[1,0].set_ylabel('Order Count')
    axes[1,0].legend(title='Prime Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    # Line chart: Order value trend before/after Prime
    prime_trend = df[df['prime_status'].isin(['Pre-Prime', 'Post-Prime'])].copy()
    if not prime_trend.empty:
        prime_trend['days_since_prime'] = (prime_trend['order_date'] - prime_trend['prime_start_date']).dt.days
        # Bin days for smoothing
        bin_edges = [-365, -180, -90, -30, 0, 30, 90, 180, 365]
        bin_labels = ['-1yr', '-6mo', '-3mo', '-1mo', 'Prime', '+1mo', '+3mo', '+6mo']
        prime_trend['bin'] = pd.cut(prime_trend['days_since_prime'], bins=bin_edges, labels=bin_labels)
        trend_data = prime_trend.groupby(['bin', 'prime_status'])[amount_col].mean().reset_index()
        sns.lineplot(x='bin', y=amount_col, hue='prime_status', data=trend_data, ax=axes[1,1], marker='o')
        axes[1,1].set_title('Order Value Trend Before/After Prime Membership')
        axes[1,1].set_xlabel('Time Relative to Prime Start')
        axes[1,1].set_ylabel('Avg Order Value (INR)')
        axes[1,1].legend(title='Prime Status')
    else:
        axes[1,1].text(0.5, 0.5, 'No Prime member trend data available', ha='center', va='center')
        axes[1,1].set_axis_off()
    plt.tight_layout()
    plt.show()

    print("[Q6] Prime membership impact analysis complete.")
    print("Average order value by Prime status:")
    print(aov)
    print("Average order frequency by Prime status:")
    print(freq_summary)

def geographic_sales_analysis(df):
    """
    Q7: Geographic analysis of sales performance across Indian cities and states.
    Build choropleth maps and bar charts showing revenue density and growth patterns by tier (Metro/Tier1/Tier2/Rural).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    try:
        import geopandas as gpd
    except ImportError:
        print("Please install geopandas for choropleth maps: pip install geopandas")
        return
    import warnings
    warnings.filterwarnings('ignore')
    amount_col = 'final_amount_inr' if 'final_amount_inr' in df.columns else 'original_price_inr'
    # City/State tier mapping (simplified, can be expanded)
    metro_cities = {'Bengaluru', 'Mumbai', 'Delhi', 'Chennai', 'Kolkata', 'Hyderabad'}
    tier1_cities = {'Pune', 'Ahmedabad', 'Coimbatore', 'Kochi', 'Thiruvananthapuram', 'Jaipur', 'Lucknow', 'Indore', 'Nagpur', 'Chandigarh', 'Goa'}
    # All others as Tier2/Rural
    def city_tier(city):
        if city in metro_cities:
            return 'Metro'
        elif city in tier1_cities:
            return 'Tier1'
        elif isinstance(city, str) and city:
            return 'Tier2/Rural'
        else:
            return 'Unknown'
    df['city_tier'] = df['customer_city'].apply(city_tier)
    # Revenue by city/state/tier
    city_rev = df.groupby('customer_city')[amount_col].sum().sort_values(ascending=False)
    tier_rev = df.groupby('city_tier')[amount_col].sum().sort_values(ascending=False)
    # Growth by tier (2015-2025)
    df['year'] = df['order_date'].dt.year
    yearly_tier = df.groupby(['year', 'city_tier'])[amount_col].sum().reset_index()
    growth = yearly_tier.groupby('city_tier').apply(lambda x: (x.loc[x['year']==x['year'].max(), amount_col].sum() - x.loc[x['year']==x['year'].min(), amount_col].sum()) / max(x.loc[x['year']==x['year'].min(), amount_col].sum(), 1)).sort_values(ascending=False)

    # Bar chart: Revenue by tier
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(x=tier_rev.index, y=tier_rev.values, ax=axes[0], palette='viridis')
    axes[0].set_title('Revenue by City Tier')
    axes[0].set_xlabel('City Tier')
    axes[0].set_ylabel('Total Revenue (INR)')

    # Bar chart: Growth by tier
    sns.barplot(x=growth.index, y=growth.values, ax=axes[1], palette='coolwarm')
    axes[1].set_title('Growth Rate by City Tier (2015-2025)')
    axes[1].set_xlabel('City Tier')
    axes[1].set_ylabel('Growth Rate')
    plt.tight_layout()
    plt.show()

    # Choropleth map: Revenue by state (requires state mapping)
    if 'customer_state' in df.columns:
        # Load India states shapefile (user must provide path to shapefile)
        try:

            # Use correct path for shapefile in DataAnalytics folder
            shapefile_path = os.path.join(os.path.dirname(__file__), 'india_states_shapefile.shp')
            india_states = gpd.read_file(shapefile_path)
            state_rev = df.groupby('customer_state')[amount_col].sum().reset_index()
            state_rev['revenue_crore'] = state_rev[amount_col] / 1e7
            india_states = india_states.merge(state_rev[['customer_state', 'revenue_crore']], left_on='ST_NM', right_on='customer_state', how='left')
            india_states['revenue'] = india_states['revenue_crore'].fillna(0)

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            # Plot with legend

            import matplotlib as mpl
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            # Plot without default legend
            india_states.plot(column='revenue', cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=False)
            ax.set_title('Choropleth Map: Revenue by State (INR Crores)')
            ax.axis('off')
            # Custom colorbar in millions
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            norm = mpl.colors.Normalize(vmin=india_states['revenue'].min(), vmax=india_states['revenue'].max())
            sm = mpl.cm.ScalarMappable(cmap='YlOrRd', norm=norm)
            sm.set_array([])
            cb = plt.colorbar(sm, cax=cax)
            cb.set_label('Total Revenue (INR Crores)', fontsize=12)
            def crores_fmt(x, pos):
                return f"₹{x:.2f} Cr"
            cb.ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(crores_fmt))
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[Q7] Could not plot choropleth map: {e}")
    else:
        print("[Q7] No 'customer_state' column found. Choropleth map skipped.")

    print("[Q7] Geographic sales analysis complete.")
    print("Revenue by city tier:")
    print(tier_rev)
    print("Growth rate by city tier:")
    print(growth)

def run_cleaning():

    # Delete existing *_cleaned.csv files before running cleaning
    for f in os.listdir(DATA_DIR):
        if f.startswith('amazon_india_') and f.endswith('_cleaned.csv'):
            try:
                os.remove(os.path.join(DATA_DIR, f))
                print(f"Deleted old cleaned file: {f}")
            except Exception as e:
                print(f"Could not delete {f}: {e}")
    # Now run cleaning
    for fname in os.listdir(DATA_DIR):
        if fname.startswith('amazon_india_') and fname.endswith('.csv') and not fname.endswith('_cleaned.csv'):
            file_path = os.path.join(DATA_DIR, fname)
            print(f'Cleaning {fname}...')
            # Read input file and count rows
            try:
                input_df = pd.read_csv(file_path)
                input_count = len(input_df)
            except Exception as e:
                print(f'Error reading {fname}: {e}')
                input_count = 0
            cleaned_df = clean_file(file_path)
            output_count = len(cleaned_df)
            out_path = os.path.join(DATA_DIR, fname.replace('.csv', '_cleaned.csv'))
            cleaned_df.to_csv(out_path, index=False)
            print(f'Wrote cleaned file: {out_path}')
            print(f'Input rows: {input_count}, Output rows: {output_count}\n')

def plot_line_with_trend(x, y, xlabel, ylabel, title, trend_label='Trend line', line_label=None, annotate_points=None, ax=None, color='tab:blue'):
    """
    Plots a line with optional trend line and annotations.
    annotate_points: list of (x, y, text, color) for annotation.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(x, y, marker='o', label=line_label or ylabel, color=color)
    # Trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), '--', color='gray', label=trend_label)
    # Annotations
    if annotate_points:
        for xp, yp, txt, c in annotate_points:
            ax.annotate(txt, (xp, yp), textcoords="offset points", xytext=(0,10), ha='center', color=c, fontsize=10, fontweight='bold')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return ax

def plot_heatmap(data, title, xlabel, ylabel, fmt='.0f', cmap='YlGnBu'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(12,6))
    sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def plot_multi_line(df, x, y, group, xlabel, ylabel, title, legend_title=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6))
    for g, group_df in df.groupby(group):
        plt.plot(group_df[x], group_df[y], marker='o', label=str(g))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend_title:
        plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def run_eda():
 
    """
    Run EDA on all *_cleaned.csv files. Concatenate all years and perform analyses.
    """
    # Load all cleaned files
    all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.startswith('amazon_india_') and f.endswith('_cleaned.csv')]
    if not all_files:
        print("No cleaned files found. Please run cleaning first.")
        return
    df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(all_files)} cleaned files.")
    # Ensure order_date is datetime
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    # Q1: Revenue trend analysis
    revenue_trend_analysis(df)

    # Q2: Seasonal patterns
    seasonal_patterns_analysis(df)

    # Q3: RFM customer segmentation
    rfm_customer_segmentation(df)

    # Q4: Payment method evolution
    payment_method_evolution(df)

    # Q5: Category-wise performance
    category_performance_analysis(df)

    # Q6: Prime membership impact
    prime_membership_impact_analysis(df)

    # Q7: Geographic sales analysis
    geographic_sales_analysis(df)

    # Q8: Festival sales impact analysis
    festival_sales_impact_analysis(df)

    # Q9: Customer age group demographic analysis
    customer_age_group_analysis(df)

    # Q10: Price vs demand analysis
    price_vs_demand_analysis(df)

    # Q11: Delivery performance analysis
    delivery_performance_analysis(df)

    # Q12: Return patterns analysis
    return_patterns_analysis(df)

    # Q13: Brand performance analysis
    brand_performance_analysis(df)

    # Q14: Customer lifetime value analysis
    customer_lifetime_value_analysis(df)

    # Q15: Discount effectiveness analysis
    discount_effectiveness_analysis(df)

    # Q16: product_rating_patterns_analysis
    product_rating_patterns_analysis(df)

    # Q17: customer_journey_analysis
    customer_journey_analysis(df)

    # Q18: Inventory lifecycle analysis
    inventory_lifecycle_analysis(df)

    # Q19: Competitive pricing analysis
    competitive_pricing_analysis(df)

    # Q20: Business health dashboard
    business_health_dashboard(df)

def product_rating_patterns_analysis(df):
    """
    Q16: Study product rating patterns and their impact on sales. Analyze rating distributions, correlation with sales performance, and identify patterns across subcategories and price ranges.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = df.copy()
    if 'customer_rating' not in df.columns:
        print("[Q16] No customer_rating column found. Skipping rating analysis.")
        return
    amount_col = 'final_amount_inr' if 'final_amount_inr' in df.columns else 'original_price_inr'
    # 1. Rating distribution
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    sns.histplot(df['customer_rating'], bins=20, kde=True, ax=axes[0,0], color='tab:blue')
    axes[0,0].set_title('Product Rating Distribution')
    axes[0,0].set_xlabel('Rating')
    axes[0,0].set_ylabel('Count')
    # 2. Correlation: Rating vs Sales Performance
    if 'quantity' in df.columns:
        corr = df[['customer_rating', 'quantity']].corr().iloc[0,1]
        axes[0,1].scatter(df['customer_rating'], df['quantity'], alpha=0.3, color='tab:orange')
        axes[0,1].set_title(f'Rating vs Sales Volume (Corr={corr:.2f})')
        axes[0,1].set_xlabel('Rating')
        axes[0,1].set_ylabel('Quantity Sold')
    else:
        axes[0,1].text(0.5, 0.5, 'No quantity data available', ha='center', va='center')
        axes[0,1].set_axis_off()
    # 3. Patterns by Subcategory
    if 'subcategory' in df.columns:
        subcat_rating = df.groupby('subcategory')['customer_rating'].mean().reset_index()
        sns.barplot(x='subcategory', y='customer_rating', data=subcat_rating, ax=axes[1,0], palette='tab20')
        axes[1,0].set_title('Avg Rating by Subcategory')
        axes[1,0].set_xlabel('Subcategory')
        axes[1,0].set_ylabel('Avg Rating')
        axes[1,0].tick_params(axis='x', rotation=45)
    else:
        axes[1,0].text(0.5, 0.5, 'No subcategory data available', ha='center', va='center')
        axes[1,0].set_axis_off()
    # 4. Patterns by Price Range
    price_col = amount_col
    df['price_bin'] = pd.qcut(df[price_col], 5, labels=[f'Q{i+1}' for i in range(5)])
    price_rating = df.groupby('price_bin')['customer_rating'].mean().reset_index()
    sns.barplot(x='price_bin', y='customer_rating', data=price_rating, ax=axes[1,1], palette='coolwarm')
    axes[1,1].set_title('Avg Rating by Price Range')
    axes[1,1].set_xlabel('Price Range (Quantile)')
    axes[1,1].set_ylabel('Avg Rating')
    plt.tight_layout()
    plt.show()
    print("[Q16] Product rating patterns analysis complete. See grouped charts for rating distribution, correlation, and patterns.")

def customer_journey_analysis(df):
    """
    Q17: Create customer journey analysis showing purchase frequency patterns, category transitions, and customer evolution from first purchase to loyal customers using flow diagrams and transition matrices.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    df = df.copy()
    # Purchase frequency
    freq = df.groupby('customer_id').size().reset_index(name='order_count')
    # Category transitions
    if 'category' not in df.columns:
        print("[Q17] No category column found. Skipping category transition analysis.")
        return
    df = df.sort_values(['customer_id', 'order_date'])
    df['prev_category'] = df.groupby('customer_id')['category'].shift(1)
    transitions = df.groupby(['prev_category', 'category']).size().reset_index(name='count')
    transition_matrix = transitions.pivot(index='prev_category', columns='category', values='count').fillna(0)
    # Evolution: first purchase to loyal
    loyal_customers = freq[freq['order_count'] >= 5]['customer_id']
    df['is_loyal'] = df['customer_id'].isin(loyal_customers)
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    # 1. Purchase frequency distribution
    sns.histplot(freq['order_count'], bins=20, kde=True, ax=axes[0,0], color='tab:blue')
    axes[0,0].set_title('Customer Purchase Frequency Distribution')
    axes[0,0].set_xlabel('Order Count')
    axes[0,0].set_ylabel('Customer Count')
    # 2. Transition matrix heatmap
    sns.heatmap(transition_matrix, annot=True, fmt='.0f', cmap='YlGnBu', ax=axes[0,1])
    axes[0,1].set_title('Category Transition Matrix')
    axes[0,1].set_xlabel('Next Category')
    axes[0,1].set_ylabel('Previous Category')
    # 3. Evolution: first purchase to loyal (flow)
    loyal_df = df[df['is_loyal']]
    if not loyal_df.empty:
        loyal_first = loyal_df.groupby('customer_id').first().reset_index()
        loyal_last = loyal_df.groupby('customer_id').last().reset_index()
        axes[1,0].scatter(loyal_first['order_date'], loyal_last['order_date'], alpha=0.3, color='tab:green')
        axes[1,0].set_title('Loyal Customer Evolution: First to Last Purchase')
        axes[1,0].set_xlabel('First Purchase Date')
        axes[1,0].set_ylabel('Last Purchase Date')
    else:
        axes[1,0].text(0.5, 0.5, 'No loyal customer data available', ha='center', va='center')
        axes[1,0].set_axis_off()
    # 4. Flow diagram: Sankey (if matplotlib >= 3.4)
    try:
        from matplotlib.sankey import Sankey
        sankey = Sankey(ax=axes[1,1], scale=0.01)
        # Use top transitions for Sankey
        top_trans = transitions.nlargest(6, 'count')
        flows = top_trans['count'].values
        labels = [f"{row['prev_category']}→{row['category']}" for _, row in top_trans.iterrows()]
        sankey.add(flows=flows, labels=labels, orientations=[0]*len(flows))
        sankey.finish()
        axes[1,1].set_title('Top Category Transitions (Sankey)')
    except Exception:
        axes[1,1].text(0.5, 0.5, 'Sankey diagram not available', ha='center', va='center')
        axes[1,1].set_axis_off()
    plt.tight_layout()
    plt.show()
    print("[Q17] Customer journey analysis complete. See grouped charts for frequency, transitions, and evolution.")

def inventory_lifecycle_analysis(df):
    """
    Q18: Analyze inventory and product lifecycle patterns. Study product success by sales count, detect decline phases, and show subcategory evolution over the decade with detailed trend analysis. No product_launch_date required.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = df.copy()
    amount_col = 'final_amount_inr' if 'final_amount_inr' in df.columns else 'original_price_inr'
    # Ensure product_id and product_name exist
    if 'product_id' not in df.columns or 'product_name' not in df.columns:
        print("[Q18] product_id or product_name column missing. Skipping lifecycle analysis.")
        return
    df['order_year'] = df['order_date'].dt.year
    # Product success: total units sold
    if 'quantity' in df.columns:
        prod_sales = df.groupby(['product_id', 'product_name'])['quantity'].sum().reset_index(name='total_sold')
    else:
        prod_sales = df.groupby(['product_id', 'product_name']).size().reset_index(name='total_sold')
    # Product sales by year
    if 'quantity' in df.columns:
        prod_yearly = df.groupby(['product_id', 'product_name', 'order_year'])['quantity'].sum().reset_index()
    else:
        prod_yearly = df.groupby(['product_id', 'product_name', 'order_year']).size().reset_index(name='quantity')
    # Detect decline: year of peak sales vs last year sales
    peak_year = prod_yearly.groupby(['product_id', 'product_name'])[['order_year', 'quantity']].apply(lambda x: x.loc[x['quantity'].idxmax()]).reset_index()
    last_year = prod_yearly.groupby(['product_id', 'product_name'])[['order_year', 'quantity']].apply(lambda x: x.loc[x['order_year'].idxmax()]).reset_index()
    # Ensure product_id and product_name are columns, not index
    if 'product_id' not in peak_year.columns:
        peak_year = peak_year.rename(columns={'product_id':'product_id', 'product_name':'product_name'})
    if 'product_id' not in last_year.columns:
        last_year = last_year.rename(columns={'product_id':'product_id', 'product_name':'product_name'})
    decline_df = peak_year.merge(last_year, on=['product_id', 'product_name'], suffixes=('_peak', '_last'))
    decline_df['decline_pct'] = (decline_df['quantity_peak'] - decline_df['quantity_last']) / decline_df['quantity_peak']
    # Subcategory evolution
    if 'subcategory' in df.columns:
        subcat_yearly = df.groupby(['order_year', 'subcategory'])['quantity'].sum().reset_index()
    else:
        subcat_yearly = None
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    # 1. Product success (barplot for top 20 products)
    top_products = prod_sales.nlargest(20, 'total_sold')
    sns.barplot(x='total_sold', y='product_name', data=top_products, ax=axes[0,0], palette='Blues_r')
    axes[0,0].set_title('Top 20 Products by Units Sold')
    axes[0,0].set_xlabel('Units Sold')
    axes[0,0].set_ylabel('Product Name')
    # 2. Decline phase (histogram)
    sns.histplot(decline_df['decline_pct'], bins=30, kde=True, ax=axes[0,1], color='tab:orange')
    axes[0,1].set_title('Product Decline Phase (% Sales Drop)')
    axes[0,1].set_xlabel('Decline %')
    axes[0,1].set_ylabel('Product Count')
    # 3. Subcategory evolution (line)
    if subcat_yearly is not None:
        sns.lineplot(x='order_year', y='quantity', hue='subcategory', data=subcat_yearly, ax=axes[1,0], marker='o')
        axes[1,0].set_title('Subcategory Sales Evolution (2015-2025)')
        axes[1,0].set_xlabel('Year')
        axes[1,0].set_ylabel('Units Sold')
        axes[1,0].legend(title='Subcategory', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        axes[1,0].text(0.5, 0.5, 'No subcategory data available', ha='center', va='center')
        axes[1,0].set_axis_off()
    # 4. Product lifecycle scatter: peak year vs decline %
    axes[1,1].scatter(decline_df['order_year_peak'], decline_df['decline_pct'], alpha=0.3, color='tab:green')
    axes[1,1].set_title('Product Lifecycle: Peak Year vs Decline %')
    axes[1,1].set_xlabel('Peak Year')
    axes[1,1].set_ylabel('Decline %')
    plt.tight_layout()
    plt.show()
    print("[Q18] Inventory and product lifecycle analysis complete. See grouped charts for product success, decline, and subcategory evolution.")

def competitive_pricing_analysis(df):
    """
    Q19: Build competitive pricing analysis comparing brand positioning, price ranges, and market penetration strategies across different product subcategories using box plots and competitive matrices.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = df.copy()
    price_col = 'final_amount_inr' if 'final_amount_inr' in df.columns else 'original_price_inr'
    # Brand positioning by price
    if 'brand' not in df.columns:
        print("[Q19] No brand column found. Skipping competitive pricing analysis.")
        return
    if 'subcategory' not in df.columns:
        print("[Q19] No subcategory column found. Skipping competitive pricing analysis.")
        return
    # Market penetration: order count by brand/subcategory
    penetration = df.groupby(['brand', 'subcategory']).size().reset_index(name='order_count')
    # Box plot: price ranges by brand
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    top_brands = df['brand'].value_counts().head(8).index.tolist()
    sns.boxplot(x='brand', y=price_col, data=df[df['brand'].isin(top_brands)], ax=axes[0,0], hue='brand', palette='tab20', legend=False)
    axes[0,0].set_title('Price Range by Top Brands')
    axes[0,0].set_xlabel('Brand')
    axes[0,0].set_ylabel('Price (INR)')
    axes[0,0].tick_params(axis='x', rotation=45)
    # Box plot: price ranges by subcategory
    top_subcats = df['subcategory'].value_counts().head(8).index.tolist()
    sns.boxplot(x='subcategory', y=price_col, data=df[df['subcategory'].isin(top_subcats)], ax=axes[0,1], hue='subcategory', palette='Set2', legend=False)
    axes[0,1].set_title('Price Range by Top Subcategories')
    axes[0,1].set_xlabel('Subcategory')
    axes[0,1].set_ylabel('Price (INR)')
    axes[0,1].tick_params(axis='x', rotation=45)
    # Competitive matrix: market penetration
    pen_pivot = penetration.pivot(index='brand', columns='subcategory', values='order_count').fillna(0)
    sns.heatmap(pen_pivot, annot=True, fmt='.0f', cmap='YlGnBu', ax=axes[1,0])
    axes[1,0].set_title('Competitive Matrix: Market Penetration')
    axes[1,0].set_xlabel('Subcategory')
    axes[1,0].set_ylabel('Brand')
    # Scatter: price vs penetration
    brand_price = df.groupby('brand')[price_col].mean().reset_index()
    brand_pen = penetration.groupby('brand')['order_count'].sum().reset_index()
    merged = brand_price.merge(brand_pen, on='brand')
    axes[1,1].scatter(merged[price_col], merged['order_count'], alpha=0.3, color='tab:orange')
    axes[1,1].set_title('Brand Positioning: Price vs Market Penetration')
    axes[1,1].set_xlabel('Avg Price (INR)')
    axes[1,1].set_ylabel('Order Count')
    plt.tight_layout()
    plt.show()
    print("[Q19] Competitive pricing analysis complete. See grouped charts for brand positioning, price ranges, and penetration.")

def business_health_dashboard(df):
    """
    Q20: Create a comprehensive business health dashboard combining key metrics like revenue growth, customer acquisition, retention rates, and operational efficiency using multi-panel visualizations with executive summary insights.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = df.copy()
    amount_col = 'final_amount_inr' if 'final_amount_inr' in df.columns else 'original_price_inr'
    # Revenue growth
    df['year'] = df['order_date'].dt.year
    yearly = df.groupby('year')[amount_col].sum().reset_index(name='revenue')
    yearly['revenue_crore'] = yearly['revenue'] / 1e7
    yearly['pct_growth'] = yearly['revenue'].pct_change() * 100
    # Customer acquisition
    acq_year = df.groupby('customer_id')['order_date'].min().dt.year
    acq_counts = acq_year.value_counts().sort_index().reset_index()
    acq_counts.columns = ['year', 'new_customers']
    # Retention rates (cohort)
    df['cohort_month'] = df.groupby('customer_id')['order_date'].transform('min').dt.to_period('M')
    df['order_month'] = df['order_date'].dt.to_period('M')
    cohort_data = df.groupby(['cohort_month', 'order_month'])['customer_id'].nunique().reset_index()
    cohort_data['period'] = (cohort_data['order_month'] - cohort_data['cohort_month']).apply(lambda x: x.n)
    cohort_pivot = cohort_data.pivot(index='cohort_month', columns='period', values='customer_id')
    retention = cohort_pivot.divide(cohort_pivot.iloc[:,0], axis=0)
    # Operational efficiency: delivery_days
    if 'delivery_days' in df.columns:
        delivery_eff = df.groupby('year')['delivery_days'].mean().reset_index()
    else:
        delivery_eff = None
    # Set up subplots: 2x3
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    # 1. Revenue growth (line)
    sns.lineplot(x='year', y='revenue_crore', data=yearly, marker='o', ax=axes[0,0], color='tab:blue')
    axes[0,0].set_title('Revenue Growth (INR Crores)')
    axes[0,0].set_xlabel('Year')
    axes[0,0].set_ylabel('Revenue (INR Crores)')
    # 2. Customer acquisition (bar)
    sns.barplot(x='year', y='new_customers', data=acq_counts, ax=axes[0,1], color='tab:green')
    axes[0,1].set_title('Customer Acquisition by Year')
    axes[0,1].set_xlabel('Year')
    axes[0,1].set_ylabel('New Customers')
    # 3. Retention rates (heatmap)
    sns.heatmap(retention, annot=True, fmt='.2%', cmap='YlGnBu', ax=axes[0,2])
    axes[0,2].set_title('Cohort Retention Rates')
    axes[0,2].set_xlabel('Months Since Acquisition')
    axes[0,2].set_ylabel('Cohort Month')
    # 4. Operational efficiency (delivery days)
    if delivery_eff is not None:
        sns.lineplot(x='year', y='delivery_days', data=delivery_eff, marker='o', ax=axes[1,0], color='tab:orange')
        axes[1,0].set_title('Operational Efficiency: Avg Delivery Days')
        axes[1,0].set_xlabel('Year')
        axes[1,0].set_ylabel('Avg Delivery Days')
    else:
        axes[1,0].text(0.5, 0.5, 'No delivery data available', ha='center', va='center')
        axes[1,0].set_axis_off()
    # 5. Executive summary (text)
    axes[1,1].axis('off')
    summary = f"Revenue 2025: ₹{yearly['revenue_crore'].iloc[-1]:.2f} Cr\nGrowth: {yearly['pct_growth'].iloc[-1]:.1f}%\nNew Customers 2025: {acq_counts['new_customers'].iloc[-1]}\nRetention (Month 1): {retention.iloc[:,-1].mean():.2%}"
    axes[1,1].text(0.1, 0.5, summary, fontsize=16, ha='left', va='center', color='navy')
    axes[1,1].set_title('Executive Summary')
    # 6. Empty panel for future metrics
    axes[1,2].axis('off')
    plt.tight_layout()
    plt.show()
    print("[Q20] Business health dashboard complete. See multi-panel visualization and executive summary.")

def discount_effectiveness_analysis(df):
    """
    Q15: Analyze discount and promotional effectiveness. Create discount impact analysis showing correlation between discount percentages, sales volumes, and revenue across subcategories and time periods.
    All plots are grouped in one window.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = df.copy()
    # Ensure discount column exists
    if 'discount_percent' not in df.columns:
        print("[Q15] No discount_percent column found. Skipping discount analysis.")
        return
    amount_col = 'final_amount_inr' if 'final_amount_inr' in df.columns else 'original_price_inr'
    # Convert revenue to crores
    df['revenue_crore'] = df[amount_col] / 1e7
    # Time period (year-month)
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df['year_month'] = df['order_date'].dt.to_period('M')
    # 1. Correlation: discount vs sales volume
    corr = df[['discount_percent', 'quantity']].corr().iloc[0,1] if 'quantity' in df.columns else None
    # 2. Discount impact by subcategory
    if 'subcategory' in df.columns:
        subcat_disc = df.groupby('subcategory')['discount_percent'].mean().reset_index()
        subcat_sales = df.groupby('subcategory')['quantity'].sum().reset_index() if 'quantity' in df.columns else None
        subcat_rev = df.groupby('subcategory')['revenue_crore'].sum().reset_index()
    else:
        subcat_disc = None
        subcat_sales = None
        subcat_rev = None
    # 3. Discount trends over time
    disc_time = df.groupby('year_month')['discount_percent'].mean().reset_index()
    sales_time = df.groupby('year_month')['quantity'].sum().reset_index() if 'quantity' in df.columns else None
    rev_time = df.groupby('year_month')['revenue_crore'].sum().reset_index()
    # Set up subplots: 2x2
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    # 1. Scatter: Discount % vs Sales Volume
    if 'quantity' in df.columns:
        axes[0,0].scatter(df['discount_percent'], df['quantity'], alpha=0.3, color='tab:blue')
        axes[0,0].set_title(f'Discount % vs Sales Volume (Corr={corr:.2f})')
        axes[0,0].set_xlabel('Discount Percent')
        axes[0,0].set_ylabel('Quantity Sold')
    else:
        axes[0,0].text(0.5, 0.5, 'No quantity data available', ha='center', va='center')
        axes[0,0].set_axis_off()
    # 2. Bar: Avg Discount % by Subcategory
    if subcat_disc is not None:
        sns.barplot(x='subcategory', y='discount_percent', data=subcat_disc, ax=axes[0,1], palette='tab20')
        axes[0,1].set_title('Avg Discount % by Subcategory')
        axes[0,1].set_xlabel('Subcategory')
        axes[0,1].set_ylabel('Avg Discount Percent')
        axes[0,1].tick_params(axis='x', rotation=45)
    else:
        axes[0,1].text(0.5, 0.5, 'No subcategory data available', ha='center', va='center')
        axes[0,1].set_axis_off()
    # 3. Line: Discount % and Sales/Revenue over Time
    ax2 = axes[1,0]
    ax2.plot(disc_time['year_month'].astype(str), disc_time['discount_percent'], label='Discount %', color='tab:blue')
    if sales_time is not None:
        ax2.plot(sales_time['year_month'].astype(str), sales_time['quantity'], label='Sales Volume', color='tab:orange')
    ax2.set_title('Discount %, Sales Volume Over Time')
    ax2.set_xlabel('Year-Month')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    # 4. Bar: Revenue by Subcategory (in Crores)
    if subcat_rev is not None:
        sns.barplot(x='subcategory', y='revenue_crore', data=subcat_rev, ax=axes[1,1], palette='viridis')
        axes[1,1].set_title('Revenue by Subcategory (INR Crores)')
        axes[1,1].set_xlabel('Subcategory')
        axes[1,1].set_ylabel('Revenue (INR Crores)')
        axes[1,1].tick_params(axis='x', rotation=45)
    else:
        axes[1,1].text(0.5, 0.5, 'No subcategory data available', ha='center', va='center')
        axes[1,1].set_axis_off()
    plt.tight_layout()
    plt.show()
    print("[Q15] Discount effectiveness analysis complete. See grouped charts for discount impact, correlation, and trends.")

def customer_lifetime_value_analysis(df):
    """
    Q14: Build customer lifetime value (CLV) analysis using cohort analysis, retention curves, and CLV distribution across different customer segments and acquisition years.
    All plots are grouped in one window.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = df.copy()
    # Ensure order_date is datetime
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    amount_col = 'final_amount_inr' if 'final_amount_inr' in df.columns else 'original_price_inr'
    # Acquisition year
    df['acq_year'] = df.groupby('customer_id')['order_date'].transform('min').dt.year
    # CLV per customer
    clv = df.groupby('customer_id')[amount_col].sum().reset_index(name='clv')
    clv['acq_year'] = df.groupby('customer_id')['acq_year'].first().values
    # CLV by segment (if available)
    if 'customer_age_group' in df.columns:
        clv['segment'] = df.groupby('customer_id')['customer_age_group'].first().values
    else:
        clv['segment'] = 'Unknown'
    # Cohort analysis: retention curves
    df['cohort_month'] = df.groupby('customer_id')['order_date'].transform('min').dt.to_period('M')
    df['order_month'] = df['order_date'].dt.to_period('M')
    cohort_data = df.groupby(['cohort_month', 'order_month'])['customer_id'].nunique().reset_index()
    cohort_data['period'] = (cohort_data['order_month'] - cohort_data['cohort_month']).apply(lambda x: x.n)
    cohort_pivot = cohort_data.pivot(index='cohort_month', columns='period', values='customer_id')
    retention = cohort_pivot.divide(cohort_pivot.iloc[:,0], axis=0)
    # Set up subplots: 2x2
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    # 1. CLV distribution (histogram)
    sns.histplot(clv['clv']/1e7, bins=30, kde=True, ax=axes[0,0], color='tab:blue')
    axes[0,0].set_title('Customer Lifetime Value (CLV) Distribution (INR Crores)')
    axes[0,0].set_xlabel('CLV (INR Crores)')
    axes[0,0].set_ylabel('Customer Count')
    # 2. CLV by acquisition year (boxplot)
    sns.boxplot(x='acq_year', y=clv['clv']/1e7, data=clv, ax=axes[0,1], palette='Set2')
    axes[0,1].set_title('CLV by Acquisition Year (INR Crores)')
    axes[0,1].set_xlabel('Acquisition Year')
    axes[0,1].set_ylabel('CLV (INR Crores)')
    # 3. CLV by segment (boxplot)
    sns.boxplot(x='segment', y=clv['clv']/1e7, data=clv, ax=axes[1,0], palette='tab20')
    axes[1,0].set_title('CLV by Customer Segment (INR Crores)')
    axes[1,0].set_xlabel('Segment')
    axes[1,0].set_ylabel('CLV (INR Crores)')
    axes[1,0].tick_params(axis='x', rotation=45)
    # 4. Retention curves (heatmap)
    sns.heatmap(retention, annot=True, fmt='.2%', cmap='YlGnBu', ax=axes[1,1])
    axes[1,1].set_title('Cohort Retention Curves')
    axes[1,1].set_xlabel('Months Since Acquisition')
    axes[1,1].set_ylabel('Cohort Month')
    plt.tight_layout()
    plt.show()
    print("[Q14] Customer lifetime value analysis complete. See grouped charts for CLV, retention, and cohorts.")

def brand_performance_analysis(df):
    """
    Q13: Study brand performance and market share evolution. Create brand comparison charts, market share trends, and competitive positioning analysis across different subcategories.
    All plots are grouped in one window.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = df.copy()
    # Check for required columns
    if 'brand' not in df.columns:
        print("[Q13] No brand column found. Skipping brand analysis.")
        return
    amount_col = 'final_amount_inr' if 'final_amount_inr' in df.columns else 'original_price_inr'
    # Convert revenue to crores
    df['revenue_crore'] = df[amount_col] / 1e7
    # Top brands overall
    top_brands = df.groupby('brand')['revenue_crore'].sum().nlargest(8).index.tolist()
    # Market share by year
    if 'order_date' in df.columns:
        df['year'] = pd.to_datetime(df['order_date'], errors='coerce').dt.year
    else:
        df['year'] = None
    market_share = df[df['brand'].isin(top_brands)].groupby(['year', 'brand'])['revenue_crore'].sum().reset_index()
    market_share_pivot = market_share.pivot(index='year', columns='brand', values='revenue_crore').fillna(0)
    market_share_pct = market_share_pivot.div(market_share_pivot.sum(axis=1), axis=0)
    # Competitive positioning by subcategory
    if 'subcategory' in df.columns:
        subcat_brand = df[df['brand'].isin(top_brands)].groupby(['subcategory', 'brand'])['revenue_crore'].sum().reset_index()
        subcat_pivot = subcat_brand.pivot(index='subcategory', columns='brand', values='revenue_crore').fillna(0)
    else:
        subcat_pivot = None
    # Set up subplots: 2x2
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    # 1. Brand comparison (bar)
    brand_rev = df.groupby('brand')['revenue_crore'].sum().loc[top_brands]
    sns.barplot(x=brand_rev.index, y=brand_rev.values, ax=axes[0,0], palette='tab20')
    axes[0,0].set_title('Top Brands by Revenue (INR Crores)')
    axes[0,0].set_xlabel('Brand')
    axes[0,0].set_ylabel('Revenue (INR Crores)')
    axes[0,0].tick_params(axis='x', rotation=45)
    # 2. Market share trends (line)
    if not market_share_pct.empty:
        market_share_pct.plot(ax=axes[0,1], marker='o')
        axes[0,1].set_title('Market Share Trends by Brand (Yearly)')
        axes[0,1].set_xlabel('Year')
        axes[0,1].set_ylabel('Market Share')
        axes[0,1].legend(title='Brand', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        axes[0,1].text(0.5, 0.5, 'No market share data available', ha='center', va='center')
        axes[0,1].set_axis_off()
    # 3. Competitive positioning by subcategory (heatmap)
    if subcat_pivot is not None:
        sns.heatmap(subcat_pivot, annot=True, fmt='.2f', cmap='YlGnBu', ax=axes[1,0])
        axes[1,0].set_title('Brand Revenue by Subcategory (INR Crores, Heatmap)')
        axes[1,0].set_xlabel('Brand')
        axes[1,0].set_ylabel('Subcategory')
    else:
        axes[1,0].text(0.5, 0.5, 'No subcategory data available', ha='center', va='center')
        axes[1,0].set_axis_off()
    # 4. Pie chart: Overall market share
    axes[1,1].pie(brand_rev.values, labels=brand_rev.index, autopct='%1.1f%%', colors=sns.color_palette('tab20', len(top_brands)), startangle=140)
    axes[1,1].set_title('Overall Market Share (Top Brands, Revenue in Crores)')
    plt.tight_layout()
    plt.show()
    print("[Q13] Brand performance analysis complete. See grouped charts for brand comparison, market share, and positioning.")

def return_patterns_analysis(df):
    """
    Q12: Analyze return patterns and customer satisfaction using return rates, reasons, and correlation with product ratings, prices, and subcategories through multiple visualization techniques.
    All plots are grouped in one window.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = df.copy()

    if 'return_status' not in df.columns:
        print("[Q12] No return_status column found. Skipping return analysis.")
        return
    # Create is_returned boolean from return_status
    df['is_returned'] = df['return_status'].str.strip().str.lower() == 'returned'
    return_rate = df['is_returned'].mean()
    subcat_rate = df.groupby('subcategory')['is_returned'].mean().reset_index() if 'subcategory' in df.columns else None
    reason_counts = df['return_reason'].value_counts().head(8) if 'return_reason' in df.columns else None
    has_rating = 'customer_rating' in df.columns
    price_col = 'final_amount_inr' if 'final_amount_inr' in df.columns else 'original_price_inr'
    has_price = price_col in df.columns
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    if subcat_rate is not None:
        sns.barplot(x='subcategory', y='is_returned', data=subcat_rate, ax=axes[0,0], palette='tab20')
        axes[0,0].set_title('Return Rate by Subcategory')
        axes[0,0].set_xlabel('Subcategory')
        axes[0,0].set_ylabel('Return Rate')
        axes[0,0].tick_params(axis='x', rotation=45)
    else:
        axes[0,0].text(0.5, 0.5, 'No subcategory data available', ha='center', va='center')
        axes[0,0].set_axis_off()
    if reason_counts is not None:
        sns.barplot(x=reason_counts.index, y=reason_counts.values, ax=axes[0,1], palette='Set2')
        axes[0,1].set_title('Top Return Reasons')
        axes[0,1].set_xlabel('Reason')
        axes[0,1].set_ylabel('Count')
        axes[0,1].tick_params(axis='x', rotation=45)
    else:
        axes[0,1].text(0.5, 0.5, 'No return reason data available', ha='center', va='center')
        axes[0,1].set_axis_off()
    if has_rating and has_price:
        axes[1,0].scatter(df[price_col], df['customer_rating'], c=df['is_returned'], cmap='coolwarm', alpha=0.3)
        axes[1,0].set_title('Return Patterns: Price vs Rating')
        axes[1,0].set_xlabel('Price (INR)')
        axes[1,0].set_ylabel('Customer Rating')
        cb = plt.colorbar(axes[1,0].collections[0], ax=axes[1,0])
        cb.set_label('Returned (1=True, 0=False)')
    else:
        axes[1,0].text(0.5, 0.5, 'No rating/price data available', ha='center', va='center')
        axes[1,0].set_axis_off()
    axes[1,1].pie([return_rate, 1-return_rate], labels=['Returned', 'Not Returned'], autopct='%1.1f%%', colors=['#e74c3c','#2ecc71'], startangle=90)
    axes[1,1].set_title('Overall Return Rate')
    plt.tight_layout()
    plt.show()
    print(f"[Q12] Return patterns analysis complete. Overall return rate: {return_rate:.2%}")

    

def delivery_performance_analysis(df):
    """
    Q11: Delivery performance analysis
    Shows delivery days distribution, on-time performance, and customer satisfaction correlation with delivery speed across different cities and customer tiers.
    All plots are grouped in one window.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = df.copy()
    # Filter valid delivery_days
    if 'delivery_days' not in df.columns:
        print("[Q11] No delivery_days column found. Skipping delivery performance analysis.")
        return
    df = df[df['delivery_days'].notnull()]
    # On-time: delivery_days <= 2
    df['on_time'] = df['delivery_days'] <= 2
    # Customer satisfaction: use customer_rating if available
    has_rating = 'customer_rating' in df.columns
    # Prepare city tiers
    metro_cities = {'Bengaluru', 'Mumbai', 'Delhi', 'Chennai', 'Kolkata', 'Hyderabad'}
    tier1_cities = {'Pune', 'Ahmedabad', 'Coimbatore', 'Kochi', 'Thiruvananthapuram', 'Jaipur', 'Lucknow', 'Indore', 'Nagpur', 'Chandigarh', 'Goa'}
    def city_tier(city):
        if city in metro_cities:
            return 'Metro'
        elif city in tier1_cities:
            return 'Tier1'
        elif isinstance(city, str) and city:
            return 'Tier2/Rural'
        else:
            return 'Unknown'
    df['city_tier'] = df['customer_city'].apply(city_tier) if 'customer_city' in df.columns else 'Unknown'
    # Set up subplots: 2x2
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    # 1. Delivery days distribution (histogram)
    sns.histplot(df['delivery_days'], bins=20, kde=True, ax=axes[0,0], color='tab:blue')
    axes[0,0].set_title('Delivery Days Distribution')
    axes[0,0].set_xlabel('Delivery Days')
    axes[0,0].set_ylabel('Order Count')
    # 2. On-time performance by city tier (bar)
    on_time_tier = df.groupby('city_tier')['on_time'].mean().reset_index()
    sns.barplot(x='city_tier', y='on_time', data=on_time_tier, ax=axes[0,1], palette='viridis')
    axes[0,1].set_title('On-Time Delivery Rate by City Tier')
    axes[0,1].set_xlabel('City Tier')
    axes[0,1].set_ylabel('On-Time Rate')
    # 3. Delivery days by city (boxplot)
    top_cities = df['customer_city'].value_counts().head(8).index.tolist() if 'customer_city' in df.columns else []
    if top_cities:
        sns.boxplot(x='customer_city', y='delivery_days', data=df[df['customer_city'].isin(top_cities)], ax=axes[1,0], palette='Set2')
        axes[1,0].set_title('Delivery Days by Top Cities')
        axes[1,0].set_xlabel('City')
        axes[1,0].set_ylabel('Delivery Days')
    else:
        axes[1,0].text(0.5, 0.5, 'No city data available', ha='center', va='center')
        axes[1,0].set_axis_off()
    # 4. Customer satisfaction vs delivery speed (scatter/correlation)
    if has_rating:
        axes[1,1].scatter(df['delivery_days'], df['customer_rating'], alpha=0.3, color='tab:orange')
        axes[1,1].set_title('Customer Satisfaction vs Delivery Speed')
        axes[1,1].set_xlabel('Delivery Days')
        axes[1,1].set_ylabel('Customer Rating')
        # Correlation
        corr = df[['delivery_days', 'customer_rating']].corr().iloc[0,1]
        axes[1,1].text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=axes[1,1].transAxes, fontsize=12, color='red', va='top')
    else:
        axes[1,1].text(0.5, 0.5, 'No customer rating data available', ha='center', va='center')
        axes[1,1].set_axis_off()
    plt.tight_layout()
    plt.show()
    print("[Q11] Delivery performance analysis complete. See grouped plots for delivery days, on-time rates, and satisfaction correlation.")

def price_vs_demand_analysis(df):
    """
    Q10: Build price vs demand analysis using scatter plots and correlation matrices.
    Analyze how pricing strategies affect sales volumes across different subcategories and customer segments.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Use final_amount_inr if present, else original_price_inr
    price_col = 'final_amount_inr' if 'final_amount_inr' in df.columns else 'original_price_inr'
    # Demand: use quantity if available, else count of orders
    demand_col = 'quantity' if 'quantity' in df.columns else None
    # Only keep rows with price and demand
    if demand_col:
        df = df.dropna(subset=[price_col, demand_col])
    else:
        df = df.dropna(subset=[price_col])
        df['quantity'] = 1
        demand_col = 'quantity'
    # Convert price to crores
    df['price_crore'] = df[price_col] / 1e7
    # Prepare subcategories and age groups
    subcats = df['subcategory'].value_counts().head(6).index.tolist() if 'subcategory' in df.columns else []
    age_groups = df['customer_age_group'].value_counts().index.tolist() if 'customer_age_group' in df.columns else []
    # Set up subplots: 2 rows x 2 cols
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    # 1. Scatter plot: Price vs Demand (All Data)
    axes[0,0].scatter(df['price_crore'], df[demand_col], alpha=0.3, color='tab:blue')
    axes[0,0].set_xlabel('Price (INR Crores)')
    axes[0,0].set_ylabel('Quantity Sold')
    axes[0,0].set_title('Price vs Demand (All Data)')
    axes[0,0].grid(True, linestyle='--', alpha=0.5)
    # 2. Correlation heatmaps: Price vs Demand by Subcategory (max 3)
    heatmap_axes = [axes[0,1], axes[1,1], axes[1,0]]
    if subcats:
        for i, subcat in enumerate(subcats[:3]):
            sub_df = df[df['subcategory'] == subcat]
            corr = sub_df[['price_crore', demand_col]].corr()
            ax = heatmap_axes[i]
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, cbar=(i==0))
            ax.set_title(f'Correlation: {subcat}')
            ax.set_xlabel('')
            ax.set_ylabel('')
        # If less than 3 subcats, hide unused heatmap axes
        for j in range(len(subcats[:3]), 3):
            heatmap_axes[j].set_axis_off()
    else:
        axes[0,1].text(0.5, 0.5, 'No subcategory data available', ha='center', va='center')
        axes[0,1].set_axis_off()
        axes[1,1].set_axis_off()
        axes[1,0].set_axis_off()
    # 3. Scatter plot: Price vs Demand by Age Group (overlay on axes[1,0] if not used by heatmap)
    age_group_ax = axes[1,0] if not (subcats and len(subcats) >= 3) else axes[1,1]
    if age_groups:
        for age in age_groups:
            sub_df = df[df['customer_age_group'] == age]
            age_group_ax.scatter(sub_df['price_crore'], sub_df[demand_col], alpha=0.3, label=age)
        age_group_ax.set_xlabel('Price (INR Crores)')
        age_group_ax.set_ylabel('Quantity Sold')
        age_group_ax.set_title('Price vs Demand by Customer Age Group')
        age_group_ax.legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
        age_group_ax.grid(True, linestyle='--', alpha=0.5)
    else:
        age_group_ax.text(0.5, 0.5, 'No age group data available', ha='center', va='center')
        age_group_ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    print("[Q10] Price vs demand analysis complete. See grouped plots for insights.")

def customer_age_group_analysis(df):
    """
    Q9: Analyze customer age group behavior and preferences.
    Create demographic analysis with subcategory preferences, spending patterns, and shopping frequency across different age segments.
    Revenue is shown in crores.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    amount_col = 'final_amount_inr' if 'final_amount_inr' in df.columns else 'original_price_inr'
    df = df.copy()
    if 'customer_age_group' not in df.columns:
        print("[Q9] No customer_age_group column found. Skipping age group analysis.")
        return
    df['age_group'] = df['customer_age_group']
    # Spending patterns by age group
    spend = df.groupby('age_group')[amount_col].sum().reset_index()
    spend['revenue_crore'] = spend[amount_col] / 1e7
    # Shopping frequency by age group
    freq = df.groupby('age_group').size().reset_index(name='order_count')
    # Subcategory preferences by age group
    if 'subcategory' in df.columns:
        subcat_pref = df.groupby(['age_group', 'subcategory'])[amount_col].sum().reset_index()
        subcat_pref['revenue_crore'] = subcat_pref[amount_col] / 1e7
        top_subcats = subcat_pref.groupby('subcategory')['revenue_crore'].sum().nlargest(6).index.tolist()
        subcat_pref = subcat_pref[subcat_pref['subcategory'].isin(top_subcats)]
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    # Bar: Spending patterns
    sns.barplot(x='age_group', y='revenue_crore', data=spend, ax=axes[0], palette='viridis')
    axes[0].set_title('Total Revenue by Age Group (INR Crores)')
    axes[0].set_xlabel('Age Group')
    axes[0].set_ylabel('Revenue (INR Crores)')
    # Bar: Shopping frequency
    sns.barplot(x='age_group', y='order_count', data=freq, ax=axes[1], palette='coolwarm')
    axes[1].set_title('Order Frequency by Age Group')
    axes[1].set_xlabel('Age Group')
    axes[1].set_ylabel('Order Count')
    # Stacked bar: Subcategory preferences
    if 'subcategory' in df.columns:
        subcat_pivot = subcat_pref.pivot(index='age_group', columns='subcategory', values='revenue_crore').fillna(0)
        subcat_pivot.plot(kind='bar', stacked=True, ax=axes[2], colormap='tab20')
        axes[2].set_title('Subcategory Preferences by Age Group (INR Crores)')
        axes[2].set_xlabel('Age Group')
        axes[2].set_ylabel('Revenue (INR Crores)')
        axes[2].legend(title='Subcategory', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        axes[2].text(0.5, 0.5, 'No subcategory data available', ha='center', va='center')
        axes[2].set_axis_off()
    plt.tight_layout()
    plt.show()
    print("[Q9] Customer age group analysis complete. See grouped bar charts for spending, frequency, and preferences.")

def festival_sales_impact_analysis(df):
    """
    Q8: Study festival sales impact using before/during/after analysis.
    Visualize revenue spikes for each festival (using festival_name where is_festival_sale is True) with detailed time series analysis.
    All festival trends are shown in one grouped window.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    amount_col = 'final_amount_inr' if 'final_amount_inr' in df.columns else 'original_price_inr'
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    # Filter festival sales
    if 'is_festival_sale' not in df.columns or 'festival_name' not in df.columns:
        print("[Q8] No festival sale columns found. Skipping festival analysis.")
        return
    fest_df = df[df['is_festival_sale'] & df['festival_name'].notnull()].copy()
    # Group by festival_name and order_date
    grouped = fest_df.groupby(['festival_name', 'order_date'])[amount_col].sum().reset_index()
    # Convert revenue to crores
    grouped['revenue_crore'] = grouped[amount_col] / 1e7
    # Get top N festivals by total revenue
    top_fests = grouped.groupby('festival_name')['revenue_crore'].sum().nlargest(8).index.tolist()
    grouped = grouped[grouped['festival_name'].isin(top_fests)]
    # Plot all festival trends in one window
    plt.figure(figsize=(18, 8))
    for fest in top_fests:
        fest_trend = grouped[grouped['festival_name'] == fest].sort_values('order_date')
        plt.plot(fest_trend['order_date'], fest_trend['revenue_crore'], marker='o', label=fest)
    plt.title('Festival Sales Revenue Trends (Top Festivals)')
    plt.xlabel('Date')
    plt.ylabel('Total Revenue (INR Crores)')
    plt.legend(title='Festival Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    print("[Q8] Festival sales impact analysis complete. See grouped time series plot for each festival (revenue in crores).")

def table_operations():
    create_postgres_tables()
    create_time_dimension_table()
    load_orders_to_postgres()
    load_customers_to_postgres()
    load_products_to_postgres()

def main():

    #run_cleaning()
    #scan_and_suggest_map_values(DATA_DIR)
    run_eda()
    # Controls for ETL/refresh
    #table_operations()

if __name__ == '__main__':
    main()
    #scan_price_patterns(DATA_DIR)
