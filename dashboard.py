import streamlit as st
import pandas as pd
import psycopg2
import numpy as np
from sqlalchemy import create_engine
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Amazon India Analytics Dashboard", layout="wide")

@st.cache_resource
def get_engine():
    return create_engine("postgresql+psycopg2://postgres:postgres@localhost:5432/data_analytics_dashboard")

@st.cache_data
def load_table(table):
    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM {table}", engine)
    return df

orders = load_table("orders")
customers = load_table("customers")
products = load_table("products")
time_dim = load_table("time_dimension")

# Debug information
st.sidebar.expander("Debug Info", expanded=False).write({
    "Orders Columns": list(orders.columns),
    "Products Columns": list(products.columns),
    "Customers Columns": list(customers.columns),
    "Time Columns": list(time_dim.columns)
})

# Add year and month columns to orders for filtering
if "order_date" in orders.columns:
    orders["order_year"] = pd.to_datetime(orders["order_date"], errors="coerce").dt.year
    orders["order_month"] = pd.to_datetime(orders["order_date"], errors="coerce").dt.strftime('%B')

st.title("Amazon India Business Analytics Dashboard")

# Create main tabs with organized structure
main_tab1, main_tab2, main_tab3, main_tab4, main_tab5, main_tab6 = st.tabs([
    "📊 Business Performance", 
    "📈 Growth & Revenue", 
    "👥 Customer Analytics", 
    "📦 Product Analytics",
    "🚚 Operations & Logistics", 
    "🔮 Advanced Analytics"
])

# Initialize all tabs variables for backward compatibility
tabs = [None] * 30

# Main Tab 1: Business Performance (Questions 1-5)
with main_tab1:
    st.header("📊 Business Performance Analytics")
    st.write("Core business performance metrics and KPIs covering Questions")
    
    # Create subtabs for Questions 1-5
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Executive Summary", 
        "Business Performance", 
        "Strategic Overview", 
        "Financial Performance", 
        "Growth Analytics"
    ])

# Question 1: Executive Summary
with tab1:
    st.header("Executive Summary Dashboard")
    with st.expander("Filters", expanded=True):
        years = orders["order_year"].dropna().unique()
        year_filter = st.multiselect("Year", sorted(years), default=sorted(years), key="q1_year_filter")
        subcategories = products["subcategory"].dropna().unique()
        subcategory_filter = st.multiselect("Subcategory", sorted(subcategories), default=sorted(subcategories), key="q1_subcategory_filter")
    q1_orders = orders.copy()
    if year_filter:
        q1_orders = q1_orders[q1_orders["order_year"].isin(year_filter)]
    if subcategory_filter:
        q1_orders = q1_orders[q1_orders["subcategory"].isin(subcategory_filter)]
    st.write(f"Filtered orders: {len(q1_orders):,} rows")
    total_revenue = q1_orders["final_amount_inr"].sum()
    active_customers = q1_orders["customer_id"].nunique()
    avg_order_value = q1_orders["final_amount_inr"].mean()
    yearly = q1_orders.groupby("order_year")["final_amount_inr"].sum().sort_index()
    growth_rate = yearly.pct_change().fillna(0) * 100
    top_subcategories = q1_orders.groupby("subcategory")["final_amount_inr"].sum().sort_values(ascending=False).head(5)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
    col2.metric("Growth Rate (YoY)", f"{growth_rate.iloc[-1]:.2f}%", delta=f"{growth_rate.iloc[-1]:.2f}%")
    col3.metric("Active Customers", f"{active_customers:,}")
    col4.metric("Avg Order Value", f"₹{avg_order_value:,.0f}")
    st.subheader("Year-over-Year Revenue Trend")
    st.line_chart(yearly)
    st.subheader("Top Performing Subcategories")
    st.bar_chart(top_subcategories)
    st.info("Use the filters above to view metrics for specific years or subcategories.")

# Real-time Business Performance Monitor tab
with tab2:
    st.header("Real-time Business Performance Monitor")
    with st.expander("Filters", expanded=True):
        months = orders["order_month"].dropna().unique()
        month_filter = st.selectbox("Month", sorted(months), index=0, key="q2_month_filter")
        years = orders["order_year"].dropna().unique()
        year_filter_q2 = st.selectbox("Year", sorted(years), index=len(sorted(years))-1, key="q2_year_filter")
    q2_orders = orders[(orders["order_month"] == month_filter) & (orders["order_year"] == year_filter_q2)]
    monthly_revenue_target = 1000000  # Example target
    customer_acquisition_target = 100  # Example target
    current_month_revenue = q2_orders["final_amount_inr"].sum()
    current_month_orders = len(q2_orders)
    
    # Calculate newly acquired customers
    if not q2_orders.empty:
        all_first_orders = orders.groupby("customer_id")['order_date'].min().reset_index()
        q2_orders_dates = pd.to_datetime(q2_orders["order_date"], errors="coerce")
        q2_customers = q2_orders["customer_id"].unique()
        new_customers = all_first_orders[all_first_orders["customer_id"].isin(q2_customers)]
        new_customers_in_period = new_customers[pd.to_datetime(new_customers["order_date"], errors="coerce").dt.month == pd.to_datetime(q2_orders["order_date"], errors="coerce").dt.month.iloc[0]]
        new_customers_in_period = new_customers_in_period[pd.to_datetime(new_customers_in_period["order_date"], errors="coerce").dt.year == pd.to_datetime(q2_orders["order_date"], errors="coerce").dt.year.iloc[0]]
        current_month_new_customers = len(new_customers_in_period)
    else:
        current_month_new_customers = 0
    
    if not q2_orders.empty and "order_date" in q2_orders.columns:
        q2_orders["order_date"] = pd.to_datetime(q2_orders["order_date"], errors="coerce")
        last_day = q2_orders["order_date"].dt.day.max()
        revenue_run_rate = (current_month_revenue / max(last_day, 1)) * 30
    else:
        revenue_run_rate = 0
    
    revenue_alert = current_month_revenue < monthly_revenue_target
    customer_alert = current_month_new_customers < customer_acquisition_target
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Month Revenue", f"₹{current_month_revenue:,.0f}", delta=f"Target: ₹{monthly_revenue_target:,.0f}")
    col2.metric("Revenue Run Rate", f"₹{revenue_run_rate:,.0f}")
    col3.metric("New Customer Acquisition", f"{current_month_new_customers}", delta=f"Target: {customer_acquisition_target}")
    col4.metric("Order Count", f"{current_month_orders}")
    
    if revenue_alert:
        st.error(f"Revenue below target! Current: ₹{current_month_revenue:,.0f}, Target: ₹{monthly_revenue_target:,.0f}")
    if customer_alert:
        st.warning(f"New customer acquisition below target! Current: {current_month_new_customers}, Target: {customer_acquisition_target}")
    
    avg_order_value_q2 = q2_orders["final_amount_inr"].mean() if not q2_orders.empty else 0
    
    # Top and underperforming products by name
    if not q2_orders.empty:
        product_revenue = q2_orders.groupby("product_id")["final_amount_inr"].sum().reset_index()
        product_revenue = product_revenue.merge(products[["product_id", "product_name"]], on="product_id", how="left")
        top_products = product_revenue.sort_values("final_amount_inr", ascending=False).head(5).set_index("product_name")
        under_products = product_revenue.sort_values("final_amount_inr", ascending=True).head(5).set_index("product_name")
    else:
        top_products = pd.DataFrame()
        under_products = pd.DataFrame()
    
    st.subheader("Key Operational Indicators")
    st.metric("Avg Order Value", f"₹{avg_order_value_q2:,.0f}")
    st.write("Top Products (by Revenue):")
    if not top_products.empty:
        st.bar_chart(top_products["final_amount_inr"])
    st.write("Underperforming Products (by Revenue):")
    if not under_products.empty:
        st.bar_chart(under_products["final_amount_inr"])
    
    st.info("Monitor current month performance, run-rate, and key metrics. Alerts will show if targets are not met.")

# Strategic Overview Dashboard
with tab3:
    st.header("Strategic Overview Dashboard")
    
    # Filters for Strategic View
    with st.expander("Filters", expanded=True):
        time_periods = ["Last Month", "Last Quarter", "Last Year", "All Time"]
        selected_period = st.selectbox("Time Period", time_periods, key="q3_time_period")
        regions = sorted(orders["customer_state"].dropna().unique())
        selected_regions = st.multiselect("Regions", regions, default=regions, key="q3_regions")
        categories = sorted(products["category"].dropna().unique())
        selected_categories = st.multiselect("Categories", categories, default=categories, key="q3_categories")

    # Filter data based on selections
    q3_orders = orders.copy()
    
    # Debug: Show counts before filtering
    st.write("Total orders before filtering:", len(q3_orders))
    
    if selected_regions:
        q3_orders = q3_orders[q3_orders["customer_state"].isin(selected_regions)]
    
    if selected_categories:
        q3_orders = q3_orders[q3_orders["category"].isin(selected_categories)]

    # Apply time period filter
    current_date = pd.to_datetime(q3_orders["order_date"]).max()
    if selected_period == "Last Month":
        q3_orders = q3_orders[pd.to_datetime(q3_orders["order_date"]) >= current_date - pd.DateOffset(months=1)]
    elif selected_period == "Last Quarter":
        q3_orders = q3_orders[pd.to_datetime(q3_orders["order_date"]) >= current_date - pd.DateOffset(months=3)]
    elif selected_period == "Last Year":
        q3_orders = q3_orders[pd.to_datetime(q3_orders["order_date"]) >= current_date - pd.DateOffset(years=1)]

    # 1. Market Share Analysis
    st.subheader("Market Share Analysis")
    col1, col2 = st.columns(2)
    
    # Calculate total revenue from unfiltered data
    total_revenue_unfiltered = orders["final_amount_inr"].sum()
    
    # Category Market Share
    category_share = q3_orders.groupby("category")["final_amount_inr"].sum().sort_values(ascending=False)
    total_revenue = category_share.sum()
    category_share_pct = (category_share / total_revenue_unfiltered * 100).round(2)
    
    with col1:
        st.write("Category Market Share (% of Total Revenue)")
        st.write(f"Showing {len(category_share)} categories out of {len(orders['category'].unique())}")
        st.bar_chart(category_share_pct)
    
    # Subcategory Market Share (Top 10)
    subcategory_share = q3_orders.groupby("subcategory")["final_amount_inr"].sum().sort_values(ascending=False).head(10)
    subcategory_share_pct = (subcategory_share / total_revenue_unfiltered * 100).round(2)
    
    with col2:
        st.write("Top 10 Subcategory Market Share (%)")
        st.bar_chart(subcategory_share_pct)

    # 2. Competitive Positioning
    st.subheader("Competitive Positioning")
    
    # Price Point Analysis
    filtered_price_metrics = q3_orders.groupby("category").agg({
        "final_amount_inr": ["mean", "min", "max", "std"]
    }).round(2)
    filtered_price_metrics.columns = ["Avg Price", "Min Price", "Max Price", "Price Std Dev"]
    
    overall_price_metrics = orders.groupby("category").agg({
        "final_amount_inr": ["mean", "min", "max", "std"]
    }).round(2)
    overall_price_metrics.columns = ["Overall Avg", "Overall Min", "Overall Max", "Overall Std"]
    
    price_metrics = pd.concat([filtered_price_metrics, overall_price_metrics], axis=1)
    
    st.write("Price Point Analysis by Category")
    st.write("(Showing filtered data vs overall data)")
    st.dataframe(price_metrics)

    # 3. Geographic Expansion Metrics
    st.subheader("Geographic Expansion Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        state_revenue = q3_orders.groupby("customer_state")["final_amount_inr"].sum().sort_values(ascending=False)
        st.write("Revenue by State")
        st.bar_chart(state_revenue)
    
    with col2:
        state_customers = q3_orders.groupby("customer_state")["customer_id"].nunique().sort_values(ascending=False)
        st.write("Customer Distribution by State")
        st.bar_chart(state_customers)

    # 4. Business Health Indicators
    st.subheader("Business Health Indicators")
    
    # Key Metrics
    total_customers = q3_orders["customer_id"].nunique()
    avg_order_value = q3_orders["final_amount_inr"].mean()
    order_frequency = len(q3_orders) / total_customers
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
    col2.metric("Unique Customers", f"{total_customers:,}")
    col3.metric("Avg Order Value", f"₹{avg_order_value:,.0f}")
    col4.metric("Orders per Customer", f"{order_frequency:.2f}")

    # Growth Trends
    st.write("Revenue Growth Trend")
    monthly_revenue = q3_orders.groupby(pd.to_datetime(q3_orders["order_date"]).dt.strftime('%Y-%m'))["final_amount_inr"].sum()
    st.line_chart(monthly_revenue)

    # Customer Retention
    repeat_customers = q3_orders.groupby("customer_id").size()
    retention_rate = (len(repeat_customers[repeat_customers > 1]) / len(repeat_customers) * 100)
    st.metric("Customer Retention Rate", f"{retention_rate:.1f}%")

    st.info("This strategic overview provides key insights for C-level decision making, including market positioning, geographic performance, and overall business health indicators.")

# Financial Performance Dashboard
with tab4:
    st.header("Financial Performance Dashboard")
    
    # Filters for Financial View
    with st.expander("Filters", expanded=True):
        years = orders["order_year"].dropna().unique()
        year_filter_q4 = st.multiselect("Year", sorted(years), default=sorted(years)[-1:], key="q4_year_filter")
        
        # Calculate financial quarters
        orders['quarter'] = pd.to_datetime(orders['order_date']).dt.quarter
        quarters = orders[orders['order_year'].isin(year_filter_q4)]['quarter'].unique()
        quarter_filter = st.multiselect("Quarter", sorted(quarters), default=sorted(quarters), key="q4_quarter_filter")
    
    # Filter data
    q4_orders = orders.copy()
    if year_filter_q4:
        q4_orders = q4_orders[q4_orders["order_year"].isin(year_filter_q4)]
    if quarter_filter:
        q4_orders = q4_orders[q4_orders["quarter"].isin(quarter_filter)]
    
    # 1. Revenue Breakdown by Subcategories
    st.subheader("Revenue Breakdown by Subcategories")
    col1, col2 = st.columns(2)
    
    with col1:
        subcategory_revenue = q4_orders.groupby("subcategory")["final_amount_inr"].sum().sort_values(ascending=True)
        st.write("Revenue by Subcategory")
        chart = pd.DataFrame({
            'Subcategory': subcategory_revenue.index,
            'Revenue (₹)': subcategory_revenue.values
        }).set_index('Subcategory')
        st.bar_chart(chart)
    
    with col2:
        total_revenue = subcategory_revenue.sum()
        revenue_contribution = (subcategory_revenue / total_revenue * 100).round(2)
        st.write("Revenue Contribution (%)")
        st.bar_chart(revenue_contribution)
    
    # 2. Profit Margin Analysis
    st.subheader("Profit Margin Analysis")
    
    # Define realistic profit margins for each category
    category_margins = {
        'Smartphones': 15,    # Competitive market, lower margins
        'Laptops': 18,       # Slightly better margins than phones
        'Audio': 35,         # High margin category
        'Tablets': 25,       # Medium margin
        'Smart Watch': 45,   # Very high margin accessory
        'TV & Entertainment': 22  # Moderate margin for TVs
    }
    
    def calculate_financials(row):
        target_margin = category_margins.get(row['category'], 20)
        selling_price = row['final_amount_inr']
        cost = selling_price / (1 + (target_margin/100))
        profit = selling_price - cost
        margin_percent = (profit / selling_price * 100)
        return pd.Series({
            'cost': cost,
            'profit': profit,
            'margin_percent': margin_percent
        })
    
    financials = q4_orders.apply(calculate_financials, axis=1)
    q4_orders['cost'] = financials['cost']
    q4_orders['profit'] = financials['profit']
    q4_orders['margin_percent'] = financials['margin_percent'].round(2)
    q4_orders['target_margin'] = q4_orders['category'].map(category_margins)
    
    col1, col2 = st.columns(2)
    
    with col1:
        margin_comparison = q4_orders.groupby('category').agg({
            'margin_percent': 'mean',
            'target_margin': 'first'
        }).round(2)
        
        margin_comparison = margin_comparison.sort_values('margin_percent', ascending=True)
        st.write("Average Margins by Category")
        st.write("Blue: Actual Margin, Red: Target Margin")
        
        chart_data = pd.DataFrame({
            'Actual Margin %': margin_comparison['margin_percent'],
            'Target Margin %': margin_comparison['target_margin']
        })
        st.bar_chart(chart_data)
        
        margin_comparison['Difference'] = margin_comparison['margin_percent'] - margin_comparison['target_margin']
        st.dataframe(margin_comparison.round(2))
    
    with col2:
        profitability = q4_orders.groupby('category').agg({
            'final_amount_inr': 'sum',
            'cost': 'sum',
            'profit': 'sum',
            'margin_percent': 'mean'
        }).round(2)
        
        profitability['Revenue Share %'] = (profitability['final_amount_inr'] / profitability['final_amount_inr'].sum() * 100).round(2)
        profitability['Profit Share %'] = (profitability['profit'] / profitability['profit'].sum() * 100).round(2)
        
        profitability.columns = ['Revenue', 'Cost', 'Profit', 'Margin %', 'Revenue Share %', 'Profit Share %']
        
        st.write("Profitability Analysis by Category")
        st.dataframe(profitability.style.format({
            'Revenue': '₹{:,.0f}',
            'Cost': '₹{:,.0f}',
            'Profit': '₹{:,.0f}',
            'Margin %': '{:.2f}%',
            'Revenue Share %': '{:.2f}%',
            'Profit Share %': '{:.2f}%'
        }))
    
    # 3. Cost Structure Visualization
    st.subheader("Cost Structure Analysis")
    
    cost_structure = pd.DataFrame({
        'Component': ['Product Cost', 'Discounts', 'Delivery Cost', 'Other Operational Costs'],
        'Percentage': [70, 
                      (q4_orders['discount_percent'].mean()),
                      5,
                      10]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Cost Structure Breakdown (%)")
        st.bar_chart(data=cost_structure.set_index('Component'))
        
    with col2:
        total_cost = q4_orders['cost'].sum()
        total_revenue = q4_orders['final_amount_inr'].sum()
        total_profit = total_revenue - total_cost
        
        st.write("Key Financial Metrics")
        st.metric("Total Revenue", f"₹{total_revenue:,.0f}")
        st.metric("Total Cost", f"₹{total_cost:,.0f}")
        st.metric("Net Profit", f"₹{total_profit:,.0f}")
    
    # 4. Financial Forecasting
    st.subheader("Financial Forecasting")
    
    monthly_revenue = q4_orders.groupby(pd.to_datetime(q4_orders['order_date']).dt.to_period('M'))['final_amount_inr'].sum()
    monthly_revenue.index = monthly_revenue.index.astype(str)
    ma_3 = monthly_revenue.rolling(window=3).mean()
    forecast_value = ma_3.iloc[-1] * 1.1 if len(ma_3) > 0 else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Historical Revenue Trend with 3-Month Moving Average")
        trend_data = pd.DataFrame({
            'Actual Revenue': monthly_revenue,
            'Moving Average': ma_3
        })
        st.line_chart(trend_data)
    
    with col2:
        st.write("Revenue Forecast")
        st.metric(
            "Next Month Forecast", 
            f"₹{forecast_value:,.0f}",
            delta=f"{((forecast_value/monthly_revenue.iloc[-1] if len(monthly_revenue) > 0 else 1) - 1)*100:.1f}%"
        )
        
        st.write("Forecast Ranges:")
        st.write(f"Conservative (5% growth): ₹{ma_3.iloc[-1] * 1.05:,.0f}")
        st.write(f"Moderate (10% growth): ₹{forecast_value:,.0f}")
        st.write(f"Optimistic (15% growth): ₹{ma_3.iloc[-1] * 1.15:,.0f}")
    
    st.info("This financial dashboard provides detailed insights into revenue patterns, profitability metrics, cost structures, and future projections. The forecasting model uses historical trends and moving averages to predict future performance.")

# Growth Analytics Dashboard
with tab5:
    st.header("Growth Analytics Dashboard")
    
    # Filters for Growth Analytics
    with st.expander("Filters", expanded=True):
        time_periods = ["Last 6 Months", "Last 12 Months", "Last 24 Months", "All Time"]
        selected_period = st.selectbox("Analysis Period", time_periods, key="q5_time_period")
        
        # Get date range based on selection
        latest_date = pd.to_datetime(orders["order_date"]).max()
        if selected_period == "Last 6 Months":
            start_date = latest_date - pd.DateOffset(months=6)
        elif selected_period == "Last 12 Months":
            start_date = latest_date - pd.DateOffset(months=12)
        elif selected_period == "Last 24 Months":
            start_date = latest_date - pd.DateOffset(months=24)
        else:
            start_date = pd.to_datetime(orders["order_date"]).min()
        
        # Filter orders for selected period
        mask = (pd.to_datetime(orders["order_date"]) >= start_date) & (pd.to_datetime(orders["order_date"]) <= latest_date)
        q5_orders = orders[mask].copy()
        
        # Calculate monthly dates for trend analysis
        q5_orders["order_month"] = pd.to_datetime(q5_orders["order_date"]).dt.to_period("M")
    
    # 1. Customer Growth Analysis
    st.subheader("Customer Growth Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly New Customer Acquisition
        monthly_new_customers = []
        months = sorted(q5_orders["order_month"].unique())
        
        for month in months:
            month_orders = q5_orders[q5_orders["order_month"] <= month]
            previous_customers = set(q5_orders[q5_orders["order_month"] < month]["customer_id"])
            current_customers = set(month_orders[month_orders["order_month"] == month]["customer_id"])
            new_customers = len(current_customers - previous_customers)
            monthly_new_customers.append({"Month": str(month), "New Customers": new_customers})
        
        monthly_new_customers_df = pd.DataFrame(monthly_new_customers)
        
        st.write("Monthly New Customer Acquisition")
        st.line_chart(monthly_new_customers_df.set_index("Month"))
        
        total_customers = len(q5_orders["customer_id"].unique())
        avg_monthly_growth = monthly_new_customers_df["New Customers"].mean()
        st.metric("Average Monthly New Customers", f"{avg_monthly_growth:.0f}")
        st.metric("Total Unique Customers", f"{total_customers:,}")
    
    with col2:
        # Customer Retention Analysis
        retention_data = []
        months = sorted(q5_orders["order_month"].unique())
        
        for i in range(1, len(months)):  # Skip first month
            current_month = months[i]
            previous_month = months[i-1]
            previous_customers = set(q5_orders[q5_orders["order_month"] == previous_month]["customer_id"])
            retained_customers = set(q5_orders[q5_orders["order_month"] == current_month]["customer_id"])
            retention_rate = len(previous_customers & retained_customers) / len(previous_customers) * 100 if len(previous_customers) > 0 else 0
            retention_data.append({"Month": str(current_month), "Retention Rate": retention_rate})
        
        retention_data_df = pd.DataFrame(retention_data)
        
        st.write("Monthly Customer Retention Rate (%)")
        st.line_chart(retention_data_df.set_index("Month"))
        
        avg_retention = retention_data_df["Retention Rate"].mean()
        st.metric("Average Monthly Retention Rate", f"{avg_retention:.1f}%")
    
    # 2. Market Penetration Analysis
    st.subheader("Market Penetration Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Geographic Penetration
        state_penetration = q5_orders.groupby("customer_state").agg({
            "customer_id": "nunique",
            "final_amount_inr": "sum"
        }).reset_index()
        
        state_penetration["Revenue per Customer"] = state_penetration["final_amount_inr"] / state_penetration["customer_id"]
        state_penetration["Market Share %"] = (state_penetration["final_amount_inr"] / state_penetration["final_amount_inr"].sum() * 100).round(2)
        
        st.write("Geographic Market Penetration")
        st.dataframe(state_penetration.sort_values("Market Share %", ascending=False).style.format({
            "customer_id": "{:,}",
            "final_amount_inr": "₹{:,.0f}",
            "Revenue per Customer": "₹{:,.0f}",
            "Market Share %": "{:.2f}%"
        }))
    
    with col2:
        # Category Penetration
        category_penetration = q5_orders.groupby("category").agg({
            "customer_id": "nunique",
            "final_amount_inr": "sum",
            "order_id": "count"
        }).reset_index()
        
        category_penetration["Orders per Customer"] = (category_penetration["order_id"] / category_penetration["customer_id"]).round(2)
        category_penetration["Category Share %"] = (category_penetration["final_amount_inr"] / category_penetration["final_amount_inr"].sum() * 100).round(2)
        
        st.write("Category Market Penetration")
        st.dataframe(category_penetration.sort_values("Category Share %", ascending=False).style.format({
            "customer_id": "{:,}",
            "final_amount_inr": "₹{:,.0f}",
            "Orders per Customer": "{:.2f}",
            "Category Share %": "{:.2f}%"
        }))
    
    # 3. Product Portfolio Analysis
    st.subheader("Product Portfolio Growth")
    
    monthly_category_revenue = q5_orders.groupby([
        pd.to_datetime(q5_orders["order_date"]).dt.to_period("M"),
        "category"
    ])["final_amount_inr"].sum().reset_index()
    
    category_trends = monthly_category_revenue.pivot(
        index="order_date",
        columns="category",
        values="final_amount_inr"
    )
    
    st.write("Category Revenue Trends")
    st.line_chart(category_trends)
    
    growth_rates = pd.DataFrame()
    for category in category_trends.columns:
        first_value = category_trends[category].iloc[0]
        last_value = category_trends[category].iloc[-1]
        growth_rate = ((last_value / first_value) - 1) * 100 if first_value > 0 else 0
        growth_rates = pd.concat([growth_rates,
                                pd.DataFrame({"Category": [category], "Growth Rate": [growth_rate]})])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Category Growth Rates (%)")
        st.bar_chart(growth_rates.set_index("Category"))
    
    with col2:
        st.write("Product Mix Evolution")
        early_mix = q5_orders[pd.to_datetime(q5_orders["order_date"]) <= (start_date + (latest_date - start_date)/2)]
        late_mix = q5_orders[pd.to_datetime(q5_orders["order_date"]) > (start_date + (latest_date - start_date)/2)]
        
        mix_evolution = pd.DataFrame({
            "Early Period": early_mix.groupby("category")["final_amount_inr"].sum() / early_mix["final_amount_inr"].sum() * 100,
            "Late Period": late_mix.groupby("category")["final_amount_inr"].sum() / late_mix["final_amount_inr"].sum() * 100
        }).round(2)
        
        st.dataframe(mix_evolution.style.format("{:.2f}%"))
    
    # 4. Strategic Growth Indicators
    st.subheader("Strategic Growth Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        customer_total_revenue = q5_orders.groupby("customer_id")["final_amount_inr"].sum()
        avg_clv = customer_total_revenue.mean()
        st.metric("Average Customer Lifetime Value", f"₹{avg_clv:,.0f}")
        
        customer_orders = q5_orders.groupby("customer_id")["order_id"].count()
        avg_frequency = customer_orders.mean()
        st.metric("Average Purchase Frequency", f"{avg_frequency:.2f} orders")
    
    with col2:
        monthly_aov = q5_orders.groupby("order_month")["final_amount_inr"].mean()
        aov_growth = ((monthly_aov.iloc[-1] / monthly_aov.iloc[0]) - 1) * 100
        st.metric("Average Order Value Growth", f"{aov_growth:.1f}%")
        
        total_states = len(q5_orders["customer_state"].unique())
        st.metric("Geographic Markets", f"{total_states} states")
    
    with col3:
        customer_categories = q5_orders.groupby("customer_id")["category"].nunique()
        avg_categories = customer_categories.mean()
        st.metric("Avg Categories per Customer", f"{avg_categories:.2f}")
        
        premium_customers = len(q5_orders[q5_orders["is_prime_member"]]["customer_id"].unique())
        premium_ratio = (premium_customers / total_customers * 100)
        st.metric("Premium Customer Ratio", f"{premium_ratio:.1f}%")
    
    # Growth Predictions
    st.subheader("Growth Predictions")
    
    monthly_metrics = q5_orders.groupby("order_month").agg({
        "final_amount_inr": "sum",
        "customer_id": "nunique",
        "order_id": "count"
    }).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        revenue_series = monthly_metrics["final_amount_inr"]
        growth_rate = (revenue_series.iloc[-1] / revenue_series.iloc[0]) ** (1/len(revenue_series)) - 1
        projected_revenue = revenue_series.iloc[-1] * (1 + growth_rate)
        
        st.write("Revenue Growth Projection")
        st.metric(
            "Next Month Projected Revenue",
            f"₹{projected_revenue:,.0f}",
            delta=f"{growth_rate*100:.1f}% projected growth"
        )
    
    with col2:
        customer_series = monthly_metrics["customer_id"]
        customer_growth = (customer_series.iloc[-1] / customer_series.iloc[0]) ** (1/len(customer_series)) - 1
        projected_customers = customer_series.iloc[-1] * (1 + customer_growth)
        
        st.write("Customer Base Growth Projection")
        st.metric(
            "Next Month Projected New Customers",
            f"{projected_customers:,.0f}",
            delta=f"{customer_growth*100:.1f}% projected growth"
        )
    
    st.info("""This Growth Analytics dashboard provides comprehensive insights into customer acquisition, 
    retention, market penetration, and product portfolio performance. The strategic indicators and 
    growth predictions help identify opportunities and track progress towards growth objectives.""")

# Main Tab 2: Growth & Revenue Analytics (Questions 6-10)
with main_tab2:
    st.header("📈 Growth & Revenue Analytics")
    st.write("Revenue trends, geographic analysis, and performance metrics covering Questions")
    
    # Create subtabs for Questions 6-10
    tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "Revenue Trends", 
        "SubCategory Performance", 
        "Geographic Revenue", 
        "Festival Sales", 
        "Price Optimization"
    ])

# Question 6: Revenue Trend Analysis
with tab6:
    st.header("Revenue Trend Analysis Dashboard")
    
    # Time Period Selection
    with st.expander("Time Period Selection", expanded=True):
        # Year range selection
        min_year = pd.to_datetime(orders['order_date']).dt.year.min()
        max_year = pd.to_datetime(orders['order_date']).dt.year.max()
        selected_years = st.slider(
            "Select Year Range",
            min_value=int(min_year),
            max_value=int(max_year),
            value=(int(min_year), int(max_year))
        )
        
        # Aggregation level selection
        agg_level = st.radio(
            "Select Time Aggregation",
            ["Monthly", "Quarterly", "Yearly"],
            horizontal=True
        )
        
        # Filter orders by selected years
        orders['date'] = pd.to_datetime(orders['order_date'])
        mask = (orders['date'].dt.year >= selected_years[0]) & \
               (orders['date'].dt.year <= selected_years[1])
        filtered_orders = orders[mask].copy()
        
        # Create time-based columns
        if agg_level == "Monthly":
            filtered_orders['period'] = filtered_orders['date'].dt.strftime('%Y-%m')
        elif agg_level == "Quarterly":
            filtered_orders['period'] = filtered_orders['date'].dt.strftime('%Y-Q%q')
        else:
            filtered_orders['period'] = filtered_orders['date'].dt.year.astype(str)
    
    # 1. Revenue Trends
    st.subheader("Revenue Trends")
    col1, col2 = st.columns(2)
    
    with col1:
        revenue_trend = filtered_orders.groupby('period')['final_amount_inr'].sum().reset_index()

        fig_revenue = go.Figure(data=[
            go.Scatter(
                x=revenue_trend['period'],
                y=revenue_trend['final_amount_inr'],
                mode='lines+markers',
                name='Revenue'
            )
        ])
        fig_revenue.update_layout(
            title=f'{agg_level} Revenue Trend',
            xaxis_title='Period',
            yaxis_title='Revenue (₹)'
        )
        st.plotly_chart(fig_revenue)
    
    with col2:
        revenue_trend['growth_rate'] = revenue_trend['final_amount_inr'].pct_change() * 100
        fig_growth = go.Figure(data=[
            go.Bar(
                x=revenue_trend['period'],
                y=revenue_trend['growth_rate'],
                name='Growth Rate'
            )
        ])
        fig_growth.update_layout(
            title=f'{agg_level} Growth Rate',
            xaxis_title='Period',
            yaxis_title='Growth Rate (%)'
        )
        st.plotly_chart(fig_growth)
    
    # 2. Seasonal Analysis
    st.subheader("Seasonal Analysis")
    
    filtered_orders['month'] = pd.Categorical(filtered_orders['date'].dt.strftime('%B'), 
        categories=['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December'],
        ordered=True)
    filtered_orders['quarter'] = filtered_orders['date'].dt.quarter
    
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_pattern = filtered_orders.groupby('month')['final_amount_inr'].mean().reindex([
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ])
        st.write("Average Monthly Revenue Pattern")
        st.line_chart(monthly_pattern)
    
    with col2:
        quarterly_pattern = filtered_orders.groupby('quarter')['final_amount_inr'].mean()
        st.write("Average Quarterly Revenue Pattern")
        st.line_chart(quarterly_pattern)
    
    # 3. Revenue Distribution Analysis
    st.subheader("Revenue Distribution Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_stats = filtered_orders.groupby('month')['final_amount_inr'].describe()
        st.write("Monthly Revenue Distribution")
        st.dataframe(monthly_stats.style.format({
            'mean': '₹{:,.0f}',
            'std': '₹{:,.0f}',
            'min': '₹{:,.0f}',
            '25%': '₹{:,.0f}',
            '50%': '₹{:,.0f}',
            '75%': '₹{:,.0f}',
            'max': '₹{:,.0f}'
        }))
    
    with col2:
        volatility = filtered_orders.groupby('period')['final_amount_inr'].std() / \
                    filtered_orders.groupby('period')['final_amount_inr'].mean() * 100
        st.write("Revenue Volatility by Period (CV%)")
        st.line_chart(volatility)
    
    # 4. Revenue Forecasting
    st.subheader("Revenue Forecasting")
    
    ts_data = filtered_orders.groupby('period')['final_amount_inr'].sum()
    ma_3 = ts_data.rolling(window=3).mean()
    ma_6 = ts_data.rolling(window=6).mean()
    
    last_value = ts_data.iloc[-1]
    growth_rate = (ts_data.iloc[-1] / ts_data.iloc[-6]) ** (1/6) - 1 if len(ts_data) >= 6 else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        ma_data = pd.DataFrame({
            'Actual': ts_data,
            '3-Period MA': ma_3,
            '6-Period MA': ma_6
        })
        st.write("Revenue Trend with Moving Averages")
        st.line_chart(ma_data)
    
    with col2:
        st.write("Revenue Forecast")
        forecast_1 = last_value * (1 + growth_rate)
        forecast_3 = last_value * (1 + growth_rate) ** 3
        
        st.metric(
            "Next Period Forecast",
            f"₹{forecast_1:,.0f}",
            delta=f"{growth_rate*100:.1f}% projected growth"
        )
        st.metric(
            "3-Period Ahead Forecast",
            f"₹{forecast_3:,.0f}",
            delta=f"{(forecast_3/last_value-1)*100:.1f}% total growth"
        )
        
# Category Performance Dashboard (Question 7)
with tab7:
    st.header("SubCategory Performance Dashboard")
    
    # Time Period Selection
    with st.expander("Filters", expanded=True):
        # Date range selection
        date_range = st.date_input(
            "Select Date Range",
            value=(
                pd.to_datetime(orders['order_date']).min(),
                pd.to_datetime(orders['order_date']).max()
            ),
            min_value=pd.to_datetime(orders['order_date']).min(),
            max_value=pd.to_datetime(orders['order_date']).max()
        )
        
        # Subcategory selection
        all_subcategories = orders['subcategory'].unique()
        selected_subcategories = st.multiselect(
            "Select Subcategories",
            sorted(all_subcategories),
            default=sorted(all_subcategories)
        )
    
    # Filter data
    q7_orders = orders[
        (pd.to_datetime(orders['order_date']) >= pd.to_datetime(date_range[0])) &
        (pd.to_datetime(orders['order_date']) <= pd.to_datetime(date_range[1])) &
        (orders['subcategory'].isin(selected_subcategories))
    ].copy()
    
    # 1. Category Revenue Contribution
    st.subheader("Category Revenue Contribution")
    col1, col2 = st.columns(2)
    
    with col1:
        # Overall subcategory contribution
        subcategory_revenue = q7_orders.groupby('subcategory')['final_amount_inr'].sum().sort_values(ascending=True)
        total_revenue = subcategory_revenue.sum()
        subcategory_share = (subcategory_revenue / total_revenue * 100).round(2)
        
        st.write("Subcategory Revenue Share")
        # Take top 15 subcategories for better visualization
        top_15_subcategories = subcategory_revenue.nlargest(15)
        fig_share = go.Figure(data=[
            go.Pie(
                values=top_15_subcategories.values,
                labels=top_15_subcategories.index,
                hole=0.4,
                name='Subcategory Share'
            )
        ])
        fig_share.update_layout(
            title='Top 15 Subcategories Revenue Distribution'
        )
        st.plotly_chart(fig_share)
    
    with col2:
        # Subcategory metrics
        subcategory_metrics = q7_orders.groupby('subcategory').agg({
            'final_amount_inr': ['sum', 'mean', 'count'],
            'order_id': 'nunique'
        }).round(2)
        subcategory_metrics.columns = ['Total Revenue', 'Avg Order Value', 'Number of Sales', 'Unique Orders']
        
        st.write("Subcategory Performance Metrics")
        st.dataframe(subcategory_metrics.style.format({
            'Total Revenue': '₹{:,.0f}',
            'Avg Order Value': '₹{:,.0f}',
            'Number of Sales': '{:,.0f}',
            'Unique Orders': '{:,.0f}'
        }))
    
    # 2. Growth Trends by Subcategory
    st.subheader("Subcategory Growth Trends")
    
    q7_orders['month_year'] = pd.to_datetime(q7_orders['order_date']).dt.strftime('%Y-%m')
    
    # Get top 10 subcategories by revenue for trend analysis
    top_10_subcategories = q7_orders.groupby('subcategory')['final_amount_inr'].sum().nlargest(10).index
    
    subcategory_trends = q7_orders[q7_orders['subcategory'].isin(top_10_subcategories)].pivot_table(
        index='month_year',
        columns='subcategory',
        values='final_amount_inr',
        aggfunc='sum'
    ).fillna(0)
    
    subcategory_growth = pd.DataFrame()
    for subcategory in subcategory_trends.columns:
        first_value = subcategory_trends[subcategory].iloc[0]
        last_value = subcategory_trends[subcategory].iloc[-1]
        growth = ((last_value / first_value) - 1) * 100 if first_value > 0 else 0
        subcategory_growth = pd.concat([
            subcategory_growth,
            pd.DataFrame({'Subcategory': [subcategory], 'Growth Rate': [growth]})
        ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Revenue Trends by Top 10 Subcategories")
        st.line_chart(subcategory_trends)
    
    with col2:
        st.write("Subcategory Growth Rates")
        fig_growth = go.Figure(data=[
            go.Bar(
                x=subcategory_growth['Subcategory'],
                y=subcategory_growth['Growth Rate'],
                name='Growth Rate'
            )
        ])
        fig_growth.update_layout(
            title='Top 10 Subcategories Growth Rates (%)',
            xaxis_title='Subcategory',
            yaxis_title='Growth Rate (%)'
        )
        st.plotly_chart(fig_growth)
    
    # 3. Market Share Analysis
    st.subheader("Market Share Analysis")
    
    mid_date = pd.to_datetime(date_range[0]) + (pd.to_datetime(date_range[1]) - pd.to_datetime(date_range[0]))/2
    early_period = q7_orders[pd.to_datetime(q7_orders['order_date']) <= mid_date]
    late_period = q7_orders[pd.to_datetime(q7_orders['order_date']) > mid_date]
    
    # Get top 15 subcategories for market share analysis
    top_15_revenue = q7_orders.groupby('subcategory')['final_amount_inr'].sum().nlargest(15).index
    
    early_share = early_period[early_period['subcategory'].isin(top_15_revenue)].groupby('subcategory')['final_amount_inr'].sum() / early_period['final_amount_inr'].sum() * 100
    late_share = late_period[late_period['subcategory'].isin(top_15_revenue)].groupby('subcategory')['final_amount_inr'].sum() / late_period['final_amount_inr'].sum() * 100
    
    market_share_change = pd.DataFrame({
        'Early Period Share': early_share,
        'Late Period Share': late_share,
        'Share Change': late_share - early_share
    }).round(2)
    
    st.write("Market Share Evolution (Top 15 Subcategories)")
    st.dataframe(market_share_change.style.format({
        'Early Period Share': '{:.2f}%',
        'Late Period Share': '{:.2f}%',
        'Share Change': '{:+.2f}%'
    }).background_gradient(subset=['Share Change'], cmap='RdYlGn'))
    
    # 4. Subcategory Profitability Analysis
    st.subheader("Subcategory Profitability Analysis")
    
    q7_orders['cost'] = q7_orders['final_amount_inr'] * 0.7  # Assuming 30% margin
    q7_orders['profit'] = q7_orders['final_amount_inr'] - q7_orders['cost']
    q7_orders['margin_percent'] = (q7_orders['profit'] / q7_orders['final_amount_inr'] * 100).round(2)
    
    profitability = q7_orders.groupby('subcategory').agg({
        'final_amount_inr': 'sum',
        'profit': 'sum',
        'margin_percent': 'mean',
        'order_id': 'count'
    }).round(2)
    
    profitability['revenue_rank'] = profitability['final_amount_inr'].rank(ascending=False)
    profitability['profit_rank'] = profitability['profit'].rank(ascending=False)
    profitability['orders_rank'] = profitability['order_id'].rank(ascending=False)
    
    # Calculate revenue per order
    profitability['revenue_per_order'] = (profitability['final_amount_inr'] / profitability['order_id']).round(2)
    
    st.write("Subcategory Profitability Matrix")
    
    # Sort by revenue and get top 20 subcategories
    top_20_profitability = profitability.nlargest(20, 'final_amount_inr')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Top 20 Subcategories by Revenue")
        st.dataframe(top_20_profitability.style.format({
            'final_amount_inr': '₹{:,.0f}',
            'profit': '₹{:,.0f}',
            'margin_percent': '{:.2f}%',
            'order_id': '{:,.0f}',
            'revenue_per_order': '₹{:,.0f}',
            'revenue_rank': '{:.0f}',
            'profit_rank': '{:.0f}',
            'orders_rank': '{:.0f}'
        }).background_gradient(subset=['margin_percent'], cmap='RdYlGn'))
    
    with col2:
        # Scatter plot of revenue vs margin
        fig = go.Figure(data=go.Scatter(
            x=profitability['final_amount_inr'],
            y=profitability['margin_percent'],
            mode='markers+text',
            text=profitability.index,
            textposition="top center",
            marker=dict(
                size=profitability['order_id'] / profitability['order_id'].max() * 50,
                color=profitability['profit'],
                colorscale='Viridis',
                showscale=True
            )
        ))
        fig.update_layout(
            title='Revenue vs Margin % (bubble size = number of orders)',
            xaxis_title='Revenue (₹)',
            yaxis_title='Margin %',
            showlegend=False
        )
        st.plotly_chart(fig)
    
    st.info("""
    This Subcategory Performance Dashboard provides detailed insights into:
    1. Revenue contribution and distribution across subcategories
    2. Growth trends and subcategory performance evolution
    3. Market share changes and competitive positioning
    4. Profitability analysis with revenue per order and ranking metrics
    
    Use the date range and subcategory filters to analyze specific periods and subcategories of interest.
    The bubble chart shows the relationship between revenue, margin percentage, and order volume.
    """)

# Geographic Revenue Dashboard (Question 8)
with tab8:
    st.header("Geographic Revenue Analysis Dashboard")
    
    # Filters
    with st.expander("Filters", expanded=True):
        # Time period selection
        years = orders["order_year"].dropna().unique()
        selected_years = st.multiselect("Select Years", sorted(years), default=sorted(years)[-3:], key="q8_year_filter")
        
        # State selection
        states = orders["customer_state"].dropna().unique()
        selected_states = st.multiselect("Select States", sorted(states), default=sorted(states), key="q8_state_filter")
        
        # Subcategory selection for drill-down
        subcategories = orders["subcategory"].dropna().unique()
        selected_subcategories = st.multiselect("Select Subcategories", sorted(subcategories), default=sorted(subcategories), key="q8_subcategory_filter")
    
    # Filter data
    q8_orders = orders.copy()
    if selected_years:
        q8_orders = q8_orders[q8_orders["order_year"].isin(selected_years)]
    if selected_states:
        q8_orders = q8_orders[q8_orders["customer_state"].isin(selected_states)]
    if selected_subcategories:
        q8_orders = q8_orders[q8_orders["subcategory"].isin(selected_subcategories)]
    
    # 1. Overall Geographic Performance
    st.subheader("Overall Geographic Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_revenue = q8_orders["final_amount_inr"].sum()
    total_states = len(q8_orders["customer_state"].unique())
    avg_revenue_per_state = total_revenue / total_states
    top_state_revenue = q8_orders.groupby("customer_state")["final_amount_inr"].sum().max()
    
    col1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
    col2.metric("Active States", f"{total_states}")
    col3.metric("Avg Revenue/State", f"₹{avg_revenue_per_state:,.0f}")
    col4.metric("Top State Revenue", f"₹{top_state_revenue:,.0f}")
    
    # 2. State-wise Revenue Distribution
    st.subheader("State-wise Revenue Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        state_revenue = q8_orders.groupby("customer_state").agg({
            "final_amount_inr": "sum",
            "order_id": "count",
            "customer_id": "nunique"
        }).round(2)
        
        state_revenue["revenue_per_order"] = (state_revenue["final_amount_inr"] / state_revenue["order_id"]).round(2)
        state_revenue["revenue_per_customer"] = (state_revenue["final_amount_inr"] / state_revenue["customer_id"]).round(2)
        state_revenue = state_revenue.sort_values("final_amount_inr", ascending=False)
        
        st.write("State-wise Performance Metrics")
        st.dataframe(state_revenue.style.format({
            "final_amount_inr": "₹{:,.0f}",
            "order_id": "{:,.0f}",
            "customer_id": "{:,.0f}",
            "revenue_per_order": "₹{:,.0f}",
            "revenue_per_customer": "₹{:,.0f}"
        }).background_gradient(subset=["final_amount_inr"], cmap="Blues"))
    
    with col2:
        # Top 10 states revenue visualization
        top_10_states = state_revenue["final_amount_inr"].nlargest(10)
        fig = go.Figure(data=[
            go.Bar(
                x=top_10_states.index,
                y=top_10_states.values,
                text=top_10_states.values.round(2),
                texttemplate='₹%{text:,.0f}',
                textposition='auto',
            )
        ])
        fig.update_layout(
            title='Top 10 States by Revenue',
            xaxis_title='State',
            yaxis_title='Revenue (₹)',
            showlegend=False
        )
        st.plotly_chart(fig)
    
    # 3. Geographic Growth Analysis
    st.subheader("Geographic Growth Analysis")
    
    # Calculate year-over-year growth for each state
    yearly_state_revenue = q8_orders.pivot_table(
        index="customer_state",
        columns="order_year",
        values="final_amount_inr",
        aggfunc="sum"
    ).fillna(0)
    
    # Calculate growth rates
    growth_rates = pd.DataFrame()
    for state in yearly_state_revenue.index:
        state_data = yearly_state_revenue.loc[state]
        if len(state_data) >= 2:
            initial_value = state_data.iloc[0]
            final_value = state_data.iloc[-1]
            cagr = (((final_value / initial_value) ** (1 / len(state_data))) - 1) * 100
        else:
            cagr = 0
        growth_rates.loc[state, 'CAGR'] = cagr
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("State-wise Growth Rates")
        growth_rates = growth_rates.sort_values("CAGR", ascending=False)
        st.dataframe(growth_rates.style.format({
            "CAGR": "{:+.2f}%"
        }).background_gradient(cmap="RdYlGn"))
    
    with col2:
        # Scatter plot of Revenue vs Growth
        fig = go.Figure(data=go.Scatter(
            x=state_revenue["final_amount_inr"],
            y=growth_rates["CAGR"],
            mode='markers+text',
            text=growth_rates.index,
            textposition="top center",
            marker=dict(
                size=state_revenue["customer_id"] / state_revenue["customer_id"].max() * 50,
                color=state_revenue["revenue_per_customer"],
                colorscale='Viridis',
                showscale=True
            )
        ))
        fig.update_layout(
            title='Revenue vs Growth Rate (bubble size = customer base)',
            xaxis_title='Revenue (₹)',
            yaxis_title='CAGR (%)',
            showlegend=False
        )
        st.plotly_chart(fig)
    
    # 4. Subcategory Performance by State
    st.subheader("Subcategory Performance by State")
    
    # Get top 5 states by revenue
    top_5_states = state_revenue.nlargest(5, "final_amount_inr").index
    
    # Calculate subcategory performance for top 5 states
    state_subcategory_data = q8_orders[q8_orders["customer_state"].isin(top_5_states)].pivot_table(
        index="customer_state",
        columns="subcategory",
        values="final_amount_inr",
        aggfunc="sum"
    ).fillna(0)
    
    # Calculate percentage distribution
    state_subcategory_pct = state_subcategory_data.div(state_subcategory_data.sum(axis=1), axis=0) * 100
    
    # Plot heatmap using plotly
    fig = go.Figure(data=go.Heatmap(
        z=state_subcategory_pct.values,
        x=state_subcategory_pct.columns,
        y=state_subcategory_pct.index,
        text=state_subcategory_pct.values.round(1),
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        colorscale="Viridis"
    ))
    fig.update_layout(
        title='Subcategory Distribution in Top 5 States (%)',
        xaxis_title='Subcategory',
        yaxis_title='State'
    )
    st.plotly_chart(fig)
    
    st.info("""
    This Geographic Revenue Dashboard provides insights into:
    1. Overall geographic performance metrics
    2. State-wise revenue distribution and performance metrics
    3. Geographic growth analysis with CAGR calculations
    4. Category performance distribution in top states
    
    Use the filters to analyze specific time periods, states, and categories.
    The bubble chart helps identify high-growth markets, while the heatmap shows category preferences by state.
    """)

# Festival Sales Analysis Dashboard (Question 9)
with tab9:
    st.header("Festival Sales Analysis Dashboard")
    
    # Define major Indian festivals and their approximate dates
    festivals = {
        "Diwali": {"month": 10, "day_range": (15, 30)},  # October
        "Dussehra": {"month": 10, "day_range": (1, 15)},  # October
        "Holi": {"month": 3, "day_range": (1, 15)},      # March
        "Raksha Bandhan": {"month": 8, "day_range": (1, 15)},  # August
        "Christmas": {"month": 12, "day_range": (15, 31)},  # December
    }
    
    # Create festival period column
    def get_festival_period(date):
        for festival, timing in festivals.items():
            if (date.month == timing["month"] and 
                timing["day_range"][0] <= date.day <= timing["day_range"][1]):
                return festival
        return "Non-Festival"
    
    # Add festival period to orders
    q9_orders = orders.copy()
    q9_orders["festival_period"] = q9_orders["order_date"].apply(get_festival_period)
    
    # Filters
    with st.expander("Filters", expanded=True):
        # Year selection
        years = q9_orders["order_year"].dropna().unique()
        selected_years = st.multiselect("Select Years", sorted(years), default=sorted(years)[-3:], key="q9_year_filter")
        
        # Festival selection
        festival_list = ["Non-Festival"] + list(festivals.keys())
        selected_festivals = st.multiselect("Select Festivals", festival_list, default=festival_list, key="q9_festival_filter")
        
        # Subcategory selection
        subcategories = q9_orders["subcategory"].dropna().unique()
        selected_subcategories = st.multiselect("Select Subcategories", sorted(subcategories), default=sorted(subcategories), key="q9_subcategory_filter")
    
    # Filter data
    if selected_years:
        q9_orders = q9_orders[q9_orders["order_year"].isin(selected_years)]
    if selected_festivals:
        q9_orders = q9_orders[q9_orders["festival_period"].isin(selected_festivals)]
    if selected_subcategories:
        q9_orders = q9_orders[q9_orders["subcategory"].isin(selected_subcategories)]
    
    # 1. Overall Festival Performance Metrics
    st.subheader("Overall Festival Performance")
    
    # Calculate metrics
    festival_metrics = q9_orders.groupby("festival_period").agg({
        "final_amount_inr": ["sum", "mean"],
        "order_id": "count",
        "customer_id": "nunique"
    }).round(2)
    
    festival_metrics.columns = ["Total Revenue", "Avg Order Value", "Order Count", "Customer Count"]
    festival_metrics["Revenue per Customer"] = (festival_metrics["Total Revenue"] / festival_metrics["Customer Count"]).round(2)
    
    # Display metrics in columns
    metrics_cols = st.columns(len(festival_metrics))
    for col, (festival, data) in zip(metrics_cols, festival_metrics.iterrows()):
        with col:
            st.metric(
                label=festival,
                value=f"₹{data['Total Revenue']:,.0f}",
                delta=f"{data['Order Count']:,.0f} orders"
            )
    
    # 2. Festival Revenue Trends
    st.subheader("Festival Revenue Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Year-over-year festival revenue
        yearly_festival_revenue = q9_orders.pivot_table(
            index="festival_period",
            columns="order_year",
            values="final_amount_inr",
            aggfunc="sum"
        ).fillna(0)
        
        # Calculate YoY growth
        yoy_growth = yearly_festival_revenue.pct_change(axis=1) * 100
        
        st.write("Festival Revenue Year-over-Year Growth (%)")
        st.dataframe(yoy_growth.style.format("{:+.2f}%").background_gradient(cmap="RdYlGn"))
    
    with col2:
        # Festival revenue distribution plot
        fig = go.Figure()
        for year in sorted(q9_orders["order_year"].unique()):
            year_data = q9_orders[q9_orders["order_year"] == year]
            fig.add_trace(go.Bar(
                name=str(year),
                x=year_data.groupby("festival_period")["final_amount_inr"].sum().index,
                y=year_data.groupby("festival_period")["final_amount_inr"].sum().values,
                text=year_data.groupby("festival_period")["final_amount_inr"].sum().round(0),
                texttemplate='₹%{text:,.0f}',
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Festival Revenue by Year',
            xaxis_title='Festival Period',
            yaxis_title='Revenue (₹)',
            barmode='group'
        )
        st.plotly_chart(fig)
    
    # 3. Subcategory Performance During Festivals
    st.subheader("Subcategory Performance During Festivals")
    
    # Calculate subcategory performance by festival
    festival_subcategory_data = q9_orders.pivot_table(
        index="festival_period",
        columns="subcategory",
        values="final_amount_inr",
        aggfunc="sum"
    ).fillna(0)
    
    # Calculate percentage distribution
    festival_subcategory_pct = festival_subcategory_data.div(festival_subcategory_data.sum(axis=1), axis=0) * 100
    
    # Plot heatmap
    fig = go.Figure(data=go.Heatmap(
        z=festival_subcategory_pct.values,
        x=festival_subcategory_pct.columns,
        y=festival_subcategory_pct.index,
        text=festival_subcategory_pct.values.round(1),
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        colorscale="Viridis"
    ))
    fig.update_layout(
        title='Subcategory Distribution by Festival Period (%)',
        xaxis_title='Subcategory',
        yaxis_title='Festival Period'
    )
    st.plotly_chart(fig)
    
    # 4. Customer Behavior Analysis
    st.subheader("Customer Behavior During Festivals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average order value comparison
        aov_comparison = q9_orders.groupby("festival_period")["final_amount_inr"].mean().round(2)
        fig = go.Figure(data=go.Bar(
            x=aov_comparison.index,
            y=aov_comparison.values,
            text=aov_comparison.values,
            texttemplate='₹%{text:,.0f}',
            textposition='auto'
        ))
        fig.update_layout(
            title='Average Order Value by Festival Period',
            xaxis_title='Festival Period',
            yaxis_title='Average Order Value (₹)',
            showlegend=False
        )
        st.plotly_chart(fig)
    
    with col2:
        # Customer retention analysis
        customer_festival_counts = q9_orders.groupby("customer_id")["festival_period"].nunique()
        festival_customer_segments = pd.cut(customer_festival_counts, 
                                          bins=[-np.inf, 1, 2, 3, np.inf],
                                          labels=["Single Festival", "Two Festivals", "Three Festivals", "All Festivals"])
        
        festival_customer_dist = festival_customer_segments.value_counts()
        fig = go.Figure(data=go.Pie(
            labels=festival_customer_dist.index,
            values=festival_customer_dist.values,
            hole=0.4
        ))
        fig.update_layout(
            title='Customer Festival Shopping Pattern'
        )
        st.plotly_chart(fig)
    
    st.info("""
    This Festival Sales Analysis Dashboard provides insights into:
    1. Overall festival period performance metrics
    2. Year-over-year festival revenue trends
    3. Category performance during different festivals
    4. Customer behavior analysis during festival periods
    
    Use the filters to analyze specific years, festivals, and categories.
    The heatmap shows category preferences during different festivals, while the customer behavior analysis reveals shopping patterns.
    """)

# Price Optimization Analysis Dashboard (Question 10)
with tab10:
    st.header("Price Optimization Analysis Dashboard")
    
    # Filters
    with st.expander("Filters", expanded=True):
        # Time period selection
        years = orders["order_year"].dropna().unique()
        selected_years = st.multiselect("Select Years", sorted(years), default=sorted(years)[-3:], key="q10_year_filter")
        
        # Subcategory selection
        subcategories = orders["subcategory"].dropna().unique()
        selected_subcategory = st.selectbox("Select Subcategory", sorted(subcategories), key="q10_subcategory_filter")
    
    # Filter data
    q10_orders = orders.copy()
    if selected_years:
        q10_orders = q10_orders[q10_orders["order_year"].isin(selected_years)]
    if selected_subcategory:
        q10_orders = q10_orders[q10_orders["subcategory"] == selected_subcategory]
    
    # 1. Price Range Analysis
    st.subheader("Price Range Analysis")
    
    # Calculate price ranges and metrics
    price_stats = q10_orders["final_amount_inr"].describe()
    price_ranges = pd.qcut(q10_orders["final_amount_inr"], q=5, duplicates="drop")
    price_range_metrics = q10_orders.groupby(price_ranges).agg({
        "final_amount_inr": ["count", "sum", "mean"],
        "order_id": "nunique",
        "customer_id": "nunique"
    }).round(2)
    
    price_range_metrics.columns = ["Order Count", "Total Revenue", "Avg Price", "Unique Orders", "Unique Customers"]
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Minimum Price", f"₹{price_stats['min']:,.2f}")
    col2.metric("Average Price", f"₹{price_stats['mean']:,.2f}")
    col3.metric("Median Price", f"₹{price_stats['50%']:,.2f}")
    col4.metric("Maximum Price", f"₹{price_stats['max']:,.2f}")
    
    # Price range distribution
    fig = go.Figure(data=[
        go.Bar(
            x=[str(x) for x in price_range_metrics.index],
            y=price_range_metrics["Order Count"],
            text=price_range_metrics["Order Count"],
            textposition='auto',
        )
    ])
    fig.update_layout(
        title='Order Distribution by Price Range',
        xaxis_title='Price Range',
        yaxis_title='Number of Orders',
        showlegend=False
    )
    st.plotly_chart(fig)
    
    # 2. Price Elasticity Analysis
    st.subheader("Price Elasticity Analysis")
    
    # Convert order_date to datetime and set as index
    q10_orders["order_date"] = pd.to_datetime(q10_orders["order_date"])
    
    # Calculate price elasticity
    price_data = q10_orders.set_index("order_date").resample("M").agg({
        "final_amount_inr": "mean",
        "order_id": "count"
    }).reset_index()
    
    # Calculate percentage changes
    price_data["price_pct_change"] = price_data["final_amount_inr"].pct_change()
    price_data["demand_pct_change"] = price_data["order_id"].pct_change()
    
    # Calculate elasticity
    price_data["elasticity"] = (price_data["demand_pct_change"] / price_data["price_pct_change"]).replace([np.inf, -np.inf], np.nan)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price vs Demand Scatter Plot
        # Create a time-based color index (0 to 1)
        min_date = price_data["order_date"].min()
        max_date = price_data["order_date"].max()
        color_values = [(date - min_date) / (max_date - min_date) 
                       for date in price_data["order_date"]]
        
        fig = go.Figure(data=go.Scatter(
            x=price_data["final_amount_inr"],
            y=price_data["order_id"],
            mode='markers',
            marker=dict(
                size=8,
                color=color_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Time Progress",
                    ticktext=["Earlier", "Later"],
                    tickvals=[0, 1]
                )
            )
        ))
        fig.update_layout(
            title='Price vs Demand Relationship',
            xaxis_title='Average Price (₹)',
            yaxis_title='Number of Orders'
        )
        st.plotly_chart(fig)
    
    with col2:
        # Price Elasticity Over Time
        fig = go.Figure(data=go.Scatter(
            x=price_data["order_date"],
            y=price_data["elasticity"].rolling(window=3).mean(),
            mode='lines+markers',
            name='Price Elasticity (3-month moving average)'
        ))
        fig.update_layout(
            title='Price Elasticity Over Time',
            xaxis_title='Date',
            yaxis_title='Price Elasticity',
            showlegend=True
        )
        st.plotly_chart(fig)
    
    # 3. Competitor Price Analysis
    st.subheader("Competitive Price Analysis")
    
    # Calculate competitive metrics
    comp_price_data = q10_orders.copy()
    comp_price_data["price_bucket"] = pd.qcut(comp_price_data["final_amount_inr"], q=5, labels=[
        "Very Low", "Low", "Medium", "High", "Very High"
    ])
    
    comp_metrics = comp_price_data.groupby("price_bucket").agg({
        "final_amount_inr": ["mean", "count"],
        "customer_id": "nunique",
        "order_id": "nunique"
    }).round(2)
    
    comp_metrics.columns = ["Avg Price", "Order Count", "Customer Count", "Unique Orders"]
    comp_metrics["Market Share"] = (comp_metrics["Order Count"] / comp_metrics["Order Count"].sum() * 100).round(2)
    
    # Display competitive metrics
    st.write("Price Bucket Analysis")
    st.dataframe(comp_metrics.style.format({
        "Avg Price": "₹{:,.2f}",
        "Order Count": "{:,.0f}",
        "Customer Count": "{:,.0f}",
        "Unique Orders": "{:,.0f}",
        "Market Share": "{:,.2f}%"
    }).background_gradient(subset=["Market Share"], cmap="Blues"))
    
    # 4. Optimal Price Recommendation
    st.subheader("Price Optimization Recommendations")
    
    # Calculate optimal price range
    optimal_price_data = comp_metrics[comp_metrics["Market Share"] == comp_metrics["Market Share"].max()]
    optimal_price = optimal_price_data["Avg Price"].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric(
        "Recommended Price Range",
        f"₹{optimal_price:,.2f}",
        f"{optimal_price_data.index[0]}"
    )
    
    col2.metric(
        "Expected Market Share",
        f"{optimal_price_data['Market Share'].iloc[0]:.2f}%",
        f"{optimal_price_data['Order Count'].iloc[0]:,.0f} orders"
    )
    
    col3.metric(
        "Customer Base",
        f"{optimal_price_data['Customer Count'].iloc[0]:,.0f}",
        f"{optimal_price_data['Unique Orders'].iloc[0]:,.0f} unique orders"
    )
    
    st.info("""
    This Price Optimization Analysis Dashboard provides insights into:
    1. Price range distribution and key pricing metrics
    2. Price elasticity analysis showing demand sensitivity
    3. Competitive price analysis with market share by price bucket
    4. Data-driven price optimization recommendations
    
    Use the filters to analyze specific categories and subcategories.
    The price elasticity chart shows how demand responds to price changes over time.
    """)

# Main Tab 3: Customer Analytics (Questions 11-15)
with main_tab3:
    st.header("👥 Customer Analytics")
    st.write("Customer behavior, segmentation, and journey analysis covering Questions")
    
    # Create subtabs for Questions 11-15
    tab11, tab12, tab13, tab14, tab15 = st.tabs([
        "Customer Segmentation", 
        "Customer Journey", 
        "Prime Analytics", 
        "Customer Retention", 
        "Demographics"
    ])

# Question 11: Customer Segmentation
with tab11:
    st.header("Customer Segmentation Dashboard")
    
    # Filters
    with st.expander("Filters", expanded=True):
        # Date range selection with default to last 6 months
        max_date = pd.to_datetime(orders["order_date"]).max()
        default_start_date = max_date - pd.DateOffset(months=6)
        
        date_range = st.date_input(
            "Select Date Range",
            value=(default_start_date.date(), max_date.date()),
            min_value=pd.to_datetime(orders["order_date"]).min().date(),
            max_value=max_date.date(),
            key="q11_date_range"
        )
        
        # Customer segment selection
        segments = ["All Segments", "Champions", "Loyal", "Potential", "New Customers", "At Risk", "Lost"]
        selected_segments = st.multiselect("Select Customer Segments", segments, default=["All Segments"])
    
    # Calculate RFM metrics
    def calculate_rfm(data, end_date):
        rfm_data = data.groupby("customer_id").agg({
            "order_date": lambda x: (end_date - x.max()).days,  # Recency
            "order_id": "count",  # Frequency
            "final_amount_inr": "sum"  # Monetary
        }).reset_index()
        
        rfm_data.columns = ["customer_id", "recency", "frequency", "monetary"]
        
        # Calculate RFM scores using percentile-based ranking (1-5)
        rfm_data["r_score"] = 5 - pd.qcut(rfm_data["recency"], q=5, labels=False, duplicates='drop')
        rfm_data["f_score"] = 1 + pd.qcut(rfm_data["frequency"], q=5, labels=False, duplicates='drop')
        rfm_data["m_score"] = 1 + pd.qcut(rfm_data["monetary"], q=5, labels=False, duplicates='drop')
        
        # Fill any NaN values with median scores
        rfm_data["r_score"] = rfm_data["r_score"].fillna(3)
        rfm_data["f_score"] = rfm_data["f_score"].fillna(3)
        rfm_data["m_score"] = rfm_data["m_score"].fillna(3)
        
        # Calculate RFM Score
        rfm_data["rfm_score"] = (rfm_data["r_score"].astype(str) + 
                                rfm_data["f_score"].astype(str) + 
                                rfm_data["m_score"].astype(str))
        
        # Segment customers
        def segment_customers(row):
            r, f, m = row["r_score"], row["f_score"], row["m_score"]
            if r >= 4 and f >= 4 and m >= 4:
                return "Champions"
            elif r >= 3 and f >= 3 and m >= 3:
                return "Loyal"
            elif r >= 3 and f >= 2 and m >= 2:
                return "Potential"
            elif r <= 2 and f <= 2:
                return "At Risk"
            elif r == 1 and f == 1:
                return "Lost"
            else:
                return "New Customers"
        
        rfm_data["customer_segment"] = rfm_data.apply(segment_customers, axis=1)
        return rfm_data
    
    # Filter data
    q11_orders = orders.copy()
    q11_orders["order_date"] = pd.to_datetime(q11_orders["order_date"]).dt.date
    if len(date_range) == 2:
        q11_orders = q11_orders[
            (q11_orders["order_date"] >= date_range[0]) &
            (q11_orders["order_date"] <= date_range[1])
        ]
    
    # Calculate RFM metrics
    rfm_data = calculate_rfm(q11_orders, date_range[1])
    
    # Filter by selected segments
    if "All Segments" not in selected_segments:
        rfm_data = rfm_data[rfm_data["customer_segment"].isin(selected_segments)]
    
    # 1. RFM Overview
    st.subheader("RFM Analysis Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(rfm_data)
    avg_frequency = rfm_data["frequency"].mean()
    avg_monetary = rfm_data["monetary"].mean()
    active_customers = len(rfm_data[rfm_data["recency"] <= 90])  # Active in last 90 days
    
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Average Purchase Frequency", f"{avg_frequency:.1f}")
    col3.metric("Average Customer Value", f"₹{avg_monetary:,.0f}")
    col4.metric("Active Customers (90d)", f"{active_customers:,}")
    
    # 2. Customer Segmentation Distribution
    st.subheader("Customer Segmentation Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        segment_dist = rfm_data["customer_segment"].value_counts()
        fig = go.Figure(data=[
            go.Pie(
                labels=segment_dist.index,
                values=segment_dist.values,
                hole=0.4
            )
        ])
        fig.update_layout(title="Customer Segment Distribution")
        st.plotly_chart(fig)
    
    with col2:
        segment_metrics = rfm_data.groupby("customer_segment").agg({
            "monetary": ["mean", "sum"],
            "frequency": "mean",
            "recency": "mean"
        }).round(2)
        
        segment_metrics.columns = ["Avg Value", "Total Value", "Avg Frequency", "Avg Recency"]
        st.write("Segment-wise Metrics")
        st.dataframe(segment_metrics.style.format({
            "Avg Value": "₹{:,.0f}",
            "Total Value": "₹{:,.0f}",
            "Avg Frequency": "{:.1f}",
            "Avg Recency": "{:.0f} days"
        }))
    
    # 3. Behavioral Analysis
    st.subheader("Customer Behavioral Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RFM Score Distribution
        fig = go.Figure()
        
        for score in ["r_score", "f_score", "m_score"]:
            score_dist = rfm_data[score].value_counts().sort_index()
            fig.add_trace(go.Bar(
                name=score.upper(),
                x=score_dist.index,
                y=score_dist.values,
                text=score_dist.values,
                textposition='auto',
            ))
        
        fig.update_layout(
            title="RFM Score Distribution",
            xaxis_title="Score",
            yaxis_title="Number of Customers",
            barmode='group'
        )
        st.plotly_chart(fig)
    
    with col2:
        # Scatter plot of Frequency vs Monetary
        fig = go.Figure(data=go.Scatter(
            x=rfm_data["frequency"],
            y=rfm_data["monetary"],
            mode='markers',
            marker=dict(
                size=8,
                color=rfm_data["recency"],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Recency (days)")
            ),
            text=rfm_data["customer_segment"]
        ))
        fig.update_layout(
            title='Customer Behavior Pattern',
            xaxis_title='Purchase Frequency',
            yaxis_title='Total Spend (₹)'
        )
        st.plotly_chart(fig)
    
    # 4. Customer Lifecycle Value Analysis
    st.subheader("Customer Lifecycle Value Analysis")
    
    lifecycle_data = q11_orders.merge(rfm_data[["customer_id", "customer_segment"]], on="customer_id")
    
    # Ensure order_date is datetime
    lifecycle_data["order_date"] = pd.to_datetime(lifecycle_data["order_date"])
    
    # Calculate days since first purchase for each customer
    lifecycle_data["days_since_first"] = lifecycle_data.groupby("customer_id").apply(
        lambda x: (x["order_date"] - x["order_date"].min()).dt.days
    ).reset_index(level=0, drop=True)
    
    # Create tenure buckets
    lifecycle_data["tenure_bucket"] = pd.cut(
        lifecycle_data["days_since_first"],
        bins=[0, 30, 90, 180, 365, float("inf")],
        labels=["0-30d", "31-90d", "91-180d", "181-365d", "365d+"]
    )
    
    # Calculate lifecycle metrics
    lifecycle_metrics = lifecycle_data.groupby(["customer_segment", "tenure_bucket"]).agg({
        "final_amount_inr": ["mean", "sum"],
        "order_id": "count"
    }).round(2)
    
    lifecycle_metrics.columns = ["Avg Order Value", "Total Revenue", "Order Count"]
    lifecycle_metrics = lifecycle_metrics.reset_index()
    
    # Create two columns for visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue by Segment and Tenure
        segment_revenue = lifecycle_metrics.pivot(
            index="customer_segment",
            columns="tenure_bucket",
            values="Total Revenue"
        ).fillna(0)
        
        fig = go.Figure(data=go.Heatmap(
            z=segment_revenue.values,
            x=segment_revenue.columns,
            y=segment_revenue.index,
            text=segment_revenue.values.round(0),
            texttemplate="₹%{text:,.0f}",
            colorscale="Viridis"
        ))
        
        fig.update_layout(
            title="Revenue by Customer Segment and Tenure",
            xaxis_title="Customer Tenure",
            yaxis_title="Customer Segment"
        )
        st.plotly_chart(fig)
    
    with col2:
        # Average Order Value Trends
        aov_trends = lifecycle_metrics.pivot(
            index="customer_segment",
            columns="tenure_bucket",
            values="Avg Order Value"
        ).fillna(0)
        
        fig = go.Figure()
        
        for segment in aov_trends.index:
            fig.add_trace(go.Scatter(
                x=aov_trends.columns,
                y=aov_trends.loc[segment],
                name=segment,
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title="Average Order Value Evolution by Segment",
            xaxis_title="Customer Tenure",
            yaxis_title="Average Order Value (₹)",
            showlegend=True
        )
        st.plotly_chart(fig)
    
    # Display detailed metrics
    st.write("Detailed Lifecycle Metrics")
    st.dataframe(lifecycle_metrics.style.format({
        "Total Revenue": "₹{:,.0f}",
        "Avg Order Value": "₹{:,.0f}",
        "Order Count": "{:,}"
    }))
    
    # 5. Marketing Recommendations
    st.subheader("Targeted Marketing Recommendations")
    
    for segment in rfm_data["customer_segment"].unique():
        segment_data = rfm_data[rfm_data["customer_segment"] == segment]
        
        with st.expander(f"{segment} ({len(segment_data):,} customers)"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Segment Characteristics:**")
                st.write(f"- Average Order Value: ₹{segment_data['monetary'].mean():,.0f}")
                st.write(f"- Purchase Frequency: {segment_data['frequency'].mean():.1f} orders")
                st.write(f"- Average Recency: {segment_data['recency'].mean():.0f} days")
            
            with col2:
                st.write("**Recommended Actions:**")
                if segment == "Champions":
                    st.write("- Exclusive VIP rewards and early access")
                    st.write("- Personal shopping assistance")
                    st.write("- Referral program incentives")
                elif segment == "Loyal":
                    st.write("- Loyalty program upgrades")
                    st.write("- Cross-sell premium products")
                    st.write("- Feedback and review requests")
                elif segment == "Potential":
                    st.write("- Personalized product recommendations")
                    st.write("- Category-specific promotions")
                    st.write("- Engagement campaigns")
                elif segment == "New Customers":
                    st.write("- Welcome series emails")
                    st.write("- First-time buyer incentives")
                    st.write("- Educational content")
                elif segment == "At Risk":
                    st.write("- Re-engagement campaigns")
                    st.write("- Special comeback offers")
                    st.write("- Satisfaction surveys")
                elif segment == "Lost":
                    st.write("- Win-back campaigns")
                    st.write("- Deep discount offers")
                    st.write("- Feedback collection")
    
    st.info("""
    This Customer Segmentation Dashboard provides insights into:
    1. RFM (Recency, Frequency, Monetary) analysis
    2. Customer segment distribution and characteristics
    3. Behavioral patterns across segments
    4. Customer lifecycle value analysis
    5. Targeted marketing recommendations
    
    Use the filters to analyze specific date ranges and customer segments.
    The behavioral analysis shows relationships between purchase frequency, value, and recency.
    """)

# Customer Journey Analytics Dashboard (Question 12)
with tab12:
    st.header("Customer Journey Analytics Dashboard")
    
    # Filters
    with st.expander("Filters", expanded=True):
        # Date range selection with default to last 6 months
        max_date = pd.to_datetime(orders["order_date"]).max()
        default_start_date = max_date - pd.DateOffset(months=6)
        
        date_range = st.date_input(
            "Select Date Range",
            value=(default_start_date.date(), max_date.date()),
            min_value=pd.to_datetime(orders["order_date"]).min().date(),
            max_value=max_date.date(),
            key="q12_date_range"
        )
        
        # Customer type filter
        customer_types = ["All", "First-time", "Returning", "Loyal"]
        selected_customer_type = st.selectbox("Customer Type", customer_types, key="q12_customer_type")
    
    # Filter data
    q12_orders = orders.copy()
    q12_orders["order_date"] = pd.to_datetime(q12_orders["order_date"])
    
    if len(date_range) == 2:
        q12_orders = q12_orders[
            (q12_orders["order_date"].dt.date >= date_range[0]) &
            (q12_orders["order_date"].dt.date <= date_range[1])
        ]
    
    # Calculate customer journey metrics
    customer_journey = q12_orders.groupby("customer_id").agg({
        "order_id": ["count", "first", "last"],
        "final_amount_inr": ["sum", "mean"],
        "order_date": ["min", "max"]
    }).reset_index()
    
    customer_journey.columns = [
        "customer_id", "total_orders", "first_order_id", "last_order_id",
        "total_spend", "avg_order_value", "first_purchase_date", "last_purchase_date"
    ]
    
    customer_journey["days_since_first"] = (customer_journey["last_purchase_date"] - 
                                          customer_journey["first_purchase_date"]).dt.days
    
    # Categorize customers
    customer_journey["customer_type"] = "First-time"
    customer_journey.loc[customer_journey["total_orders"] > 1, "customer_type"] = "Returning"
    customer_journey.loc[customer_journey["total_orders"] > 5, "customer_type"] = "Loyal"
    
    if selected_customer_type != "All":
        customer_journey = customer_journey[customer_journey["customer_type"] == selected_customer_type]
    
    # 1. Customer Acquisition Overview
    st.subheader("Customer Acquisition Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(customer_journey)
    avg_orders = customer_journey["total_orders"].mean()
    avg_ltv = customer_journey["total_spend"].mean()
    avg_tenure = customer_journey["days_since_first"].mean()
    
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Avg Orders/Customer", f"{avg_orders:.1f}")
    col3.metric("Avg Customer LTV", f"₹{avg_ltv:,.0f}")
    col4.metric("Avg Customer Tenure", f"{avg_tenure:.0f} days")
    
    # 2. Customer Journey Timeline
    st.subheader("Customer Journey Timeline")
    
    # Calculate acquisition by month
    monthly_acquisition = pd.DataFrame({
        "date": pd.date_range(
            start=customer_journey["first_purchase_date"].min(),
            end=customer_journey["first_purchase_date"].max(),
            freq="M"
        )
    })
    
    acquisition_data = customer_journey.groupby(
        pd.Grouper(key="first_purchase_date", freq="M")
    ).size().reset_index()
    acquisition_data.columns = ["date", "new_customers"]
    
    monthly_acquisition = monthly_acquisition.merge(
        acquisition_data, on="date", how="left"
    ).fillna(0)
    
    # Plot customer acquisition trend
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monthly_acquisition["date"],
        y=monthly_acquisition["new_customers"].cumsum(),
        name="Cumulative Customers",
        mode="lines"
    ))
    
    fig.add_trace(go.Bar(
        x=monthly_acquisition["date"],
        y=monthly_acquisition["new_customers"],
        name="New Customers"
    ))
    
    fig.update_layout(
        title="Customer Acquisition Timeline",
        xaxis_title="Date",
        yaxis_title="Number of Customers",
        showlegend=True
    )
    st.plotly_chart(fig)
    
    # 3. Purchase Pattern Analysis
    st.subheader("Purchase Pattern Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Time between orders
        order_intervals = q12_orders.groupby("customer_id")["order_date"].apply(
            lambda x: x.sort_values().diff().dt.days
        ).dropna()
        
        fig = go.Figure(data=[
            go.Histogram(
                x=order_intervals[order_intervals <= order_intervals.quantile(0.95)],
                nbinsx=30
            )
        ])
        fig.update_layout(
            title="Time Between Orders Distribution",
            xaxis_title="Days Between Orders",
            yaxis_title="Count"
        )
        st.plotly_chart(fig)
    
    with col2:
        # Order value progression
        customer_order_values = q12_orders.groupby(
            ["customer_id", "order_id"]
        )["final_amount_inr"].sum().reset_index()
        
        customer_order_values["order_sequence"] = customer_order_values.groupby(
            "customer_id"
        )["order_id"].cumcount() + 1
        
        order_value_progression = customer_order_values.groupby(
            "order_sequence"
        )["final_amount_inr"].agg(["mean", "count"]).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=order_value_progression["order_sequence"],
            y=order_value_progression["mean"],
            mode="lines+markers",
            name="Average Order Value",
            yaxis="y"
        ))
        
        fig.add_trace(go.Bar(
            x=order_value_progression["order_sequence"],
            y=order_value_progression["count"],
            name="Number of Orders",
            yaxis="y2",
            opacity=0.3
        ))
        
        fig.update_layout(
            title="Order Value Progression",
            xaxis_title="Order Sequence",
            yaxis_title="Average Order Value (₹)",
            yaxis2=dict(
                title="Number of Orders",
                overlaying="y",
                side="right"
            ),
            showlegend=True
        )
        st.plotly_chart(fig)
    
    # 4. Subcategory Transition Analysis
    st.subheader("Subcategory Transition Analysis")
    
    # Calculate subcategory transitions
    transitions = []
    for _, customer_data in q12_orders.sort_values(["customer_id", "order_date"]).groupby("customer_id"):
        if len(customer_data) > 1:
            for i in range(len(customer_data) - 1):
                transitions.append((
                    customer_data.iloc[i]["subcategory"],
                    customer_data.iloc[i + 1]["subcategory"]
                ))
    
    if transitions:
        transitions_df = pd.DataFrame(transitions, columns=["from_subcategory", "to_subcategory"])
        transition_matrix = pd.crosstab(
            transitions_df["from_subcategory"],
            transitions_df["to_subcategory"],
            normalize="index"
        ) * 100
        
        fig = go.Figure(data=go.Heatmap(
            z=transition_matrix.values,
            x=transition_matrix.columns,
            y=transition_matrix.index,
            text=transition_matrix.values.round(1),
            texttemplate="%{text:.1f}%",
            colorscale="Viridis"
        ))
        
        fig.update_layout(
            title="Subcategory Transition Patterns",
            xaxis_title="Next Subcategory",
            yaxis_title="Current Subcategory"
        )
    else:
        fig = go.Figure()
        fig.add_annotation(
            text="No transition data available for selected period",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False
        )
    
    fig.update_layout(
        title="Subcategory Transition Patterns",
        xaxis_title="Next Subcategory",
        yaxis_title="Current Subcategory"
    )
    st.plotly_chart(fig)
    
    # 5. Customer Evolution Analysis
    st.subheader("Customer Evolution Analysis")
    
    # Calculate customer evolution stages
    def get_customer_stage(row):
        if row["total_orders"] == 1:
            return "First-time"
        elif row["total_orders"] <= 3:
            return "Early Stage"
        elif row["total_orders"] <= 5:
            return "Regular"
        else:
            return "Loyal"
    
    customer_journey["customer_stage"] = customer_journey.apply(get_customer_stage, axis=1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer stage distribution
        stage_dist = customer_journey["customer_stage"].value_counts()
        fig = go.Figure(data=[
            go.Pie(
                labels=stage_dist.index,
                values=stage_dist.values,
                hole=0.4
            )
        ])
        fig.update_layout(title="Customer Stage Distribution")
        st.plotly_chart(fig)
    
    with col2:
        # Stage metrics
        stage_metrics = customer_journey.groupby("customer_stage").agg({
            "total_spend": "mean",
            "avg_order_value": "mean",
            "days_since_first": "mean",
            "customer_id": "count"
        }).round(2)
        
        stage_metrics.columns = ["Avg LTV", "Avg Order Value", "Avg Tenure", "Customer Count"]
        st.write("Stage-wise Metrics")
        st.dataframe(stage_metrics.style.format({
            "Avg LTV": "₹{:,.0f}",
            "Avg Order Value": "₹{:,.0f}",
            "Avg Tenure": "{:.0f} days",
            "Customer Count": "{:,.0f}"
        }))
    
    st.info("""
    This Customer Journey Analytics Dashboard provides insights into:
    1. Customer acquisition trends and patterns
    2. Purchase behavior analysis
    3. Subcategory transition patterns
    4. Customer evolution stages
    5. Key metrics across customer lifecycle
    
    Use the filters to analyze specific time periods and customer types.
    The transition analysis shows how customers move between subcategories.
    """)

# Prime Membership Analytics Dashboard (Question 13)
with tab13:
    st.header("Prime Membership Analytics Dashboard")
    
    # Filters
    with st.expander("Filters", expanded=True):
        # Date range selection with default to last 6 months
        max_date = pd.to_datetime(orders["order_date"]).max()
        default_start_date = max_date - pd.DateOffset(months=6)
        
        date_range = st.date_input(
            "Select Date Range",
            value=(default_start_date.date(), max_date.date()),
            min_value=pd.to_datetime(orders["order_date"]).min().date(),
            max_value=max_date.date(),
            key="q13_date_range"
        )
        
        # Subcategory selection
        subcategories = orders["subcategory"].dropna().unique()
        selected_subcategories = st.multiselect(
            "Select Subcategories",
            sorted(subcategories),
            default=sorted(subcategories)[:5],
            key="q13_subcategory_filter"
        )
    
    # Filter data
    q13_orders = orders.copy()
    q13_orders["order_date"] = pd.to_datetime(q13_orders["order_date"])
    
    if len(date_range) == 2:
        q13_orders = q13_orders[
            (q13_orders["order_date"].dt.date >= date_range[0]) &
            (q13_orders["order_date"].dt.date <= date_range[1])
        ]
    
    if selected_subcategories:
        q13_orders = q13_orders[q13_orders["subcategory"].isin(selected_subcategories)]
    
    # Simulate Prime membership status based on customer behavior
    customer_stats = q13_orders.groupby("customer_id").agg({
        "order_id": "count",
        "final_amount_inr": "sum",
        "order_date": ["min", "max"]
    }).reset_index()
    
    customer_stats.columns = ["customer_id", "order_count", "total_spend", "first_order", "last_order"]
    customer_stats["days_active"] = (customer_stats["last_order"] - customer_stats["first_order"]).dt.days
    customer_stats["avg_order_value"] = customer_stats["total_spend"] / customer_stats["order_count"]
    
    # Define Prime customers (for simulation: frequent buyers with high average order value)
    aov_threshold = customer_stats["avg_order_value"].median()
    frequency_threshold = customer_stats["order_count"].median()
    
    customer_stats["is_prime"] = (
        (customer_stats["avg_order_value"] > aov_threshold) &
        (customer_stats["order_count"] > frequency_threshold)
    )
    
    # 1. Prime vs Non-Prime Overview
    st.subheader("Prime vs Non-Prime Performance")
    
    # Merge Prime status back to orders
    q13_orders = q13_orders.merge(
        customer_stats[["customer_id", "is_prime"]],
        on="customer_id",
        how="left"
    )
    
    # Calculate key metrics
    prime_metrics = q13_orders.groupby("is_prime").agg({
        "order_id": "count",
        "customer_id": "nunique",
        "final_amount_inr": ["sum", "mean"]
    }).round(2)
    
    prime_metrics.columns = ["Orders", "Customers", "Total Revenue", "Avg Order Value"]
    
    col1, col2, col3, col4 = st.columns(4)
    
    prime_pct = len(customer_stats[customer_stats["is_prime"]]) / len(customer_stats) * 100
    prime_revenue_pct = q13_orders[q13_orders["is_prime"]]["final_amount_inr"].sum() / q13_orders["final_amount_inr"].sum() * 100
    prime_orders_pct = q13_orders[q13_orders["is_prime"]]["order_id"].count() / len(q13_orders) * 100
    prime_aov = q13_orders[q13_orders["is_prime"]]["final_amount_inr"].mean()
    non_prime_aov = q13_orders[~q13_orders["is_prime"]]["final_amount_inr"].mean()
    
    col1.metric("Prime Members", f"{prime_pct:.1f}%")
    col2.metric("Prime Revenue Share", f"{prime_revenue_pct:.1f}%")
    col3.metric("Prime Orders Share", f"{prime_orders_pct:.1f}%")
    col4.metric("Prime AOV Premium", f"+{((prime_aov/non_prime_aov)-1)*100:.1f}%")
    
    # 2. Subcategory Preferences
    st.subheader("Subcategory Preferences by Membership")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Subcategory preference heatmap
        subcategory_pref = pd.crosstab(
            q13_orders["subcategory"],
            q13_orders["is_prime"],
            values=q13_orders["final_amount_inr"],
            aggfunc="sum",
            normalize="columns"
        ) * 100
        
        fig = go.Figure(data=go.Heatmap(
            z=subcategory_pref.values,
            x=["Non-Prime", "Prime"],
            y=subcategory_pref.index,
            text=subcategory_pref.values.round(1),
            texttemplate="%{text:.1f}%",
            colorscale="Viridis"
        ))
        
        fig.update_layout(
            title="Subcategory Revenue Distribution",
            xaxis_title="Membership Type",
            yaxis_title="Subcategory"
        )
        st.plotly_chart(fig)
    
    with col2:
        # Order value distribution
        fig = go.Figure()
        
        for is_prime in [True, False]:
            member_type = "Prime" if is_prime else "Non-Prime"
            order_values = q13_orders[q13_orders["is_prime"] == is_prime]["final_amount_inr"]
            
            fig.add_trace(go.Box(
                y=order_values,
                name=member_type,
                boxpoints="outliers"
            ))
        
        fig.update_layout(
            title="Order Value Distribution by Membership",
            yaxis_title="Order Value (₹)",
            showlegend=True
        )
        st.plotly_chart(fig)
    
    # 3. Purchase Patterns
    st.subheader("Purchase Patterns Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly order frequency
        monthly_orders = q13_orders.groupby([
            "is_prime",
            pd.Grouper(key="order_date", freq="M")
        ]).size().reset_index()
        monthly_orders.columns = ["is_prime", "month", "orders"]
        
        fig = go.Figure()
        
        for is_prime in [True, False]:
            member_data = monthly_orders[monthly_orders["is_prime"] == is_prime]
            member_type = "Prime" if is_prime else "Non-Prime"
            
            fig.add_trace(go.Scatter(
                x=member_data["month"],
                y=member_data["orders"],
                name=member_type,
                mode="lines+markers"
            ))
        
        fig.update_layout(
            title="Monthly Order Volume by Membership",
            xaxis_title="Month",
            yaxis_title="Number of Orders",
            showlegend=True
        )
        st.plotly_chart(fig)
    
    with col2:
        # Day of week analysis
        q13_orders["day_of_week"] = q13_orders["order_date"].dt.day_name()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        day_analysis = pd.crosstab(
            q13_orders["day_of_week"],
            q13_orders["is_prime"],
            values=q13_orders["order_id"],
            aggfunc="count",
            normalize="columns"
        ) * 100
        
        day_analysis = day_analysis.reindex(day_order)
        
        fig = go.Figure()
        
        for is_prime in [True, False]:
            member_type = "Prime" if is_prime else "Non-Prime"
            
            fig.add_trace(go.Scatter(
                x=day_analysis.index,
                y=day_analysis[is_prime],
                name=member_type,
                mode="lines+markers",
                line=dict(shape="spline")
            ))
        
        fig.update_layout(
            title="Order Distribution by Day of Week",
            xaxis_title="Day of Week",
            yaxis_title="Percentage of Orders",
            showlegend=True
        )
        st.plotly_chart(fig)
    
    # 4. Value Analysis
    st.subheader("Membership Value Analysis")
    
    # Calculate customer-level metrics
    customer_value = q13_orders.groupby(["customer_id", "is_prime"]).agg({
        "order_id": "count",
        "final_amount_inr": ["sum", "mean"]
    }).round(2)
    
    customer_value.columns = ["Order Count", "Total Spend", "Avg Order Value"]
    customer_value = customer_value.reset_index()
    
    # Display value metrics
    value_metrics = customer_value.groupby("is_prime").agg({
        "Order Count": ["mean", "median"],
        "Total Spend": ["mean", "median"],
        "Avg Order Value": ["mean", "median"]
    }).round(2)
    
    value_metrics.columns = [
        "Avg Orders/Customer", "Median Orders/Customer",
        "Avg Spend/Customer", "Median Spend/Customer",
        "Avg Order Value", "Median Order Value"
    ]
    
    st.write("Value Metrics by Membership Type")
    st.dataframe(value_metrics.style.format({
        "Avg Orders/Customer": "{:.1f}",
        "Median Orders/Customer": "{:.1f}",
        "Avg Spend/Customer": "₹{:,.0f}",
        "Median Spend/Customer": "₹{:,.0f}",
        "Avg Order Value": "₹{:,.0f}",
        "Median Order Value": "₹{:,.0f}"
    }))
    
    st.info("""
    This Prime Membership Analytics Dashboard provides insights into:
    1. Prime vs Non-Prime performance metrics
    2. Subcategory preferences by membership type
    3. Purchase patterns and timing analysis
    4. Membership value and customer behavior analysis
    
    Note: Prime membership status is simulated based on customer behavior patterns
    for demonstration purposes.
    """)

# Customer Retention Dashboard (Question 14)
with tab14:
    st.header("Customer Retention Dashboard")
    
    # Filters
    with st.expander("Filters", expanded=True):
        # Date range selection with default to last 6 months
        max_date = pd.to_datetime(orders["order_date"]).max()
        default_start_date = max_date - pd.DateOffset(months=6)
        
        date_range = st.date_input(
            "Select Date Range",
            value=(default_start_date.date(), max_date.date()),
            min_value=pd.to_datetime(orders["order_date"]).min().date(),
            max_value=max_date.date(),
            key="q14_date_range"
        )
        
        # Cohort type selection
        cohort_type = st.selectbox(
            "Cohort Analysis Type",
            ["Monthly", "Quarterly", "Yearly"],
            key="q14_cohort_type"
        )
    
    # Filter data
    q14_orders = orders.copy()
    q14_orders["order_date"] = pd.to_datetime(q14_orders["order_date"])
    
    if len(date_range) == 2:
        q14_orders = q14_orders[
            (q14_orders["order_date"].dt.date >= date_range[0]) &
            (q14_orders["order_date"].dt.date <= date_range[1])
        ]
    
    # 1. Retention Overview
    st.subheader("Retention Overview")
    
    # Calculate key retention metrics
    customer_orders = q14_orders.groupby("customer_id").agg({
        "order_id": "count",
        "order_date": ["min", "max"],
        "final_amount_inr": "sum"
    }).reset_index()
    
    customer_orders.columns = ["customer_id", "order_count", "first_order", "last_order", "total_spend"]
    customer_orders["days_between_orders"] = (customer_orders["last_order"] - customer_orders["first_order"]).dt.days
    
    # Calculate retention metrics
    total_customers = len(customer_orders)
    repeat_customers = len(customer_orders[customer_orders["order_count"] > 1])
    avg_order_frequency = customer_orders["order_count"].mean()
    retention_rate = (repeat_customers / total_customers) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Repeat Customers", f"{repeat_customers:,}")
    col3.metric("Retention Rate", f"{retention_rate:.1f}%")
    col4.metric("Avg Orders/Customer", f"{avg_order_frequency:.1f}")
    
    # 2. Cohort Analysis
    st.subheader("Cohort Analysis")
    
    # Prepare cohort data
    def get_cohort_period(df, cohort_type):
        if cohort_type == "Monthly":
            return df["order_date"].dt.to_period("M")
        elif cohort_type == "Quarterly":
            return df["order_date"].dt.to_period("Q")
        else:  # Yearly
            return df["order_date"].dt.to_period("Y")
    
    q14_orders["cohort"] = get_cohort_period(q14_orders, cohort_type)
    q14_orders["cohort_month"] = q14_orders["cohort"].astype(str)
    
    # Calculate customer first purchase cohort
    customer_first_purchase = q14_orders.groupby("customer_id")["cohort"].min()
    q14_orders = q14_orders.merge(
        customer_first_purchase.to_frame("first_cohort"),
        left_on="customer_id",
        right_index=True
    )
    
    # Calculate periods since first purchase
    def get_period_number(cohort_period, first_cohort):
        if cohort_type == "Monthly":
            return ((cohort_period.year - first_cohort.year) * 12 +
                    cohort_period.month - first_cohort.month)
        elif cohort_type == "Quarterly":
            return ((cohort_period.year - first_cohort.year) * 4 +
                    cohort_period.quarter - first_cohort.quarter)
        else:  # Yearly
            return cohort_period.year - first_cohort.year
    
    q14_orders["period_number"] = q14_orders.apply(
        lambda x: get_period_number(x["cohort"], x["first_cohort"]),
        axis=1
    )
    
    # Create cohort table
    cohort_data = q14_orders.groupby(["first_cohort", "period_number"])["customer_id"].nunique().reset_index()
    cohort_table = cohort_data.pivot(index="first_cohort", columns="period_number", values="customer_id")
    
    # Calculate retention rates
    retention_rates = cohort_table.div(cohort_table[0], axis=0) * 100
    
    # Plot cohort heatmap
    fig = go.Figure(data=go.Heatmap(
        z=retention_rates.values,
        x=retention_rates.columns,
        y=[str(x) for x in retention_rates.index],
        text=retention_rates.values.round(1),
        texttemplate="%{text:.1f}%",
        colorscale="Viridis"
    ))
    
    fig.update_layout(
        title=f"Customer Retention by {cohort_type} Cohort",
        xaxis_title="Periods Since First Purchase",
        yaxis_title="Cohort Period"
    )
    
    st.plotly_chart(fig)
    
    # 3. Churn Analysis
    st.subheader("Churn Analysis")
    
    # Define churn (no activity in last 90 days)
    last_date = q14_orders["order_date"].max()
    customer_orders["days_since_last"] = (last_date - customer_orders["last_order"]).dt.days
    customer_orders["is_churned"] = customer_orders["days_since_last"] > 90
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn by order count
        churn_by_orders = customer_orders.groupby(
            pd.qcut(customer_orders["order_count"], q=5, duplicates='drop')
        )["is_churned"].mean() * 100
        
        fig = go.Figure(data=go.Bar(
            x=[str(x) for x in churn_by_orders.index],
            y=churn_by_orders.values,
            text=churn_by_orders.values.round(1),
            texttemplate="%{text:.1f}%",
            textposition="auto"
        ))
        
        fig.update_layout(
            title="Churn Rate by Order Frequency",
            xaxis_title="Order Count Range",
            yaxis_title="Churn Rate (%)"
        )
        st.plotly_chart(fig)
    
    with col2:
        # Churn by total spend
        churn_by_spend = customer_orders.groupby(
            pd.qcut(customer_orders["total_spend"], q=5)
        )["is_churned"].mean() * 100
        
        fig = go.Figure(data=go.Bar(
            x=[f"₹{x.left:,.0f}-{x.right:,.0f}" for x in churn_by_spend.index],
            y=churn_by_spend.values,
            text=churn_by_spend.values.round(1),
            texttemplate="%{text:.1f}%",
            textposition="auto"
        ))
        
        fig.update_layout(
            title="Churn Rate by Customer Spend",
            xaxis_title="Total Spend Range",
            yaxis_title="Churn Rate (%)"
        )
        st.plotly_chart(fig)
    
    # 4. Customer Lifecycle Value
    st.subheader("Customer Lifecycle Value")
    
    # Calculate CLV metrics
    clv_data = customer_orders.copy()
    clv_data["monthly_value"] = clv_data["total_spend"] / ((clv_data["days_between_orders"] + 1) / 30)
    
    # Segment customers by value
    clv_data["value_segment"] = pd.qcut(
        clv_data["monthly_value"],
        q=4,
        labels=["Low Value", "Medium Value", "High Value", "Premium"]
    )
    
    # Calculate segment metrics
    segment_metrics = clv_data.groupby("value_segment").agg({
        "customer_id": "count",
        "monthly_value": "mean",
        "total_spend": "mean",
        "order_count": "mean",
        "is_churned": "mean"
    }).round(2)
    
    segment_metrics.columns = [
        "Customer Count", "Monthly Value",
        "Total Spend", "Order Count",
        "Churn Rate"
    ]
    
    st.write("Customer Segment Analysis")
    st.dataframe(segment_metrics.style.format({
        "Customer Count": "{:,.0f}",
        "Monthly Value": "₹{:,.2f}",
        "Total Spend": "₹{:,.2f}",
        "Order Count": "{:.1f}",
        "Churn Rate": "{:.1%}"
    }))
    
    st.info("""
    This Customer Retention Dashboard provides insights into:
    1. Overall retention metrics and customer loyalty
    2. Cohort analysis showing customer retention patterns
    3. Churn analysis by customer segments
    4. Customer lifecycle value and segmentation
    
    Use the filters to analyze retention patterns across different time periods
    and cohort types.
    """)

# Demographics & Behavior Dashboard (Question 15)
with tab15:
    st.header("Demographics & Behavior Dashboard")
    
    # Filters
    with st.expander("Filters", expanded=True):
        # Date range selection with default to last 6 months
        max_date = pd.to_datetime(orders["order_date"]).max()
        default_start_date = max_date - pd.DateOffset(months=6)
        
        date_range = st.date_input(
            "Select Date Range",
            value=(default_start_date.date(), max_date.date()),
            min_value=pd.to_datetime(orders["order_date"]).min().date(),
            max_value=max_date.date(),
            key="q15_date_range"
        )
        
        # State selection
        states = orders["customer_state"].dropna().unique()
        selected_states = st.multiselect(
            "Select States",
            sorted(states),
            default=sorted(states)[:5],
            key="q15_state_filter"
        )
    
    # Filter data
    q15_orders = orders.copy()
    q15_orders["order_date"] = pd.to_datetime(q15_orders["order_date"])
    
    if len(date_range) == 2:
        q15_orders = q15_orders[
            (q15_orders["order_date"].dt.date >= date_range[0]) &
            (q15_orders["order_date"].dt.date <= date_range[1])
        ]
    
    if selected_states:
        q15_orders = q15_orders[q15_orders["customer_state"].isin(selected_states)]
    
    # 1. Geographic Distribution
    st.subheader("Geographic Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # State-wise distribution
        state_metrics = q15_orders.groupby("customer_state").agg({
            "customer_id": "nunique",
            "order_id": "count",
            "final_amount_inr": "sum"
        }).round(2)
        
        state_metrics.columns = ["Customers", "Orders", "Revenue"]
        state_metrics = state_metrics.sort_values("Revenue", ascending=False)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=state_metrics.index,
            y=state_metrics["Revenue"],
            name="Revenue",
            yaxis="y"
        ))
        
        fig.add_trace(go.Scatter(
            x=state_metrics.index,
            y=state_metrics["Customers"],
            name="Customers",
            yaxis="y2",
            mode="lines+markers"
        ))
        
        fig.update_layout(
            title="State-wise Performance",
            xaxis_title="State",
            yaxis_title="Revenue (₹)",
            yaxis2=dict(
                title="Number of Customers",
                overlaying="y",
                side="right"
            ),
            showlegend=True
        )
        st.plotly_chart(fig)
    
    with col2:
        # Revenue per customer by state
        state_metrics["Revenue per Customer"] = state_metrics["Revenue"] / state_metrics["Customers"]
        state_metrics["Orders per Customer"] = state_metrics["Orders"] / state_metrics["Customers"]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=state_metrics.index,
            y=state_metrics["Revenue per Customer"],
            name="Revenue/Customer"
        ))
        
        fig.update_layout(
            title="Revenue per Customer by State",
            xaxis_title="State",
            yaxis_title="Revenue per Customer (₹)",
            showlegend=True
        )
        st.plotly_chart(fig)
    
    # 2. Subcategory Preferences by Region
    st.subheader("Subcategory Preferences by Region")
    
    # Calculate subcategory preferences
    region_subcategory = pd.pivot_table(
        q15_orders,
        values="final_amount_inr",
        index="customer_state",
        columns="subcategory",
        aggfunc="sum",
        fill_value=0
    )
    
    # Convert to percentages
    region_subcategory_pct = region_subcategory.div(region_subcategory.sum(axis=1), axis=0) * 100
    
    fig = go.Figure(data=go.Heatmap(
        z=region_subcategory_pct.values,
        x=region_subcategory_pct.columns,
        y=region_subcategory_pct.index,
        text=region_subcategory_pct.values.round(1),
        texttemplate="%{text:.1f}%",
        colorscale="Viridis"
    ))
    
    fig.update_layout(
        title="Subcategory Preferences by State",
        xaxis_title="Subcategory",
        yaxis_title="State"
    )
    st.plotly_chart(fig)
    
    # 3. Purchase Timing Analysis
    st.subheader("Purchase Timing Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hour of day analysis
        q15_orders["hour"] = q15_orders["order_date"].dt.hour
        hourly_orders = q15_orders.groupby("hour")["order_id"].count()
        
        fig = go.Figure(data=go.Scatter(
            x=hourly_orders.index,
            y=hourly_orders.values,
            mode="lines+markers",
            line=dict(shape="spline")
        ))
        
        fig.update_layout(
            title="Orders by Hour of Day",
            xaxis_title="Hour",
            yaxis_title="Number of Orders"
        )
        st.plotly_chart(fig)
    
    with col2:
        # Day of week analysis
        q15_orders["day_of_week"] = q15_orders["order_date"].dt.day_name()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        daily_orders = q15_orders.groupby("day_of_week")["order_id"].count()
        daily_orders = daily_orders.reindex(day_order)
        
        fig = go.Figure(data=go.Bar(
            x=daily_orders.index,
            y=daily_orders.values,
            text=daily_orders.values,
            textposition="auto"
        ))
        
        fig.update_layout(
            title="Orders by Day of Week",
            xaxis_title="Day",
            yaxis_title="Number of Orders"
        )
        st.plotly_chart(fig)
    
    # 4. Customer Value Analysis
    st.subheader("Customer Value Analysis by Region")
    
    # Calculate customer value metrics by state
    customer_value = q15_orders.groupby(["customer_state", "customer_id"]).agg({
        "order_id": "count",
        "final_amount_inr": ["sum", "mean"]
    }).round(2)
    
    customer_value.columns = ["Orders", "Total Spend", "Avg Order Value"]
    customer_value = customer_value.reset_index()
    
    state_value = customer_value.groupby("customer_state").agg({
        "Orders": ["mean", "median"],
        "Total Spend": ["mean", "median"],
        "Avg Order Value": ["mean", "median"]
    }).round(2)
    
    state_value.columns = [
        "Avg Orders", "Median Orders",
        "Avg Spend", "Median Spend",
        "Avg Order Value", "Median Order Value"
    ]
    
    st.write("Customer Value Metrics by State")
    st.dataframe(state_value.style.format({
        "Avg Orders": "{:.1f}",
        "Median Orders": "{:.1f}",
        "Avg Spend": "₹{:,.0f}",
        "Median Spend": "₹{:,.0f}",
        "Avg Order Value": "₹{:,.0f}",
        "Median Order Value": "₹{:,.0f}"
    }))
    
    st.info("""
    This Demographics & Behavior Dashboard provides insights into:
    1. Geographic distribution of customers and revenue
    2. Regional subcategory preferences
    3. Purchase timing patterns
    4. Customer value analysis by region
    
    Use the filters to analyze specific states and time periods.
    """)

# Main Tab 4: Product Analytics (Questions 16-20)
with main_tab4:
    st.header("📦 Product Analytics")
    st.write("Product performance, inventory, and ratings analysis covering Questions")
    
    # Create subtabs for Questions 16-20
    tab16, tab17, tab18, tab19, tab20 = st.tabs([
        "Product Performance", 
        "Manufacturer Analytics", 
        "Inventory Optimization", 
        "Product Ratings", 
        "New Products"
    ])

# Question 16: Product Performance Analytics
with tab16:
    st.header("Product Performance Analytics Dashboard")
    
    # Filters
    with st.expander("Filters", expanded=True):
        # Date range selection
        max_date = pd.to_datetime(orders["order_date"]).max()
        default_start_date = max_date - pd.DateOffset(months=6)
        
        date_range = st.date_input(
            "Select Date Range",
            value=(default_start_date.date(), max_date.date()),
            min_value=pd.to_datetime(orders["order_date"]).min().date(),
            max_value=max_date.date(),
            key="q16_date_range"
        )
        
        # Category and subcategory selection
        categories = orders["category"].dropna().unique()
        selected_categories = st.multiselect(
            "Select Categories",
            sorted(categories),
            default=sorted(categories),
            key="q16_category_filter"
        )
        
        subcategories = orders["subcategory"].dropna().unique()
        selected_subcategories = st.multiselect(
            "Select Subcategories",
            sorted(subcategories),
            default=sorted(subcategories)[:5],
            key="q16_subcategory_filter"
        )
    
    # Filter data
    q16_orders = orders.copy()
    q16_orders["order_date"] = pd.to_datetime(q16_orders["order_date"])
    
    if len(date_range) == 2:
        q16_orders = q16_orders[
            (q16_orders["order_date"].dt.date >= date_range[0]) &
            (q16_orders["order_date"].dt.date <= date_range[1])
        ]
    
    if selected_categories:
        q16_orders = q16_orders[q16_orders["category"].isin(selected_categories)]
    
    if selected_subcategories:
        q16_orders = q16_orders[q16_orders["subcategory"].isin(selected_subcategories)]
    
    # 1. Overall Product Performance
    st.subheader("Overall Product Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_revenue = q16_orders["final_amount_inr"].sum()
    avg_order_value = q16_orders["final_amount_inr"].mean()
    total_products = len(q16_orders["product_id"].unique())
    total_orders = len(q16_orders["order_id"].unique())
    
    col1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
    col2.metric("Average Order Value", f"₹{avg_order_value:,.0f}")
    col3.metric("Total Products", f"{total_products:,}")
    col4.metric("Total Orders", f"{total_orders:,}")
    
    # 2. Product Performance Analysis
    st.subheader("Product Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top selling products
        product_sales = q16_orders.groupby("product_name").agg({
            "order_id": "count",
            "final_amount_inr": "sum"
        }).round(2)
        
        product_sales.columns = ["Orders", "Revenue"]
        product_sales = product_sales.sort_values("Revenue", ascending=False)
        
        st.write("Top 10 Products by Revenue")
        fig = go.Figure(data=[
            go.Bar(
                x=product_sales["Revenue"].head(10).index,
                y=product_sales["Revenue"].head(10),
                text=product_sales["Revenue"].head(10).round(0),
                texttemplate="₹%{text:,.0f}",
                textposition="auto"
            )
        ])
        
        fig.update_layout(
            xaxis_title="Product",
            yaxis_title="Revenue (₹)",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    with col2:
        # Product performance heatmap
        product_metrics = q16_orders.groupby("product_name").agg({
            "order_id": "count",
            "final_amount_inr": ["sum", "mean"]
        }).round(2)
        
        product_metrics.columns = ["Order Count", "Total Revenue", "Avg Price"]
        product_metrics = product_metrics.nlargest(15, "Total Revenue")
        
        fig = go.Figure(data=go.Heatmap(
            z=np.log1p(product_metrics.values),  # Log transform for better visualization
            x=["Order Count", "Total Revenue", "Avg Price"],
            y=product_metrics.index,
            text=product_metrics.values.round(1),
            texttemplate="%{text:.1f}",
            colorscale="Viridis"
        ))
        
        fig.update_layout(
            title="Top 15 Products Performance Metrics",
            xaxis_title="Metrics",
            yaxis_title="Product"
        )
        st.plotly_chart(fig)
    
    # 3. Category and Subcategory Analysis
    st.subheader("Category and Subcategory Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category performance
        category_performance = q16_orders.groupby("category").agg({
            "order_id": "count",
            "final_amount_inr": ["sum", "mean"],
            "product_id": "nunique"
        }).round(2)
        
        category_performance.columns = ["Orders", "Revenue", "Avg Price", "Products"]
        
        st.write("Category Performance")
        st.dataframe(category_performance.style.format({
            "Orders": "{:,.0f}",
            "Revenue": "₹{:,.0f}",
            "Avg Price": "₹{:,.0f}",
            "Products": "{:,.0f}"
        }))
        
        # Category revenue distribution
        fig = go.Figure(data=[
            go.Pie(
                labels=category_performance.index,
                values=category_performance["Revenue"],
                hole=0.4
            )
        ])
        fig.update_layout(title="Revenue Distribution by Category")
        st.plotly_chart(fig)
    
    with col2:
        # Subcategory performance
        subcategory_performance = q16_orders.groupby("subcategory").agg({
            "order_id": "count",
            "final_amount_inr": ["sum", "mean"],
            "product_id": "nunique"
        }).round(2)
        
        subcategory_performance.columns = ["Orders", "Revenue", "Avg Price", "Products"]
        subcategory_performance = subcategory_performance.sort_values("Revenue", ascending=False)
        
        st.write("Top 10 Subcategories")
        st.dataframe(subcategory_performance.head(10).style.format({
            "Orders": "{:,.0f}",
            "Revenue": "₹{:,.0f}",
            "Avg Price": "₹{:,.0f}",
            "Products": "{:,.0f}"
        }))
    
    # 4. Price Analysis
    st.subheader("Price Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution
        fig = go.Figure(data=[
            go.Histogram(
                x=q16_orders["final_amount_inr"],
                nbinsx=50,
                name="Price Distribution"
            )
        ])
        
        fig.update_layout(
            title="Product Price Distribution",
            xaxis_title="Price (₹)",
            yaxis_title="Count"
        )
        st.plotly_chart(fig)
    
    with col2:
        # Price ranges by category
        price_by_category = q16_orders.groupby("category")["final_amount_inr"].agg([
            "min", "max", "mean", "median"
        ]).round(2)
        
        st.write("Price Ranges by Category")
        st.dataframe(price_by_category.style.format({
            "min": "₹{:,.0f}",
            "max": "₹{:,.0f}",
            "mean": "₹{:,.0f}",
            "median": "₹{:,.0f}"
        }))
    
    # 5. Trend Analysis
    st.subheader("Product Trends")
    
    # Monthly trends for top products
    top_5_products = product_sales.head().index
    monthly_product_sales = q16_orders[q16_orders["product_name"].isin(top_5_products)].groupby(
        [pd.Grouper(key="order_date", freq="M"), "product_name"]
    )["final_amount_inr"].sum().reset_index()
    
    fig = go.Figure()
    
    for product in top_5_products:
        product_data = monthly_product_sales[monthly_product_sales["product_name"] == product]
        fig.add_trace(go.Scatter(
            x=product_data["order_date"],
            y=product_data["final_amount_inr"],
            name=product,
            mode="lines+markers"
        ))
    
    fig.update_layout(
        title="Monthly Revenue Trends - Top 5 Products",
        xaxis_title="Month",
        yaxis_title="Revenue (₹)",
        showlegend=True
    )
    st.plotly_chart(fig)
    
    st.info("""
    This Product Performance Analytics Dashboard provides insights into:
    1. Overall product performance metrics
    2. Detailed product-level analysis
    3. Category and subcategory performance
    4. Price distribution and analysis
    5. Product revenue trends
    
    Use the filters to analyze specific categories, subcategories, and time periods.
    """)

# Manufacturer Analytics Dashboard (Question 17)
with tab17:
    st.header("Manufacturer Analytics Dashboard")
    
    # Check available product and order attributes for manufacturer/brand info
    product_columns = products.columns
    order_columns = orders.columns
    brand_column = None
    
    # First check orders table since it contains the direct data
    if 'brand' in order_columns:
        brand_column = 'brand'
    # Then check products table for alternative columns
    elif 'manufacturer' in product_columns:
        brand_column = 'manufacturer'
    elif 'vendor' in product_columns:
        brand_column = 'vendor'
    elif 'brand' in product_columns:
        brand_column = 'brand'
    
    if brand_column is None:
        st.error("No brand, manufacturer, or vendor information available in the data.")
        st.info("Please ensure either the orders or products table includes brand-related information to use this dashboard.")
    else:
        # Filters
        with st.expander("Filters", expanded=True):
            # Date range selection
            max_date = pd.to_datetime(orders["order_date"]).max()
            default_start_date = max_date - pd.DateOffset(months=6)
            
            date_range = st.date_input(
                "Select Date Range",
                value=(default_start_date.date(), max_date.date()),
                min_value=pd.to_datetime(orders["order_date"]).min().date(),
                max_value=max_date.date(),
                key="q17_date_range"
            )
            
            # Get unique values from the appropriate table
            if brand_column in orders.columns:
                brand_values = orders[brand_column].dropna().unique()
            else:
                brand_values = products[brand_column].dropna().unique()
            
            selected_brands = st.multiselect(
                f"Select {brand_column.title()}s",
                sorted(brand_values),
                default=sorted(brand_values)[:5],
                key="q17_brand_filter"
            )
    
    # Filter data
    if brand_column:
        q17_orders = orders.copy()
        
        # If brand column is in products table but not in orders, merge it in
        if brand_column not in orders.columns and brand_column in products.columns:
            q17_orders = q17_orders.merge(
                products[["product_id", brand_column]], 
                on="product_id", 
                how="inner"  # Only keep orders that have brand information
            )
        
        # Apply brand filter if selected
        if selected_brands:
            q17_orders = q17_orders[q17_orders[brand_column].isin(selected_brands)]
            
        # Convert order_date to datetime and apply date filter
        q17_orders["order_date"] = pd.to_datetime(q17_orders["order_date"])
        
        if len(date_range) == 2:
            q17_orders = q17_orders[
                (q17_orders["order_date"].dt.date >= date_range[0]) &
                (q17_orders["order_date"].dt.date <= date_range[1])
            ]
    
    if brand_column:
        # 1. Performance Overview
        st.subheader(f"{brand_column.title()} Performance Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_revenue = q17_orders["final_amount_inr"].sum()
        total_brands = len(q17_orders[brand_column].unique())
        avg_brand_revenue = total_revenue / total_brands
        market_leader = q17_orders.groupby(brand_column)["final_amount_inr"].sum().idxmax()
    
        col1.metric(f"Total {brand_column.title()} Revenue", f"₹{total_revenue:,.0f}")
        col2.metric(f"Total Active {brand_column.title()}s", f"{total_brands}")
        col3.metric(f"Avg Revenue per {brand_column.title()}", f"₹{avg_brand_revenue:,.0f}")
        col4.metric("Market Leader", f"{market_leader}")
    
    # 2. Market Share Analysis
    st.subheader(f"{brand_column.title()} Market Share Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue share
        brand_revenue = q17_orders.groupby(brand_column)["final_amount_inr"].sum().sort_values(ascending=True)
        brand_share = (brand_revenue / brand_revenue.sum() * 100).round(2)
        
        fig = go.Figure(data=[
            go.Bar(
                x=brand_share.values,
                y=brand_share.index,
                orientation='h',
                text=brand_share.values,
                texttemplate="%{text:.1f}%",
                textposition="auto",
            )
        ])
        
        fig.update_layout(
            title=f"{brand_column.title()} Market Share (%)",
            xaxis_title="Market Share (%)",
            yaxis_title=brand_column.title(),
            showlegend=False
        )
        st.plotly_chart(fig)
    
    with col2:
        # Performance metrics
        brand_metrics = q17_orders.groupby(brand_column).agg({
            "order_id": "count",
            "final_amount_inr": ["sum", "mean"],
            "product_id": "nunique"
        }).round(2)
        
        brand_metrics.columns = ["Orders", "Revenue", "Avg Price", "Products"]
        brand_metrics = brand_metrics.sort_values("Revenue", ascending=False)
        
        st.write(f"{brand_column.title()} Performance Metrics")
        st.dataframe(brand_metrics.head(10).style.format({
            "Orders": "{:,.0f}",
            "Revenue": "₹{:,.0f}",
            "Avg Price": "₹{:,.0f}",
            "Products": "{:,.0f}"
        }))
    
    # 3. Subcategory Presence
    st.subheader(f"{brand_column.title()} Subcategory Presence")
    
    # Calculate presence across subcategories
    brand_subcategory = pd.crosstab(
        q17_orders[brand_column],
        q17_orders["subcategory"],
        values=q17_orders["final_amount_inr"],
        aggfunc="sum",
        normalize="index"
    ) * 100
    
    # Sort subcategories by total revenue for better visualization
    subcategory_totals = q17_orders.groupby("subcategory")["final_amount_inr"].sum()
    top_subcategories = subcategory_totals.nlargest(10).index
    brand_subcategory = brand_subcategory[top_subcategories]
    
    fig = go.Figure(data=go.Heatmap(
        z=brand_subcategory.values,
        x=brand_subcategory.columns,
        y=brand_subcategory.index,
        text=brand_subcategory.values.round(1),
        texttemplate="%{text:.1f}%",
        colorscale="Viridis"
    ))
    
    fig.update_layout(
        title=f"{brand_column.title()} Revenue Distribution Across Top Subcategories (%)",
        xaxis_title="Subcategory",
        yaxis_title=brand_column.title()
    )
    st.plotly_chart(fig)
    
    # 4. Performance Trends
    st.subheader(f"{brand_column.title()} Performance Trends")
    
    # Monthly revenue trends for top brands
    top_5_brands = brand_metrics.head().index
    monthly_brand_revenue = q17_orders[q17_orders[brand_column].isin(top_5_brands)].groupby(
        [pd.Grouper(key="order_date", freq="M"), brand_column]
    )["final_amount_inr"].sum().reset_index()
    
    fig = go.Figure()
    
    for brand in top_5_brands:
        brand_data = monthly_brand_revenue[monthly_brand_revenue[brand_column] == brand]
        fig.add_trace(go.Scatter(
            x=brand_data["order_date"],
            y=brand_data["final_amount_inr"],
            name=brand,
            mode="lines+markers"
        ))
    
    fig.update_layout(
        title=f"Monthly Revenue Trends - Top 5 {brand_column.title()}s",
        xaxis_title="Month",
        yaxis_title="Revenue (₹)",
        showlegend=True
    )
    st.plotly_chart(fig)
    
    # 5. Brand Customer Base Analysis
    st.subheader("Brand Customer Base Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer loyalty by brand
        brand_customers = q17_orders.groupby("brand")["customer_id"].nunique()
        brand_orders = q17_orders.groupby("brand")["order_id"].count()
        brand_loyalty = (brand_orders / brand_customers).round(2)
        
        fig = go.Figure(data=[
            go.Bar(
                x=brand_loyalty.sort_values(ascending=True).index,
                y=brand_loyalty.sort_values(ascending=True).values,
                text=brand_loyalty.sort_values(ascending=True).values,
                texttemplate="%{text:.2f}",
                textposition="auto"
            )
        ])
        
        fig.update_layout(
            title="Orders per Customer by Brand",
            xaxis_title="Brand",
            yaxis_title="Orders per Customer",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    with col2:
        # Customer value by brand
        brand_customer_value = q17_orders.groupby("brand").agg({
            "customer_id": "nunique",
            "final_amount_inr": "sum"
        })
        brand_customer_value["value_per_customer"] = (
            brand_customer_value["final_amount_inr"] / brand_customer_value["customer_id"]
        ).round(2)
        
        st.write("Customer Value by Brand")
        st.dataframe(brand_customer_value.sort_values("value_per_customer", ascending=False).style.format({
            "customer_id": "{:,.0f}",
            "final_amount_inr": "₹{:,.0f}",
            "value_per_customer": "₹{:,.0f}"
        }))
    
    st.info("""
    This Brand Analytics Dashboard provides insights into:
    1. Overall brand performance metrics
    2. Brand market share analysis
    3. Category presence and distribution
    4. Revenue trends for top brands
    5. Customer base and loyalty analysis
    
    Use the filters to analyze specific brands and time periods.
    """)

# Inventory Optimization Dashboard (Question 18)
with tab18:
    st.header("Inventory Optimization Dashboard")
    
    # Filters
    with st.expander("Filters", expanded=True):
        # Date range selection
        max_date = pd.to_datetime(orders["order_date"]).max()
        default_start_date = max_date - pd.DateOffset(months=6)
        
        date_range = st.date_input(
            "Select Date Range",
            value=(default_start_date.date(), max_date.date()),
            min_value=pd.to_datetime(orders["order_date"]).min().date(),
            max_value=max_date.date(),
            key="q18_date_range"
        )
        
        # Subcategory selection
        subcategories = products["subcategory"].dropna().unique()
        selected_subcategories = st.multiselect(
            "Select Subcategories",
            sorted(subcategories),
            default=sorted(subcategories),
            key="q18_subcategory_filter"
        )
    
    # Filter data
    q18_orders = orders.copy()
    q18_orders["order_date"] = pd.to_datetime(q18_orders["order_date"])
    
    if len(date_range) == 2:
        q18_orders = q18_orders[
            (q18_orders["order_date"].dt.date >= date_range[0]) &
            (q18_orders["order_date"].dt.date <= date_range[1])
        ]
    
    if selected_subcategories:
        q18_orders = q18_orders[q18_orders["subcategory"].isin(selected_subcategories)]
    
    # 1. Inventory Overview
    st.subheader("Inventory Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_products = len(q18_orders["product_id"].unique())
    total_orders = len(q18_orders)
    avg_daily_orders = total_orders / len(q18_orders["order_date"].dt.date.unique())
    active_subcategories = len(q18_orders["subcategory"].unique())
    
    col1.metric("Total Products", f"{total_products:,}")
    col2.metric("Total Orders", f"{total_orders:,}")
    col3.metric("Avg Daily Orders", f"{avg_daily_orders:.1f}")
    col4.metric("Active Subcategories", f"{active_subcategories}")
    
    # 2. Product Movement Analysis
    st.subheader("Product Movement Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily order volume
        daily_orders = q18_orders.groupby(
            q18_orders["order_date"].dt.date
        )["order_id"].count().reset_index()
        
        fig = go.Figure(data=[
            go.Scatter(
                x=daily_orders["order_date"],
                y=daily_orders["order_id"],
                mode="lines",
                name="Daily Orders"
            )
        ])
        
        # Add moving average
        daily_orders["MA7"] = daily_orders["order_id"].rolling(window=7).mean()
        fig.add_trace(go.Scatter(
            x=daily_orders["order_date"],
            y=daily_orders["MA7"],
            mode="lines",
            name="7-day Moving Average",
            line=dict(dash="dash")
        ))
        
        fig.update_layout(
            title="Daily Order Volume",
            xaxis_title="Date",
            yaxis_title="Number of Orders",
            showlegend=True
        )
        st.plotly_chart(fig)
    
    with col2:
        # Product frequency distribution
        product_frequency = q18_orders["product_id"].value_counts()
        
        fig = go.Figure(data=[
            go.Histogram(
                x=product_frequency,
                nbinsx=30,
                name="Product Order Frequency"
            )
        ])
        
        fig.update_layout(
            title="Product Order Frequency Distribution",
            xaxis_title="Number of Orders",
            yaxis_title="Number of Products"
        )
        st.plotly_chart(fig)
    
    # 3. Subcategory-wise Inventory Analysis
    st.subheader("Subcategory-wise Inventory Analysis")
    
    # Calculate inventory metrics by subcategory
    subcategory_metrics = q18_orders.groupby("subcategory").agg({
        "order_id": "count",
        "product_id": "nunique",
        "final_amount_inr": "sum"
    }).round(2)
    
    subcategory_metrics["orders_per_product"] = (
        subcategory_metrics["order_id"] / subcategory_metrics["product_id"]
    ).round(2)
    
    subcategory_metrics.columns = ["Orders", "Products", "Revenue", "Orders/Product"]
    
    st.write("Subcategory Inventory Metrics")
    st.dataframe(subcategory_metrics.style.format({
        "Orders": "{:,.0f}",
        "Products": "{:,.0f}",
        "Revenue": "₹{:,.0f}",
        "Orders/Product": "{:.2f}"
    }))
    
    # 4. Inventory Turnover Analysis
    st.subheader("Inventory Turnover Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Product turnover by subcategory
        turnover_data = subcategory_metrics.sort_values("Orders/Product", ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(
                x=turnover_data.index,
                y=turnover_data["Orders/Product"],
                text=turnover_data["Orders/Product"],
                texttemplate="%{text:.2f}",
                textposition="auto"
            )
        ])
        
        fig.update_layout(
            title="Product Turnover Rate by Subcategory",
            xaxis_title="Subcategory",
            yaxis_title="Orders per Product",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    with col2:
        # Revenue per product by subcategory
        revenue_per_product = (subcategory_metrics["Revenue"] / subcategory_metrics["Products"]).round(2)
        
        fig = go.Figure(data=[
            go.Bar(
                x=revenue_per_product.sort_values(ascending=True).index,
                y=revenue_per_product.sort_values(ascending=True).values,
                text=revenue_per_product.sort_values(ascending=True).values,
                texttemplate="₹%{text:,.0f}",
                textposition="auto"
            )
        ])
        
        fig.update_layout(
            title="Revenue per Product by Subcategory",
            xaxis_title="Subcategory",
            yaxis_title="Revenue per Product (₹)",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    # 5. Stock Movement Patterns
    st.subheader("Stock Movement Patterns")
    
    # Daily order patterns
    hourly_orders = q18_orders.groupby(
        q18_orders["order_date"].dt.hour
    )["order_id"].count()
    
    daily_pattern = q18_orders.groupby(
        q18_orders["order_date"].dt.day_name()
    )["order_id"].count()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly pattern
        fig = go.Figure(data=[
            go.Scatter(
                x=hourly_orders.index,
                y=hourly_orders.values,
                mode="lines+markers",
                name="Hourly Orders"
            )
        ])
        
        fig.update_layout(
            title="Hourly Order Pattern",
            xaxis_title="Hour of Day",
            yaxis_title="Number of Orders"
        )
        st.plotly_chart(fig)
    
    with col2:
        # Daily pattern
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        daily_pattern = daily_pattern.reindex(day_order)
        
        fig = go.Figure(data=[
            go.Bar(
                x=daily_pattern.index,
                y=daily_pattern.values,
                text=daily_pattern.values,
                textposition="auto"
            )
        ])
        
        fig.update_layout(
            title="Daily Order Pattern",
            xaxis_title="Day of Week",
            yaxis_title="Number of Orders"
        )
        st.plotly_chart(fig)
    
    st.info("""
    This Inventory Optimization Dashboard provides insights into:
    1. Overall inventory metrics and product movement
    2. Product order frequency analysis
    3. Subcategory-wise inventory performance
    4. Product turnover analysis
    5. Stock movement patterns
    
    Use the filters to analyze specific categories and time periods.
    The insights can help optimize inventory levels and improve stock management.
    """)

# Product Ratings Analytics (Question 19)
with tab19:
    st.header("Product Ratings Analytics")
    
    # Filters
    with st.expander("Filters", expanded=True):
        # Date range selection
        max_date = pd.to_datetime(orders["order_date"]).max()
        default_start_date = max_date - pd.DateOffset(months=6)
        
        date_range = st.date_input(
            "Select Date Range",
            value=(default_start_date.date(), max_date.date()),
            min_value=pd.to_datetime(orders["order_date"]).min().date(),
            max_value=max_date.date(),
            key="q19_date_range"
        )
        
        # Subcategory selection
        subcategories = orders["subcategory"].unique()
        selected_subcategories = st.multiselect(
            "Select Subcategories",
            sorted(subcategories),
            default=sorted(subcategories),
            key="q19_subcategory_filter"
        )
    
    # Filter data
    q19_orders = orders.copy()
    q19_orders["order_date"] = pd.to_datetime(q19_orders["order_date"])
    
    if len(date_range) == 2:
        q19_orders = q19_orders[
            (q19_orders["order_date"].dt.date >= date_range[0]) &
            (q19_orders["order_date"].dt.date <= date_range[1])
        ]
    
    if selected_subcategories:
        q19_orders = q19_orders[q19_orders["subcategory"].isin(selected_subcategories)]
    
    # 1. Overall Rating Metrics
    st.subheader("Overall Rating Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Check if customer_rating exists in the DataFrame, then fallback to other rating columns
    if 'customer_rating' in q19_orders.columns:
        rating_column = 'customer_rating'
    elif 'star_rating' in q19_orders.columns:
        rating_column = 'star_rating'
    elif 'rating' in q19_orders.columns:
        rating_column = 'rating'
    else:
        rating_column = None
    
    if rating_column is None or rating_column not in q19_orders.columns:
        st.error("Rating data not found. Please make sure your orders table includes a 'customer_rating', 'rating', or 'star_rating' column.")
        avg_rating = 0
        total_reviews = 0
        five_star_reviews = 0
        low_ratings = 0
    else:
        avg_rating = q19_orders[rating_column].mean()
        total_reviews = len(q19_orders[q19_orders[rating_column].notna()])
        five_star_reviews = len(q19_orders[q19_orders[rating_column] == 5])
        low_ratings = len(q19_orders[q19_orders[rating_column] <= 2])
    
    col1.metric("Average Rating", f"{avg_rating:.2f}⭐")
    col2.metric("Total Reviews", f"{total_reviews:,}")
    col3.metric("5-Star Reviews", f"{five_star_reviews:,}")
    col4.metric("Low Ratings (≤2)", f"{low_ratings:,}")
    
    # 2. Rating Distribution Analysis
    st.subheader("Rating Distribution")
    
    col1, col2 = st.columns(2)
    
    if rating_column is not None and rating_column in q19_orders.columns:
        # Calculate rating distribution
        rating_dist = q19_orders[rating_column].value_counts().sort_index()
        rating_pct = (rating_dist / rating_dist.sum() * 100).round(2)
        
        with col1:
            # Rating distribution bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=rating_dist.index,
                    y=rating_dist.values,
                    text=rating_dist.values,
                    textposition="auto",
                )
            ])
            
            fig.update_layout(
                title="Rating Distribution",
                xaxis_title="Rating",
                yaxis_title="Number of Reviews",
                showlegend=False
            )
            st.plotly_chart(fig)
        
        with col2:
            # Rating percentage pie chart
            fig = go.Figure(data=[
                go.Pie(
                    labels=rating_pct.index,
                    values=rating_pct.values,
                    hole=0.4,
                    texttemplate="%{value:.1f}%"
                )
            ])
            
            fig.update_layout(
                title="Rating Distribution (%)"
            )
            st.plotly_chart(fig)
    else:
        with col1:
            st.warning("Rating data not available for distribution analysis")
        with col2:
            st.warning("Rating data not available for percentage analysis")
    
    # 3. Subcategory Rating Analysis
    st.subheader("Subcategory Rating Analysis")
    
    # Calculate subcategory metrics
    if rating_column is not None and rating_column in q19_orders.columns:
        subcategory_ratings = q19_orders.groupby("subcategory").agg({
            rating_column: ["mean", "count", "std"],
            "final_amount_inr": "sum"
        }).round(2)
    else:
        subcategory_ratings = q19_orders.groupby("subcategory").agg({
            "final_amount_inr": "sum"
        }).round(2)
        subcategory_ratings["rating_mean"] = 0
        subcategory_ratings["rating_count"] = 0
        subcategory_ratings["rating_std"] = 0
    
    subcategory_ratings.columns = ["Avg Rating", "Review Count", "Rating Std", "Revenue"]
    
    st.write("Subcategory Rating Performance")
    st.dataframe(subcategory_ratings.style.format({
        "Avg Rating": "{:.2f}",
        "Review Count": "{:,.0f}",
        "Rating Std": "{:.2f}",
        "Revenue": "₹{:,.0f}"
    }))
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average rating by subcategory
        subcategory_avg = subcategory_ratings.sort_values("Avg Rating", ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(
                x=subcategory_avg["Avg Rating"],
                y=subcategory_avg.index,
                orientation="h",
                text=subcategory_avg["Avg Rating"],
                texttemplate="%{text:.2f}⭐",
                textposition="auto",
            )
        ])
        
        fig.update_layout(
            title="Average Rating by Subcategory",
            xaxis_title="Average Rating",
            yaxis_title="Subcategory",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    with col2:
        # Rating consistency (inverse of standard deviation)
        rating_consistency = subcategory_ratings.sort_values("Rating Std", ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(
                x=rating_consistency["Rating Std"],
                y=rating_consistency.index,
                orientation="h",
                text=rating_consistency["Rating Std"],
                texttemplate="%{text:.2f}",
                textposition="auto",
            )
        ])
        
        fig.update_layout(
            title="Rating Consistency by Subcategory (Lower is Better)",
            xaxis_title="Rating Standard Deviation",
            yaxis_title="Subcategory",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    # 4. Rating Trends
    st.subheader("Rating Trends")
    
    # Monthly rating trends
    if rating_column is not None and rating_column in q19_orders.columns:
        monthly_ratings = q19_orders.groupby(q19_orders["order_date"].dt.to_period("M")).agg({
            rating_column: ["mean", "count"]
        }).reset_index()
    else:
        monthly_ratings = q19_orders.groupby(q19_orders["order_date"].dt.to_period("M")).size().reset_index()
        monthly_ratings.columns = ["Month", "count"]
        monthly_ratings["mean"] = 0
    
    monthly_ratings.columns = ["Month", "Average Rating", "Review Count"]
    monthly_ratings["Month"] = monthly_ratings["Month"].astype(str)
    
    fig = go.Figure(data=[
        go.Scatter(
            x=monthly_ratings["Month"],
            y=monthly_ratings["Average Rating"],
            mode="lines+markers",
            name="Average Rating",
            text=monthly_ratings["Review Count"],
            hovertemplate="Rating: %{y:.2f}<br>Reviews: %{text}"
        )
    ])
    
    fig.update_layout(
        title="Rating Trends Over Time",
        xaxis_title="Month",
        yaxis_title="Average Rating",
        showlegend=True
    )
    st.plotly_chart(fig)
    
    st.info("""
    This Product Ratings Analytics Dashboard provides insights into:
    1. Overall rating performance metrics
    2. Rating distribution and patterns
    3. Subcategory-wise rating analysis
    4. Rating trends over time
    
    Use the filters to analyze ratings for specific subcategories and time periods.
    """)

# New Products Analytics (Question 20)
with tab20:
    st.header("New Products Analytics")
    
    # Filters
    with st.expander("Filters", expanded=True):
        # Date range selection
        max_date = pd.to_datetime(orders["order_date"]).max()
        default_start_date = max_date - pd.DateOffset(months=6)
        
        date_range = st.date_input(
            "Select Date Range",
            value=(default_start_date.date(), max_date.date()),
            min_value=pd.to_datetime(orders["order_date"]).min().date(),
            max_value=max_date.date(),
            key="q20_date_range"
        )
        
        # Subcategory selection
        subcategories = orders["subcategory"].unique()
        selected_subcategories = st.multiselect(
            "Select Subcategories",
            sorted(subcategories),
            default=sorted(subcategories),
            key="q20_subcategory_filter"
        )
        
        # New product definition
        new_product_days = st.slider(
            "Define New Product Period (Days)",
            min_value=30,
            max_value=180,
            value=90,
            step=30,
            key="q20_days_filter"
        )
    
    # Filter data
    q20_orders = orders.copy()
    q20_orders["order_date"] = pd.to_datetime(q20_orders["order_date"])
    
    if len(date_range) == 2:
        q20_orders = q20_orders[
            (q20_orders["order_date"].dt.date >= date_range[0]) &
            (q20_orders["order_date"].dt.date <= date_range[1])
        ]
    
    if selected_subcategories:
        q20_orders = q20_orders[q20_orders["subcategory"].isin(selected_subcategories)]
    
    # Identify new products
    product_first_order = q20_orders.groupby("product_id")["order_date"].min()
    new_products = product_first_order[
        product_first_order >= (max_date - pd.Timedelta(days=new_product_days))
    ].index
    
    q20_orders["is_new_product"] = q20_orders["product_id"].isin(new_products)
    
    # 1. New Product Overview
    st.subheader("New Product Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_new_products = len(new_products)
    new_product_revenue = q20_orders[q20_orders["is_new_product"]]["final_amount_inr"].sum()
    new_product_orders = len(q20_orders[q20_orders["is_new_product"]])
    revenue_contribution = (new_product_revenue / q20_orders["final_amount_inr"].sum() * 100)
    
    col1.metric("New Products", f"{total_new_products:,}")
    col2.metric("New Product Revenue", f"₹{new_product_revenue:,.0f}")
    col3.metric("New Product Orders", f"{new_product_orders:,}")
    col4.metric("Revenue Contribution", f"{revenue_contribution:.1f}%")
    
    # 2. New Product Revenue Analysis
    st.subheader("New Product Revenue Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily revenue trend
        daily_revenue = q20_orders[q20_orders["is_new_product"]].groupby(
            q20_orders["order_date"].dt.date
        )["final_amount_inr"].sum().reset_index()
        
        fig = go.Figure(data=[
            go.Scatter(
                x=daily_revenue["order_date"],
                y=daily_revenue["final_amount_inr"],
                mode="lines",
                name="Daily Revenue"
            )
        ])
        
        # Add moving average
        daily_revenue["MA7"] = daily_revenue["final_amount_inr"].rolling(window=7).mean()
        fig.add_trace(go.Scatter(
            x=daily_revenue["order_date"],
            y=daily_revenue["MA7"],
            mode="lines",
            name="7-day Moving Average",
            line=dict(dash="dash")
        ))
        
        fig.update_layout(
            title="New Product Daily Revenue Trend",
            xaxis_title="Date",
            yaxis_title="Revenue (₹)",
            showlegend=True
        )
        st.plotly_chart(fig)
    
    with col2:
        # Revenue distribution
        new_product_dist = q20_orders[q20_orders["is_new_product"]].groupby(
            "product_id"
        )["final_amount_inr"].sum().sort_values(ascending=True)
        
        fig = go.Figure(data=[
            go.Box(
                y=new_product_dist.values,
                name="Revenue Distribution",
                boxpoints="all",
                pointpos=-1.5
            )
        ])
        
        fig.update_layout(
            title="New Product Revenue Distribution",
            yaxis_title="Revenue (₹)"
        )
        st.plotly_chart(fig)
    
    # 3. Subcategory Performance
    st.subheader("New Product Subcategory Performance")
    
    # Calculate subcategory metrics
    subcategory_metrics = q20_orders[q20_orders["is_new_product"]].groupby("subcategory").agg({
        "product_id": "nunique",
        "final_amount_inr": ["sum", "mean"],
        "order_id": "count"
    }).round(2)
    
    subcategory_metrics.columns = ["New Products", "Total Revenue", "Avg Revenue", "Orders"]
    subcategory_metrics["Orders/Product"] = (
        subcategory_metrics["Orders"] / subcategory_metrics["New Products"]
    ).round(2)
    
    st.write("New Product Performance by Subcategory")
    st.dataframe(subcategory_metrics.style.format({
        "New Products": "{:,.0f}",
        "Total Revenue": "₹{:,.0f}",
        "Avg Revenue": "₹{:,.0f}",
        "Orders": "{:,.0f}",
        "Orders/Product": "{:.2f}"
    }))
    
    col1, col2 = st.columns(2)
    
    with col1:
        # New products by subcategory
        subcategory_products = subcategory_metrics.sort_values("New Products", ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(
                x=subcategory_products["New Products"],
                y=subcategory_products.index,
                orientation="h",
                text=subcategory_products["New Products"],
                textposition="auto",
            )
        ])
        
        fig.update_layout(
            title="New Products by Subcategory",
            xaxis_title="Number of New Products",
            yaxis_title="Subcategory",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    with col2:
        # Average revenue by subcategory
        subcategory_revenue = subcategory_metrics.sort_values("Avg Revenue", ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(
                x=subcategory_revenue["Avg Revenue"],
                y=subcategory_revenue.index,
                orientation="h",
                text=subcategory_revenue["Avg Revenue"],
                texttemplate="₹%{text:,.0f}",
                textposition="auto",
            )
        ])
        
        fig.update_layout(
            title="Average Revenue per Product by Subcategory",
            xaxis_title="Average Revenue (₹)",
            yaxis_title="Subcategory",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    # 4. New Product Rating Analysis
    st.subheader("New Product Rating Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating comparison
        if 'customer_rating' in q20_orders.columns:
            rating_column = 'customer_rating'
        elif 'star_rating' in q20_orders.columns:
            rating_column = 'star_rating'
        elif 'rating' in q20_orders.columns:
            rating_column = 'rating'
        else:
            rating_column = None
            
        if rating_column is not None and rating_column in q20_orders.columns:
            old_product_rating = q20_orders[~q20_orders["is_new_product"]][rating_column].mean()
            new_product_rating = q20_orders[q20_orders["is_new_product"]][rating_column].mean()
        else:
            old_product_rating = 0
            new_product_rating = 0
            st.warning("Rating data not available for comparison")
        
        comparison_data = pd.DataFrame({
            "Product Type": ["Existing Products", "New Products"],
            "Average Rating": [old_product_rating, new_product_rating]
        })
        
        fig = go.Figure(data=[
            go.Bar(
                x=comparison_data["Product Type"],
                y=comparison_data["Average Rating"],
                text=comparison_data["Average Rating"],
                texttemplate="%{text:.2f}⭐",
                textposition="auto",
            )
        ])
        
        fig.update_layout(
            title="Rating Comparison: New vs Existing Products",
            xaxis_title="Product Type",
            yaxis_title="Average Rating",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    with col2:
        # Rating distribution for new products
        if rating_column is not None and rating_column in q20_orders.columns:
            new_product_ratings = q20_orders[q20_orders["is_new_product"]][rating_column].value_counts()
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=new_product_ratings.index,
                    values=new_product_ratings.values,
                    hole=0.4,
                    texttemplate="%{percent:.1f}%"
                )
            ])
        else:
            st.warning("Rating data not available for distribution analysis")
            fig = go.Figure()
        
        fig.update_layout(
            title="New Product Rating Distribution"
        )
        st.plotly_chart(fig)
    
    st.info("""
    This New Products Analytics Dashboard provides insights into:
    1. Overall new product performance metrics
    2. New product revenue trends and distribution
    3. Subcategory-wise new product analysis
    4. New product rating performance
    
    Use the filters to analyze specific subcategories and adjust the new product definition period.
    """)

# Main Tab 5: Operations & Logistics (Questions 21-25)
with main_tab5:
    st.header("🚚 Operations & Logistics")
    st.write("Delivery, payments, returns, and supply chain analytics covering Questions")
    
    # Create subtabs for Questions 21-25
    tab21, tab22, tab23, tab24, tab25 = st.tabs([
        "Delivery Performance", 
        "Payment Analytics", 
        "Returns & Cancellations", 
        "Customer Service", 
        "Supply Chain"
    ])

# Question 21: Delivery Performance
with tab21:
    st.header("Delivery Performance Dashboard")
    
    # Filters
    with st.expander("Filters", expanded=True):
        # Date range selection
        max_date = pd.to_datetime(orders["order_date"]).max()
        default_start_date = max_date - pd.DateOffset(months=3)
        
        date_range = st.date_input(
            "Select Date Range",
            value=(default_start_date.date(), max_date.date()),
            min_value=pd.to_datetime(orders["order_date"]).min().date(),
            max_value=max_date.date(),
            key="q21_date_range"
        )
        
        # State selection
        states = orders["customer_state"].dropna().unique()
        selected_states = st.multiselect(
            "Select States",
            sorted(states),
            default=sorted(states),
            key="q21_state_filter"
        )
        
        # Subcategory selection
        subcategories = orders["subcategory"].dropna().unique()
        selected_subcategories = st.multiselect(
            "Select Subcategories",
            sorted(subcategories),
            default=sorted(subcategories),
            key="q21_subcategory_filter"
        )
    
    # Filter data
    q21_orders = orders.copy()
    q21_orders["order_date"] = pd.to_datetime(q21_orders["order_date"])
    
    if len(date_range) == 2:
        q21_orders = q21_orders[
            (q21_orders["order_date"].dt.date >= date_range[0]) &
            (q21_orders["order_date"].dt.date <= date_range[1])
        ]
    
    if selected_states:
        q21_orders = q21_orders[q21_orders["customer_state"].isin(selected_states)]
    
    if selected_subcategories:
        q21_orders = q21_orders[q21_orders["subcategory"].isin(selected_subcategories)]
    
    # Simulate delivery data (since we don't have actual delivery dates)
    np.random.seed(42)
    q21_orders = q21_orders.copy()
    q21_orders["delivery_days"] = np.random.normal(5, 2, len(q21_orders)).clip(1, 15).round().astype(int)
    q21_orders["promised_delivery_days"] = np.random.choice([3, 5, 7], len(q21_orders))
    q21_orders["on_time"] = q21_orders["delivery_days"] <= q21_orders["promised_delivery_days"]
    q21_orders["delivery_status"] = np.random.choice(["Delivered", "In Transit", "Delayed"], len(q21_orders), p=[0.85, 0.10, 0.05])
    
    # 1. Overall Delivery Performance
    st.subheader("Overall Delivery Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    avg_delivery_time = q21_orders["delivery_days"].mean()
    on_time_rate = (q21_orders["on_time"].sum() / len(q21_orders)) * 100
    total_deliveries = len(q21_orders[q21_orders["delivery_status"] == "Delivered"])
    delayed_deliveries = len(q21_orders[q21_orders["delivery_status"] == "Delayed"])
    
    col1.metric("Average Delivery Time", f"{avg_delivery_time:.1f} days")
    col2.metric("On-Time Delivery Rate", f"{on_time_rate:.1f}%")
    col3.metric("Total Deliveries", f"{total_deliveries:,}")
    col4.metric("Delayed Deliveries", f"{delayed_deliveries:,}")
    
    # 2. Delivery Time Analysis
    st.subheader("Delivery Time Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Delivery time distribution
        fig = go.Figure(data=[
            go.Histogram(
                x=q21_orders["delivery_days"],
                nbinsx=10,
                name="Delivery Days"
            )
        ])
        
        fig.update_layout(
            title="Delivery Time Distribution",
            xaxis_title="Delivery Days",
            yaxis_title="Number of Orders",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    with col2:
        # On-time vs delayed comparison
        delivery_performance = pd.DataFrame({
            "Status": ["On Time", "Delayed"],
            "Count": [q21_orders["on_time"].sum(), (~q21_orders["on_time"]).sum()]
        })
        
        fig = go.Figure(data=[
            go.Pie(
                labels=delivery_performance["Status"],
                values=delivery_performance["Count"],
                hole=0.4,
                marker_colors=['#2E8B57', '#DC143C']
            )
        ])
        
        fig.update_layout(
            title="On-Time vs Delayed Deliveries"
        )
        st.plotly_chart(fig)
    
    # 3. Geographic Performance Analysis
    st.subheader("Geographic Performance Analysis")
    
    state_performance = q21_orders.groupby("customer_state").agg({
        "delivery_days": "mean",
        "on_time": ["sum", "count"],
        "order_id": "count"
    })
    
    # Flatten MultiIndex columns
    state_performance.columns = state_performance.columns.droplevel(0) if state_performance.columns.nlevels > 1 else state_performance.columns
    state_performance.columns = ["Avg Delivery Days", "On Time Orders", "Total Orders", "Order Count"]
    
    # Ensure numeric types and apply rounding
    state_performance["Avg Delivery Days"] = pd.to_numeric(state_performance["Avg Delivery Days"], errors='coerce').round(2)
    state_performance["On Time Orders"] = pd.to_numeric(state_performance["On Time Orders"], errors='coerce')
    state_performance["Total Orders"] = pd.to_numeric(state_performance["Total Orders"], errors='coerce')
    state_performance["Order Count"] = pd.to_numeric(state_performance["Order Count"], errors='coerce')
    
    state_performance["On Time Rate %"] = (
        state_performance["On Time Orders"] / state_performance["Total Orders"] * 100
    ).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("State-wise Delivery Performance")
        st.dataframe(state_performance.style.format({
            "Avg Delivery Days": "{:.1f}",
            "On Time Orders": "{:,.0f}",
            "Total Orders": "{:,.0f}",
            "Order Count": "{:,.0f}",
            "On Time Rate %": "{:.1f}%"
        }).background_gradient(subset=["On Time Rate %"], cmap="RdYlGn"))
    
    with col2:
        # Top 10 states by on-time rate
        top_states = state_performance.nlargest(10, "On Time Rate %")
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_states["On Time Rate %"],
                y=top_states.index,
                orientation="h",
                text=top_states["On Time Rate %"],
                texttemplate="%{text:.1f}%",
                textposition="auto"
            )
        ])
        
        fig.update_layout(
            title="Top 10 States by On-Time Delivery Rate",
            xaxis_title="On-Time Rate (%)",
            yaxis_title="State",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    # 4. Subcategory Performance Analysis
    st.subheader("Subcategory Delivery Performance")
    
    subcategory_performance = q21_orders.groupby("subcategory").agg({
        "delivery_days": "mean",
        "on_time": ["sum", "count"],
        "final_amount_inr": "sum"
    })
    
    # Flatten MultiIndex columns
    subcategory_performance.columns = subcategory_performance.columns.droplevel(0) if subcategory_performance.columns.nlevels > 1 else subcategory_performance.columns
    subcategory_performance.columns = ["Avg Delivery Days", "On Time Orders", "Total Orders", "Revenue"]
    
    # Ensure numeric types and apply rounding
    subcategory_performance["Avg Delivery Days"] = pd.to_numeric(subcategory_performance["Avg Delivery Days"], errors='coerce').round(2)
    subcategory_performance["On Time Orders"] = pd.to_numeric(subcategory_performance["On Time Orders"], errors='coerce')
    subcategory_performance["Total Orders"] = pd.to_numeric(subcategory_performance["Total Orders"], errors='coerce')
    subcategory_performance["Revenue"] = pd.to_numeric(subcategory_performance["Revenue"], errors='coerce').round(2)
    
    subcategory_performance["On Time Rate %"] = (
        subcategory_performance["On Time Orders"] / subcategory_performance["Total Orders"] * 100
    ).round(2)
    
    # Scatter plot of delivery performance vs revenue
    fig = go.Figure(data=go.Scatter(
        x=subcategory_performance["Avg Delivery Days"],
        y=subcategory_performance["On Time Rate %"],
        mode='markers+text',
        text=subcategory_performance.index,
        textposition="top center",
        marker=dict(
            size=subcategory_performance["Revenue"] / subcategory_performance["Revenue"].max() * 50,
            color=subcategory_performance["Total Orders"],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Total Orders")
        )
    ))
    
    fig.update_layout(
        title="Subcategory Performance: Delivery Days vs On-Time Rate (bubble size = revenue)",
        xaxis_title="Average Delivery Days",
        yaxis_title="On-Time Rate (%)",
        showlegend=False
    )
    st.plotly_chart(fig)
    
    st.info("""
    This Delivery Performance Dashboard provides insights into:
    1. Overall delivery performance metrics
    2. Delivery time distribution and analysis
    3. Geographic performance variations
    4. Subcategory-wise delivery performance
    
    Use the filters to analyze specific regions and subcategories.
    Note: Delivery data is simulated for demonstration purposes.
    """)

# Payment Analytics Dashboard (Question 22)
with tab22:
    st.header("Payment Analytics Dashboard")
    
    # Filters
    with st.expander("Filters", expanded=True):
        # Date range selection
        max_date = pd.to_datetime(orders["order_date"]).max()
        default_start_date = max_date - pd.DateOffset(months=6)
        
        date_range = st.date_input(
            "Select Date Range",
            value=(default_start_date.date(), max_date.date()),
            min_value=pd.to_datetime(orders["order_date"]).min().date(),
            max_value=max_date.date(),
            key="q22_date_range"
        )
        
        # Subcategory selection
        subcategories = orders["subcategory"].dropna().unique()
        selected_subcategories = st.multiselect(
            "Select Subcategories",
            sorted(subcategories),
            default=sorted(subcategories),
            key="q22_subcategory_filter"
        )
    
    # Filter data and simulate payment data
    q22_orders = orders.copy()
    q22_orders["order_date"] = pd.to_datetime(q22_orders["order_date"])
    
    if len(date_range) == 2:
        q22_orders = q22_orders[
            (q22_orders["order_date"].dt.date >= date_range[0]) &
            (q22_orders["order_date"].dt.date <= date_range[1])
        ]
    
    if selected_subcategories:
        q22_orders = q22_orders[q22_orders["subcategory"].isin(selected_subcategories)]
    
    # Simulate payment method data
    np.random.seed(42)
    payment_methods = ["Credit Card", "Debit Card", "UPI", "Net Banking", "Wallet", "COD"]
    q22_orders = q22_orders.copy()
    q22_orders["payment_method"] = np.random.choice(
        payment_methods, len(q22_orders), 
        p=[0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
    )
    q22_orders["payment_success"] = np.random.choice([True, False], len(q22_orders), p=[0.95, 0.05])
    q22_orders["transaction_fee"] = q22_orders["final_amount_inr"] * np.random.uniform(0.01, 0.03, len(q22_orders))
    
    # 1. Payment Method Overview
    st.subheader("Payment Method Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_transactions = len(q22_orders)
    success_rate = (q22_orders["payment_success"].sum() / len(q22_orders)) * 100
    total_transaction_value = q22_orders["final_amount_inr"].sum()
    avg_transaction_value = q22_orders["final_amount_inr"].mean()
    
    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Success Rate", f"{success_rate:.1f}%")
    col3.metric("Total Transaction Value", f"₹{total_transaction_value:,.0f}")
    col4.metric("Avg Transaction Value", f"₹{avg_transaction_value:,.0f}")
    
    # 2. Payment Method Preferences
    st.subheader("Payment Method Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Payment method distribution
        payment_dist = q22_orders["payment_method"].value_counts()
        
        fig = go.Figure(data=[
            go.Pie(
                labels=payment_dist.index,
                values=payment_dist.values,
                hole=0.4,
                texttemplate="%{percent:.1f}%"
            )
        ])
        
        fig.update_layout(
            title="Payment Method Distribution"
        )
        st.plotly_chart(fig)
    
    with col2:
        # Payment method revenue
        payment_revenue = q22_orders.groupby("payment_method")["final_amount_inr"].sum().sort_values(ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(
                x=payment_revenue.values,
                y=payment_revenue.index,
                orientation="h",
                text=payment_revenue.values,
                texttemplate="₹%{text:,.0f}",
                textposition="auto"
            )
        ])
        
        fig.update_layout(
            title="Revenue by Payment Method",
            xaxis_title="Revenue (₹)",
            yaxis_title="Payment Method",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    # 3. Transaction Success Analysis
    st.subheader("Transaction Success Analysis")
    
    payment_success_analysis = q22_orders.groupby("payment_method").agg({
        "payment_success": ["sum", "count"],
        "final_amount_inr": ["sum", "mean"],
        "transaction_fee": "sum"
    })
    
    # Flatten MultiIndex columns
    payment_success_analysis.columns = payment_success_analysis.columns.droplevel(0) if payment_success_analysis.columns.nlevels > 1 else payment_success_analysis.columns
    payment_success_analysis.columns = ["Successful", "Total", "Revenue", "Avg Amount", "Total Fees"]
    
    # Ensure numeric types and apply rounding
    payment_success_analysis["Successful"] = pd.to_numeric(payment_success_analysis["Successful"], errors='coerce')
    payment_success_analysis["Total"] = pd.to_numeric(payment_success_analysis["Total"], errors='coerce')
    payment_success_analysis["Revenue"] = pd.to_numeric(payment_success_analysis["Revenue"], errors='coerce').round(2)
    payment_success_analysis["Avg Amount"] = pd.to_numeric(payment_success_analysis["Avg Amount"], errors='coerce').round(2)
    payment_success_analysis["Total Fees"] = pd.to_numeric(payment_success_analysis["Total Fees"], errors='coerce').round(2)
    
    payment_success_analysis["Success Rate %"] = (
        payment_success_analysis["Successful"] / payment_success_analysis["Total"] * 100
    ).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Payment Method Performance Analysis")
        st.dataframe(payment_success_analysis.style.format({
            "Successful": "{:,.0f}",
            "Total": "{:,.0f}",
            "Revenue": "₹{:,.0f}",
            "Avg Amount": "₹{:,.0f}",
            "Total Fees": "₹{:,.0f}",
            "Success Rate %": "{:.1f}%"
        }).background_gradient(subset=["Success Rate %"], cmap="RdYlGn"))
    
    with col2:
        # Success rate comparison
        fig = go.Figure(data=[
            go.Bar(
                x=payment_success_analysis.index,
                y=payment_success_analysis["Success Rate %"],
                text=payment_success_analysis["Success Rate %"],
                texttemplate="%{text:.1f}%",
                textposition="auto"
            )
        ])
        
        fig.update_layout(
            title="Success Rate by Payment Method",
            xaxis_title="Payment Method",
            yaxis_title="Success Rate (%)",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    # 4. Payment Trends Evolution
    st.subheader("Payment Trends Evolution")
    
    # Monthly payment trends
    monthly_payments = q22_orders.groupby([
        pd.to_datetime(q22_orders["order_date"]).dt.to_period("M"),
        "payment_method"
    ])["final_amount_inr"].sum().reset_index()
    
    monthly_payments["order_date"] = monthly_payments["order_date"].astype(str)
    
    payment_trends = monthly_payments.pivot(
        index="order_date",
        columns="payment_method",
        values="final_amount_inr"
    ).fillna(0)
    
    fig = go.Figure()
    
    for method in payment_trends.columns:
        fig.add_trace(go.Scatter(
            x=payment_trends.index,
            y=payment_trends[method],
            mode='lines+markers',
            name=method
        ))
    
    fig.update_layout(
        title="Payment Method Revenue Trends",
        xaxis_title="Month",
        yaxis_title="Revenue (₹)",
        showlegend=True
    )
    st.plotly_chart(fig)
    
    # 5. Subcategory Payment Preferences
    st.subheader("Subcategory Payment Preferences")
    
    subcategory_payment = q22_orders.pivot_table(
        index="subcategory",
        columns="payment_method",
        values="final_amount_inr",
        aggfunc="sum"
    ).fillna(0)
    
    # Calculate percentage distribution
    subcategory_payment_pct = subcategory_payment.div(subcategory_payment.sum(axis=1), axis=0) * 100
    
    fig = go.Figure(data=go.Heatmap(
        z=subcategory_payment_pct.values,
        x=subcategory_payment_pct.columns,
        y=subcategory_payment_pct.index,
        text=subcategory_payment_pct.values.round(1),
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        colorscale="Viridis"
    ))
    
    fig.update_layout(
        title="Payment Method Preferences by Subcategory (%)",
        xaxis_title="Payment Method",
        yaxis_title="Subcategory"
    )
    st.plotly_chart(fig)
    
    st.info("""
    This Payment Analytics Dashboard provides insights into:
    1. Overall payment method performance and preferences
    2. Transaction success rates across different methods
    3. Payment trends evolution over time
    4. Subcategory-wise payment preferences
    
    Use the filters to analyze specific subcategories and time periods.
    Note: Payment data is simulated for demonstration purposes.
    """)

# Returns & Cancellations Dashboard (Question 23)
with tab23:
    st.header("Returns & Cancellations Dashboard")
    
    # Filters
    with st.expander("Filters", expanded=True):
        # Date range selection
        max_date = pd.to_datetime(orders["order_date"]).max()
        default_start_date = max_date - pd.DateOffset(months=6)
        
        date_range = st.date_input(
            "Select Date Range",
            value=(default_start_date.date(), max_date.date()),
            min_value=pd.to_datetime(orders["order_date"]).min().date(),
            max_value=max_date.date(),
            key="q23_date_range"
        )
        
        # Subcategory selection
        subcategories = orders["subcategory"].dropna().unique()
        selected_subcategories = st.multiselect(
            "Select Subcategories",
            sorted(subcategories),
            default=sorted(subcategories),
            key="q23_subcategory_filter"
        )
    
    # Filter data and simulate returns/cancellations data
    q23_orders = orders.copy()
    q23_orders["order_date"] = pd.to_datetime(q23_orders["order_date"])
    
    if len(date_range) == 2:
        q23_orders = q23_orders[
            (q23_orders["order_date"].dt.date >= date_range[0]) &
            (q23_orders["order_date"].dt.date <= date_range[1])
        ]
    
    if selected_subcategories:
        q23_orders = q23_orders[q23_orders["subcategory"].isin(selected_subcategories)]
    
    # Simulate returns and cancellations data
    np.random.seed(42)
    q23_orders = q23_orders.copy()
    q23_orders["is_returned"] = np.random.choice([True, False], len(q23_orders), p=[0.08, 0.92])
    q23_orders["is_cancelled"] = np.random.choice([True, False], len(q23_orders), p=[0.05, 0.95])
    
    return_reasons = ["Defective Product", "Wrong Item", "Size Issue", "Quality Issue", "Changed Mind", "Delivery Issue"]
    cancel_reasons = ["Payment Failed", "Changed Mind", "Found Better Price", "Delivery Issue", "Product Unavailable"]
    
    q23_orders["return_reason"] = np.where(
        q23_orders["is_returned"],
        np.random.choice(return_reasons, len(q23_orders)),
        None
    )
    
    q23_orders["cancel_reason"] = np.where(
        q23_orders["is_cancelled"],
        np.random.choice(cancel_reasons, len(q23_orders)),
        None
    )
    
    q23_orders["return_cost"] = np.where(
        q23_orders["is_returned"],
        q23_orders["final_amount_inr"] * np.random.uniform(0.1, 0.3, len(q23_orders)),
        0
    )
    
    # 1. Overall Returns & Cancellations Metrics
    st.subheader("Overall Returns & Cancellations Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_orders = len(q23_orders)
    return_rate = (q23_orders["is_returned"].sum() / total_orders) * 100
    cancellation_rate = (q23_orders["is_cancelled"].sum() / total_orders) * 100
    total_return_cost = q23_orders["return_cost"].sum()
    
    col1.metric("Total Orders", f"{total_orders:,}")
    col2.metric("Return Rate", f"{return_rate:.1f}%")
    col3.metric("Cancellation Rate", f"{cancellation_rate:.1f}%")
    col4.metric("Return Cost Impact", f"₹{total_return_cost:,.0f}")
    
    # 2. Returns Analysis
    st.subheader("Returns Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Return reasons distribution
        return_reasons_data = q23_orders[q23_orders["is_returned"]]["return_reason"].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=return_reasons_data.index,
                y=return_reasons_data.values,
                text=return_reasons_data.values,
                textposition="auto"
            )
        ])
        
        fig.update_layout(
            title="Return Reasons Distribution",
            xaxis_title="Return Reason",
            yaxis_title="Number of Returns",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    with col2:
        # Cancellation reasons distribution
        cancel_reasons_data = q23_orders[q23_orders["is_cancelled"]]["cancel_reason"].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=cancel_reasons_data.index,
                y=cancel_reasons_data.values,
                text=cancel_reasons_data.values,
                textposition="auto"
            )
        ])
        
        fig.update_layout(
            title="Cancellation Reasons Distribution",
            xaxis_title="Cancellation Reason",
            yaxis_title="Number of Cancellations",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    # 3. Subcategory-wise Analysis
    st.subheader("Subcategory-wise Returns & Cancellations Analysis")
    
    subcategory_analysis = q23_orders.groupby("subcategory").agg({
        "order_id": "count",
        "is_returned": "sum",
        "is_cancelled": "sum",
        "return_cost": "sum",
        "final_amount_inr": "sum"
    }).round(2)
    
    subcategory_analysis.columns = ["Total Orders", "Returns", "Cancellations", "Return Cost", "Revenue"]
    subcategory_analysis["Return Rate %"] = (
        subcategory_analysis["Returns"] / subcategory_analysis["Total Orders"] * 100
    ).round(2)
    subcategory_analysis["Cancel Rate %"] = (
        subcategory_analysis["Cancellations"] / subcategory_analysis["Total Orders"] * 100
    ).round(2)
    subcategory_analysis["Cost Impact %"] = (
        subcategory_analysis["Return Cost"] / subcategory_analysis["Revenue"] * 100
    ).round(2)
    
    st.write("Subcategory Performance Analysis")
    st.dataframe(subcategory_analysis.style.format({
        "Total Orders": "{:,.0f}",
        "Returns": "{:,.0f}",
        "Cancellations": "{:,.0f}",
        "Return Cost": "₹{:,.0f}",
        "Revenue": "₹{:,.0f}",
        "Return Rate %": "{:.1f}%",
        "Cancel Rate %": "{:.1f}%",
        "Cost Impact %": "{:.1f}%"
    }).background_gradient(subset=["Return Rate %", "Cancel Rate %"], cmap="Reds_r"))
    
    # 4. Trends Analysis
    st.subheader("Returns & Cancellations Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly return trends
        monthly_returns = q23_orders.groupby(q23_orders["order_date"].dt.to_period("M")).agg({
            "order_id": "count",
            "is_returned": "sum",
            "is_cancelled": "sum"
        }).reset_index()
        
        monthly_returns["return_rate"] = (monthly_returns["is_returned"] / monthly_returns["order_id"] * 100).round(2)
        monthly_returns["cancel_rate"] = (monthly_returns["is_cancelled"] / monthly_returns["order_id"] * 100).round(2)
        monthly_returns["order_date"] = monthly_returns["order_date"].astype(str)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_returns["order_date"],
            y=monthly_returns["return_rate"],
            mode='lines+markers',
            name='Return Rate %'
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly_returns["order_date"],
            y=monthly_returns["cancel_rate"],
            mode='lines+markers',
            name='Cancellation Rate %'
        ))
        
        fig.update_layout(
            title="Monthly Return & Cancellation Trends",
            xaxis_title="Month",
            yaxis_title="Rate (%)",
            showlegend=True
        )
        st.plotly_chart(fig)
    
    with col2:
        # Cost impact analysis
        cost_impact = q23_orders.groupby("subcategory").agg({
            "return_cost": "sum",
            "final_amount_inr": "sum"
        }).round(2)
        
        cost_impact["impact_percentage"] = (cost_impact["return_cost"] / cost_impact["final_amount_inr"] * 100).round(2)
        cost_impact = cost_impact.sort_values("impact_percentage", ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(
                x=cost_impact["impact_percentage"],
                y=cost_impact.index,
                orientation="h",
                text=cost_impact["impact_percentage"],
                texttemplate="%{text:.1f}%",
                textposition="auto"
            )
        ])
        
        fig.update_layout(
            title="Return Cost Impact by Subcategory",
            xaxis_title="Cost Impact (%)",
            yaxis_title="Subcategory",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    # 5. Quality Improvement Opportunities
    st.subheader("Quality Improvement Opportunities")
    
    # Focus on high return rate subcategories
    high_return_subcategories = subcategory_analysis.nlargest(10, "Return Rate %")
    
    st.write("Top 10 Subcategories Requiring Quality Focus")
    
    opportunities_data = []
    for subcategory, data in high_return_subcategories.iterrows():
        subcategory_returns = q23_orders[
            (q23_orders["subcategory"] == subcategory) & 
            (q23_orders["is_returned"] == True)
        ]
        
        top_return_reason = subcategory_returns["return_reason"].value_counts().index[0] if len(subcategory_returns) > 0 else "N/A"
        
        opportunities_data.append({
            "Subcategory": subcategory,
            "Return Rate": f"{data['Return Rate %']:.1f}%",
            "Total Returns": int(data["Returns"]),
            "Primary Return Reason": top_return_reason,
            "Cost Impact": f"₹{data['Return Cost']:,.0f}",
            "Recommendation": f"Focus on {top_return_reason.lower()} issues"
        })
    
    opportunities_df = pd.DataFrame(opportunities_data)
    st.dataframe(opportunities_df)
    
    st.info("""
    This Returns & Cancellations Dashboard provides insights into:
    1. Overall return and cancellation rates and metrics
    2. Return and cancellation reasons analysis
    3. Subcategory-wise performance breakdown
    4. Trends and patterns over time
    5. Quality improvement opportunities and recommendations
    
    Use the filters to analyze specific subcategories and time periods.
    Note: Returns and cancellation data is simulated for demonstration purposes.
    """)

# Customer Service Dashboard (Question 24)
with tab24:
    st.header("Customer Service Dashboard")
    
    # Filters
    with st.expander("Filters", expanded=True):
        # Date range selection
        max_date = pd.to_datetime(orders["order_date"]).max()
        default_start_date = max_date - pd.DateOffset(months=3)
        
        date_range = st.date_input(
            "Select Date Range",
            value=(default_start_date.date(), max_date.date()),
            min_value=pd.to_datetime(orders["order_date"]).min().date(),
            max_value=max_date.date(),
            key="q24_date_range"
        )
        
        # Subcategory selection
        subcategories = orders["subcategory"].dropna().unique()
        selected_subcategories = st.multiselect(
            "Select Subcategories",
            sorted(subcategories),
            default=sorted(subcategories),
            key="q24_subcategory_filter"
        )
    
    # Filter data and simulate customer service data
    q24_orders = orders.copy()
    q24_orders["order_date"] = pd.to_datetime(q24_orders["order_date"])
    
    if len(date_range) == 2:
        q24_orders = q24_orders[
            (q24_orders["order_date"].dt.date >= date_range[0]) &
            (q24_orders["order_date"].dt.date <= date_range[1])
        ]
    
    if selected_subcategories:
        q24_orders = q24_orders[q24_orders["subcategory"].isin(selected_subcategories)]
    
    # Simulate customer service data
    np.random.seed(42)
    q24_orders = q24_orders.copy()
    q24_orders["has_complaint"] = np.random.choice([True, False], len(q24_orders), p=[0.12, 0.88])
    
    complaint_categories = ["Delivery Issue", "Product Quality", "Payment Issue", "Return/Refund", "Account Issue", "Technical Support"]
    resolution_times = np.random.normal(24, 12, len(q24_orders)).clip(1, 72).round().astype(int)  # hours
    satisfaction_scores = np.random.normal(4.2, 0.8, len(q24_orders)).clip(1, 5).round(1)
    
    q24_orders["complaint_category"] = np.where(
        q24_orders["has_complaint"],
        np.random.choice(complaint_categories, len(q24_orders)),
        None
    )
    
    q24_orders["resolution_time_hours"] = np.where(
        q24_orders["has_complaint"],
        resolution_times,
        None
    )
    
    q24_orders["satisfaction_score"] = satisfaction_scores
    q24_orders["is_resolved"] = np.where(
        q24_orders["has_complaint"],
        np.random.choice([True, False], len(q24_orders), p=[0.92, 0.08]),
        None
    )
    
    # 1. Overall Customer Service Metrics
    st.subheader("Overall Customer Service Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_complaints = q24_orders["has_complaint"].sum()
    complaint_rate = (total_complaints / len(q24_orders)) * 100
    avg_satisfaction = q24_orders["satisfaction_score"].mean()
    resolution_rate = (q24_orders[q24_orders["has_complaint"]]["is_resolved"].sum() / total_complaints * 100) if total_complaints > 0 else 0
    
    col1.metric("Total Complaints", f"{total_complaints:,}")
    col2.metric("Complaint Rate", f"{complaint_rate:.1f}%")
    col3.metric("Avg Satisfaction Score", f"{avg_satisfaction:.2f}/5.0")
    col4.metric("Resolution Rate", f"{resolution_rate:.1f}%")
    
    # 2. Complaint Categories Analysis
    st.subheader("Complaint Categories Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Complaint categories distribution
        complaint_dist = q24_orders[q24_orders["has_complaint"]]["complaint_category"].value_counts()
        
        fig = go.Figure(data=[
            go.Pie(
                labels=complaint_dist.index,
                values=complaint_dist.values,
                hole=0.4,
                texttemplate="%{percent:.1f}%"
            )
        ])
        
        fig.update_layout(
            title="Complaint Categories Distribution"
        )
        st.plotly_chart(fig)
    
    with col2:
        # Resolution time by category
        resolution_by_category = q24_orders[q24_orders["has_complaint"]].groupby("complaint_category")["resolution_time_hours"].mean().sort_values(ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(
                x=resolution_by_category.values,
                y=resolution_by_category.index,
                orientation="h",
                text=resolution_by_category.values,
                texttemplate="%{text:.1f}h",
                textposition="auto"
            )
        ])
        
        fig.update_layout(
            title="Average Resolution Time by Category",
            xaxis_title="Resolution Time (Hours)",
            yaxis_title="Complaint Category",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    # 3. Customer Satisfaction Analysis
    st.subheader("Customer Satisfaction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Satisfaction score distribution
        satisfaction_dist = q24_orders["satisfaction_score"].value_counts().sort_index()
        
        fig = go.Figure(data=[
            go.Bar(
                x=satisfaction_dist.index,
                y=satisfaction_dist.values,
                text=satisfaction_dist.values,
                textposition="auto"
            )
        ])
        
        fig.update_layout(
            title="Customer Satisfaction Score Distribution",
            xaxis_title="Satisfaction Score",
            yaxis_title="Number of Customers",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    with col2:
        # Satisfaction by subcategory
        subcategory_satisfaction = q24_orders.groupby("subcategory")["satisfaction_score"].mean().sort_values(ascending=True).tail(10)
        
        fig = go.Figure(data=[
            go.Bar(
                x=subcategory_satisfaction.values,
                y=subcategory_satisfaction.index,
                orientation="h",
                text=subcategory_satisfaction.values,
                texttemplate="%{text:.2f}",
                textposition="auto"
            )
        ])
        
        fig.update_layout(
            title="Top 10 Subcategories by Satisfaction Score",
            xaxis_title="Average Satisfaction Score",
            yaxis_title="Subcategory",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    # 4. Resolution Performance Analysis
    st.subheader("Resolution Performance Analysis")
    
    # Create resolution performance metrics
    resolution_metrics = q24_orders[q24_orders["has_complaint"]].groupby("complaint_category").agg({
        "is_resolved": ["sum", "count"],
        "resolution_time_hours": "mean",
        "satisfaction_score": "mean"
    })
    
    # Flatten MultiIndex columns
    resolution_metrics.columns = resolution_metrics.columns.droplevel(0) if resolution_metrics.columns.nlevels > 1 else resolution_metrics.columns
    resolution_metrics.columns = ["Resolved", "Total", "Avg Resolution Time", "Avg Satisfaction"]
    
    # Ensure numeric types and apply rounding
    resolution_metrics["Resolved"] = pd.to_numeric(resolution_metrics["Resolved"], errors='coerce')
    resolution_metrics["Total"] = pd.to_numeric(resolution_metrics["Total"], errors='coerce')
    resolution_metrics["Avg Resolution Time"] = pd.to_numeric(resolution_metrics["Avg Resolution Time"], errors='coerce').round(2)
    resolution_metrics["Avg Satisfaction"] = pd.to_numeric(resolution_metrics["Avg Satisfaction"], errors='coerce').round(2)
    
    resolution_metrics["Resolution Rate %"] = (
        resolution_metrics["Resolved"] / resolution_metrics["Total"] * 100
    ).round(2)
    
    st.write("Resolution Performance by Category")
    st.dataframe(resolution_metrics.style.format({
        "Resolved": "{:,.0f}",
        "Total": "{:,.0f}",
        "Avg Resolution Time": "{:.1f}h",
        "Avg Satisfaction": "{:.2f}",
        "Resolution Rate %": "{:.1f}%"
    }).background_gradient(subset=["Resolution Rate %"], cmap="RdYlGn"))
    
    # 5. Trends and Improvements
    st.subheader("Service Quality Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly satisfaction trends
        monthly_satisfaction = q24_orders.groupby(q24_orders["order_date"].dt.to_period("M")).agg({
            "satisfaction_score": "mean",
            "has_complaint": ["sum", "count"]
        }).reset_index()
        
        monthly_satisfaction.columns = ["Month", "Avg Satisfaction", "Complaints", "Total Orders"]
        monthly_satisfaction["Complaint Rate"] = (monthly_satisfaction["Complaints"] / monthly_satisfaction["Total Orders"] * 100).round(2)
        monthly_satisfaction["Month"] = monthly_satisfaction["Month"].astype(str)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_satisfaction["Month"],
            y=monthly_satisfaction["Avg Satisfaction"],
            mode='lines+markers',
            name='Avg Satisfaction',
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly_satisfaction["Month"],
            y=monthly_satisfaction["Complaint Rate"],
            mode='lines+markers',
            name='Complaint Rate %',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Monthly Satisfaction & Complaint Trends",
            xaxis_title="Month",
            yaxis=dict(title="Satisfaction Score", side="left"),
            yaxis2=dict(title="Complaint Rate (%)", side="right", overlaying="y"),
            showlegend=True
        )
        st.plotly_chart(fig)
    
    with col2:
        # Subcategory service performance
        subcategory_service = q24_orders.groupby("subcategory").agg({
            "satisfaction_score": "mean",
            "has_complaint": ["sum", "count"]
        })
        
        # Flatten MultiIndex columns
        subcategory_service.columns = subcategory_service.columns.droplevel(0) if subcategory_service.columns.nlevels > 1 else subcategory_service.columns
        subcategory_service.columns = ["Avg Satisfaction", "Complaints", "Total Orders"]
        
        # Ensure numeric types and apply rounding
        subcategory_service["Avg Satisfaction"] = pd.to_numeric(subcategory_service["Avg Satisfaction"], errors='coerce').round(2)
        subcategory_service["Complaints"] = pd.to_numeric(subcategory_service["Complaints"], errors='coerce')
        subcategory_service["Total Orders"] = pd.to_numeric(subcategory_service["Total Orders"], errors='coerce')
        
        subcategory_service["Complaint Rate"] = (subcategory_service["Complaints"] / subcategory_service["Total Orders"] * 100).round(2)
        
        # Scatter plot of satisfaction vs complaint rate
        fig = go.Figure(data=go.Scatter(
            x=subcategory_service["Complaint Rate"],
            y=subcategory_service["Avg Satisfaction"],
            mode='markers+text',
            text=subcategory_service.index,
            textposition="top center",
            marker=dict(
                size=subcategory_service["Total Orders"] / subcategory_service["Total Orders"].max() * 50,
                color=subcategory_service["Avg Satisfaction"],
                colorscale='Viridis',
                showscale=True
            )
        ))
        
        fig.update_layout(
            title="Subcategory Performance: Satisfaction vs Complaints",
            xaxis_title="Complaint Rate (%)",
            yaxis_title="Average Satisfaction Score",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    # 6. Service Quality Recommendations
    st.subheader("Service Quality Improvement Recommendations")
    
    # Identify areas for improvement
    high_complaint_categories = resolution_metrics.nlargest(3, "Total")
    low_satisfaction_subcategories = subcategory_service.nsmallest(5, "Avg Satisfaction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**High Volume Complaint Categories (Priority Focus)**")
        for category, data in high_complaint_categories.iterrows():
            st.write(f"• **{category}**: {int(data['Total'])} complaints, {data['Avg Resolution Time']:.1f}h avg resolution")
            if data["Resolution Rate %"] < 90:
                st.write(f"  ⚠️ Low resolution rate: {data['Resolution Rate %']:.1f}%")
    
    with col2:
        st.write("**Low Satisfaction Subcategories (Quality Focus)**")
        for subcategory, data in low_satisfaction_subcategories.iterrows():
            st.write(f"• **{subcategory}**: {data['Avg Satisfaction']:.2f}/5.0 satisfaction")
            st.write(f"  📈 {data['Complaint Rate']:.1f}% complaint rate")
    
    st.info("""
    This Customer Service Dashboard provides insights into:
    1. Overall customer service performance metrics
    2. Complaint categories and resolution analysis
    3. Customer satisfaction trends and patterns
    4. Service quality improvement opportunities
    
    Use the filters to analyze specific subcategories and time periods.
    Note: Customer service data is simulated for demonstration purposes.
    """)

# Supply Chain Dashboard (Question 25)
with tab25:
    st.header("Supply Chain Dashboard")
    
    # Filters
    with st.expander("Filters", expanded=True):
        # Date range selection
        max_date = pd.to_datetime(orders["order_date"]).max()
        default_start_date = max_date - pd.DateOffset(months=6)
        
        date_range = st.date_input(
            "Select Date Range",
            value=(default_start_date.date(), max_date.date()),
            min_value=pd.to_datetime(orders["order_date"]).min().date(),
            max_value=max_date.date(),
            key="q25_date_range"
        )
        
        # Subcategory selection
        subcategories = orders["subcategory"].dropna().unique()
        selected_subcategories = st.multiselect(
            "Select Subcategories",
            sorted(subcategories),
            default=sorted(subcategories),
            key="q25_subcategory_filter"
        )
    
    # Filter data and simulate supply chain data
    q25_orders = orders.copy()
    q25_orders["order_date"] = pd.to_datetime(q25_orders["order_date"])
    
    if len(date_range) == 2:
        q25_orders = q25_orders[
            (q25_orders["order_date"].dt.date >= date_range[0]) &
            (q25_orders["order_date"].dt.date <= date_range[1])
        ]
    
    if selected_subcategories:
        q25_orders = q25_orders[q25_orders["subcategory"].isin(selected_subcategories)]
    
    # Simulate supply chain data
    np.random.seed(42)
    suppliers = [f"Supplier_{i}" for i in range(1, 21)]  # 20 suppliers
    q25_orders = q25_orders.copy()
    q25_orders["supplier"] = np.random.choice(suppliers, len(q25_orders))
    q25_orders["procurement_cost"] = q25_orders["final_amount_inr"] * np.random.uniform(0.6, 0.8, len(q25_orders))
    q25_orders["delivery_reliability"] = np.random.choice([True, False], len(q25_orders), p=[0.88, 0.12])
    q25_orders["quality_score"] = np.random.normal(4.0, 0.7, len(q25_orders)).clip(1, 5).round(1)
    q25_orders["lead_time_days"] = np.random.normal(7, 3, len(q25_orders)).clip(1, 20).round().astype(int)
    
    # 1. Overall Supply Chain Metrics
    st.subheader("Overall Supply Chain Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_suppliers = q25_orders["supplier"].nunique()
    avg_procurement_cost = q25_orders["procurement_cost"].sum()
    delivery_reliability_rate = (q25_orders["delivery_reliability"].sum() / len(q25_orders)) * 100
    avg_quality_score = q25_orders["quality_score"].mean()
    
    col1.metric("Active Suppliers", f"{total_suppliers}")
    col2.metric("Total Procurement Cost", f"₹{avg_procurement_cost:,.0f}")
    col3.metric("Delivery Reliability", f"{delivery_reliability_rate:.1f}%")
    col4.metric("Avg Quality Score", f"{avg_quality_score:.2f}/5.0")
    
    # 2. Supplier Performance Analysis
    st.subheader("Supplier Performance Analysis")
    
    supplier_performance = q25_orders.groupby("supplier").agg({
        "procurement_cost": "sum",
        "delivery_reliability": ["sum", "count"],
        "quality_score": "mean",
        "lead_time_days": "mean",
        "final_amount_inr": "sum"
    })
    
    # Flatten MultiIndex columns
    supplier_performance.columns = supplier_performance.columns.droplevel(0) if supplier_performance.columns.nlevels > 1 else supplier_performance.columns
    supplier_performance.columns = ["Procurement Cost", "Reliable Deliveries", "Total Orders", "Avg Quality", "Avg Lead Time", "Revenue"]
    
    # Ensure numeric types and apply rounding
    supplier_performance["Procurement Cost"] = pd.to_numeric(supplier_performance["Procurement Cost"], errors='coerce').round(2)
    supplier_performance["Reliable Deliveries"] = pd.to_numeric(supplier_performance["Reliable Deliveries"], errors='coerce')
    supplier_performance["Total Orders"] = pd.to_numeric(supplier_performance["Total Orders"], errors='coerce')
    supplier_performance["Avg Quality"] = pd.to_numeric(supplier_performance["Avg Quality"], errors='coerce').round(2)
    supplier_performance["Avg Lead Time"] = pd.to_numeric(supplier_performance["Avg Lead Time"], errors='coerce').round(2)
    supplier_performance["Revenue"] = pd.to_numeric(supplier_performance["Revenue"], errors='coerce').round(2)
    
    supplier_performance["Reliability %"] = (
        supplier_performance["Reliable Deliveries"] / supplier_performance["Total Orders"] * 100
    ).round(2)
    supplier_performance["Cost Efficiency"] = (
        supplier_performance["Procurement Cost"] / supplier_performance["Revenue"] * 100
    ).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Top 10 Suppliers by Performance")
        top_suppliers = supplier_performance.nlargest(10, "Revenue")
        st.dataframe(top_suppliers.style.format({
            "Procurement Cost": "₹{:,.0f}",
            "Reliable Deliveries": "{:,.0f}",
            "Total Orders": "{:,.0f}",
            "Avg Quality": "{:.2f}",
            "Avg Lead Time": "{:.1f} days",
            "Revenue": "₹{:,.0f}",
            "Reliability %": "{:.1f}%",
            "Cost Efficiency": "{:.1f}%"
        }).background_gradient(subset=["Reliability %"], cmap="RdYlGn"))
    
    with col2:
        # Supplier performance scatter plot
        fig = go.Figure(data=go.Scatter(
            x=supplier_performance["Reliability %"],
            y=supplier_performance["Avg Quality"],
            mode='markers',
            marker=dict(
                size=supplier_performance["Revenue"] / supplier_performance["Revenue"].max() * 50,
                color=supplier_performance["Cost Efficiency"],
                colorscale='Viridis_r',
                showscale=True,
                colorbar=dict(title="Cost Efficiency %")
            ),
            text=supplier_performance.index,
            hovertemplate="Supplier: %{text}<br>Reliability: %{x:.1f}%<br>Quality: %{y:.2f}<br>Cost Efficiency: %{marker.color:.1f}%"
        ))
        
        fig.update_layout(
            title="Supplier Performance Matrix",
            xaxis_title="Delivery Reliability (%)",
            yaxis_title="Quality Score",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    # 3. Cost Analysis
    st.subheader("Supply Chain Cost Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cost distribution by subcategory
        cost_by_subcategory = q25_orders.groupby("subcategory").agg({
            "procurement_cost": "sum",
            "final_amount_inr": "sum"
        }).round(2)
        
        cost_by_subcategory["margin"] = cost_by_subcategory["final_amount_inr"] - cost_by_subcategory["procurement_cost"]
        cost_by_subcategory["margin_percentage"] = (cost_by_subcategory["margin"] / cost_by_subcategory["final_amount_inr"] * 100).round(2)
        cost_by_subcategory = cost_by_subcategory.sort_values("margin_percentage", ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(
                x=cost_by_subcategory["margin_percentage"],
                y=cost_by_subcategory.index,
                orientation="h",
                text=cost_by_subcategory["margin_percentage"],
                texttemplate="%{text:.1f}%",
                textposition="auto"
            )
        ])
        
        fig.update_layout(
            title="Profit Margin by Subcategory",
            xaxis_title="Margin Percentage (%)",
            yaxis_title="Subcategory",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    with col2:
        # Monthly cost trends
        monthly_costs = q25_orders.groupby(q25_orders["order_date"].dt.to_period("M")).agg({
            "procurement_cost": "sum",
            "final_amount_inr": "sum"
        }).reset_index()
        
        monthly_costs["cost_ratio"] = (monthly_costs["procurement_cost"] / monthly_costs["final_amount_inr"] * 100).round(2)
        monthly_costs["order_date"] = monthly_costs["order_date"].astype(str)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_costs["order_date"],
            y=monthly_costs["procurement_cost"],
            mode='lines+markers',
            name='Procurement Cost',
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly_costs["order_date"],
            y=monthly_costs["cost_ratio"],
            mode='lines+markers',
            name='Cost Ratio %',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Monthly Procurement Cost Trends",
            xaxis_title="Month",
            yaxis=dict(title="Procurement Cost (₹)", side="left"),
            yaxis2=dict(title="Cost Ratio (%)", side="right", overlaying="y"),
            showlegend=True
        )
        st.plotly_chart(fig)
    
    # 4. Delivery Performance & Lead Times
    st.subheader("Delivery Performance & Lead Times")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Lead time distribution by subcategory
        subcategory_leadtime = q25_orders.groupby("subcategory")["lead_time_days"].mean().sort_values(ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(
                x=subcategory_leadtime.values,
                y=subcategory_leadtime.index,
                orientation="h",
                text=subcategory_leadtime.values,
                texttemplate="%{text:.1f} days",
                textposition="auto"
            )
        ])
        
        fig.update_layout(
            title="Average Lead Time by Subcategory",
            xaxis_title="Lead Time (Days)",
            yaxis_title="Subcategory",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    with col2:
        # Reliability vs lead time correlation
        supplier_metrics = q25_orders.groupby("supplier").agg({
            "delivery_reliability": "mean",
            "lead_time_days": "mean",
            "final_amount_inr": "sum"
        }).round(2)
        
        fig = go.Figure(data=go.Scatter(
            x=supplier_metrics["lead_time_days"],
            y=supplier_metrics["delivery_reliability"] * 100,
            mode='markers',
            marker=dict(
                size=supplier_metrics["final_amount_inr"] / supplier_metrics["final_amount_inr"].max() * 50,
                color=supplier_metrics["final_amount_inr"],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Revenue (₹)")
            ),
            text=supplier_metrics.index,
            hovertemplate="Supplier: %{text}<br>Lead Time: %{x:.1f} days<br>Reliability: %{y:.1f}%"
        ))
        
        fig.update_layout(
            title="Supplier Lead Time vs Reliability",
            xaxis_title="Average Lead Time (Days)",
            yaxis_title="Delivery Reliability (%)",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    # 5. Vendor Management Insights
    st.subheader("Vendor Management Insights")
    
    # Categorize suppliers based on performance
    supplier_categories = supplier_performance.copy()
    supplier_categories["category"] = "Standard"
    
    # Strategic suppliers: High revenue, high reliability
    strategic_mask = (supplier_categories["Revenue"] > supplier_categories["Revenue"].quantile(0.7)) & \
                    (supplier_categories["Reliability %"] > 85)
    supplier_categories.loc[strategic_mask, "category"] = "Strategic"
    
    # Leverage suppliers: High revenue, lower reliability
    leverage_mask = (supplier_categories["Revenue"] > supplier_categories["Revenue"].quantile(0.7)) & \
                   (supplier_categories["Reliability %"] <= 85)
    supplier_categories.loc[leverage_mask, "category"] = "Leverage"
    
    # Bottleneck suppliers: Low revenue, high reliability
    bottleneck_mask = (supplier_categories["Revenue"] <= supplier_categories["Revenue"].quantile(0.7)) & \
                     (supplier_categories["Reliability %"] > 85)
    supplier_categories.loc[bottleneck_mask, "category"] = "Bottleneck"
    
    # Non-critical suppliers: Low revenue, low reliability
    noncritical_mask = (supplier_categories["Revenue"] <= supplier_categories["Revenue"].quantile(0.7)) & \
                      (supplier_categories["Reliability %"] <= 85)
    supplier_categories.loc[noncritical_mask, "category"] = "Non-Critical"
    
    category_summary = supplier_categories["category"].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Supplier categorization
        fig = go.Figure(data=[
            go.Pie(
                labels=category_summary.index,
                values=category_summary.values,
                hole=0.4,
                texttemplate="%{percent:.1f}%"
            )
        ])
        
        fig.update_layout(
            title="Supplier Categorization"
        )
        st.plotly_chart(fig)
    
    with col2:
        st.write("**Vendor Management Recommendations:**")
        
        strategic_count = category_summary.get("Strategic", 0)
        leverage_count = category_summary.get("Leverage", 0)
        bottleneck_count = category_summary.get("Bottleneck", 0)
        noncritical_count = category_summary.get("Non-Critical", 0)
        
        st.write(f"🎯 **Strategic Suppliers ({strategic_count})**: Maintain partnership, negotiate better terms")
        st.write(f"⚖️ **Leverage Suppliers ({leverage_count})**: Improve reliability, consider alternatives")
        st.write(f"🔧 **Bottleneck Suppliers ({bottleneck_count})**: Secure supply, find backup options")
        st.write(f"📋 **Non-Critical Suppliers ({noncritical_count})**: Standardize products, reduce costs")
        
        # Performance improvement opportunities
        st.write("**Priority Improvement Areas:**")
        low_reliability = supplier_performance[supplier_performance["Reliability %"] < 80]
        high_cost = supplier_performance[supplier_performance["Cost Efficiency"] > 75]
        
        if len(low_reliability) > 0:
            st.write(f"• {len(low_reliability)} suppliers need reliability improvement")
        if len(high_cost) > 0:
            st.write(f"• {len(high_cost)} suppliers have high cost ratios")
    
    st.info("""
    This Supply Chain Dashboard provides insights into:
    1. Overall supply chain performance metrics
    2. Supplier performance analysis and benchmarking
    3. Cost analysis and margin optimization
    4. Delivery performance and lead time management
    5. Vendor management insights and recommendations
    
    Use the filters to analyze specific subcategories and time periods.
    Note: Supply chain data is simulated for demonstration purposes.
    """)

# Main Tab 6: Advanced Analytics (Questions 26-30)
with main_tab6:
    st.header("🔮 Advanced Analytics")
    st.write("Predictive modeling, market intelligence, and executive dashboards covering Questions")
    
    # Create subtabs for Questions 26-30
    tab26, tab27, tab28, tab29, tab30 = st.tabs([
        "redictive Analytics", 
        "Market Intelligence", 
        "Cross-selling & Upselling", 
        "Seasonal Planning", 
        "BI Command Center"
    ])

# Question 26: Predictive Analytics
with tab26:
    st.header("Predictive Analytics Dashboard")
    
    # Import additional libraries for advanced analytics
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    import warnings
    warnings.filterwarnings('ignore')
    
    # Filters
    with st.expander("Filters & Model Configuration", expanded=True):
        # Date range selection
        max_date = pd.to_datetime(orders["order_date"]).max()
        default_start_date = max_date - pd.DateOffset(months=12)
        
        date_range = st.date_input(
            "Select Historical Data Range",
            value=(default_start_date.date(), max_date.date()),
            min_value=pd.to_datetime(orders["order_date"]).min().date(),
            max_value=max_date.date(),
            key="q26_date_range"
        )
        
        # Forecasting parameters
        forecast_days = st.slider("Forecast Horizon (Days)", 30, 180, 90, key="q26_forecast_days")
        
        # Model selection
        model_type = st.selectbox("Prediction Model", 
                                ["Linear Regression", "Random Forest"], 
                                key="q26_model_type")
        
        # Subcategory selection
        subcategories = orders["subcategory"].dropna().unique()
        selected_subcategories = st.multiselect(
            "Select Subcategories for Analysis",
            sorted(subcategories),
            default=sorted(subcategories)[:5],
            key="q26_subcategory_filter"
        )
    
    # Filter and prepare data
    q26_orders = orders.copy()
    q26_orders["order_date"] = pd.to_datetime(q26_orders["order_date"])
    
    if len(date_range) == 2:
        q26_orders = q26_orders[
            (q26_orders["order_date"].dt.date >= date_range[0]) &
            (q26_orders["order_date"].dt.date <= date_range[1])
        ]
    
    if selected_subcategories:
        q26_orders = q26_orders[q26_orders["subcategory"].isin(selected_subcategories)]
    
    # 1. Sales Forecasting
    st.subheader("Sales Forecasting")
    
    # Prepare time series data
    daily_sales = q26_orders.groupby(q26_orders["order_date"].dt.date).agg({
        "final_amount_inr": "sum",
        "order_id": "count"
    }).reset_index()
    
    daily_sales.columns = ["date", "revenue", "orders"]
    daily_sales["date"] = pd.to_datetime(daily_sales["date"])
    daily_sales = daily_sales.sort_values("date").reset_index(drop=True)
    
    # Create features for forecasting
    daily_sales["day_of_week"] = daily_sales["date"].dt.dayofweek
    daily_sales["month"] = daily_sales["date"].dt.month
    daily_sales["day_of_month"] = daily_sales["date"].dt.day
    daily_sales["quarter"] = daily_sales["date"].dt.quarter
    daily_sales["days_since_start"] = (daily_sales["date"] - daily_sales["date"].min()).dt.days
    
    # Calculate moving averages
    daily_sales["ma_7"] = daily_sales["revenue"].rolling(window=7, min_periods=1).mean()
    daily_sales["ma_30"] = daily_sales["revenue"].rolling(window=30, min_periods=1).mean()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue forecasting
        if len(daily_sales) > 30:  # Ensure we have enough data
            # Prepare features and target
            feature_cols = ["day_of_week", "month", "day_of_month", "quarter", "days_since_start", "ma_7", "ma_30"]
            X = daily_sales[feature_cols].fillna(0)
            y = daily_sales["revenue"]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            if model_type == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Model performance
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            st.write("**Revenue Forecasting Model Performance**")
            st.metric("Mean Absolute Error", f"₹{mae:,.0f}")
            st.metric("R² Score", f"{r2:.3f}")
            
            # Generate future predictions
            last_date = daily_sales["date"].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                       periods=forecast_days, freq='D')
            
            future_features = []
            for date in future_dates:
                days_since = (date - daily_sales["date"].min()).days
                features = [
                    date.dayofweek,
                    date.month,
                    date.day,
                    date.quarter,
                    days_since,
                    daily_sales["revenue"].tail(7).mean(),  # Recent 7-day average
                    daily_sales["revenue"].tail(30).mean()  # Recent 30-day average
                ]
                future_features.append(features)
            
            future_X = pd.DataFrame(future_features, columns=feature_cols)
            future_pred = model.predict(future_X)
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                "date": future_dates,
                "predicted_revenue": future_pred
            })
            
            # Plot historical and forecast
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=daily_sales["date"].tail(60),  # Last 60 days
                y=daily_sales["revenue"].tail(60),
                mode='lines+markers',
                name='Historical Revenue',
                line=dict(color='blue')
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["predicted_revenue"],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f"Revenue Forecast - Next {forecast_days} Days",
                xaxis_title="Date",
                yaxis_title="Revenue (₹)",
                showlegend=True
            )
            st.plotly_chart(fig)
            
        else:
            st.warning("Insufficient data for forecasting. Need at least 30 days of historical data.")
    
    with col2:
        # Demand planning by subcategory
        subcategory_demand = q26_orders.groupby([
            q26_orders["order_date"].dt.date,
            "subcategory"
        ]).agg({
            "order_id": "count",
            "final_amount_inr": "sum"
        }).reset_index()
        
        subcategory_demand.columns = ["date", "subcategory", "orders", "revenue"]
        
        # Calculate average daily demand by subcategory
        avg_demand = subcategory_demand.groupby("subcategory").agg({
            "orders": "mean",
            "revenue": "mean"
        }).round(2)
        
        avg_demand = avg_demand.sort_values("revenue", ascending=False).head(10)
        
        st.write("**Top 10 Subcategories - Average Daily Demand**")
        st.dataframe(avg_demand.style.format({
            "orders": "{:.1f}",
            "revenue": "₹{:,.0f}"
        }))
        
        # Demand volatility analysis
        demand_volatility = subcategory_demand.groupby("subcategory")["orders"].std().sort_values(ascending=False).head(10)
        
        fig = go.Figure(data=[
            go.Bar(
                x=demand_volatility.values,
                y=demand_volatility.index,
                orientation='h',
                text=demand_volatility.values,
                texttemplate='%{text:.1f}',
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Top 10 Most Volatile Subcategories (Std Dev of Daily Orders)",
            xaxis_title="Standard Deviation",
            yaxis_title="Subcategory",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    # 2. Customer Churn Prediction
    st.subheader("Customer Churn Prediction")
    
    # Calculate customer metrics for churn analysis
    customer_metrics = q26_orders.groupby("customer_id").agg({
        "order_date": ["min", "max", "count"],
        "final_amount_inr": ["sum", "mean"],
        "subcategory": "nunique"
    })
    
    # Flatten MultiIndex columns
    customer_metrics.columns = customer_metrics.columns.droplevel(0) if customer_metrics.columns.nlevels > 1 else customer_metrics.columns
    customer_metrics.columns = ["first_order", "last_order", "order_frequency", "total_spent", "avg_order_value", "category_diversity"]
    
    # Calculate recency (days since last order)
    customer_metrics["recency"] = (max_date - pd.to_datetime(customer_metrics["last_order"])).dt.days
    customer_metrics["tenure"] = (pd.to_datetime(customer_metrics["last_order"]) - pd.to_datetime(customer_metrics["first_order"])).dt.days
    
    # Define churn (customers who haven't ordered in 90+ days)
    customer_metrics["is_churned"] = customer_metrics["recency"] > 90
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn rate metrics
        total_customers = len(customer_metrics)
        churned_customers = customer_metrics["is_churned"].sum()
        churn_rate = (churned_customers / total_customers) * 100
        
        st.metric("Total Customers", f"{total_customers:,}")
        st.metric("Churned Customers", f"{churned_customers:,}")
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
        
        # Churn by customer segments
        customer_metrics["spending_segment"] = pd.qcut(
            customer_metrics["total_spent"], 
            q=4, 
            labels=["Low", "Medium", "High", "Premium"],
            duplicates='drop'
        )
        
        churn_by_segment = customer_metrics.groupby("spending_segment")["is_churned"].agg(
            ["sum", "count"]
        )
        churn_by_segment.columns = ["churned", "total"]
        churn_by_segment["churn_rate"] = (churn_by_segment["churned"] / churn_by_segment["total"] * 100).round(1)
        
        st.write("**Churn Rate by Spending Segment**")
        fig = go.Figure(data=[
            go.Bar(
                x=churn_by_segment.index,
                y=churn_by_segment["churn_rate"],
                text=churn_by_segment["churn_rate"],
                texttemplate='%{text:.1f}%',
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Churn Rate by Customer Segment",
            xaxis_title="Spending Segment",
            yaxis_title="Churn Rate (%)",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    with col2:
        # Churn risk factors analysis
        risk_factors = customer_metrics[customer_metrics["is_churned"] == False].copy()
        
        # Risk scoring based on recency and frequency
        risk_factors["risk_score"] = (
            (risk_factors["recency"] / risk_factors["recency"].max()) * 0.5 +
            (1 - risk_factors["order_frequency"] / risk_factors["order_frequency"].max()) * 0.3 +
            (1 - risk_factors["total_spent"] / risk_factors["total_spent"].max()) * 0.2
        ) * 100
        
        high_risk_customers = len(risk_factors[risk_factors["risk_score"] > 70])
        medium_risk_customers = len(risk_factors[(risk_factors["risk_score"] > 40) & (risk_factors["risk_score"] <= 70)])
        low_risk_customers = len(risk_factors[risk_factors["risk_score"] <= 40])
        
        risk_distribution = pd.DataFrame({
            "Risk Level": ["High Risk", "Medium Risk", "Low Risk"],
            "Count": [high_risk_customers, medium_risk_customers, low_risk_customers]
        })
        
        fig = go.Figure(data=[
            go.Pie(
                labels=risk_distribution["Risk Level"],
                values=risk_distribution["Count"],
                hole=0.4,
                marker_colors=['#FF4444', '#FFA500', '#90EE90']
            )
        ])
        
        fig.update_layout(
            title="Customer Churn Risk Distribution"
        )
        st.plotly_chart(fig)
        
        # Top risk customers
        top_risk = risk_factors.nlargest(10, "risk_score")[["recency", "order_frequency", "total_spent", "risk_score"]]
        
        st.write("**Top 10 At-Risk Customers**")
        st.dataframe(top_risk.style.format({
            "recency": "{:.0f} days",
            "order_frequency": "{:.0f}",
            "total_spent": "₹{:,.0f}",
            "risk_score": "{:.1f}"
        }).background_gradient(subset=["risk_score"], cmap="Reds"))
    
    # 3. Business Scenario Analysis
    st.subheader("Business Scenario Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**What-If Scenario Planning**")
        
        # Current baseline metrics
        current_revenue = q26_orders["final_amount_inr"].sum()
        current_orders = len(q26_orders)
        current_aov = current_revenue / current_orders
        
        # Scenario parameters
        price_change = st.slider("Price Change (%)", -30, 30, 0, key="price_scenario")
        marketing_spend = st.slider("Marketing Spend Increase (%)", 0, 100, 0, key="marketing_scenario")
        
        # Calculate scenario impacts (simplified model)
        # Price elasticity assumption: 1% price increase = 0.5% demand decrease
        demand_impact = -0.5 * price_change / 100
        
        # Marketing impact assumption: 1% marketing increase = 0.3% demand increase
        marketing_impact = 0.3 * marketing_spend / 100
        
        new_demand = current_orders * (1 + demand_impact + marketing_impact)
        new_price = current_aov * (1 + price_change / 100)
        new_revenue = new_demand * new_price
        
        revenue_change = ((new_revenue / current_revenue) - 1) * 100
        
        st.write("**Scenario Results:**")
        st.metric("Projected Revenue", f"₹{new_revenue:,.0f}", f"{revenue_change:+.1f}%")
        st.metric("Projected Orders", f"{new_demand:,.0f}", f"{((new_demand/current_orders)-1)*100:+.1f}%")
        st.metric("New Average Order Value", f"₹{new_price:,.0f}", f"{price_change:+.1f}%")
        
    with col2:
        st.write("**Sensitivity Analysis**")
        
        # Create sensitivity analysis for price changes
        price_scenarios = np.arange(-20, 25, 5)
        scenario_results = []
        
        for price_change in price_scenarios:
            demand_impact = -0.5 * price_change / 100
            new_demand = current_orders * (1 + demand_impact)
            new_price = current_aov * (1 + price_change / 100)
            new_revenue = new_demand * new_price
            revenue_change = ((new_revenue / current_revenue) - 1) * 100
            
            scenario_results.append({
                "Price Change %": price_change,
                "Revenue Change %": revenue_change
            })
        
        scenario_df = pd.DataFrame(scenario_results)
        
        fig = go.Figure(data=[
            go.Scatter(
                x=scenario_df["Price Change %"],
                y=scenario_df["Revenue Change %"],
                mode='lines+markers',
                name='Revenue Impact'
            )
        ])
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title="Price Sensitivity Analysis",
            xaxis_title="Price Change (%)",
            yaxis_title="Revenue Change (%)",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    st.info("""
    This Predictive Analytics Dashboard provides insights into:
    1. Sales forecasting using machine learning models
    2. Customer churn prediction and risk assessment
    3. Demand planning and volatility analysis
    4. Business scenario analysis and sensitivity testing
    
    Use the filters and model configuration to adjust predictions.
    Note: Predictive models are simplified for demonstration purposes.
    """)

# Market Intelligence Dashboard (Question 27)
with tab27:
    st.header("Market Intelligence Dashboard")
    
    # Filters
    with st.expander("Filters", expanded=True):
        # Date range selection
        max_date = pd.to_datetime(orders["order_date"]).max()
        default_start_date = max_date - pd.DateOffset(months=6)
        
        date_range = st.date_input(
            "Select Analysis Period",
            value=(default_start_date.date(), max_date.date()),
            min_value=pd.to_datetime(orders["order_date"]).min().date(),
            max_value=max_date.date(),
            key="q27_date_range"
        )
        
        # Subcategory selection for competitive analysis
        subcategories = orders["subcategory"].dropna().unique()
        selected_subcategories = st.multiselect(
            "Select Subcategories for Market Analysis",
            sorted(subcategories),
            default=sorted(subcategories)[:10],
            key="q27_subcategory_filter"
        )
    
    # Filter data and simulate competitive intelligence data
    q27_orders = orders.copy()
    q27_orders["order_date"] = pd.to_datetime(q27_orders["order_date"])
    
    if len(date_range) == 2:
        q27_orders = q27_orders[
            (q27_orders["order_date"].dt.date >= date_range[0]) &
            (q27_orders["order_date"].dt.date <= date_range[1])
        ]
    
    if selected_subcategories:
        q27_orders = q27_orders[q27_orders["subcategory"].isin(selected_subcategories)]
    
    # Simulate competitive data
    np.random.seed(42)
    competitors = ["Amazon", "Flipkart", "Myntra", "Meesho", "Snapdeal"]
    
    # 1. Market Share Analysis
    st.subheader("Market Share & Competitive Positioning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Simulate market share data
        our_revenue = q27_orders["final_amount_inr"].sum()
        
        # Generate competitive revenue data
        competitive_data = []
        for competitor in competitors:
            if competitor == "Amazon":
                revenue = our_revenue * np.random.uniform(1.2, 1.5)  # Market leader
            elif competitor == "Flipkart":
                revenue = our_revenue * np.random.uniform(0.8, 1.0)  # Close competitor
            else:
                revenue = our_revenue * np.random.uniform(0.3, 0.7)  # Smaller players
            
            competitive_data.append({
                "Company": competitor,
                "Revenue": revenue
            })
        
        # Add our company
        competitive_data.append({"Company": "Our Company", "Revenue": our_revenue})
        
        market_df = pd.DataFrame(competitive_data)
        market_df["Market Share %"] = (market_df["Revenue"] / market_df["Revenue"].sum() * 100).round(2)
        market_df = market_df.sort_values("Market Share %", ascending=False)
        
        fig = go.Figure(data=[
            go.Pie(
                labels=market_df["Company"],
                values=market_df["Market Share %"],
                hole=0.4,
                texttemplate='%{percent}',
                marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
            )
        ])
        
        fig.update_layout(
            title="Market Share Distribution",
            showlegend=True
        )
        st.plotly_chart(fig)
    
    with col2:
        # Market position analysis
        st.write("**Competitive Positioning Matrix**")
        
        # Simulate positioning data (Price vs Quality perception)
        positioning_data = []
        for _, row in market_df.iterrows():
            company = row["Company"]
            if company == "Our Company":
                price_pos = 65  # Mid-premium
                quality_pos = 78  # High quality
            elif company == "Amazon":
                price_pos = 70  # Premium
                quality_pos = 85  # Highest quality
            elif company == "Flipkart":
                price_pos = 60  # Mid-market
                quality_pos = 75  # Good quality
            elif company == "Myntra":
                price_pos = 75  # Premium fashion
                quality_pos = 80  # High quality
            else:
                price_pos = np.random.uniform(40, 60)  # Budget
                quality_pos = np.random.uniform(50, 70)  # Average quality
            
            positioning_data.append({
                "Company": company,
                "Price Position": price_pos,
                "Quality Position": quality_pos,
                "Market Share": row["Market Share %"]
            })
        
        positioning_df = pd.DataFrame(positioning_data)
        
        fig = go.Figure(data=go.Scatter(
            x=positioning_df["Price Position"],
            y=positioning_df["Quality Position"],
            mode='markers+text',
            text=positioning_df["Company"],
            textposition="top center",
            marker=dict(
                size=positioning_df["Market Share"] * 2,  # Size represents market share
                color=positioning_df["Market Share"],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Market Share %")
            )
        ))
        
        fig.update_layout(
            title="Competitive Positioning Map",
            xaxis_title="Price Position (Budget ← → Premium)",
            yaxis_title="Quality Position (Low ← → High)",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    # 2. Market Trends Analysis
    st.subheader("Market Trends & Growth Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Subcategory growth trends
        monthly_subcategory = q27_orders.groupby([
            q27_orders["order_date"].dt.to_period("M"),
            "subcategory"
        ])["final_amount_inr"].sum().reset_index()
        
        monthly_subcategory["order_date"] = monthly_subcategory["order_date"].astype(str)
        
        # Calculate growth rates for top subcategories
        top_subcategories = q27_orders.groupby("subcategory")["final_amount_inr"].sum().nlargest(5).index
        
        growth_analysis = {}
        for subcategory in top_subcategories:
            subcat_data = monthly_subcategory[monthly_subcategory["subcategory"] == subcategory]
            if len(subcat_data) > 1:
                first_month = subcat_data["final_amount_inr"].iloc[0]
                last_month = subcat_data["final_amount_inr"].iloc[-1]
                growth_rate = ((last_month / first_month) ** (1 / len(subcat_data))) - 1
                growth_analysis[subcategory] = growth_rate * 100
        
        growth_df = pd.DataFrame(list(growth_analysis.items()), columns=["Subcategory", "Monthly Growth Rate"])
        growth_df = growth_df.sort_values("Monthly Growth Rate", ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(
                x=growth_df["Monthly Growth Rate"],
                y=growth_df["Subcategory"],
                orientation='h',
                text=growth_df["Monthly Growth Rate"],
                texttemplate='%{text:.1f}%',
                textposition='auto',
                marker_color=np.where(growth_df["Monthly Growth Rate"] > 0, 'green', 'red')
            )
        ])
        
        fig.update_layout(
            title="Subcategory Growth Trends (Monthly %)",
            xaxis_title="Monthly Growth Rate (%)",
            yaxis_title="Subcategory",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    with col2:
        # Market size and opportunity analysis
        st.write("**Market Opportunity Analysis**")
        
        # Calculate market opportunity metrics
        opportunity_metrics = q27_orders.groupby("subcategory").agg({
            "final_amount_inr": ["sum", "mean", "count"],
            "customer_id": "nunique"
        })
        
        # Flatten MultiIndex columns
        opportunity_metrics.columns = opportunity_metrics.columns.droplevel(0) if opportunity_metrics.columns.nlevels > 1 else opportunity_metrics.columns
        opportunity_metrics.columns = ["Total Revenue", "Avg Order Value", "Order Count", "Unique Customers"]
        
        # Calculate opportunity score (combination of revenue potential and customer base)
        opportunity_metrics["Opportunity Score"] = (
            (opportunity_metrics["Total Revenue"] / opportunity_metrics["Total Revenue"].max()) * 0.5 +
            (opportunity_metrics["Unique Customers"] / opportunity_metrics["Unique Customers"].max()) * 0.3 +
            (opportunity_metrics["Avg Order Value"] / opportunity_metrics["Avg Order Value"].max()) * 0.2
        ) * 100
        
        top_opportunities = opportunity_metrics.nlargest(8, "Opportunity Score")
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_opportunities["Opportunity Score"],
                y=top_opportunities.index,
                orientation='h',
                text=top_opportunities["Opportunity Score"],
                texttemplate='%{text:.1f}',
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Market Opportunity Score by Subcategory",
            xaxis_title="Opportunity Score",
            yaxis_title="Subcategory",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    # 3. Pricing Intelligence
    st.subheader("Pricing Intelligence & Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price positioning analysis
        price_analysis = q27_orders.groupby("subcategory")["final_amount_inr"].describe()
        price_analysis = price_analysis.round(2)
        
        # Simulate competitor pricing
        competitive_pricing = {}
        for subcategory in price_analysis.index:
            our_avg_price = price_analysis.loc[subcategory, "mean"]
            
            # Generate competitor prices around our price
            comp_prices = {
                "Our Price": our_avg_price,
                "Market Leader": our_avg_price * np.random.uniform(1.05, 1.15),
                "Direct Competitor": our_avg_price * np.random.uniform(0.95, 1.05),
                "Budget Option": our_avg_price * np.random.uniform(0.7, 0.9)
            }
            competitive_pricing[subcategory] = comp_prices
        
        # Display pricing comparison for top subcategories
        top_subcats = q27_orders.groupby("subcategory")["final_amount_inr"].sum().nlargest(5).index
        
        st.write("**Competitive Pricing Analysis (Top 5 Subcategories)**")
        
        pricing_comparison = []
        for subcat in top_subcats:
            prices = competitive_pricing[subcat]
            pricing_comparison.append({
                "Subcategory": subcat,
                "Our Price": f"₹{prices['Our Price']:,.0f}",
                "Market Leader": f"₹{prices['Market Leader']:,.0f}",
                "Direct Competitor": f"₹{prices['Direct Competitor']:,.0f}",
                "Budget Option": f"₹{prices['Budget Option']:,.0f}"
            })
        
        pricing_df = pd.DataFrame(pricing_comparison)
        st.dataframe(pricing_df)
    
    with col2:
        # Price elasticity and opportunity
        st.write("**Pricing Opportunity Analysis**")
        
        # Calculate price gaps and opportunities
        price_opportunities = []
        for subcat in top_subcats:
            prices = competitive_pricing[subcat]
            our_price = prices["Our Price"]
            market_leader_price = prices["Market Leader"]
            
            # Calculate potential uplift if we can command premium pricing
            potential_uplift = ((market_leader_price - our_price) / our_price) * 100
            current_revenue = q27_orders[q27_orders["subcategory"] == subcat]["final_amount_inr"].sum()
            
            # Assume 50% of customers would accept 25% of the price gap
            conservative_uplift = potential_uplift * 0.25 * 0.5
            revenue_opportunity = current_revenue * (conservative_uplift / 100)
            
            price_opportunities.append({
                "Subcategory": subcat,
                "Price Gap %": potential_uplift,
                "Conservative Uplift %": conservative_uplift,
                "Revenue Opportunity": revenue_opportunity
            })
        
        opportunities_df = pd.DataFrame(price_opportunities)
        opportunities_df = opportunities_df.sort_values("Revenue Opportunity", ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(
                x=opportunities_df["Revenue Opportunity"],
                y=opportunities_df["Subcategory"],
                orientation='h',
                text=opportunities_df["Revenue Opportunity"],
                texttemplate='₹%{text:,.0f}',
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Revenue Opportunity from Pricing Optimization",
            xaxis_title="Revenue Opportunity (₹)",
            yaxis_title="Subcategory",
            showlegend=False
        )
        st.plotly_chart(fig)
    
    # 4. Strategic Insights & Recommendations
    st.subheader("Strategic Intelligence & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Market Position Assessment**")
        
        # Calculate our market position metrics
        our_market_share = market_df[market_df["Company"] == "Our Company"]["Market Share %"].iloc[0]
        market_rank = market_df[market_df["Company"] == "Our Company"].index[0] + 1
        
        st.metric("Market Share", f"{our_market_share:.1f}%")
        st.metric("Market Ranking", f"#{market_rank}")
        
        # Growth vs Market Share analysis
        total_revenue_growth = q27_orders.groupby(q27_orders["order_date"].dt.to_period("M"))["final_amount_inr"].sum()
        if len(total_revenue_growth) > 1:
            recent_growth = ((total_revenue_growth.iloc[-1] / total_revenue_growth.iloc[0]) ** (1/len(total_revenue_growth)) - 1) * 100
        else:
            recent_growth = 0
        
        st.metric("Monthly Growth Rate", f"{recent_growth:.1f}%")
        
        # Strategic recommendations based on position
        st.write("**Strategic Recommendations:**")
        
        if market_rank <= 2:
            st.success("🏆 **Market Leader Strategy**")
            st.write("- Focus on market expansion and innovation")
            st.write("- Defend market share through differentiation")
            st.write("- Consider premium positioning opportunities")
        elif market_rank <= 4:
            st.warning("⚡ **Challenger Strategy**")
            st.write("- Target market leader's weaknesses")
            st.write("- Focus on specific subcategory dominance")
            st.write("- Improve price-value proposition")
        else:
            st.info("🚀 **Growth Strategy**")
            st.write("- Focus on niche market segments")
            st.write("- Aggressive pricing and customer acquisition")
            st.write("- Build market share in high-opportunity categories")
    
    with col2:
        st.write("**Competitive Intelligence Summary**")
        
        # Key competitive insights
        total_market_size = market_df["Revenue"].sum()
        growth_categories = len([g for g in growth_analysis.values() if g > 0])
        declining_categories = len([g for g in growth_analysis.values() if g < 0])
        
        insights = [
            f"📊 Total addressable market: ₹{total_market_size:,.0f}",
            f"📈 Growing subcategories: {growth_categories}",
            f"📉 Declining subcategories: {declining_categories}",
            f"💰 Average pricing gap vs leader: {opportunities_df['Price Gap %'].mean():.1f}%",
            f"🎯 Top opportunity subcategory: {opportunities_df.nlargest(1, 'Revenue Opportunity').iloc[0]['Subcategory']}"
        ]
        
        for insight in insights:
            st.write(insight)
        
        # Threat and opportunity matrix
        st.write("**🎯 Priority Action Items:**")
        
        # High opportunity categories
        high_opp = opportunities_df.nlargest(2, "Revenue Opportunity")
        for _, row in high_opp.iterrows():
            st.write(f"• **{row['Subcategory']}**: ₹{row['Revenue Opportunity']:,.0f} pricing opportunity")
        
        # Fast growing categories
        fast_growth = growth_df.nlargest(2, "Monthly Growth Rate")
        for _, row in fast_growth.iterrows():
            st.write(f"• **{row['Subcategory']}**: {row['Monthly Growth Rate']:.1f}% monthly growth - invest more")
        
        # Market share gaps
        competitors_ahead = market_df[market_df["Market Share %"] > our_market_share]
        if len(competitors_ahead) > 0:
            biggest_competitor = competitors_ahead.iloc[0]
            share_gap = biggest_competitor["Market Share %"] - our_market_share
            st.write(f"• Target **{biggest_competitor['Company']}**: {share_gap:.1f}% market share gap to close")
    
    st.info("""
    This Market Intelligence Dashboard provides insights into:
    1. Competitive market share analysis and positioning
    2. Market trends and growth opportunity identification
    3. Pricing intelligence and optimization opportunities
    4. Strategic recommendations based on market position
    
    Use the filters to analyze specific subcategories and time periods.
    Note: Competitive data is simulated for demonstration purposes.
    """)

# Question 28: Cross-selling & Upselling Dashboard
with tab28:

    st.header("Question 28: Cross-selling & Upselling Analytics")
    
    # Sample data for cross-selling analysis
    np.random.seed(42)
    
    # Generate cross-selling data
    q28_orders = orders.copy()
    q28_orders["basket_size"] = np.random.randint(1, 8, len(q28_orders))
    q28_orders["cross_sell_items"] = np.random.randint(0, 4, len(q28_orders))
    q28_orders["upsell_revenue"] = np.random.uniform(0, 500, len(q28_orders))
    q28_orders["customer_ltv"] = np.random.uniform(1000, 10000, len(q28_orders))
    q28_orders["purchase_frequency"] = np.random.randint(1, 24, len(q28_orders))
    
    # Create product affinity matrix
    product_categories = ["Electronics", "Clothing", "Home", "Books", "Sports", "Beauty"]
    q28_orders["primary_category"] = np.random.choice(product_categories, len(q28_orders))
    q28_orders["secondary_category"] = np.random.choice(product_categories, len(q28_orders))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cross-selling Performance")
        
        # Cross-selling metrics by category
        cross_sell_metrics = q28_orders.groupby("primary_category").agg({
            "cross_sell_items": "mean",
            "basket_size": "mean", 
            "upsell_revenue": "sum",
            "final_amount_inr": "mean"
        }).round(2)
        
        cross_sell_metrics.columns = ["Avg Cross-sell Items", "Avg Basket Size", "Total Upsell Revenue", "Avg Order Value"]
        
        # Calculate cross-sell rate
        cross_sell_metrics["Cross-sell Rate %"] = (
            q28_orders.groupby("primary_category")["cross_sell_items"].apply(lambda x: (x > 0).sum() / len(x) * 100)
        ).round(1)
        
        st.dataframe(cross_sell_metrics.style.format({
            "Avg Cross-sell Items": "{:.1f}",
            "Avg Basket Size": "{:.1f}",
            "Total Upsell Revenue": "₹{:,.0f}",
            "Avg Order Value": "₹{:,.0f}",
            "Cross-sell Rate %": "{:.1f}%"
        }).background_gradient(subset=["Cross-sell Rate %"], cmap="Greens"))
        
        # Product affinity heatmap
        st.subheader("Product Affinity Matrix")
        
        # Create affinity matrix
        affinity_data = []
        for primary in product_categories:
            row = []
            for secondary in product_categories:
                if primary != secondary:
                    # Calculate how often secondary is bought with primary
                    primary_orders = q28_orders[q28_orders["primary_category"] == primary]
                    affinity_score = (primary_orders["secondary_category"] == secondary).sum() / len(primary_orders) * 100
                else:
                    affinity_score = 0
                row.append(affinity_score)
            affinity_data.append(row)
        
        affinity_df = pd.DataFrame(affinity_data, index=product_categories, columns=product_categories)
        
        fig = px.imshow(
            affinity_df,
            labels=dict(x="Secondary Category", y="Primary Category", color="Affinity %"),
            title="Product Cross-selling Affinity Matrix (%)",
            aspect="auto",
            color_continuous_scale="Blues"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Customer Segmentation for Upselling")
        
        # Customer segments based on value and frequency
        q28_customers = q28_orders.groupby("customer_id").agg({
            "final_amount_inr": "sum",
            "purchase_frequency": "mean",
            "customer_ltv": "first",
            "upsell_revenue": "sum"
        }).reset_index()
        
        # Create segments
        q28_customers["value_segment"] = pd.cut(q28_customers["final_amount_inr"], 
                                              bins=3, labels=["Low", "Medium", "High"])
        q28_customers["frequency_segment"] = pd.cut(q28_customers["purchase_frequency"], 
                                                   bins=3, labels=["Occasional", "Regular", "Frequent"])
        
        # Segment analysis
        segment_analysis = q28_customers.groupby(["value_segment", "frequency_segment"]).agg({
            "customer_id": "count",
            "upsell_revenue": "mean",
            "customer_ltv": "mean"
        }).reset_index()
        
        segment_analysis.columns = ["Value Segment", "Frequency Segment", "Customer Count", "Avg Upsell Revenue", "Avg LTV"]
        
        st.write("Customer Segmentation Matrix")
        st.dataframe(segment_analysis.style.format({
            "Customer Count": "{:,.0f}",
            "Avg Upsell Revenue": "₹{:,.0f}",
            "Avg LTV": "₹{:,.0f}"
        }))
        
        # Upselling opportunity by segment
        fig = px.scatter(
            q28_customers,
            x="purchase_frequency",
            y="final_amount_inr",
            size="upsell_revenue",
            color="customer_ltv",
            title="Upselling Opportunities by Customer Segment",
            labels={
                "purchase_frequency": "Purchase Frequency (per year)",
                "final_amount_inr": "Total Order Value (₹)",
                "upsell_revenue": "Upsell Revenue (₹)",
                "customer_ltv": "Customer LTV (₹)"
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations engine
        st.subheader("Cross-sell Recommendations")
        
        # Top product pairs
        top_pairs = []
        for i, primary in enumerate(product_categories):
            for j, secondary in enumerate(product_categories):
                if i != j and affinity_df.iloc[i, j] > 10:  # Threshold for strong affinity
                    top_pairs.append({
                        "Primary Category": primary,
                        "Recommended Category": secondary,
                        "Affinity Score": affinity_df.iloc[i, j],
                        "Revenue Potential": np.random.uniform(50000, 200000)
                    })
        
        if top_pairs:
            recommendations_df = pd.DataFrame(top_pairs).sort_values("Affinity Score", ascending=False).head(5)
            
            st.write("**Top Cross-selling Opportunities:**")
            for _, rec in recommendations_df.iterrows():
                st.write(f"• When customers buy **{rec['Primary Category']}**, recommend **{rec['Recommended Category']}** "
                        f"(Affinity: {rec['Affinity Score']:.1f}%, Revenue Potential: ₹{rec['Revenue Potential']:,.0f})")
    
    # Performance summary
    st.subheader("Cross-selling & Upselling Summary")
    
    total_cross_sell = q28_orders["cross_sell_items"].sum()
    total_upsell_revenue = q28_orders["upsell_revenue"].sum()
    avg_basket_increase = q28_orders[q28_orders["cross_sell_items"] > 0]["final_amount_inr"].mean() - q28_orders[q28_orders["cross_sell_items"] == 0]["final_amount_inr"].mean()
    cross_sell_rate = (q28_orders["cross_sell_items"] > 0).sum() / len(q28_orders) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cross-sell Items", f"{total_cross_sell:,.0f}")
    
    with col2:
        st.metric("Total Upsell Revenue", f"₹{total_upsell_revenue:,.0f}")
    
    with col3:
        st.metric("Avg Basket Increase", f"₹{avg_basket_increase:,.0f}")
    
    with col4:
        st.metric("Cross-sell Rate", f"{cross_sell_rate:.1f}%")
    
    st.info("""
    This Cross-selling & Upselling Dashboard provides insights into:
    1. Product affinity and cross-selling opportunities
    2. Customer segmentation for targeted upselling
    3. Performance metrics and revenue impact
    4. Automated recommendation engine for sales teams
    
    Use these insights to develop targeted marketing campaigns and improve sales strategies.
    """)

# Question 29: Seasonal Planning Dashboard  
with tab29:

    st.header("Question 29: Seasonal Planning & Demand Forecasting")
    
    # Sample seasonal data
    np.random.seed(42)
    
    # Generate seasonal patterns
    q29_orders = orders.copy()
    q29_orders["month"] = pd.to_datetime(q29_orders["order_date"]).dt.month
    q29_orders["quarter"] = pd.to_datetime(q29_orders["order_date"]).dt.quarter
    q29_orders["season"] = q29_orders["month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring", 
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall"
    })
    
    # Add seasonal multipliers
    seasonal_multipliers = {
        "Winter": {"Electronics": 1.8, "Clothing": 1.4, "Home": 1.2, "Books": 0.9, "Sports": 0.7, "Beauty": 1.1},
        "Spring": {"Electronics": 1.0, "Clothing": 1.3, "Home": 1.4, "Books": 1.1, "Sports": 1.2, "Beauty": 1.2}, 
        "Summer": {"Electronics": 0.9, "Clothing": 0.8, "Home": 0.8, "Books": 0.8, "Sports": 1.6, "Beauty": 1.3},
        "Fall": {"Electronics": 1.3, "Clothing": 1.5, "Home": 1.0, "Books": 1.4, "Sports": 1.1, "Beauty": 1.0}
    }
    
    q29_orders["category"] = np.random.choice(list(seasonal_multipliers["Winter"].keys()), len(q29_orders))
    
    # Apply seasonal effects
    for idx, row in q29_orders.iterrows():
        multiplier = seasonal_multipliers[row["season"]][row["category"]]
        q29_orders.at[idx, "seasonal_order_value"] = row["final_amount_inr"] * multiplier
        q29_orders.at[idx, "seasonal_demand"] = np.random.poisson(multiplier * 10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Seasonal Demand Patterns")
        
        # Seasonal analysis by category
        seasonal_analysis = q29_orders.groupby(["season", "category"]).agg({
            "seasonal_order_value": "mean",
            "seasonal_demand": "sum",
            "order_id": "count"
        }).reset_index()
        
        seasonal_analysis.columns = ["Season", "Category", "Avg Order Value", "Total Demand", "Order Count"]
        
        # Season-over-season growth
        seasonal_pivot = seasonal_analysis.pivot(index="Category", columns="Season", values="Total Demand")
        seasonal_pivot = seasonal_pivot[["Spring", "Summer", "Fall", "Winter"]]  # Order seasons
        
        fig = px.line(
            seasonal_pivot.T,
            title="Seasonal Demand Trends by Category",
            labels={"index": "Season", "value": "Total Demand", "variable": "Category"}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonality index
        st.subheader("Seasonality Index by Category")
        
        seasonality_df = q29_orders.groupby(["category", "season"])["seasonal_order_value"].mean().reset_index()
        category_avg = q29_orders.groupby("category")["seasonal_order_value"].mean()
        
        seasonality_index = []
        for category in seasonality_df["category"].unique():
            cat_data = seasonality_df[seasonality_df["category"] == category]
            avg_value = category_avg[category]
            
            for _, row in cat_data.iterrows():
                seasonality_index.append({
                    "Category": category,
                    "Season": row["season"],
                    "Seasonality Index": (row["seasonal_order_value"] / avg_value) * 100
                })
        
        seasonality_df = pd.DataFrame(seasonality_index)
        seasonality_pivot = seasonality_df.pivot(index="Category", columns="Season", values="Seasonality Index")
        

    st.dataframe(seasonality_pivot.style.format("{:.1f}").background_gradient(cmap="RdYlGn"))
    
    with col2:
        st.subheader("Demand Forecasting")
        
        # Generate forecast data
        forecast_months = pd.date_range(start="2024-01-01", end="2024-12-31", freq="MS")
        
        forecast_data = []
        for month in forecast_months:
            season = {1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 
                     6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall", 12: "Winter"}[month.month]
            
            for category in seasonal_multipliers[season].keys():
                base_demand = np.random.uniform(1000, 5000)
                seasonal_effect = seasonal_multipliers[season][category]
                trend = 1 + (month.month - 1) * 0.02  # 2% monthly growth
                noise = np.random.normal(1, 0.1)
                
                forecasted_demand = base_demand * seasonal_effect * trend * noise
                
                forecast_data.append({
                    "Month": month,
                    "Category": category,
                    "Forecasted Demand": max(0, forecasted_demand),
                    "Season": season,
                    "Seasonal Multiplier": seasonal_effect
                })
        
        forecast_df = pd.DataFrame(forecast_data)
        
        # Plot forecast
        fig = px.line(
            forecast_df,
            x="Month",
            y="Forecasted Demand",
            color="Category",
            title="12-Month Demand Forecast by Category"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Peak season preparation
        st.subheader("Peak Season Preparation")
        
        peak_seasons = forecast_df.groupby(["Category", "Season"])["Forecasted Demand"].sum().reset_index()
        peak_seasons = peak_seasons.loc[peak_seasons.groupby("Category")["Forecasted Demand"].idxmax()]
        
        st.write("**Peak Season by Category:**")
        for _, row in peak_seasons.iterrows():
            demand_increase = (row["Forecasted Demand"] / peak_seasons[peak_seasons["Category"] == row["Category"]]["Forecasted Demand"].min() - 1) * 100
            st.write(f"• **{row['Category']}**: Peak in **{row['Season']}** - {demand_increase:.0f}% above low season")
        
        # Inventory recommendations
        st.subheader("Inventory Planning Recommendations")
        
        # Calculate recommended stock levels
        inventory_recs = []
        for category in forecast_df["Category"].unique():
            cat_forecast = forecast_df[forecast_df["Category"] == category]
            
            # Next quarter forecast
            next_quarter = cat_forecast.head(3)["Forecasted Demand"].sum()
            peak_demand = cat_forecast["Forecasted Demand"].max()
            avg_demand = cat_forecast["Forecasted Demand"].mean()
            
            inventory_recs.append({
                "Category": category,
                "Recommended Stock": peak_demand * 1.2,  # 20% buffer
                "Next Quarter Demand": next_quarter,
                "Safety Stock": avg_demand * 0.3,
                "Reorder Point": avg_demand * 0.5
            })
        
        inventory_df = pd.DataFrame(inventory_recs)
        
        st.dataframe(inventory_df.style.format({
            "Recommended Stock": "{:,.0f} units",
            "Next Quarter Demand": "{:,.0f} units", 
            "Safety Stock": "{:,.0f} units",
            "Reorder Point": "{:,.0f} units"
        }))
    
    # Planning calendar
    st.subheader("Seasonal Planning Calendar")
    
    # Create planning timeline
    planning_events = []
    current_month = 1
    
    seasonal_events = {
        "Winter": ["New Year Sale", "Valentine's Day", "Winter Clearance"],
        "Spring": ["Spring Collection Launch", "Easter Sale", "Mother's Day"],
        "Summer": ["Summer Sale", "Back to School", "Vacation Season"],
        "Fall": ["Fall Fashion", "Halloween", "Black Friday", "Holiday Prep"]
    }
    
    for season, events in seasonal_events.items():
        season_months = {"Winter": [12, 1, 2], "Spring": [3, 4, 5], "Summer": [6, 7, 8], "Fall": [9, 10, 11]}
        
        for event in events:
            month = np.random.choice(season_months[season])
            planning_events.append({
                "Month": month,
                "Season": season, 
                "Event": event,
                "Prep Time": "2-3 months prior",
                "Impact": np.random.choice(["High", "Medium", "Low"])
            })
    
    planning_df = pd.DataFrame(planning_events).sort_values("Month")
    
    fig = px.timeline(
        planning_df,
        x_start="Month",
        x_end="Month", 
        y="Event",
        color="Impact",
        title="Seasonal Planning Timeline"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    This Seasonal Planning Dashboard provides insights into:
    1. Historical seasonal demand patterns and trends
    2. Category-specific seasonality indices and forecasts
    3. Inventory planning and stock level recommendations
    4. Seasonal event calendar and preparation timeline
    
    Use this data for better inventory management, marketing planning, and resource allocation.
    """)

# Question 30: BI Command Center (Executive Dashboard)
with tab30:

    st.header("Question 30: Executive BI Command Center")
    
    # Real-time KPI simulation
    np.random.seed(42)
    
    # Generate executive metrics
    current_date = pd.Timestamp.now()
    
    # Key business metrics
    total_revenue = orders["final_amount_inr"].sum()
    total_orders = len(orders)
    avg_order_value = orders["final_amount_inr"].mean()
    total_customers = orders["customer_id"].nunique()
    
    # Growth calculations (simulated)
    revenue_growth = np.random.uniform(5, 25)  
    order_growth = np.random.uniform(10, 30)
    customer_growth = np.random.uniform(8, 20)
    
    # Real-time alerts simulation
    alerts = [
        {"Type": "Revenue", "Message": "Q4 revenue target 95% achieved", "Priority": "Medium", "Time": "2 hours ago"},
        {"Type": "Inventory", "Message": "Low stock alert: Electronics category", "Priority": "High", "Time": "1 hour ago"},
        {"Type": "Performance", "Message": "Customer satisfaction dipped to 4.2", "Priority": "High", "Time": "30 min ago"},
        {"Type": "Opportunity", "Message": "Upselling opportunity identified in Books", "Priority": "Low", "Time": "15 min ago"}
    ]
    
    # Executive Summary Cards
    st.subheader("🎯 Executive KPI Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Revenue", 
            f"₹{total_revenue:,.0f}",
            delta=f"{revenue_growth:.1f}% vs last quarter"
        )
    
    with col2:
        st.metric(
            "Total Orders",
            f"{total_orders:,.0f}",
            delta=f"{order_growth:.1f}% vs last quarter"
        )
    
    with col3:
        st.metric(
            "Average Order Value",
            f"₹{avg_order_value:,.0f}",
            delta="₹150 vs last quarter"
        )
    
    with col4:
        st.metric(
            "Total Customers",
            f"{total_customers:,.0f}",
            delta=f"{customer_growth:.1f}% vs last quarter"
        )
    
    # Real-time alerts
    st.subheader("🚨 Real-time Business Alerts")
    
    for alert in alerts:
        priority_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
        st.write(f"{priority_color[alert['Priority']]} **{alert['Type']}**: {alert['Message']} _{alert['Time']}_")
    
    # Executive charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Revenue Trend (Last 12 Months)")
        
        # Generate monthly revenue data
        months = pd.date_range(start="2024-01-01", end="2024-12-31", freq="MS")
        monthly_revenue = []
        
        for i, month in enumerate(months):
            base_revenue = 5000000  # 5M base
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 12)  # Seasonal pattern
            growth_factor = 1 + 0.02 * i  # 2% monthly growth
            noise = np.random.uniform(0.9, 1.1)
            
            revenue = base_revenue * seasonal_factor * growth_factor * noise
            monthly_revenue.append({"Month": month, "Revenue": revenue})
        
        revenue_df = pd.DataFrame(monthly_revenue)
        
        fig = px.line(
            revenue_df,
            x="Month",
            y="Revenue",
            title="Monthly Revenue Trend",
            line_shape="spline"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🎯 Goal vs Achievement")
        
        # Goals tracking
        goals_data = [
            {"Metric": "Revenue", "Target": 100000000, "Achieved": total_revenue, "Progress": min(100, total_revenue/100000000*100)},
            {"Metric": "Orders", "Target": 50000, "Achieved": total_orders, "Progress": min(100, total_orders/50000*100)},
            {"Metric": "Customers", "Target": 25000, "Achieved": total_customers, "Progress": min(100, total_customers/25000*100)},
            {"Metric": "Satisfaction", "Target": 4.5, "Achieved": 4.2, "Progress": min(100, 4.2/4.5*100)}
        ]
        
        for goal in goals_data:
            progress_color = "🟢" if goal["Progress"] >= 90 else "🟡" if goal["Progress"] >= 70 else "🔴"
            st.write(f"{progress_color} **{goal['Metric']}**: {goal['Progress']:.1f}% ({goal['Achieved']:,.0f}/{goal['Target']:,.0f})")
    
    with col2:
        st.subheader("🗺️ Geographic Performance Heatmap")
        
        # Generate state-wise performance
        indian_states = ["Maharashtra", "Karnataka", "Tamil Nadu", "Delhi", "Gujarat", 
                        "Rajasthan", "Uttar Pradesh", "West Bengal", "Punjab", "Haryana"]
        
        state_performance = []
        for state in indian_states:
            performance_score = np.random.uniform(70, 95)
            revenue_share = np.random.uniform(5, 20)
            
            state_performance.append({
                "State": state,
                "Performance Score": performance_score,
                "Revenue Share %": revenue_share,
                "Status": "Excellent" if performance_score >= 90 else "Good" if performance_score >= 80 else "Needs Attention"
            })
        
        performance_df = pd.DataFrame(state_performance)
        

        # Using bar chart instead of treemap to avoid compatibility issues
        fig = px.bar(
            performance_df.sort_values("Performance Score", ascending=True),
            x="Performance Score",
            y="State",
            color="Revenue Share %",
            orientation="h",
            title="State-wise Performance & Revenue Share",
            color_continuous_scale="RdYlGn"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("⚡ Quick Actions")
        
        # Quick action buttons
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("📊 Generate Report"):
                st.success("Executive report generation initiated!")
            
            if st.button("📧 Send Alert"):
                st.info("Alert sent to management team!")
        
        with col_b:
            if st.button("🔄 Refresh Data"):
                st.success("Data refreshed successfully!")
                
            if st.button("⚙️ Settings"):
                st.info("Dashboard settings opened!")
    
    # Strategic insights
    st.subheader("💡 Strategic Business Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**🎯 Growth Opportunities**")
        opportunities = [
            "Electronics showing 25% growth - expand inventory",
            "Delhi market underperforming - investigate causes", 
            "Customer acquisition cost decreasing - scale marketing",
            "Mobile orders up 40% - optimize mobile experience"
        ]
        
        for opp in opportunities:
            st.write(f"• {opp}")
    
    with col2:
        st.write("**⚠️ Risk Factors**")
        risks = [
            "Customer satisfaction trending down",
            "Competition increasing in Maharashtra", 
            "Supply chain delays in 2 categories",
            "Seasonal inventory buildup needed"
        ]
        
        for risk in risks:
            st.write(f"• {risk}")
    
    with col3:
        st.write("**🚀 Recommended Actions**")
        actions = [
            "Launch customer retention program",
            "Increase marketing spend in Delhi",
            "Negotiate backup suppliers", 
            "Start holiday season preparation"
        ]
        
        for action in actions:
            st.write(f"• {action}")
    
    # Market comparison
    st.subheader("🏆 Market Position Analysis")
    
    market_position = {
        "Our Company": {"Market Share": 15.2, "Growth Rate": 18.5, "Customer Satisfaction": 4.2},
        "Competitor A": {"Market Share": 22.1, "Growth Rate": 12.3, "Customer Satisfaction": 4.1}, 
        "Competitor B": {"Market Share": 18.7, "Growth Rate": 8.9, "Customer Satisfaction": 4.3},
        "Competitor C": {"Market Share": 12.4, "Growth Rate": 15.2, "Customer Satisfaction": 3.9},
        "Others": {"Market Share": 31.6, "Growth Rate": 5.1, "Customer Satisfaction": 3.8}
    }
    
    position_df = pd.DataFrame(market_position).T.reset_index()
    position_df.columns = ["Company", "Market Share %", "Growth Rate %", "Customer Satisfaction"]
    
    fig = px.scatter(
        position_df,
        x="Market Share %",
        y="Growth Rate %",
        size="Customer Satisfaction",
        color="Company",
        title="Competitive Position Matrix",
        hover_data=["Customer Satisfaction"]
    )
    
    # Add quadrant labels
    fig.add_hline(y=position_df["Growth Rate %"].mean(), line_dash="dash", line_color="gray")
    fig.add_vline(x=position_df["Market Share %"].mean(), line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Executive summary
    st.subheader("📋 Executive Summary")
    
    summary_points = [
        f"🎯 **Performance**: Achieved {revenue_growth:.1f}% revenue growth, {order_growth:.1f}% order growth",
        f"📊 **Market Position**: 3rd largest player with {market_position['Our Company']['Market Share']:.1f}% market share",
        f"🚀 **Growth**: Leading growth rate at {market_position['Our Company']['Growth Rate']:.1f}% vs industry average 10.1%",
        f"⭐ **Customer Experience**: Satisfaction at {market_position['Our Company']['Customer Satisfaction']:.1f}/5.0, need improvement",
        f"💰 **Financial**: Revenue target 95% achieved with strong Q4 performance expected",
        f"🎪 **Opportunities**: Electronics expansion, Delhi market potential, mobile optimization"
    ]
    
    for point in summary_points:
        st.write(point)
    