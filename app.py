import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Market EDA Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .section-header {
        color: #1f77b4;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0 10px 0;
        border-bottom: 2px solid #e6f3ff;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Color palette for consistent theming
colors = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9800',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

# Helper functions
@st.cache_data
def load_data(file_path):
    """Load and preprocess stock data"""
    try:
        data = pd.read_csv(file_path)
        
        # Try different date column names
        date_cols = ['Date', 'date', 'DATE', 'Timestamp', 'timestamp']
        date_col = None
        for col in date_cols:
            if col in data.columns:
                date_col = col
                break
        
        if date_col is None:
            st.error("No date column found. Expected columns: Date, date, DATE, Timestamp, or timestamp")
            return None
            
        data[date_col] = pd.to_datetime(data[date_col])
        data = data.sort_values(date_col).set_index(date_col)
        
        # Forward fill numeric columns if they exist
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in numeric_cols if col in data.columns]
        if available_cols:
            data[available_cols] = data[available_cols].ffill()
        
        # Create return column if Close price exists
        close_cols = ['Close', 'close', 'CLOSE']
        close_col = None
        for col in close_cols:
            if col in data.columns:
                close_col = col
                break
                
        if close_col:
            data['Return'] = data[close_col].pct_change()
            # Standardize column name
            if close_col != 'Close':
                data['Close'] = data[close_col]
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def load_uploaded_data(uploaded_file):
    """Load data from uploaded file"""
    try:
        data = pd.read_csv(uploaded_file)
        
        # Try different date column names
        date_cols = ['Date', 'date', 'DATE', 'Timestamp', 'timestamp']
        date_col = None
        for col in date_cols:
            if col in data.columns:
                date_col = col
                break
        
        if date_col is None:
            st.error("No date column found. Expected columns: Date, date, DATE, Timestamp, or timestamp")
            return None
            
        data[date_col] = pd.to_datetime(data[date_col])
        data = data.sort_values(date_col).set_index(date_col)
        
        # Forward fill numeric columns if they exist
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in numeric_cols if col in data.columns]
        if available_cols:
            data[available_cols] = data[available_cols].ffill()
        
        # Create return column if Close price exists
        close_cols = ['Close', 'close', 'CLOSE']
        close_col = None
        for col in close_cols:
            if col in data.columns:
                close_col = col
                break
                
        if close_col:
            data['Return'] = data[close_col].pct_change()
            # Standardize column name
            if close_col != 'Close':
                data['Close'] = data[close_col]
        
        return data
    except Exception as e:
        st.error(f"Error loading uploaded data: {str(e)}")
        return None

def adf_test_results(series, title=''):
    """Perform Augmented Dickey-Fuller test"""
    try:
        result = adfuller(series.dropna(), autolag='AIC')
        return {
            'title': title,
            'adf_stat': result[0],
            'p_value': result[1],
            'lags_used': result[2],
            'n_obs': result[3],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    except:
        return None

def create_price_volume_chart(data, stock_name):
    """Create price and volume chart"""
    has_volume = 'Volume' in data.columns and not data['Volume'].isna().all()
    
    if has_volume:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'{stock_name} Stock Price', 'Trading Volume'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                name='Close Price',
                line=dict(color=colors['primary'], width=2),
                hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Volume chart
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors['secondary'],
                opacity=0.7,
                hovertemplate='<b>Date</b>: %{x}<br><b>Volume</b>: %{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text=f"{stock_name} Price and Volume Analysis",
            title_x=0.5,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
    else:
        # Only price chart if no volume data
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                name='Close Price',
                line=dict(color=colors['primary'], width=2),
                hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
            )
        )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            title_text=f"{stock_name} Stock Price",
            title_x=0.5,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title="Date",
            yaxis_title="Price ($)"
        )
    
    # Common styling for both cases
    fig.update_xaxes(
        gridcolor='lightgray',
        gridwidth=0.5,
        showgrid=True
    )
    fig.update_yaxes(
        gridcolor='lightgray',
        gridwidth=0.5,
        showgrid=True
    )
    
    return fig

def create_rolling_stats_chart(data, stock_name, window=30):
    """Create rolling mean and volatility bands chart"""
    rolling_mean = data['Close'].rolling(window).mean()
    rolling_std = data['Close'].rolling(window).std()
    
    fig = go.Figure()
    
    # Upper and lower bands
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=rolling_mean + 2*rolling_std,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=rolling_mean - 2*rolling_std,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.2)',
            name='±2σ Bands',
            hoverinfo='skip'
        )
    )
    
    # Close price
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'],
            name='Close Price',
            line=dict(color=colors['primary'], width=1),
            opacity=0.7
        )
    )
    
    # Rolling mean
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=rolling_mean,
            name=f'{window}-Day Rolling Mean',
            line=dict(color=colors['danger'], width=2)
        )
    )
    
    fig.update_layout(
        title=f'{stock_name} Rolling Mean & Volatility Bands',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_returns_distribution_chart(data, stock_name):
    """Create returns distribution chart"""
    returns = data['Return'].dropna()
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=50,
            name='Distribution',
            marker_color=colors['primary'],
            opacity=0.7,
            yaxis='y'
        )
    )
    
    # Add normal distribution overlay
    x_range = np.linspace(returns.min(), returns.max(), 100)
    normal_dist = (1/(returns.std() * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - returns.mean()) / returns.std())**2)
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=normal_dist * len(returns) * (returns.max() - returns.min()) / 50,
            mode='lines',
            name='Normal Distribution',
            line=dict(color=colors['danger'], width=2),
            yaxis='y'
        )
    )
    
    fig.update_layout(
        title=f'{stock_name} Daily Returns Distribution',
        xaxis_title='Daily Return',
        yaxis_title='Frequency',
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_seasonal_decomposition_chart(data, stock_name, period=90):
    """Create seasonal decomposition chart"""
    try:
        decomposition = seasonal_decompose(data['Close'], model='multiplicative', period=period)
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.05
        )
        
        components = [
            (decomposition.observed, 'Original', colors['primary']),
            (decomposition.trend, 'Trend', colors['success']),
            (decomposition.seasonal, 'Seasonal', colors['warning']),
            (decomposition.resid, 'Residual', colors['secondary'])
        ]
        
        for i, (component, name, color) in enumerate(components, 1):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=component,
                    name=name,
                    line=dict(color=color, width=1),
                    showlegend=False
                ),
                row=i, col=1
            )
        
        fig.update_layout(
            height=800,
            title_text=f'{stock_name} Seasonal Decomposition',
            title_x=0.5,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating seasonal decomposition: {str(e)}")
        return None

def create_correlation_heatmap(data, stock_name):
    """Create correlation heatmap"""
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr_matrix = data[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{stock_name} Correlation Matrix',
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_seasonality_charts(data, stock_name):
    """Create monthly and day-of-week seasonality charts"""
    data_copy = data.copy()
    data_copy['Month'] = data_copy.index.month
    data_copy['DayOfWeek'] = data_copy.index.dayofweek
    
    # Monthly seasonality
    monthly_avg = data_copy.groupby('Month')['Return'].mean()
    
    fig1 = go.Figure(data=[
        go.Bar(
            x=monthly_avg.index,
            y=monthly_avg.values,
            marker_color=colors['primary'],
            hovertemplate='<b>Month</b>: %{x}<br><b>Avg Return</b>: %{y:.4f}<extra></extra>'
        )
    ])
    
    fig1.update_layout(
        title=f'{stock_name} Average Monthly Returns',
        xaxis_title='Month',
        yaxis_title='Average Return',
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Day of week seasonality
    dow_avg = data_copy.groupby('DayOfWeek')['Return'].mean()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    fig2 = go.Figure(data=[
        go.Bar(
            x=day_names,
            y=dow_avg.values,
            marker_color=colors['secondary'],
            hovertemplate='<b>Day</b>: %{x}<br><b>Avg Return</b>: %{y:.4f}<extra></extra>'
        )
    ])
    
    fig2.update_layout(
        title=f'{stock_name} Average Returns by Day of Week',
        xaxis_title='Day of Week',
        yaxis_title='Average Return',
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig1, fig2

# Main application
def main():
    st.title("📈 Stock Market Exploratory Data Analysis Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🔧 Dashboard Controls")
    
    # Data loading options
    st.sidebar.markdown("### 📂 Data Source")
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Upload CSV File", "Use Sample Data"]
    )
    
    data = None
    selected_stock = "Custom Data"
    
    if data_source == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file with stock data. Should contain Date and Close price columns."
        )
        
        if uploaded_file is not None:
            data = load_uploaded_data(uploaded_file)
            selected_stock = uploaded_file.name.replace('.csv', '').title()
        else:
            st.info("👆 Please upload a CSV file to begin analysis.")
            return
            
    else:
        # Stock selection for sample data
        stock_options = {
            'Amazon': 'data/merged-data/amazon-merged.csv',
            'Apple': 'data/merged-data/apple-merged.csv',
            'Google': 'data/merged-data/google-merged.csv',
            'Microsoft': 'data/merged-data/microsoft-merged.csv',
            'NVIDIA': 'data/merged-data/nvidia-merged.csv'
        }
        
        selected_stock = st.sidebar.selectbox(
            "Select Stock:",
            options=list(stock_options.keys()),
            index=0
        )
        
        # Load data
        data_path = stock_options[selected_stock]
        data = load_data(data_path)
        
        if data is None:
            st.warning(f"⚠️ Could not load {selected_stock} data from {data_path}")
            st.info("💡 Try uploading your own CSV file instead!")
            return
    
    if data is None:
        st.error("Failed to load data. Please check the file format and try again.")
        return
    
    # Check if required columns exist
    if 'Close' not in data.columns:
        st.error("❌ No 'Close' price column found in the data. Please ensure your CSV has a Close price column.")
        st.info("Expected columns: Date (index), Close, and optionally: Open, High, Low, Volume")
        return
    
    # Parameters
    st.sidebar.markdown("### 📊 Analysis Parameters")
    rolling_window = st.sidebar.slider("Rolling Window (days)", 10, 100, 30)
    seasonal_period = st.sidebar.slider("Seasonal Period (days)", 30, min(252, len(data)//2), min(90, len(data)//3))
    
    # Display data info
    st.sidebar.markdown("### 📋 Data Info")
    st.sidebar.write(f"**Rows:** {len(data):,}")
    st.sidebar.write(f"**Columns:** {len(data.columns)}")
    st.sidebar.write(f"**Date Range:** {data.index.min().date()} to {data.index.max().date()}")
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📅 Data Points</h4>
            <h2>{len(data):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>💰 Latest Price</h4>
            <h2>${data['Close'].iloc[-1]:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Total Return</h4>
            <h2>{total_return:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if 'Return' in data.columns:
            avg_daily_return = data['Return'].mean() * 100
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Avg Daily Return</h4>
                <h2>{avg_daily_return:.3f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Price Range</h4>
                <h2>${data['Close'].min():.2f} - ${data['Close'].max():.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Price & Volume", 
        "📊 Statistical Analysis", 
        "🔄 Seasonal Patterns", 
        "🧮 Correlation Analysis",
        "⚡ Stationarity Tests",
        "📋 Data Summary"
    ])
    
    with tab1:
        st.markdown('<div class="section-header">Price and Volume Analysis</div>', unsafe_allow_html=True)
        
        # Price and volume chart
        price_vol_fig = create_price_volume_chart(data, selected_stock)
        st.plotly_chart(price_vol_fig, use_container_width=True)
        
        # Rolling statistics
        st.markdown("### Rolling Statistics and Volatility Bands")
        rolling_fig = create_rolling_stats_chart(data, selected_stock, rolling_window)
        st.plotly_chart(rolling_fig, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="section-header">Statistical Analysis</div>', unsafe_allow_html=True)
        
        if 'Return' in data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Returns distribution
                st.markdown("### Daily Returns Distribution")
                dist_fig = create_returns_distribution_chart(data, selected_stock)
                st.plotly_chart(dist_fig, use_container_width=True)
            
            with col2:
                # Returns statistics
                st.markdown("### Returns Statistics")
                returns_stats = data['Return'].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99])
                
                stats_df = pd.DataFrame({
                    'Metric': returns_stats.index,
                    'Value': returns_stats.values
                })
                
                st.dataframe(stats_df, hide_index=True)
        else:
            st.warning("Returns analysis requires daily return data.")
        
        # Seasonal decomposition
        if len(data) >= seasonal_period * 2:
            st.markdown("### Seasonal Decomposition")
            decomp_fig = create_seasonal_decomposition_chart(data, selected_stock, seasonal_period)
            if decomp_fig:
                st.plotly_chart(decomp_fig, use_container_width=True)
        else:
            st.warning(f"Seasonal decomposition requires at least {seasonal_period * 2} data points.")
    
    with tab3:
        st.markdown('<div class="section-header">Seasonal Patterns</div>', unsafe_allow_html=True)
        
        if 'Return' in data.columns:
            col1, col2 = st.columns(2)
            
            monthly_fig, dow_fig = create_seasonality_charts(data, selected_stock)
            
            with col1:
                st.plotly_chart(monthly_fig, use_container_width=True)
            
            with col2:
                st.plotly_chart(dow_fig, use_container_width=True)
        else:
            st.warning("Seasonal pattern analysis requires daily return data.")
    
    with tab4:
        st.markdown('<div class="section-header">Correlation Analysis</div>', unsafe_allow_html=True)
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_fig = create_correlation_heatmap(data, selected_stock)
            st.plotly_chart(corr_fig, use_container_width=True)
        else:
            st.warning("Correlation analysis requires multiple numeric columns.")
    
    with tab5:
        st.markdown('<div class="section-header">Stationarity Tests</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        # ADF test for Close price
        close_adf = adf_test_results(data['Close'], 'Close Price')
        
        with col1:
            if close_adf:
                st.markdown("### Close Price Stationarity")
                st.markdown(f"**ADF Statistic:** {close_adf['adf_stat']:.4f}")
                st.markdown(f"**p-value:** {close_adf['p_value']:.4f}")
                st.markdown(f"**Is Stationary:** {'✅ Yes' if close_adf['is_stationary'] else '❌ No'}")
        
        # ADF test for Returns
        if 'Return' in data.columns:
            returns_adf = adf_test_results(data['Return'], 'Returns')
            
            with col2:
                if returns_adf:
                    st.markdown("### Returns Stationarity")
                    st.markdown(f"**ADF Statistic:** {returns_adf['adf_stat']:.4f}")
                    st.markdown(f"**p-value:** {returns_adf['p_value']:.4f}")
                    st.markdown(f"**Is Stationary:** {'✅ Yes' if returns_adf['is_stationary'] else '❌ No'}")
        else:
            with col2:
                st.info("Returns stationarity test requires daily return data.")
    
    with tab6:
        st.markdown('<div class="section-header">Data Summary</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Dataset Information")
            st.write(f"**Shape:** {data.shape}")
            st.write(f"**Date Range:** {data.index.min().date()} to {data.index.max().date()}")
            st.write(f"**Missing Values:** {data.isnull().sum().sum()}")
            
            # Duplicate check
            dupes = data.index.duplicated(keep=False).sum()
            st.write(f"**Duplicate Dates:** {dupes}")
        
        with col2:
            st.markdown("### Data Quality")
            missing_data = data.isnull().sum()
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing %': (missing_data.values / len(data)) * 100
            })
            
            if len(missing_df[missing_df['Missing Count'] > 0]) > 0:
                st.dataframe(missing_df[missing_df['Missing Count'] > 0], hide_index=True)
            else:
                st.success("✅ No missing values found!")
        
        # Raw data sample
        st.markdown("### Data Sample")
        st.dataframe(data.head(10))
        
        # Show column information
        st.markdown("### Available Columns")
        col_info = pd.DataFrame({
            'Column': data.columns,
            'Data Type': [str(dtype) for dtype in data.dtypes],
            'Non-Null Count': [data[col].count() for col in data.columns]
        })
        st.dataframe(col_info, hide_index=True)

if __name__ == "__main__":
    main()