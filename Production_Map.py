import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from datetime import datetime

st.set_page_config(layout="wide", page_title="Oil & Gas Production Map", 
                   initial_sidebar_state="expanded")

st.title("Oil and Gas Production Map")

# Helper function to check and fix column names (if they're joined without spaces)
def check_and_fix_column_names(df):
    # Check for column names that might be joined together
    # Common patterns in the dataset
    patterns = [
        ('LEASE_NUMBER', r'LEASE_?NUMBER'),
        ('PROD_MONTH', r'PROD_?MONTH'),
        ('PROD_YEAR', r'PROD_?YEAR'),
        ('LEASE_OIL_PROD', r'LEASE_?OIL_?PROD'),
        ('LEASE_GWG_PROD', r'LEASE_?GWG_?PROD'),
        ('LEASE_CONDN_PROD', r'LEASE_?CONDN_?PROD'),
        ('LEASE_OWG_PROD', r'LEASE_?OWG_?PROD'),
        ('LEASE_WTR_PROD', r'LEASE_?WTR_?PROD'),
        ('SURF_LATITUDE', r'SURF_?LATITUDE'),
        ('SURF_LONGITUDE', r'SURF_?LONGITUDE'),
        ('WATER_DEPTH', r'WATER_?DEPTH'),
        ('WELL_BORE_TVD', r'WELL_?BORE_?TVD'),
        ('well_count', r'well_?count'),
        ('LEASE_PROD_COMP', r'LEASE_?PROD_?COMP'),
    ]
    
    renamed_columns = {}
    for standard_name, pattern in patterns:
        for col in df.columns:
            if re.match(pattern, col, re.IGNORECASE) and col != standard_name:
                renamed_columns[col] = standard_name
    
    if renamed_columns:
        return df.rename(columns=renamed_columns)
    return df

# Load data
@st.cache_data
def load_data():
    try:
        # First try the data folder path
        df_part1 = pd.read_csv('data/joined_data_agg_part1.csv')
        df_part2 = pd.read_csv('data/joined_data_agg_part2.csv')
        df = pd.concat([df_part1, df_part2], ignore_index=True)
    except FileNotFoundError:
        try:
            # If not found, try the root directory
            df_part1 = pd.read_csv('data/joined_data_agg_part1.csv')
            df_part2 = pd.read_csv('data/joined_data_agg_part2.csv')
            df = pd.concat([df_part1, df_part2], ignore_index=True)
        except FileNotFoundError:
            st.error("Could not find 'joined_data_agg.csv' in either the 'data' folder or the root directory. Please check the file path.")
            return pd.DataFrame()
    
    # Fix column names if needed
    df = check_and_fix_column_names(df)
    
    # Create DATE column from PROD_YEAR and PROD_MONTH
    if 'PROD_YEAR' in df.columns and 'PROD_MONTH' in df.columns:
        # Create a proper date from year and month
        df['DATE'] = pd.to_datetime(df['PROD_YEAR'].astype(str) + '-' + 
                                    df['PROD_MONTH'].astype(str) + '-01')
        df['YEAR'] = df['PROD_YEAR']
    elif 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'])
        df['YEAR'] = df['DATE'].dt.year
    else:
        st.warning("Neither DATE nor PROD_YEAR/PROD_MONTH columns found. Time series analysis will not be available.")
    
    return df

# Load the data and display a spinner while loading
with st.spinner('Loading data...'):
    df = load_data()

if df.empty:
    st.error("No data available. Please check your data file.")
    st.stop()

# Sidebar Filters
st.sidebar.header("Map Filters")

# Get min and max years if year column exists
if 'YEAR' in df.columns:
    min_year = int(df['YEAR'].min())
    max_year = int(df['YEAR'].max())
    
    # Add year range slider
    year_range = st.sidebar.slider(
        "Select Production Period",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    # Filter data based on selected years
    filtered_df = df[(df['YEAR'] >= year_range[0]) & (df['YEAR'] <= year_range[1])]
else:
    filtered_df = df

# Select which production type to visualize
production_type = st.sidebar.selectbox(
    "Select Production Type",
    ['Oil', 'Gas'],
    index=0
)

# Map production type to column name
production_map = {
    'Oil': 'LEASE_OIL_PROD',
    'Gas': 'LEASE_GWG_PROD'
}

production_field = production_map[production_type]

# Data preparation - group by lease and location 
@st.cache_data
def prepare_data(df, production_field):
    # Create a list of all expected columns
    expected_columns = [
        'LEASE_NUMBER', 'SURF_LATITUDE', 'SURF_LONGITUDE',
        'LEASE_OIL_PROD', 'LEASE_GWG_PROD', 'LEASE_CONDN_PROD', 
        'LEASE_OWG_PROD', 'LEASE_WTR_PROD', 'well_count',
        'WATER_DEPTH', 'WELL_BORE_TVD'
    ]
    
    # Check which columns actually exist in the dataframe
    agg_dict = {}
    for col in df.columns:
        if col in ['LEASE_OIL_PROD', 'LEASE_GWG_PROD', 'LEASE_CONDN_PROD', 'LEASE_OWG_PROD', 'LEASE_WTR_PROD']:
            agg_dict[col] = 'sum'
        elif col == 'well_count':
            agg_dict[col] = 'max'
        elif col in ['WATER_DEPTH', 'WELL_BORE_TVD']:
            agg_dict[col] = 'mean'
    
    # Make sure we have the required groupby columns
    groupby_cols = []
    for col in ['LEASE_NUMBER', 'SURF_LATITUDE', 'SURF_LONGITUDE']:
        if col in df.columns:
            groupby_cols.append(col)
    
    if not groupby_cols:
        st.error("Required columns for grouping (LEASE_NUMBER, SURF_LATITUDE, SURF_LONGITUDE) not found in the dataset.")
        return pd.DataFrame()
    
    # Group by lease and location to get total production per lease
    lease_summary = df.groupby(groupby_cols).agg(agg_dict).reset_index()
    
    # Remove any rows with missing coordinates
    coord_cols = [col for col in ['SURF_LATITUDE', 'SURF_LONGITUDE'] if col in lease_summary.columns]
    if coord_cols:
        lease_summary = lease_summary.dropna(subset=coord_cols)
    
    return lease_summary

lease_summary = prepare_data(filtered_df, production_field)

# Color scales for different production types
colorscales = {
    'Oil': 'Viridis',
    'Gas': 'Plasma'
}

# Create a function to generate maps
def create_production_map(data, production_field, title, colorscale):
    # Normalize to reasonable marker sizes
    max_size = data[production_field].max()
    if max_size > 0:
        marker_sizes = 5 + (data[production_field] / max_size) * 25
    else:
        marker_sizes = 5
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter mapbox trace
    fig.add_trace(go.Scattergeo(
        lon=data['SURF_LONGITUDE'],
        lat=data['SURF_LATITUDE'],
        text=data['LEASE_NUMBER'],
        marker=dict(
            size=marker_sizes,  # Use calculated sizes
            color=data[production_field],
            colorscale=colorscale,
            colorbar_title=production_field.replace('LEASE_', '').replace('_PROD', '') + " Production",
            opacity=0.7,
            cmin=data[production_field].min(),
            cmax=data[production_field].max(),
        ),
        hoverinfo='text',
        hovertext=[
            f"Lease: {row['LEASE_NUMBER']}<br>" +
            f"Oil: {row['LEASE_OIL_PROD']:,.0f} bbl<br>" +
            f"Gas: {row['LEASE_GWG_PROD']:,.0f} MCF<br>" +
            f"Wells: {row['well_count']}<br>" +
            f"Water Depth: {row['WATER_DEPTH']:.1f} ft<br>" +
            f"Well Depth: {row['WELL_BORE_TVD']:.1f} ft"
            for _, row in data.iterrows()
        ]
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        geo=dict(
            scope='usa',
            projection_type='albers usa',
            showland=True,
            landcolor='rgb(240, 240, 240)',
            showocean=True,
            oceancolor='rgb(198, 236, 255)',
            showlakes=True,
            lakecolor='rgb(198, 236, 255)',
            showcoastlines=True,
            coastlinecolor='rgb(80, 80, 80)',
            showframe=False
        ),
        height=600,
        width=900,
        margin={"r":0, "t":50, "l":0, "b":0}
    )
    
    return fig

# Display the map
st.header(f"{production_type} Production by Lease")

if lease_summary.empty:
    st.warning("No data available with the current filters.")
else:
    # Create production map
    fig = create_production_map(
        lease_summary, 
        production_field, 
        f'{production_type} Production Hotspots by Lease', 
        colorscale=colorscales[production_type]
    )
    
    # Display the map
    st.plotly_chart(fig, use_container_width=True)
    
    # Print some statistics about the production data
    st.subheader("Production Statistics")
    
    # Custom CSS to make all stats uniform font size
    st.markdown("""
    <style>
    .stat-label {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 400;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lease_count = f"{lease_summary.shape[0]:,}"
        st.markdown(f"<div class='stat-label'>Total Leases</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stat-value'>{lease_count}</div>", unsafe_allow_html=True)
        
        well_count = f"{lease_summary['well_count'].sum():,.0f}"
        st.markdown(f"<div class='stat-label'>Total Wells</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stat-value'>{well_count}</div>", unsafe_allow_html=True)
    
    with col2:
        oil_prod = f"{lease_summary['LEASE_OIL_PROD'].sum():,.0f} barrels"
        st.markdown(f"<div class='stat-label'>Total Oil Production</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stat-value'>{oil_prod}</div>", unsafe_allow_html=True)
        
        water_depth = f"{lease_summary['WATER_DEPTH'].mean():.1f} feet"
        st.markdown(f"<div class='stat-label'>Average Water Depth</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stat-value'>{water_depth}</div>", unsafe_allow_html=True)
    
    with col3:
        gas_prod = f"{lease_summary['LEASE_GWG_PROD'].sum():,.0f} MCF"
        st.markdown(f"<div class='stat-label'>Total Gas Production</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stat-value'>{gas_prod}</div>", unsafe_allow_html=True)
        
        well_depth = f"{lease_summary['WELL_BORE_TVD'].mean():.1f} feet"
        st.markdown(f"<div class='stat-label'>Average Well Depth</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stat-value'>{well_depth}</div>", unsafe_allow_html=True)
    
    # Show top producing leases
    st.subheader(f"Top 10 {production_type} Producing Leases")
    top_producers = lease_summary.sort_values(production_field, ascending=False).head(10)
    
    # Format the DataFrame for display
    display_df = top_producers[['LEASE_NUMBER', 'LEASE_OIL_PROD', 'LEASE_GWG_PROD', 'well_count', 'WATER_DEPTH']].copy()
    display_df.columns = ['Lease Number', 'Oil (bbl)', 'Gas (MCF)', 'Well Count', 'Water Depth (ft)']
    
    # Format numbers with commas
    display_df['Oil (bbl)'] = display_df['Oil (bbl)'].map('{:,.0f}'.format)
    display_df['Gas (MCF)'] = display_df['Gas (MCF)'].map('{:,.0f}'.format)
    
    st.dataframe(display_df)

# Add information about the Time Series Analysis page
st.sidebar.markdown("---")
st.sidebar.info("👉 Navigate to the Time Series Analysis page to view detailed lease performance over time.")