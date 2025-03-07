import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime

st.set_page_config(layout="wide", page_title="Lease Time Series Analysis", 
                   initial_sidebar_state="expanded")

st.title("Lease Time Series Analysis")

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

# Get most recent year for sorting leases by activity
if 'YEAR' in df.columns:
    most_recent_year = int(df['YEAR'].max())
    
    # Calculate total production for each lease in the most recent year for sorting
    @st.cache_data
    def get_sorted_leases(df, year):
        recent_df = df[df['YEAR'] == year]
        if recent_df.empty:
            return []
            
        # Calculate total production
        lease_activity = recent_df.groupby('LEASE_NUMBER').agg({
            'LEASE_OIL_PROD': 'sum',
            'LEASE_GWG_PROD': 'sum',
            'LEASE_OWG_PROD': 'sum' if 'LEASE_OWG_PROD' in df.columns else None
        }).reset_index()
        
        # Remove None values from the dict
        lease_activity = lease_activity.dropna(axis=1)
        
        # Calculate total hydrocarbon production
        production_cols = [col for col in ['LEASE_OIL_PROD', 'LEASE_GWG_PROD', 'LEASE_OWG_PROD'] 
                           if col in lease_activity.columns]
        
        lease_activity['TOTAL_PRODUCTION'] = lease_activity[production_cols].sum(axis=1)
        
        # Sort by total production
        lease_activity = lease_activity.sort_values('TOTAL_PRODUCTION', ascending=False)
        
        # Get ordered list of lease numbers
        sorted_leases = lease_activity['LEASE_NUMBER'].tolist()
        
        # Add any other leases that aren't in the most recent year
        all_leases = df['LEASE_NUMBER'].unique().tolist()
        for lease in all_leases:
            if lease not in sorted_leases:
                sorted_leases.append(lease)
                
        return sorted_leases
    
    sorted_leases = get_sorted_leases(df, most_recent_year)
    
    if sorted_leases:
        # Controls in sidebar
        st.sidebar.header("Lease Selection")
        
        # Create a dropdown for lease selection
        selected_lease = st.sidebar.selectbox(
            "Select Lease (Sorted by Recent Activity)",
            sorted_leases
        )
        
        # Filter data for the selected lease
        lease_df = df[df['LEASE_NUMBER'] == selected_lease].sort_values('DATE')
        
        if not lease_df.empty and 'DATE' in lease_df.columns:
            # Date range selection
            st.sidebar.header("Time Range")
            
            # Get min and max years for this lease
            min_year = int(lease_df['YEAR'].min())
            max_year = int(lease_df['YEAR'].max())
            
            # Add year range slider like main page
            year_range = st.sidebar.slider(
                "Select Production Period",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )

            # Filter data based on selected years
            date_filtered_df = lease_df[(lease_df['YEAR'] >= year_range[0]) & (lease_df['YEAR'] <= year_range[1])]
            
            # Calculate derived metrics
            if 'LEASE_GWG_PROD' in date_filtered_df.columns and 'LEASE_OWG_PROD' in date_filtered_df.columns:
                date_filtered_df['TOTAL_GAS'] = date_filtered_df['LEASE_GWG_PROD'] + date_filtered_df['LEASE_OWG_PROD']
            else:
                date_filtered_df['TOTAL_GAS'] = date_filtered_df['LEASE_GWG_PROD'] if 'LEASE_GWG_PROD' in date_filtered_df.columns else 0
            
            # Calculate GOR and WOR
            if 'LEASE_OIL_PROD' in date_filtered_df.columns:
                date_filtered_df['GOR'] = date_filtered_df['TOTAL_GAS'] / date_filtered_df['LEASE_OIL_PROD'].replace(0, np.nan)
                if 'LEASE_WTR_PROD' in date_filtered_df.columns:
                    date_filtered_df['WOR'] = date_filtered_df['LEASE_WTR_PROD'] / date_filtered_df['LEASE_OIL_PROD'].replace(0, np.nan)
            
            # Get lease metadata for the title
            well_count = "N/A"
            tvd = "N/A"
            water_depth = "N/A"
            
            if 'well_count' in date_filtered_df.columns:
                well_count = f"{date_filtered_df['well_count'].iloc[0]:.0f}" if not pd.isna(date_filtered_df['well_count'].iloc[0]) else "N/A"
            if 'WELL_BORE_TVD' in date_filtered_df.columns:
                tvd = f"{date_filtered_df['WELL_BORE_TVD'].iloc[0]:.1f}" if not pd.isna(date_filtered_df['WELL_BORE_TVD'].iloc[0]) else "N/A"
            if 'WATER_DEPTH' in date_filtered_df.columns:
                water_depth = f"{date_filtered_df['WATER_DEPTH'].iloc[0]:.1f}" if not pd.isna(date_filtered_df['WATER_DEPTH'].iloc[0]) else "N/A"
            
            # Display lease metadata
            st.header(f"Lease Information")
            col1, col2, col3 = st.columns(3)
            col1.metric("Lease Number", selected_lease)
            col2.metric("Well Count", well_count)
            col3.metric("Water Depth", f"{water_depth} ft" if water_depth != "N/A" else "N/A")
            
            # Chart selection in the sidebar
            st.sidebar.header("Visualization")
            chart_type = st.sidebar.radio(
                "Select Chart Type",
                ["Production Dashboard", "Location Map", "Cumulative Production"]
            )
            
            # Production Dashboard
            if chart_type == "Production Dashboard":
                st.subheader("Production Dashboard")
                
                # Single toggle for oil or gas
                production_type = st.radio(
                    "",
                    ["Oil", "Gas"],
                    horizontal=True
                )
                
                # Create figure for selected production type
                fig1 = go.Figure()
                
                if production_type == "Oil" and 'LEASE_OIL_PROD' in date_filtered_df.columns:
                    fig1.add_trace(
                        go.Scatter(
                            x=date_filtered_df['DATE'], 
                            y=date_filtered_df['LEASE_OIL_PROD'],
                            mode='lines', 
                            name=None,  # Remove name to prevent auto-legend
                            line=dict(color='green', width=2)
                        )
                    )
                    y_axis_title = "Oil (Barrels)"
                    
                elif production_type == "Gas" and 'TOTAL_GAS' in date_filtered_df.columns:
                    fig1.add_trace(
                        go.Scatter(
                            x=date_filtered_df['DATE'], 
                            y=date_filtered_df['TOTAL_GAS'],
                            mode='lines', 
                            name=None,  # Remove name to prevent auto-legend
                            line=dict(color='red', width=2)
                        )
                    )
                    y_axis_title = "Gas (MCF)"
                
                # Update layout - add a simple "Production" title
                fig1.update_layout(
                    title="Production",  # Simple fixed title
                    height=500,
                    hovermode="x unified",
                    xaxis_title="Date",
                    yaxis_title=y_axis_title,
                    showlegend=False,  # Remove legend completely
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                
                # Add time range buttons that match the screenshot - positioned better to avoid overlap
                fig1.update_xaxes(
                    rangeslider=dict(visible=False),  # Remove the range slider
                    rangeselector=dict(
                        buttons=list([
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(count=10, label="10y", step="year", stepmode="backward"),
                            dict(step="all", label="all")
                        ]),
                        x=0.05,  # Position better to avoid overlap with legend
                        y=1.15,  # Position higher above the chart
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        bordercolor="rgba(0, 0, 0, 0.2)",
                        borderwidth=1
                    )
                )
                
                # Show the production dashboard
                st.plotly_chart(fig1, use_container_width=True)
            
            # Removed Ratio Analysis section as requested
            
            # Location Map
            elif chart_type == "Location Map":
                st.subheader("Lease Location")
                
                # Extract location data for map
                if 'SURF_LATITUDE' in date_filtered_df.columns and 'SURF_LONGITUDE' in date_filtered_df.columns:
                    lat = date_filtered_df['SURF_LATITUDE'].iloc[0]
                    lon = date_filtered_df['SURF_LONGITUDE'].iloc[0]
                    
                    if pd.notna(lat) and pd.notna(lon):
                        # Create a map figure
                        map_fig = go.Figure()
                        
                        # Add the lease location marker
                        map_fig.add_trace(go.Scattergeo(
                            lon=[lon],
                            lat=[lat],
                            text=[f"Lease: {selected_lease}"],
                            mode='markers',
                            marker=dict(
                                size=15,
                                color='red',
                                symbol='circle',
                                line=dict(width=1, color='black')
                            ),
                            name=selected_lease
                        ))
                        
                        # Update layout
                        map_fig.update_layout(
                            title=f"Location of Lease {selected_lease}",
                            geo=dict(
                                scope='usa',
                                projection_type='albers usa',
                                showland=True,
                                landcolor='rgb(240, 240, 240)',
                                showocean=True,
                                oceancolor='rgb(198, 236, 255)',
                                showcoastlines=True,
                                coastlinecolor='rgb(80, 80, 80)',
                                showframe=False
                            ),
                            height=500,
                            margin={"r":0, "t":50, "l":0, "b":0}
                        )
                        
                        # Display the map
                        st.plotly_chart(map_fig, use_container_width=True)
                    else:
                        st.warning("Invalid coordinates for this lease.")
                else:
                    st.warning("Location data not available for this lease.")
            
            # Cumulative Production
            elif chart_type == "Cumulative Production":
                st.subheader("Cumulative Production")
                
                # Only create the cumulative chart if we have oil production data
                if 'LEASE_OIL_PROD' in date_filtered_df.columns:
                    # Create cumulative dataframe
                    cum_df = date_filtered_df.sort_values('DATE').copy()
                    cum_df['CUM_OIL'] = cum_df['LEASE_OIL_PROD'].cumsum()
                    
                    if 'TOTAL_GAS' in cum_df.columns:
                        cum_df['CUM_GAS'] = cum_df['TOTAL_GAS'].cumsum()
                    
                    if 'LEASE_WTR_PROD' in cum_df.columns:
                        cum_df['CUM_WATER'] = cum_df['LEASE_WTR_PROD'].cumsum()
                    
                    # Create figure
                    cum_fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Add oil cumulative
                    cum_fig.add_trace(
                        go.Scatter(
                            x=cum_df['DATE'], 
                            y=cum_df['CUM_OIL'],
                            name="Cumulative Oil",
                            line=dict(color="green", width=2)
                        )
                    )
                    
                    # Add gas cumulative on secondary axis if available
                    if 'CUM_GAS' in cum_df.columns:
                        cum_fig.add_trace(
                            go.Scatter(
                                x=cum_df['DATE'], 
                                y=cum_df['CUM_GAS'],
                                name="Cumulative Gas",
                                line=dict(color="red", width=2, dash="dash")
                            ),
                            secondary_y=True
                        )
                    
                    # Add water cumulative if requested
                    show_water = st.checkbox("Show Cumulative Water Production", value=False)
                    
                    if show_water and 'CUM_WATER' in cum_df.columns:
                        cum_fig.add_trace(
                            go.Scatter(
                                x=cum_df['DATE'], 
                                y=cum_df['CUM_WATER'],
                                name="Cumulative Water",
                                line=dict(color="blue", width=2, dash="dot")
                            )
                        )
                    
                    # Update layout
                    cum_fig.update_layout(
                        title="Cumulative Production Over Time",
                        hovermode="x unified",
                        height=500,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    # Set axis titles
                    cum_fig.update_yaxes(title_text="Cumulative Oil/Water (bbl)", secondary_y=False)
                    cum_fig.update_yaxes(title_text="Cumulative Gas (MCF)", secondary_y=True)
                    cum_fig.update_xaxes(title_text="Date")
                    
                    # Display the chart
                    st.plotly_chart(cum_fig, use_container_width=True)
                else:
                    st.warning("Insufficient data to generate cumulative production chart.")
                    
            # Production Summary Statistics - shown below all chart types
            st.subheader("Production Summary")
            if not date_filtered_df.empty:
                summary_cols = st.columns(3)
                
                # Calculate summary statistics
                total_oil = date_filtered_df['LEASE_OIL_PROD'].sum() if 'LEASE_OIL_PROD' in date_filtered_df.columns else 0
                total_gas = date_filtered_df['TOTAL_GAS'].sum() if 'TOTAL_GAS' in date_filtered_df.columns else 0
                total_water = date_filtered_df['LEASE_WTR_PROD'].sum() if 'LEASE_WTR_PROD' in date_filtered_df.columns else 0
                
                avg_monthly_oil = date_filtered_df['LEASE_OIL_PROD'].mean() if 'LEASE_OIL_PROD' in date_filtered_df.columns else 0
                avg_monthly_gas = date_filtered_df['TOTAL_GAS'].mean() if 'TOTAL_GAS' in date_filtered_df.columns else 0
                avg_monthly_water = date_filtered_df['LEASE_WTR_PROD'].mean() if 'LEASE_WTR_PROD' in date_filtered_df.columns else 0
                
                # Display summary statistics
                with summary_cols[0]:
                    st.metric("Total Oil Production", f"{total_oil:,.0f} bbl")
                    st.metric("Avg Monthly Oil", f"{avg_monthly_oil:,.0f} bbl")
                
                with summary_cols[1]:
                    st.metric("Total Gas Production", f"{total_gas:,.0f} MCF")
                    st.metric("Avg Monthly Gas", f"{avg_monthly_gas:,.0f} MCF")
                
                with summary_cols[2]:
                    st.metric("Total Water Production", f"{total_water:,.0f} bbl")
                    st.metric("Avg Monthly Water", f"{avg_monthly_water:,.0f} bbl")
        else:
            st.warning("No data available for the selected lease.")
    else:
        st.warning("No lease data available.")
else:
    st.warning("No year data available for time series analysis.")