import streamlit as st
import pandas as pd
import plotly.express as px
import os
from pathlib import Path

st.set_page_config(
    page_title="Ionic Emissions Analysis",
    page_icon="📊",
    layout="wide"
)

#  description section
st.title("Ionic Protocol Emissions Analysis")

with st.expander("Overview", expanded=True):
    st.markdown("""
    This dashboard presents several analyses related to [Ionic Money](https://ionic.money/). The data tracks changes 
    in supply and borrowing across vaults on Mode, OP, and Base networks, along with emissions tracking, revenue analysis, and regressions. Vault analysis is currently Base only.
    
    📊 **Data Source**: Data primarily sourced from Dune Analytics. Click the repo for links to queries.
    
    🔍 **View Source**: [Ionic Research Github](https://github.com/Unit-Zero-Labs/ionic-research/tree/main)
    
    *For more details about the analysis methodology and assumptions, please visit the GitHub repository.*
    """)

st.markdown("---")

#  absolute path to the data directory
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = SCRIPT_DIR / "data"

# load CSVs
@st.cache_data
def load_data():
    try:
        # List all required files
        required_files = {
            'emissions_results': DATA_DIR / "ionic_emissions_results.csv",
            'vault_analysis': DATA_DIR / "ionic_vault_analysis.csv",
            'age_size_analysis': DATA_DIR / "ionic_vault_analysis_age_and_size.csv",
            'revenue_analysis': DATA_DIR / "ionic_revenue_analysis.csv",
            'regression_results': DATA_DIR / "ionic_emissions_regression_results.csv",
            'apr_analysis': DATA_DIR / "ionic_apr_analysis.csv",
            'apr_summary': DATA_DIR / "ionic_apr_summary.csv"

        }
        
        # Check if data directory exists
        if not DATA_DIR.exists():
            raise FileNotFoundError(f"Data directory not found at {DATA_DIR}")
        
        # Check each required file
        missing_files = []
        for name, path in required_files.items():
            if not path.exists():
                missing_files.append(str(path))
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files:\n" + "\n".join(missing_files))
            
        # Load all the data
        emissions_results = pd.read_csv(required_files['emissions_results'])
        vault_analysis = pd.read_csv(required_files['vault_analysis'])
        age_size_analysis = pd.read_csv(required_files['age_size_analysis'])
        revenue_analysis = pd.read_csv(required_files['revenue_analysis'])
        regression_results = pd.read_csv(required_files['regression_results'])
        apr_analysis = pd.read_csv(required_files['apr_analysis'])
        apr_summary = pd.read_csv(required_files['apr_summary'])

        
        return emissions_results, vault_analysis, age_size_analysis, revenue_analysis, regression_results, apr_analysis, apr_summary

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error(f"Current working directory: {os.getcwd()}")
        st.error(f"Data directory path: {DATA_DIR}")
        st.error(f"Files in data directory: {os.listdir(DATA_DIR) if DATA_DIR.exists() else 'Directory not found'}")
        return None, None, None, None, None

#  unpacking of the returned values
emissions_results, vault_analysis, age_size_analysis, revenue_analysis, regression_results, apr_analysis, apr_summary = load_data()
#  check for all dataframes
if all(df is not None for df in [emissions_results, vault_analysis, age_size_analysis, revenue_analysis, regression_results, apr_analysis, apr_summary]):

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Base Vaults", "Vault Analysis", "Revenue Analysis", "Raw Data", "Emissions Regressions", "APR Analysis"])

    # TAB 1
    with tab1:
        # Display key metrics in columns
        metrics = emissions_results.set_index('Metric')['Value']
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total ION Emissions (Base only)", f"{metrics['Total ION Emissions']:,.2f}")
            st.metric("Days Analyzed", f"{metrics['Days Analyzed']:.0f}")
            st.metric("Daily Emission Rate", f"{metrics['Daily Emission Rate']:,.2f}")
        
        with col2:
            st.metric("Total Deposit Change", f"${metrics['Total Deposit Change ($)']:,.2f}")
            st.metric("Total Borrow Change", f"${metrics['Total Borrow Change ($)']:,.2f}")

        # Create visualization of changes
        st.subheader("Vault Changes Analysis")
        
        # Prepare data for visualization
        vault_analysis_melted = pd.melt(
            vault_analysis.reset_index(),
            id_vars=['vaultName'],
            value_vars=['Deposit Change', 'Borrow Change'],
            var_name='Metric',
            value_name='Change'
        )

        # Create bar chart
        fig = px.bar(
            vault_analysis_melted,
            x='vaultName',
            y='Change',
            color='Metric',
            barmode='group',
            title='Deposit and Borrow Changes by Vault (Since Emissions Start)',
            labels={'vaultName': 'Vault', 'Change': 'Change in USD'},
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # TAB 2
    with tab2:
        st.subheader("Vault Analysis Details")
        
        # Create size/age distribution table
        st.subheader("Vault Distribution by Size and Age")
        
        # Calculate the number of vaults in each category
        size_age_dist = pd.crosstab(
            age_size_analysis['Size Category'],
            age_size_analysis['Age Category'],
            values=age_size_analysis['Current Deposits'],
            aggfunc='sum'
        ).round(2)
        
        # Format the values as currency
        size_age_dist = size_age_dist.applymap(lambda x: f"${x:,.2f}")
        
        # Display the crosstab
        st.dataframe(size_age_dist, use_container_width=True)
        
        # Original vault analysis table
        st.subheader("Individual Vault Details")
        formatted_vault_analysis = vault_analysis.copy()
        for col in formatted_vault_analysis.select_dtypes(include=['float64']).columns:
            formatted_vault_analysis[col] = formatted_vault_analysis[col].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(
            formatted_vault_analysis,
            use_container_width=True,
            height=400
        )

    # TAB 3
    with tab3:
        st.subheader("Revenue Analysis (90 Day Lookback)")
        
        try:
            revenue_path = DATA_DIR / "ionic_revenue_analysis.csv"
            revenue_df = pd.read_csv(revenue_path)
            
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Protocol Revenue during Period", f"${revenue_df['Period_Revenue'].sum():,.2f}")
                st.metric("Total Vaults", len(revenue_df))
            with col2:
                st.metric("Annualized Revenue", f"${revenue_df['Annual_Revenue'].sum():,.2f}")
                st.metric("Revenue-Generating Vaults", len(revenue_df[revenue_df['Period_Revenue'] > 0]))
            with col3:
                avg_util = revenue_df['Avg_Utilization'].mean() * 100
                avg_apr = revenue_df['Effective_Borrow_APR'].mean()
                st.metric("Average Utilization", f"{avg_util:.1f}%")
                st.metric("Average Borrow APR", f"{avg_apr:.1f}%")

            # Network breakdown
            st.subheader("Revenue by Network")
            network_metrics = []
            for network in revenue_df['network'].unique():
                network_data = revenue_df[revenue_df['network'] == network]
                network_metrics.append({
                    'Network': network,
                    'Period Revenue': f"${network_data['Period_Revenue'].sum():,.2f}",
                    'Annualized Revenue': f"${network_data['Annual_Revenue'].sum():,.2f}",
                    'Avg Utilization': f"{network_data['Avg_Utilization'].mean() * 100:.1f}%",
                    'Avg Borrow APR': f"{network_data['Effective_Borrow_APR'].mean():.1f}%",
                    'Active/Total Vaults': f"{len(network_data[network_data['Period_Revenue'] > 0])}/{len(network_data)}"
                })
            st.dataframe(pd.DataFrame(network_metrics).set_index('Network'))

            # Visualizations
            network_colors = {'Mode': '#1f77b4', 'Base': '#2ca02c', 'Optimism': '#ff7f0e'}
            
            #1. Revenue Distribution Treemap
            fig1 = px.treemap(
                revenue_df[revenue_df['Period_Revenue'] > 0],
                path=[px.Constant("All Networks"), 'network', 'vaultName'],
                values='Period_Revenue',
                color='Effective_Borrow_APR',
                color_continuous_scale='Viridis',
                title='Protocol Revenue Distribution'
            )
            st.plotly_chart(fig1, use_container_width=True)

            ## 2. Revenue vs Utilization
            fig2 = px.scatter(
                revenue_df,
                x='Avg_Utilization',
                y='Period_Revenue',
                color='network',
                color_discrete_map=network_colors,
                title='Revenue vs Utilization',
                hover_data=['vaultName']
            )
            st.plotly_chart(fig2, use_container_width=True)

            # 3. APR Distribution
            fig3 = px.violin(
                revenue_df,
                x='network',
                y='Effective_Borrow_APR',
                color='network',
                color_discrete_map=network_colors,
                box=True,
                title='APR Distribution by Network'
            )
            st.plotly_chart(fig3, use_container_width=True)

            # 4. Network Revenue Share
            fig4 = px.pie(
                values=revenue_df.groupby('network')['Period_Revenue'].sum(),
                names=revenue_df.groupby('network')['Period_Revenue'].sum().index,
                title='Network Revenue Share',
                color=revenue_df.groupby('network')['Period_Revenue'].sum().index,
                color_discrete_map=network_colors
            )
            st.plotly_chart(fig4, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading revenue analysis data: {str(e)}") 
         

    #TAB 4
    with tab4:
        st.subheader("Raw Data")
        
        # Create dropdown for CSV selection
        csv_option = st.selectbox(
            'Select CSV to view',
            ('Emissions Results', 'Vault Analysis', 'Age and Size Analysis')
        )
        
        if csv_option == 'Emissions Results':
            st.dataframe(
                emissions_results,
                use_container_width=True,
                height=400
            )
            
            st.download_button(
                label="Download Emissions Results CSV",
                data=emissions_results.to_csv(index=False),
                file_name="ionic_emissions_results.csv",
                mime="text/csv"
            )
        elif csv_option == 'Vault Analysis':
            st.dataframe(
                vault_analysis,
                use_container_width=True,
                height=400
            )
            
            st.download_button(
                label="Download Vault Analysis CSV",
                data=vault_analysis.to_csv(index=False),
                file_name="ionic_vault_analysis.csv",
                mime="text/csv"
            )
        else:  # Age and Size Analysis
            st.dataframe(
                age_size_analysis,
                use_container_width=True,
                height=400
            )
            
            st.download_button(
                label="Download Age and Size Analysis CSV",
                data=age_size_analysis.to_csv(index=False),
                file_name="ionic_vault_analysis_age_and_size.csv",
                mime="text/csv"
            )

    #TAB 5
    with tab5:
        st.subheader("Emissions Regression Analysis")
        
        try:
            # Load regression results
            regression_path = DATA_DIR / "ionic_emissions_regression_results.csv"
            regression_df = pd.read_csv(regression_path)
            
            # Create epoch selector
            epoch = st.selectbox(
                "Select Epoch",
                ["Previous (Oct 15 - Nov 15)", "Current (Nov 15 - Dec 15)"]
            )
            
            # Convert selection to dataframe filter
            epoch_filter = 'prev' if 'Previous' in epoch else 'curr'
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Supply-Side Impact")
                supply_data = regression_df[
                    (regression_df['epoch'] == epoch_filter) & 
                    (regression_df['side'] == 'supply')
                ]
                
                # Get R-squared and n_obs
                r2 = supply_data['r_squared'].iloc[0]
                n_obs = supply_data['n_observations'].iloc[0]
                
                # Display metrics
                st.metric("R-squared", f"{r2:.4f}")
                st.metric("Number of Observations", n_obs)
                
                # Create coefficient table
                st.markdown("**Regression Coefficients:**")
                supply_table = supply_data[['variable', 'coefficient', 'p_value']].copy()
                supply_table['coefficient'] = supply_table['coefficient'].round(4)
                supply_table['p_value'] = supply_table['p_value'].round(4)
                supply_table['Significant'] = supply_table['p_value'] < 0.05
                st.dataframe(supply_table, use_container_width=True)
                
            with col2:
                st.subheader("Borrow-Side Impact")
                borrow_data = regression_df[
                    (regression_df['epoch'] == epoch_filter) & 
                    (regression_df['side'] == 'borrow')
                ]
                
                # Get R-squared and n_obs
                r2 = borrow_data['r_squared'].iloc[0]
                n_obs = borrow_data['n_observations'].iloc[0]
                
                # Display metrics
                st.metric("R-squared", f"{r2:.4f}")
                st.metric("Number of Observations", n_obs)
                
                # Create coefficient table
                st.markdown("**Regression Coefficients (Log-transformed):**")
                borrow_table = borrow_data[['variable', 'coefficient', 'p_value']].copy()
                borrow_table['coefficient'] = borrow_table['coefficient'].round(4)
                borrow_table['p_value'] = borrow_table['p_value'].round(4)
                borrow_table['Significant'] = borrow_table['p_value'] < 0.05
                st.dataframe(borrow_table, use_container_width=True)
            
            # Add interpretation section
            st.markdown("---")
            st.markdown("### Interpretation Guide")
            st.markdown("""
            - **Coefficients** show the impact of each variable on deposit/borrow changes
            - **P-values < 0.05** indicate statistically significant relationships
            - **R-squared** shows how much of the variation is explained by the model
            - **Borrow-side** analysis uses log-transformed changes for better model fit
            
            **Variables:**
            - `const`: Base level change
            - `emissions_usd`: Impact of emissions in USD
            - `net_apr`: Impact of net APR
            - `TVL_start`: Impact of initial Total Value Locked
            - `chain_encoded`: Impact of different chains
            """)
            
        except Exception as e:
            st.error(f"Error loading regression analysis data: {str(e)}")
    
    # TAB 6
    with tab6:
        st.subheader("APR Analysis")
        
        try:
            # Create three columns for key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_borrow_apr = apr_summary[('borrow_apr', 'mean')].mean() * 100
                st.metric("Average Borrow APR", f"{avg_borrow_apr:.2f}%")
                
            with col2:
                avg_supply_apr = apr_summary[('supply_apr', 'mean')].mean() * 100
                st.metric("Average Supply APR", f"{avg_supply_apr:.2f}%")
                
            with col3:
                avg_util = apr_summary[('utilization_rate', 'mean')].mean() * 100
                st.metric("Average Utilization Rate", f"{avg_util:.2f}%")

            # Borrow APR Chart
            st.subheader("Borrow APRs Over Time")
            fig1 = px.line(
                apr_analysis,
                x='date',
                y='borrow_apr',
                color='vaultName',
                title='Borrow APRs by Vault',
                labels={'borrow_apr': 'Borrow APR', 'date': 'Date'}
            )
            fig1.update_layout(height=500)
            st.plotly_chart(fig1, use_container_width=True)

            # Supply APR Chart
            st.subheader("Supply APRs Over Time")
            fig2 = px.line(
                apr_analysis,
                x='date',
                y='supply_apr',
                color='vaultName',
                title='Supply APRs by Vault',
                labels={'supply_apr': 'Supply APR', 'date': 'Date'}
            )
            fig2.update_layout(height=500)
            st.plotly_chart(fig2, use_container_width=True)

            # Utilization Rate Chart
            st.subheader("Utilization Rates Over Time")
            fig3 = px.line(
                apr_analysis,
                x='date',
                y='utilization_rate',
                color='vaultName',
                title='Utilization Rates by Vault',
                labels={'utilization_rate': 'Utilization Rate', 'date': 'Date'}
            )
            fig3.update_layout(height=500)
            st.plotly_chart(fig3, use_container_width=True)

            # Summary Statistics Table
            st.subheader("Summary Statistics by Vault")
            
            # Format percentages in summary stats
            formatted_summary = apr_summary.copy()
            # Convert to percentages
            for metric in ['borrow_apr', 'supply_apr', 'utilization_rate']:
                for stat in ['mean', 'min', 'max']:
                    formatted_summary[(metric, stat)] = formatted_summary[(metric, stat)] * 100
            
            st.dataframe(formatted_summary, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading APR analysis data: {str(e)}")