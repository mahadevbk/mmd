# tennis_partner_app.py (or mmdstreamlit.py)

import streamlit as st
import pandas as pd
from collections import defaultdict

# Check for required visualization packages
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    visualization_available = True
except ImportError:
    visualization_available = False
    st.warning("Visualization features disabled - install matplotlib and seaborn with: pip install matplotlib seaborn")

# Load the CSV you uploaded
@st.cache_data
def load_data():
    return pd.read_csv("mmdmatches.csv")

df = load_data()

# Function to get all players
def get_all_players(df):
    return (
        set(df['WINNER 1'].dropna()) | 
        set(df['WINNER 2'].dropna()) | 
        set(df['CHALLENGER 1'].dropna()) | 
        set(df['CHALLENGER 2'].dropna())
    )

all_players = get_all_players(df)

# Sidebar for navigation
st.sidebar.title("🎾 Tennis Partner Analysis")
analysis_type = st.sidebar.radio(
    "Select Analysis Type",
    ["Player Partnership Stats", "Player Frequency", "All Players Overview"]
)

# Player frequency analysis
def analyze_player_frequency(df):
    player_counts = defaultdict(int)
    
    for col in ['WINNER 1', 'WINNER 2', 'CHALLENGER 1', 'CHALLENGER 2']:
        for player in df[col].dropna():
            player_counts[player] += 1
            
    return pd.DataFrame({
        'Player': list(player_counts.keys()),
        'Matches Played': list(player_counts.values())
    }).sort_values(by='Matches Played', ascending=False)

# Analyze match data for partnerships
def analyze_partnerships(df, player_name):
    stats = defaultdict(lambda: {'matches': 0, 'wins': 0})

    for _, row in df.iterrows():
        if player_name in (row['WINNER 1'], row['WINNER 2']):
            partner = row['WINNER 2'] if row['WINNER 1'] == player_name else row['WINNER 1']
            stats[partner]['matches'] += 1
            stats[partner]['wins'] += 1
        elif player_name in (row['CHALLENGER 1'], row['CHALLENGER 2']):
            partner = row['CHALLENGER 2'] if row['CHALLENGER 1'] == player_name else row['CHALLENGER 1']
            stats[partner]['matches'] += 1

    return pd.DataFrame([
        {'Partner': partner, 'Total Matches': stat['matches'], 'Wins Together': stat['wins']}
        for partner, stat in stats.items()
    ]).sort_values(by='Wins Together', ascending=False)

# Main content area
if analysis_type == "Player Partnership Stats":
    st.header("📊 Player Partnership Statistics")
    
    selected_player = st.selectbox("🎾 Select a Player", sorted(all_players))
    
    if selected_player:
        result_df = analyze_partnerships(df, selected_player)

        st.subheader(f"Partner Stats for {selected_player}")
        st.dataframe(result_df)
        
        if visualization_available and not result_df.empty:
            # Create charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Total Matches Played Together")
                fig, ax = plt.subplots(figsize=(8, 4))
                result_df.head(10).plot.barh(y='Total Matches', x='Partner', ax=ax)
                ax.set_xlabel("Matches Played Together")
                st.pyplot(fig)
                
            with col2:
                st.write("### Wins Together")
                fig, ax = plt.subplots(figsize=(8, 4))
                result_df.head(10).plot.barh(y='Wins Together', x='Partner', ax=ax, color='green')
                ax.set_xlabel("Wins Together")
                st.pyplot(fig)
            
            # Win percentage
            result_df['Win Percentage'] = (result_df['Wins Together'] / result_df['Total Matches'] * 100).round(1)
            st.write("### Win Percentage with Partners")
            fig, ax = plt.subplots(figsize=(10, 5))
            result_df[result_df['Total Matches'] >= 3].sort_values('Win Percentage', ascending=False).head(10).plot.barh(
                y='Win Percentage', 
                x='Partner',
                ax=ax
            )
            ax.set_xlabel("Win Percentage (%)")
            ax.set_xlim(0, 100)
            st.pyplot(fig)
            
            top_partner = result_df.iloc[0]
            st.success(f"🥇 Most effective partner: **{top_partner['Partner']}** with **{top_partner['Wins Together']}** wins")
        elif not visualization_available:
            st.info("Install matplotlib and seaborn for visualizations: pip install matplotlib seaborn")

elif analysis_type == "Player Frequency":
    st.header("📈 Player Frequency Analysis")
    
    frequency_df = analyze_player_frequency(df)
    
    st.subheader("Most Active Players")
    st.dataframe(frequency_df)
    
    if visualization_available:
        # Top players chart
        st.write("### Top 20 Most Active Players")
        fig, ax = plt.subplots(figsize=(10, 6))
        frequency_df.head(20).plot.barh(y='Matches Played', x='Player', ax=ax)
        ax.set_xlabel("Number of Matches Played")
        st.pyplot(fig)
    else:
        st.info("Install matplotlib for visualizations: pip install matplotlib")

elif analysis_type == "All Players Overview":
    st.header("👥 All Players Overview")
    st.subheader("Player List")
    st.write(f"Total unique players: {len(all_players)}")
    st.write(sorted(all_players))
