# tennis_partner_app.py (or mmdstreamlit.py)

import streamlit as st
import pandas as pd
from collections import defaultdict

# Load the CSV you uploaded
@st.cache_data
def load_data():
    return pd.read_csv("mmdmatches.csv")

df = load_data()

# Assume these are the correct column names — adjust if needed!
player_columns = ['WINNER 1', 'WINNER 2', 'CHALLENGER 1', 'CHALLENGER 2']
# Filter out NaN values from each column before creating the set
all_players = (
    set(df['WINNER 1'].dropna()) | 
    set(df['WINNER 2'].dropna()) | 
    set(df['CHALLENGER 1'].dropna()) | 
    set(df['CHALLENGER 2'].dropna())
)

selected_player = st.selectbox("🎾 Select a Player", sorted(all_players))

# Analyze match data
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

# Generate output
if selected_player:
    result_df = analyze_partnerships(df, selected_player)

    st.subheader(f"📊 Partner Stats for {selected_player}")
    st.dataframe(result_df)

    if not result_df.empty:
        top_partner = result_df.iloc[0]
        st.success(f"🥇 Most effective partner: **{top_partner['Partner']}** with **{top_partner['Wins Together']}** wins")
