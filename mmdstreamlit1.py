import streamlit as st
import pandas as pd
import datetime
import os
import io
import json
from collections import defaultdict
import plotly.express as px

# --- CONFIG ---
VALID_SCORES = [f"{i}-{j}" for i, j in [
    (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (7, 5), (7, 6),
    (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 7), (6, 7)
]]
DATA_FILE = "match_data.json"

# --- PAGE SETUP ---
st.set_page_config(page_title="Tennis Community App", layout="wide")
st.title("🎾 Tennis Community App")

# --- FUNCTIONS ---
def save_local():
    with open(DATA_FILE, "w") as f:
        json.dump(st.session_state.matches, f, default=str)

def load_local():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

# --- DATA INIT ---
if "matches" not in st.session_state:
    st.session_state.matches = load_local()

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload CSV of Player Names", type="csv")

if uploaded_file:
    players_df = pd.read_csv(uploaded_file)
    player_names = players_df.iloc[:, 0].dropna().unique().tolist()

    st.header("➕ Enter a Match Result")
    match_type = st.radio("Match Type", ["Singles", "Doubles"])
    match_date = st.date_input("Date of Match", datetime.date.today())

    if match_type == "Singles":
        col1, col2 = st.columns(2)
        with col1:
            player1 = st.selectbox("Player 1", player_names, key="s1")
        with col2:
            player2 = st.selectbox("Player 2", [p for p in player_names if p != player1], key="s2")
        team1 = [player1]
        team2 = [player2]
    else:
        st.markdown("#### Select Doubles Players")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            p1 = st.selectbox("Team 1 - Player 1", player_names, key="d1p1")
        with col2:
            p2 = st.selectbox("Team 1 - Player 2", [p for p in player_names if p != p1], key="d1p2")
        with col3:
            p3 = st.selectbox("Team 2 - Player 1", [p for p in player_names if p not in [p1, p2]], key="d2p1")
        with col4:
            p4 = st.selectbox("Team 2 - Player 2", [p for p in player_names if p not in [p1, p2, p3]], key="d2p2")
        team1 = [p1, p2]
        team2 = [p3, p4]

    st.subheader("🎾 Set Scores")
    set1 = st.selectbox("Set 1", VALID_SCORES, key="set1")
    set2 = st.selectbox("Set 2", VALID_SCORES, key="set2")
    set3 = st.selectbox("Set 3 (Optional)", [""] + VALID_SCORES, key="set3")
    winner = st.radio("Who Won?", ["Team 1", "Team 2"])

    if st.button("Submit Match"):
        if set1 and set2:
            match = {
                "date": match_date,
                "type": match_type,
                "team1": team1,
                "team2": team2,
                "set1": set1,
                "set2": set2,
                "set3": set3,
                "winner": winner
            }
            st.session_state.matches.append(match)
            save_local()
            st.success("✅ Match recorded!")
        else:
            st.error("Please enter Set 1 and Set 2 scores.")

    st.header("🛠 Edit or Delete a Match")
    if st.session_state.matches:
        match_labels = [
            f"{i+1}. {m['date']} - {', '.join(m['team1'])} vs {', '.join(m['team2'])}"
            for i, m in enumerate(st.session_state.matches)
        ]
        selected_match_idx = st.selectbox("Select Match", range(len(st.session_state.matches)),
                                          format_func=lambda x: match_labels[x])
        if st.button("Delete Match"):
            del st.session_state.matches[selected_match_idx]
            save_local()
            st.success("🗑 Match deleted!")

    st.header("📋 Match Records (with Filters)")
    filter_player = st.selectbox("🔍 Filter by Player (optional)", ["All"] + player_names)
    filter_date = st.date_input("📅 Filter by Date (optional)", value=None)

    filtered_matches = st.session_state.matches
    if filter_player != "All":
        filtered_matches = [m for m in filtered_matches if filter_player in m["team1"] + m["team2"]]
    if filter_date:
        filtered_matches = [m for m in filtered_matches if m["date"] == filter_date]

    match_records = []
    for m in filtered_matches:
        match_records.append({
            "Date": m["date"],
            "Type": m["type"],
            "Team 1": ", ".join(m["team1"]),
            "Team 2": ", ".join(m["team2"]),
            "Set 1": m["set1"],
            "Set 2": m["set2"],
            "Set 3": m["set3"],
            "Winner": m["winner"]
        })

    if match_records:
        match_df = pd.DataFrame(match_records)
        st.dataframe(match_df)

        csv = match_df.to_csv(index=False).encode("utf-8")
        st.download_button("📤 Download Match Records", csv, "match_records.csv", "text/csv")
    else:
        st.info("No matches match your filter criteria.")

    st.header("🏅 Player Points")
    points = defaultdict(int)
    for match in st.session_state.matches:
        winners = match["team1"] if match["winner"] == "Team 1" else match["team2"]
        losers = match["team2"] if match["winner"] == "Team 1" else match["team1"]
        for p in winners:
            points[p] += 3
        for p in losers:
            points[p] += 1

    points_df = pd.DataFrame(points.items(), columns=["Player", "Points"]).sort_values(by="Points", ascending=False)
    st.dataframe(points_df)

    points_csv = points_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Points Table", points_csv, "player_points.csv", "text/csv")

    # 📊 Charts
    st.subheader("📈 Points Distribution")
    fig = px.bar(points_df, x="Player", y="Points", title="Player Points", text="Points")
    st.plotly_chart(fig)

    st.header("📊 Player Statistics")
    selected_player = st.selectbox("Select a Player", player_names)
    player_matches = [m for m in st.session_state.matches if selected_player in m["team1"] + m["team2"]]
    st.write(f"Total Matches Played: {len(player_matches)}")

    if player_matches:
        dates = sorted([m["date"] for m in player_matches])
        if len(dates) > 1:
            frequency = (dates[-1] - dates[0]).days / (len(dates) - 1)
            st.write(f"Average Frequency: {round(frequency, 1)} days")
        else:
            st.write("Only one match played.")

        partners = []
        partner_wins = defaultdict(int)
        partner_total = defaultdict(int)

        for m in player_matches:
            is_team1 = selected_player in m["team1"]
            team = m["team1"] if is_team1 else m["team2"]
            if m["type"] == "Doubles":
                partners_in_team = [p for p in team if p != selected_player]
                partners.extend(partners_in_team)
                for p in partners_in_team:
                    partner_total[p] += 1
                    if m["winner"] == ("Team 1" if is_team1 else "Team 2"):
                        partner_wins[p] += 1

        if partners:
            unique_partners = list(set(partners))
            partner_df = pd.DataFrame({
                "Partner": unique_partners,
                "Times Played": [partner_total[p] for p in unique_partners],
                "Win Rate (%)": [
                    round(100 * partner_wins[p] / partner_total[p], 1) if partner_total[p] > 0 else 0
                    for p in unique_partners
                ]
            }).sort_values(by="Win Rate (%)", ascending=False)
            st.subheader("Partner Stats")
            st.dataframe(partner_df)

            best_partner = partner_df.iloc[0]["Partner"]
            st.success(f"🏆 Best Partner: {best_partner}")
        else:
            st.info("No doubles data available for this player.")

        player_points = points.get(selected_player, 0)
        st.write(f"Total Points: **{player_points}**")
    else:
        st.warning("No matches found for this player.")
else:
    st.info("⬆️ Please upload a CSV file with player names (one per row).")
