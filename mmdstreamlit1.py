import streamlit as st
import pandas as pd
import datetime
import os
import json
from collections import defaultdict
import plotly.express as px

# --- CONFIG ---
VALID_SCORES = [f"{i}-{j}" for i, j in [
    (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (7, 5), (7, 6),
    (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 7), (6, 7)
]]
DATA_FILE = "match_data.json"
PLAYER_FILE = "players.csv"  # Using a fixed file for players

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

# --- Load Players from players.csv ---
if os.path.exists(PLAYER_FILE):
    players_df = pd.read_csv(PLAYER_FILE)
    player_names = players_df.iloc[:, 0].dropna().unique().tolist()
else:
    st.error(f"Player file `{PLAYER_FILE}` not found. Please make sure it exists in the same directory.")
    st.stop()  # Stop the app if player file is not found

# --- ENTER MATCH RESULTS ---
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

# --- Edit or Delete Match ---
st.header("🛠 Edit or Delete a Match")
if st.session_state.matches:
    match_labels = [
        f"{i+1}. {m['date']} - {', '.join(m['team1'])} vs {', '.join(m['team2'])}"
        for i, m in enumerate(st.session_state.matches)
    ]
    selected_match_idx = st.selectbox("Select Match", range(len(st.session_state.matches)),
                                      format_func=lambda x: match_labels[x])

    # --- Edit Match
    st.subheader("✏️ Edit Match Details")
    match = st.session_state.matches[selected_match_idx]

    if match["type"] == "Singles":
        player1 = st.selectbox("Player 1", player_names, index=player_names.index(match["team1"][0]), key="edit_s1")
        player2 = st.selectbox("Player 2", [p for p in player_names if p != player1], index=player_names.index(match["team2"][0]), key="edit_s2")
        team1 = [player1]
        team2 = [player2]
    else:
        p1 = st.selectbox("Team 1 - Player 1", player_names, index=player_names.index(match["team1"][0]), key="edit_d1p1")
        p2 = st.selectbox("Team 1 - Player 2", [p for p in player_names if p != p1], index=player_names.index(match["team1"][1]), key="edit_d1p2")
        p3 = st.selectbox("Team 2 - Player 1", [p for p in player_names if p not in [p1, p2]], index=player_names.index(match["team2"][0]), key="edit_d2p1")
        p4 = st.selectbox("Team 2 - Player 2", [p for p in player_names if p not in [p1, p2, p3]], index=player_names.index(match["team2"][1]), key="edit_d2p2")
        team1 = [p1, p2]
        team2 = [p3, p4]

    set1 = st.selectbox("Set 1", VALID_SCORES, index=VALID_SCORES.index(match["set1"]), key="edit_set1")
    set2 = st.selectbox("Set 2", VALID_SCORES, index=VALID_SCORES.index(match["set2"]), key="edit_set2")
    set3 = st.selectbox("Set 3 (Optional)", [""] + VALID_SCORES, index=VALID_SCORES.index(match["set3"]) if match["set3"] else 0, key="edit_set3")
    winner = st.radio("Who Won?", ["Team 1", "Team 2"], index=0 if match["winner"] == "Team 1" else 1, key="edit_winner")

    if st.button("Save Changes"):
        updated_match = {
            "date": match["date"],
            "type": match["type"],
            "team1": team1,
            "team2": team2,
            "set1": set1,
            "set2": set2,
            "set3": set3,
            "winner": winner
        }
        st.session_state.matches[selected_match_idx] = updated_match
        save_local()
        st.success("✅ Match updated!")

    # --- Delete Match
    if st.button("Delete Match"):
        del st.session_state.matches[selected_match_idx]
        save_local()
        st.success("🗑 Match deleted!")

# --- Match Records Table ---
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

# --- Player Points ---
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

