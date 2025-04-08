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
    
    # For each match, display an edit and delete button
    for i, match in enumerate(st.session_state.matches):
        with st.expander(f"Match {i + 1}: {', '.join(match['team1'])} vs {', '.join(match['team2'])}"):
            st.write(f"Date: {match['date']}")
            st.write(f"Set 1: {match['set1']}, Set 2: {match['set2']}, Set 3: {match['set3']}")
            st.write(f"Winner: {match['winner']}")

            # Edit and delete buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Edit Match {i + 1}"):
                    with st.expander("Edit Match Details"):
                        # Pre-fill existing match details for editing
                        match_type = st.radio("Match Type", ["Singles", "Doubles"], index=0 if match['type'] == "Singles" else 1)
                        match_date = st.date_input("Date of Match", match["date"])

                        if match_type == "Singles":
                            player1 = st.selectbox("Player 1", player_names, index=player_names.index(match["team1"][0]), key=f"edit_s1_{i}")
                            player2 = st.selectbox("Player 2", [p for p in player_names if p != player1], index=player_names.index(match["team2"][0]), key=f"edit_s2_{i}")
                            team1 = [player1]
                            team2 = [player2]
                        else:
                            p1 = st.selectbox("Team 1 - Player 1", player_names, index=player_names.index(match["team1"][0]), key=f"edit_d1p1_{i}")
                            p2 = st.selectbox("Team 1 - Player 2", [p for p in player_names if p != p1], index=player_names.index(match["team1"][1]), key=f"edit_d1p2_{i}")
                            p3 = st.selectbox("Team 2 - Player 1", [p for p in player_names if p not in [p1, p2]], index=player_names.index(match["team2"][0]), key=f"edit_d2p1_{i}")
                            p4 = st.selectbox("Team 2 - Player 2", [p for p in player_names if p not in [p1, p2, p3]], index=player_names.index(match["team2"][1]), key=f"edit_d2p2_{i}")
                            team1 = [p1, p2]
                            team2 = [p3, p4]

                        set1 = st.selectbox("Set 1", VALID_SCORES, index=VALID_SCORES.index(match["set1"]), key=f"edit_set1_{i}")
                        set2 = st.selectbox("Set 2", VALID_SCORES, index=VALID_SCORES.index(match["set2"]), key=f"edit_set2_{i}")
                        set3 = st.selectbox("Set 3 (Optional)", [""] + VALID_SCORES, index=VALID_SCORES.index(match["set3"]) if match["set3"] else 0, key=f"edit_set3_{i}")
                        winner = st.radio("Who Won?", ["Team 1", "Team 2"], index=0 if match["winner"] == "Team 1" else 1, key=f"edit_winner_{i}")

                        if st.button(f"Save Changes for Match {i + 1}"):
                            updated_match = {
                                "date": match_date,
                                "type": match_type,
                                "team1": team1,
                                "team2": team2,
                                "set1": set1,
                                "set2": set2,
                                "set3": set3,
                                "winner": winner
                            }
                            st.session_state.matches[i] = updated_match
                            save_local()
                            st.success("✅ Match updated!")

            with col2:
                if st.button(f"Delete Match {i + 1}"):
                    if st.confirm("Are you sure you want to delete this match?"):
                        del st.session_state.matches[i]
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
