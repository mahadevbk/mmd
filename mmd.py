import streamlit as st
import pandas as pd
import numpy as np
import uuid
import random
import io
import zipfile
import base64
import os
import re
import time
import json
import requests
import urllib.parse
from datetime import datetime, timedelta
from collections import defaultdict
from itertools import combinations
from dateutil import parser
from PIL import Image, ImageOps

# Plotting
import plotly.graph_objects as go
import plotly.express as px

# Supabase & AI SDKs
from supabase import create_client, Client
from openai import OpenAI
from xai_sdk import Client as XAIClient
from xai_sdk.chat import user, system

# PDF Generation
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# --- Configuration ---
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
st.set_page_config(page_title="MMD Mira Mixed Doubles Tennis League", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
.stApp { background: linear-gradient(to bottom, #041136, #21000a); background-attachment: scroll; }
@media print {
  html, body { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
  body { background: linear-gradient(to bottom, #21000a, #041136) !important; height: 100vh; margin: 0; padding: 0; }
  header, .stToolbar { display: none; }
}
[data-testid="stHeader"] { background: linear-gradient(to top, #041136 , #21000a) !important; }
.profile-image {
    width: 100px; height: 100px; object-fit: cover; border: 1px solid #fff500; border-radius: 10%;
    margin-right: 10px; vertical-align: middle; transition: transform 0.2s;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4), 0 0 10px rgba(255, 245, 0, 0.6);
}
.profile-image:hover { transform: scale(1.1); }
.birthday-banner {
    background: linear-gradient(45deg, #FFFF00, #EEE8AA); color: #950606; padding: 15px;
    border-radius: 10px; text-align: center; font-size: 1.2em; font-weight: bold;
    margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    display: flex; justify-content: center; align-items: center;
}
.court-card {
    background: linear-gradient(to bottom, #031827, #07314f); border: 1px solid #fff500;
    border-radius: 10px; padding: 15px; margin: 10px 0;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); transition: transform 0.2s, box-shadow 0.2s;
    text-align: center;
}
.court-card:hover { transform: scale(1.05); box-shadow: 0 6px 12px rgba(255, 245, 0, 0.3); }
.court-card h4 { color: #fff500; margin-bottom: 10px; }
.court-card a {
    background-color: #fff500; color: #031827; padding: 8px 16px; border-radius: 5px;
    text-decoration: none; font-weight: bold; display: inline-block; margin-top: 10px;
    transition: background-color 0.2s;
}
.court-card a:hover { background-color: #ffd700; }
@import url('https://fonts.googleapis.com/css2?family=Offside&display=swap');
html, body, [class*="st-"], h1, h2, h3, h4, h5, h6 { font-family: 'Offside', sans-serif !important; }
.rank-col { color: #fff500; font-size: 1.3em; font-weight: bold; }
[data-testid="stMetric"] > div:nth-of-type(1) { color: #FF7518 !important; }
.block-container { display: flex; flex-wrap: wrap; justify-content: center; }
</style>
""", unsafe_allow_html=True)

# --- Supabase Setup ---
try:
    supabase_url = st.secrets["supabase"]["supabase_url"]
    supabase_key = st.secrets["supabase"]["supabase_key"]
    supabase: Client = create_client(supabase_url, supabase_key)
except KeyError:
    st.error("Supabase secrets not found. Please check your .streamlit/secrets.toml file.")
    st.stop()

# Table names
players_table_name = "players"
matches_table_name = "matches"
bookings_table_name = "bookings"
hall_of_fame_table_name="hall_of_fame"

# --- Session State ---
if 'players_df' not in st.session_state: st.session_state.players_df = pd.DataFrame(columns=["name", "profile_image_url", "birthday"])
if 'matches_df' not in st.session_state: st.session_state.matches_df = pd.DataFrame(columns=["match_id", "date", "match_type", "team1_player1", "team1_player2", "team2_player1", "team2_player2", "set1", "set2", "set3", "winner", "match_image_url"])
if 'bookings_df' not in st.session_state: st.session_state.bookings_df = pd.DataFrame(columns=["booking_id", "date", "time", "match_type", "court_name", "player1", "player2", "player3", "player4", "screenshot_url"])
if 'form_key_suffix' not in st.session_state: st.session_state.form_key_suffix = 0
if 'last_match_submit_time' not in st.session_state: st.session_state.last_match_submit_time = 0
if 'image_urls' not in st.session_state: st.session_state.image_urls = {}

# --- Functions ---

def load_players():
    try:
        response = supabase.table(players_table_name).select("name, profile_image_url, birthday, gender").execute()
        df = pd.DataFrame(response.data)
        for col in ["name", "profile_image_url", "birthday", "gender"]:
            if col not in df.columns: df[col] = ""
        df['name'] = df['name'].str.upper().str.strip()
        st.session_state.players_df = df
    except Exception as e:
        st.error(f"Error loading players: {e}")
        st.session_state.players_df = pd.DataFrame(columns=["name", "profile_image_url", "birthday", "gender"])

def save_players(players_df):
    try:
        df_save = players_df[["name", "profile_image_url", "birthday", "gender"]].copy()
        df_save['name'] = df_save['name'].str.upper().str.strip()
        df_save = df_save.where(pd.notna(df_save), None).drop_duplicates(subset=['name'], keep='last')
        supabase.table(players_table_name).upsert(df_save.to_dict("records")).execute()
    except Exception as e: st.error(f"Error saving players: {e}")

def delete_player_from_db(player_name):
    try: supabase.table(players_table_name).delete().eq("name", player_name).execute()
    except Exception as e: st.error(f"Error deleting player: {e}")

def load_matches():
    try:
        response = supabase.table(matches_table_name).select("*").execute()
        df = pd.DataFrame(response.data)
        cols = ["match_id", "date", "match_type", "team1_player1", "team1_player2", "team2_player1", "team2_player2", "set1", "set2", "set3", "winner", "match_image_url"]
        for col in cols: 
            if col not in df.columns: df[col] = ""
        df['raw_date'] = df['date']
        df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce').dt.tz_localize(None)
        if df['date'].isna().any():
            df['date'] = df['date'].fillna(pd.Timestamp('1970-01-01'))
        st.session_state.matches_df = df
    except Exception as e:
        st.error(f"Error loading matches: {e}")
        st.session_state.matches_df = pd.DataFrame()

def save_matches(matches_df):
    try:
        df_save = matches_df.copy()
        df_save = df_save.where(pd.notna(df_save), None)
        df_save['date'] = df_save['date'].apply(lambda d: pd.to_datetime(d).strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(pd.to_datetime(d, errors='coerce')) else datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        df_save = df_save[df_save["match_id"].notnull() & (df_save["match_id"] != "")]
        df_save = df_save.drop_duplicates(subset=['match_id'], keep='last')
        if df_save.empty: return False
        supabase.table(matches_table_name).upsert(df_save.to_dict("records")).execute()
        return True
    except Exception as e:
        st.error(f"Error saving matches: {e}"); return False

def delete_match_from_db(match_id):
    try:
        supabase.table(matches_table_name).delete().eq("match_id", match_id).execute()
        st.session_state.matches_df = st.session_state.matches_df[st.session_state.matches_df["match_id"] != match_id].reset_index(drop=True)
        save_matches(st.session_state.matches_df)
    except Exception as e: st.error(f"Error deleting match: {e}")

def load_bookings():
    try:
        response = supabase.table("bookings").select("*").execute()
        df = pd.DataFrame(response.data)
        cols = ['booking_id', 'date', 'time', 'match_type', 'court_name', 'player1', 'player2', 'player3', 'player4', 'standby_player', 'screenshot_url']
        for c in cols: 
            if c not in df: df[c] = None
        if not df.empty:
            df['b_dt'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), format="%Y-%m-%d %H:%M:%S", errors='coerce').dt.tz_localize('Asia/Dubai', ambiguous='infer')
            cutoff = pd.Timestamp.now(tz='Asia/Dubai') - timedelta(hours=4)
            expired = df[df['b_dt'].notnull() & (df['b_dt'] < cutoff)]
            for _, row in expired.iterrows():
                try: supabase.table("bookings").delete().eq("booking_id", row['booking_id']).execute()
                except: pass
            df = df[df['b_dt'].isnull() | (df['b_dt'] >= cutoff)]
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d').fillna("")
        df = df.reindex(columns=cols).fillna("")
        st.session_state.bookings_df = df
    except Exception: st.session_state.bookings_df = pd.DataFrame()

def save_bookings(df):
    try:
        df_save = df.copy()
        if 'date' in df_save.columns:
            df_save['date'] = pd.to_datetime(df_save['date']).dt.strftime("%Y-%m-%d %H:%M:%S")
        df_save = df_save.where(pd.notna(df_save), None).drop_duplicates(subset=['booking_id'], keep='last')
        supabase.table(bookings_table_name).upsert(df_save.to_dict("records")).execute()
    except Exception as e: st.error(str(e))

def delete_booking_from_db(booking_id):
    try:
        supabase.table(bookings_table_name).delete().eq("booking_id", booking_id).execute()
        st.session_state.bookings_df = st.session_state.bookings_df[st.session_state.bookings_df["booking_id"] != booking_id].reset_index(drop=True)
        save_bookings(st.session_state.bookings_df)
    except: pass

def upload_image_to_github(file, file_name, image_type="match"):
    if not file: return ""
    try:
        token, repo, branch = st.secrets["github"]["token"], st.secrets["github"]["repo"], st.secrets["github"]["branch"]
        path = f"assets/{image_type}_images/{file_name}.jpg" if image_type in ["profile", "match", "booking"] else f"assets/others/{file_name}.jpg"
        img = Image.open(file)
        img = ImageOps.exif_transpose(img)
        if image_type == "match": img.thumbnail((1200, 1200), Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        content_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        api_url = f"https://api.github.com/repos/{repo}/contents/{path}"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
        sha = requests.get(api_url, headers=headers).json().get('sha')
        payload = {"message": f"Upload {image_type} {file_name}", "branch": branch, "content": content_b64}
        if sha: payload["sha"] = sha
        requests.put(api_url, headers=headers, json=payload).raise_for_status()
        return f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
    except Exception as e: st.error(f"Image upload failed: {e}"); return ""

def tennis_scores():
    scores = ["6-0", "6-1", "6-2", "6-3", "6-4", "7-5", "7-6", "0-6", "1-6", "2-6", "3-6", "4-6", "5-7", "6-7"]
    for i in range(10): scores.extend([f"Tie Break 7-{i}", f"Tie Break {i}-7"])
    for i in range(6): scores.extend([f"Tie Break 10-{i}", f"Tie Break {i}-10"])
    return scores

def get_quarter(month): return f"Q{(month-1)//3 + 1}"

def generate_match_id(matches_df, match_datetime):
    year, quarter = match_datetime.year, get_quarter(match_datetime.month)
    if not matches_df.empty:
        matches_df['date'] = pd.to_datetime(matches_df['date'], errors='coerce')
        count = len(matches_df[(matches_df['date'].dt.year == year) & (matches_df['date'].apply(lambda d: get_quarter(d.month) == quarter))])
        new_id = f"MMD{quarter}{year}-{count+1:02d}"
        while new_id in matches_df['match_id'].values: count+=1; new_id = f"MMD{quarter}{year}-{count+1:02d}"
        return new_id
    return f"MMD{quarter}{year}-01"

def generate_booking_id(bookings_df, booking_date):
    year, quarter = booking_date.year, get_quarter(booking_date.month)
    serial = len(bookings_df) + 1 if not bookings_df.empty else 1
    return f"BK{quarter}{year}-{serial:02d}"

def get_player_trend(player, matches, max_matches=5):
    pm = matches[(matches['team1_player1'] == player) | (matches['team1_player2'] == player) | (matches['team2_player1'] == player) | (matches['team2_player2'] == player)].copy()
    pm['date'] = pd.to_datetime(pm['date'], errors='coerce').fillna(pd.Timestamp('1970-01-01'))
    pm = pm.sort_values(by='date', ascending=False).head(max_matches)
    trend = []
    for _, r in pm.iterrows():
        t1 = [r['team1_player1'], r['team1_player2']] if r['match_type']=='Doubles' else [r['team1_player1']]
        t2 = [r['team2_player1'], r['team2_player2']] if r['match_type']=='Doubles' else [r['team2_player1']]
        if (player in t1 and r['winner']=='Team 1') or (player in t2 and r['winner']=='Team 2'): trend.append('W')
        elif r['winner'] != 'Tie': trend.append('L')
    return ' '.join(trend) if trend else 'No recent matches'

def _calculate_performance_score(player_stats, full_dataset):
    w_wp, w_agd, w_ef = 0.50, 0.35, 0.15
    max_wp = full_dataset['Win %'].max() or 1
    max_agd = full_dataset['Game Diff Avg'].max()
    min_agd = full_dataset['Game Diff Avg'].min()
    max_m = full_dataset['Matches'].max() or 1
    
    wp_n = player_stats['Win %'] / max_wp
    agd_n = (player_stats['Game Diff Avg'] - min_agd) / (max_agd - min_agd) if max_agd != min_agd else 0.5
    ef_n = player_stats['Matches'] / max_m
    return (w_wp * wp_n) + (w_agd * agd_n) + (w_ef * ef_n)


def calculate_rankings(matches_to_rank):
    scores, wins, losses, matches_played = defaultdict(float), defaultdict(int), defaultdict(int), defaultdict(int)
    games_won, cumulative_game_diff = defaultdict(int), defaultdict(int)
    clutch_wins, clutch_matches = defaultdict(int), defaultdict(int)
    partner_stats = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'ties': 0, 'matches': 0, 'game_diff_sum': 0}))
    game_diffs = defaultdict(list)
    singles_matches, doubles_matches = defaultdict(int), defaultdict(int)
    players_df = st.session_state.players_df

    for _, row in matches_to_rank.iterrows():
        t1 = [p for p in [row['team1_player1'], row.get('team1_player2')] if p and p != "Visitor"]
        t2 = [p for p in [row['team2_player1'], row.get('team2_player2')] if p and p != "Visitor"]
        
        is_mixed_doubles = False
        if row['match_type'] == 'Doubles' and len(t1)==2 and len(t2)==2:
            try:
                g1 = [players_df[players_df['name']==p]['gender'].iloc[0] for p in t1 if p in players_df['name'].values]
                g2 = [players_df[players_df['name']==p]['gender'].iloc[0] for p in t2 if p in players_df['name'].values]
                if sorted(g1) == ['F', 'M'] and sorted(g2) == ['F', 'M']: is_mixed_doubles = True
            except: pass

        match_gd = 0
        is_clutch = "Tie Break" in str(row['set1']) or "Tie Break" in str(row['set2']) or (row.get('set3') and str(row['set3']).strip())
        
        for s in [row['set1'], row['set2'], row['set3']]:
            if not s: continue
            try:
                g1, g2 = map(int, re.findall(r'\d+', str(s))[:2])
                if "Tie Break" in str(s): g1, g2 = (7, 6) if g1 > g2 else (6, 7)
                diff = g1 - g2; match_gd += diff
                for p in t1: games_won[p]+=g1; cumulative_game_diff[p]+=diff; game_diffs[p].append(diff)
                for p in t2: games_won[p]+=g2; cumulative_game_diff[p]-=diff; game_diffs[p].append(-diff)
            except: pass

        win_pts = 3 if is_mixed_doubles else 2
        for p in t1: 
            matches_played[p]+=1; 
            if row['match_type']=='Doubles': doubles_matches[p]+=1 
            else: singles_matches[p]+=1
            if is_clutch: clutch_matches[p]+=1
            if row["winner"]=="Team 1": scores[p]+=win_pts; wins[p]+=1; clutch_wins[p]+=1 if is_clutch else 0
            elif row["winner"]=="Team 2": scores[p]+=1; losses[p]+=1
            else: scores[p]+=1.5
            
        for p in t2:
            matches_played[p]+=1;
            if row['match_type']=='Doubles': doubles_matches[p]+=1 
            else: singles_matches[p]+=1
            if is_clutch: clutch_matches[p]+=1
            if row["winner"]=="Team 2": scores[p]+=win_pts; wins[p]+=1; clutch_wins[p]+=1 if is_clutch else 0
            elif row["winner"]=="Team 1": scores[p]+=1; losses[p]+=1
            else: scores[p]+=1.5

        if row['match_type'] == 'Doubles':
            for p1, p2 in combinations(t1, 2):
                partner_stats[p1][p2]['matches']+=1; partner_stats[p1][p2]['game_diff_sum']+=match_gd
                if row["winner"]=="Team 1": partner_stats[p1][p2]['wins']+=1
                elif row["winner"]=="Team 2": partner_stats[p1][p2]['losses']+=1
                else: partner_stats[p1][p2]['ties']+=1
            for p1, p2 in combinations(t2, 2):
                partner_stats[p1][p2]['matches']+=1; partner_stats[p1][p2]['game_diff_sum']-=match_gd
                if row["winner"]=="Team 2": partner_stats[p1][p2]['wins']+=1
                elif row["winner"]=="Team 1": partner_stats[p1][p2]['losses']+=1
                else: partner_stats[p1][p2]['ties']+=1

    rank_data = []
    for p in scores:
        if p == "Visitor": continue
        m = matches_played[p]
        if m == 0: continue
        
        trend = get_player_trend(p, matches_to_rank)
        consist = np.std(game_diffs[p]) if game_diffs[p] else 0
        clutch_pct = (clutch_wins[p]/clutch_matches[p]*100) if clutch_matches[p]>0 else 0
        
        # --- RESTORED LOGIC: Get Profile Image ---
        profile_url = ""
        if not players_df.empty and "profile_image_url" in players_df.columns:
            player_row = players_df[players_df["name"] == p]
            if not player_row.empty:
                profile_url = player_row.iloc[0]["profile_image_url"]
        # -----------------------------------------

        badges = []
        if clutch_pct > 70 and clutch_matches[p] >= 3: badges.append("🎯 Tie-break Monster")
        if wins[p] >= 5 and trend.startswith("W W W W W"): badges.append("🔥 Hot Streak")
        if consist < 2 and m >= 5: badges.append("📈 Consistent Performer")
        if games_won[p] == max(games_won.values()): badges.append("🥇 Game Hog")
        
        rank_data.append({
            "Rank": 0, 
            "Player": p,
            "Profile": profile_url, # Key restored
            "Points": scores[p], "Win %": (wins[p]/m)*100,
            "Matches": m, "Doubles Matches": doubles_matches[p], "Singles Matches": singles_matches[p],
            "Wins": wins[p], "Losses": losses[p], "Games Won": games_won[p],
            "Game Diff Avg": cumulative_game_diff[p]/m, "Cumulative Game Diff": cumulative_game_diff[p],
            "Recent Trend": trend, "Clutch Factor": clutch_pct, "Consistency Index": consist, "Badges": badges
        })
        
    rank_df = pd.DataFrame(rank_data)
    if not rank_df.empty:
        rank_df = rank_df.sort_values(by=["Points", "Win %", "Game Diff Avg"], ascending=False).reset_index(drop=True)
        rank_df["Rank"] = [f"🏆 {i}" for i in range(1, len(rank_df)+1)]
    return rank_df, partner_stats


def create_trend_chart(trend):
    # Fixed indentation and logic
    if not trend or trend == 'No recent matches': return None
    res = trend.split()
    colors = ['#00FF00' if r == 'W' else '#FF0000' for r in res]
    fig = go.Figure(go.Bar(x=list(range(len(res))), y=[1]*len(res), marker_color=colors))
    fig.update_layout(height=50, margin=dict(l=0,r=0,t=0,b=0), xaxis={'visible':False}, yaxis={'visible':False}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

# Restore the missing function as an alias (Fixes NameError)
def create_trend_sparkline(trend):
    return create_trend_chart(trend)

def create_win_loss_donut(wins, losses):
    if wins+losses == 0: return None
    fig = go.Figure(data=[go.Pie(labels=['Wins', 'Losses'], values=[wins, losses], hole=.6, marker_colors=['#00a86b', '#ff4136'])])
    fig.update_layout(showlegend=False, height=120, margin=dict(t=0,b=0,l=0,r=0), paper_bgcolor='rgba(0,0,0,0)')
    return fig

def create_nerd_stats_chart(rank_df):
    if rank_df is None or rank_df.empty: return None
    df = rank_df.copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Player'], y=df['Matches'], name='Matches', marker_color='#1E90FF'))
    fig.add_trace(go.Bar(x=df['Player'], y=df['Wins'], name='Wins', marker_color='#FFD700'))
    fig.add_trace(go.Bar(x=df['Player'], y=df['Points'], name='Points', marker_color='#9A5BE2'))
    fig.update_layout(barmode='stack', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#fff500'))
    return fig

def create_partnership_chart(player_name, partner_stats, players_df):
    if player_name not in partner_stats: return None
    data = []
    for p, stats in partner_stats[player_name].items():
        if p == "Visitor": continue
        data.append({'Partner': p, 'Win %': (stats['wins']/stats['matches']*100) if stats['matches']>0 else 0, 
                     'Text': f"{stats['wins']}W-{stats['losses']}L"})
    if not data: return None
    df = pd.DataFrame(data).sort_values('Win %')
    fig = go.Figure(go.Bar(y=df['Partner'], x=df['Win %'], orientation='h', text=df['Text'], marker=dict(color=df['Win %'], colorscale='Viridis')))
    fig.update_layout(title=f'Partnership: {player_name}', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#fff500'))
    return fig

def check_birthdays(players_df):
    today = datetime.now().date()
    return [row["name"] for _, row in players_df.iterrows() if row.get("birthday") and re.match(r'^\d{2}-\d{2}$', row["birthday"]) and datetime(today.year, int(row["birthday"].split("-")[1]), int(row["birthday"].split("-")[0])).date() == today]

def display_birthday_message(birthday_players):
    for p in birthday_players: st.success(f"Happy Birthday {p}!")

def display_player_insights(selected_players, players_df, matches_df, doubles_rank_df, singles_rank_df, key_prefix=""):
    if isinstance(selected_players, str): selected_players = [selected_players]
    selected_players = [p for p in selected_players if p != "Visitor"]
    if not selected_players: st.info("No players selected."); return

    view_option = st.radio("Select View", ["Player Insights", "Birthdays"], horizontal=True, key=f"{key_prefix}view_selector")
    
    if view_option == "Birthdays":
        birthday_data = []
        for player in selected_players:
            p_info = players_df[players_df["name"] == player].iloc[0] if player in players_df["name"].values else None
            if not p_info: continue
            bday = p_info.get("birthday", "")
            if bday and re.match(r'^\d{2}-\d{2}$', bday):
                try:
                    d, m = map(int, bday.split("-"))
                    birthday_data.append({"Player": player, "Birthday": datetime(2001, m, d).strftime("%b %d"), "Profile": p_info.get("profile_image_url")})
                except: continue
        if not birthday_data: st.info("No birthday data."); return
        for row in birthday_data: st.markdown(f"**{row['Player']}**: {row['Birthday']}")
        return

    rank_df_combined, _ = calculate_rankings(matches_df)
    active_players = [p for p in selected_players if p in rank_df_combined["Player"].values]
    
    for idx, player in enumerate(sorted(active_players)):
        p_data = rank_df_combined[rank_df_combined["Player"] == player].iloc[0]
        st.markdown(f"### {player} (Rank: {p_data['Rank']})")
        col1, col2 = st.columns([1, 2])
        with col1:
             st.plotly_chart(create_win_loss_donut(p_data["Wins"], p_data["Losses"]), key=f"{key_prefix}_{idx}_wl")
             st.caption(f"Trend: {p_data['Recent Trend']}")
        with col2:
             st.metric("Points", f"{p_data['Points']:.1f}")
             st.metric("Win Rate", f"{p_data['Win %']:.1f}%")

def display_community_stats(matches_df):
    matches_df['date'] = pd.to_datetime(matches_df['date'], utc=True, errors='coerce').dt.tz_localize(None)
    seven_days_ago = datetime.now() - timedelta(days=7)
    recent = matches_df[matches_df['date'] >= seven_days_ago]
    if recent.empty: st.info("No matches in last 7 days."); return
    st.metric("Matches Played", len(recent))
    players = pd.unique(recent[['team1_player1', 'team1_player2', 'team2_player1', 'team2_player2']].values.ravel('K'))
    st.metric("Active Players", len([p for p in players if pd.notna(p) and p]))



def calculate_enhanced_doubles_odds(players, doubles_rank_df):
    if len(players) != 4 or doubles_rank_df.empty: return ("Select 4 players", None, None)
    p_scores = {p: _calculate_performance_score(doubles_rank_df[doubles_rank_df["Player"]==p].iloc[0], doubles_rank_df) if p in doubles_rank_df["Player"].values else 0 for p in players}
    
    best, min_diff = None, float('inf')
    for t1 in combinations(players, 2):
        t2 = tuple(p for p in players if p not in t1)
        diff = abs(sum(p_scores[p] for p in t1) - sum(p_scores[p] for p in t2))
        if diff < min_diff: min_diff = diff; best = (t1, t2)
    
    t1, t2 = best
    s1, s2 = sum(p_scores[p] for p in t1), sum(p_scores[p] for p in t2)
    total = s1 + s2 or 1
    return (f"Team 1: {t1[0]}&{t1[1]} vs Team 2: {t2[0]}&{t2[1]}", (s1/total)*100, (s2/total)*100)

def calculate_enhanced_singles_odds(players, singles_rank_df):
    if len(players) != 2 or singles_rank_df.empty: return (None, None)
    s1 = _calculate_performance_score(singles_rank_df[singles_rank_df["Player"]==players[0]].iloc[0], singles_rank_df) if players[0] in singles_rank_df["Player"].values else 0
    s2 = _calculate_performance_score(singles_rank_df[singles_rank_df["Player"]==players[1]].iloc[0], singles_rank_df) if players[1] in singles_rank_df["Player"].values else 0
    total = s1 + s2 or 1
    return ((s1/total)*100, (s2/total)*100)

# Aliases for compatibility
def suggest_balanced_pairing(players, rank_df): return calculate_enhanced_doubles_odds(players, rank_df)
def suggest_singles_odds(players, rank_df): return calculate_enhanced_singles_odds(players, rank_df)

def generate_pdf_reportlab(rank_df_combined, rank_df_doubles, rank_df_singles):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4))
    elements = [Paragraph("AR Tennis League Rankings", getSampleStyleSheet()['Title'])]
    def add_table(df, title):
        if df.empty: return
        elements.append(Paragraph(title, getSampleStyleSheet()['Heading2']))
        data = [df.columns.to_list()] + df.values.tolist()
        t = Table(data)
        t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.grey)]))
        elements.append(t); elements.append(Spacer(1, 0.2*inch))
    
    add_table(rank_df_combined[["Rank", "Player", "Points", "Win %"]].head(20), "Combined")
    doc.build(elements)
    return buffer.getvalue()

def download_image(url):
    try: return requests.get(url, timeout=10).content if url else None
    except: return None

# --- Initialization ---
load_players()
load_matches()
load_bookings()
# ... [Your existing UI code starts below this line] ...




    


# -----------------------Main App Logic ------------------------------------------------------




load_players()
load_matches()
load_bookings()

# Check for and display birthday messages
todays_birthdays = check_birthdays(st.session_state.players_df)
if todays_birthdays:
    display_birthday_message(todays_birthdays)


court_names = [
    "Alvorado 1","Alvorado 2", "Palmera 2", "Palmera 4", "Saheel", "Hattan",
    "MLC Mirador La Colleccion", "Al Mahra", "Mirador", "Reem 1", "Reem 2",
    "Reem 3", "Alma", "Mira 2", "Mira 4", "Mira 5 A", "Mira 5 B", "Mira Oasis 1",
    "Mira Oasis 2", "Mira Oasis 3 A","Mira Oasis 3 B", "Mira Oasis 3 C","Mudon Main courts",
    "Mudon Arabella","Mudon Arabella 3","AR2 Rosa","AR2 Palma","AR 2 Fitness First","Dubai Hills Maple"
]

# List of fun verbs for match results
fun_verbs = [
    "defeated", "thrashed", "crushed", "beat the hell out of",
    "smashed", "obliterated", "demolished", "outplayed",
    "vanquished", "dominated", "trounced", "routed", "got the better of",  "inexplicably, def.", "in an upset, def.", 
    "destroyed"
]


players_df = st.session_state.players_df
matches = st.session_state.matches_df
players = sorted([p for p in players_df["name"].dropna().tolist() if p != "Visitor"]) if "name" in players_df.columns else []

#-------------ADDING SESSION STATE TO CHECK FOR DUPLICATE MATCHES -------------------------

if 'pending_match' not in st.session_state:
    st.session_state.pending_match = None
if 'duplicate_flag' not in st.session_state:
    st.session_state.duplicate_flag = False



if not matches.empty and ("match_id" not in matches.columns or matches["match_id"].isnull().any()):
    matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
    for i in matches.index:
        if pd.isna(matches.at[i, "match_id"]):
            match_date_for_id = matches.at[i, "date"] if pd.notna(matches.at[i, "date"]) else datetime.now()
            matches.at[i, "match_id"] = generate_match_id(matches, match_date_for_id)
    save_matches(matches)

st.image("https://raw.githubusercontent.com/mahadevbk/mmd/main/mmdheaderQ12026.png", width='stretch')

tab_names = ["Rankings", "Matches", "Player Profile", "Maps", "Bookings","Hall of Fame","Mini Tourney","MMD AI"]

tabs = st.tabs(tab_names)


#-------------START OF TABS -----------------------------------------------------------------







with tabs[0]:

    load_players()
    load_matches()
    available_players = sorted([name for name in st.session_state.players_df["name"].values if name]) if not st.session_state.players_df.empty else []

    
    st.header(f"Rankings as of {datetime.now().strftime('%d %b %Y')}")
    with st.expander("MMD Points Award System", expanded=False, icon="➡️"):
      st.info("3 Points for a Mixed Doubles win, 2 Points for a Doubles game win, 2 Points for a Singles game win, 1 Point for a loss(of any kind of game) & 1.5 Points for a Tie in any kind of game.")
    ranking_type = st.radio(
        "Select Ranking View",
        ["Combined", "Doubles", "Singles", "Nerd Stuff", "Table View"],
        horizontal=True,
        key="ranking_type_selector"
    )

    # Ensure matches_df is loaded
    load_matches()
    matches_df = st.session_state.matches_df

    # --- PRE-CALCULATE ALL RANKING DATAFRAMES FOR PERFORMANCE SCORES ---
    if matches_df.empty:
        st.warning("No match data available. Please add matches to generate rankings.")
        doubles_rank_df = pd.DataFrame(columns=["Rank", "Profile", "Player", "Points", "Win %", "Matches", "Doubles Matches", "Singles Matches", "Wins", "Losses", "Games Won", "Game Diff Avg", "Cumulative Game Diff", "Recent Trend", "Clutch Factor", "Consistency Index", "Badges"])
        singles_rank_df = pd.DataFrame(columns=["Rank", "Profile", "Player", "Points", "Win %", "Matches", "Doubles Matches", "Singles Matches", "Wins", "Losses", "Games Won", "Game Diff Avg", "Cumulative Game Diff", "Recent Trend", "Clutch Factor", "Consistency Index", "Badges"])
        partner_stats = {}
    else:
        doubles_matches_df = matches_df[matches_df['match_type'] == 'Doubles'].copy()
        singles_matches_df = matches_df[matches_df['match_type'] == 'Singles'].copy()
        doubles_rank_df, _ = calculate_rankings(doubles_matches_df)
        singles_rank_df, _ = calculate_rankings(singles_matches_df)
        # Calculate partner stats for Combined view
        _, partner_stats = calculate_rankings(matches_df)

    # Helper function to generate a single player card
    def display_ranking_card(player_data, players_df, matches_df, partner_stats, doubles_rank_df, singles_rank_df, key_prefix=""):
        player_name = player_data["Player"]
        player_info = players_df[players_df["name"] == player_name].iloc[0] if player_name in players_df.name.values else None

        if player_info is None:
            st.warning(f"Could not find profile information for {player_name}")
            return

        # --- Data Calculation & Formatting ---
        profile_image = player_info.get("profile_image_url", "")
        wins, losses = int(player_data["Wins"]), int(player_data["Losses"])
        trend = player_data["Recent Trend"]
        rank_value = player_data['Rank']
        rank_display = re.sub(r'[^0-9]', '', str(rank_value))

        # --- Performance Score Calculation ---
        if not doubles_rank_df.empty and 'Player' in doubles_rank_df.columns and player_name in doubles_rank_df['Player'].values:
            doubles_perf_score = _calculate_performance_score(doubles_rank_df[doubles_rank_df['Player'] == player_name].iloc[0], doubles_rank_df)
        else:
            doubles_perf_score = 0.0

        if not singles_rank_df.empty and 'Player' in singles_rank_df.columns and player_name in singles_rank_df['Player'].values:
            singles_perf_score = _calculate_performance_score(singles_rank_df[singles_rank_df['Player'] == player_name].iloc[0], singles_rank_df)
        else:
            singles_perf_score = 0.0

        # --- Birthday Calculation ---
        birthday_str = ""
        raw_birthday = player_info.get("birthday")
        if raw_birthday and isinstance(raw_birthday, str) and re.match(r'^\d{2}-\d{2}$', raw_birthday):
            try:
                bday_obj = datetime.strptime(f"{raw_birthday}-2000", "%d-%m-%Y")
                birthday_str = bday_obj.strftime("%d %b")
            except ValueError:
                birthday_str = ""

        # --- Partner Calculation Logic ---
        partners_list_str = "No doubles matches played."
        best_partner_str = "N/A"
        if player_name in partner_stats and partner_stats[player_name]:
            partners_list_items = [
                f'<li><b>{p}</b>: {item["wins"]}W - {item["losses"]}L ({item["matches"]} played)</li>'
                for p, item in partner_stats[player_name].items() if p != "Visitor"
            ]
            partners_list_str = f"<ul>{''.join(partners_list_items)}</ul>"

            sorted_partners = sorted(
                [(p, item) for p, item in partner_stats[player_name].items() if p != "Visitor"],
                key=lambda item: (
                    item[1]['wins'] / item[1]['matches'] if item[1]['matches'] > 0 else 0,
                    item[1]['game_diff_sum'] / item[1]['matches'] if item[1]['matches'] > 0 else 0,
                    item[1]['wins']
                ),
                reverse=True
            )
            if sorted_partners:
                best_partner_name = sorted_partners[0][0]
                best_stats = sorted_partners[0][1]
                best_win_percent = (best_stats['wins'] / best_stats['matches'] * 100) if best_stats['matches'] > 0 else 0
                best_partner_str = f"{best_partner_name} ({best_win_percent:.1f}% Win Rate)"

        # --- Card Layout ---
        st.markdown("---")

        header_html = f"""
        <div style="margin-bottom: 15px;">
            <h2 style="color: #fff500; margin-bottom: 5px; font-size: 2.0em; font-weight: bold;">{player_name}</h2>
            <span style="font-weight: bold; color: #bbbbbb; font-size: 1.1em;">
                Rank: <span style="color: #fff500;">#{rank_display}</span>
            </span>
            {f' | <span style="font-weight: bold; color: #bbbbbb; font-size: 1.1em;">🎂 Birthday: <span style="color: #fff500;">{birthday_str}</span></span>' if birthday_str else ''}
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])

        with col1:  # Left column for visuals
            if profile_image:
                st.image(profile_image, width=150)

            st.markdown("##### Win/Loss")
            win_loss_chart = create_win_loss_donut(wins, losses)
            if win_loss_chart:
                st.plotly_chart(win_loss_chart, config={"responsive": True}, key=f"{key_prefix}_win_loss_{player_name}")

            st.markdown("##### Trend")
            trend_chart = create_trend_sparkline(trend)
            if trend_chart:
                st.plotly_chart(trend_chart, config={"responsive": True}, key=f"{key_prefix}_trend_{player_name}")
                st.markdown(f"<div class='trend-col' style='text-align: center; margin-top: -15px;'>{trend}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='trend-col'>{trend}</div>", unsafe_allow_html=True)

        with col2:  # Right column for stats
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Points", f"{player_data['Points']:.1f}")
            m_col2.metric("Win Rate", f"{player_data['Win %']:.1f}%")
            m_col3.metric("Matches", f"{int(player_data['Matches'])}")

            st.markdown(f"""
            <div style="line-height: 2;">
                <span class="games-won-col" style="display: block;"> {int(player_data['Games Won'])}</span>
                <span class="game-diff-avg-col" style="display: block;"> {player_data['Game Diff Avg']:.2f}</span>
                <span class="cumulative-game-diff-col" style="display: block;"> {int(player_data['Cumulative Game Diff'])}</span>
                <span class="performance-score-col" style="display: block;">
                    <span style='font-weight:bold; color:#bbbbbb;'>Performance Score: </span>
                    <span style='font-weight:bold; color:#fff500;'>Doubles: {doubles_perf_score:.1f} ({int(player_data["Doubles Matches"])}), Singles: {singles_perf_score:.1f} ({int(player_data["Singles Matches"])})</span>
                </span>
                <span class="best-partner-col" style="display: block;">
                    <span style='font-weight:bold; color:#bbbbbb;'>Most Effective Partner: </span>{best_partner_str}
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Only show partner stats expander for views that have doubles matches
            if ranking_type != "Singles":
                with st.expander("View Full Partner Stats", expanded=False, icon="➡️"):
                    st.markdown(partners_list_str, unsafe_allow_html=True)

    # --- Ranking Views ---
    if ranking_type == "Doubles":
        rank_df, partner_stats = calculate_rankings(doubles_matches_df)
        if rank_df.empty:
            st.info("No ranking data available for this view.")
        else:
            for index, row in rank_df.iterrows():
                display_ranking_card(row, players_df, doubles_matches_df, partner_stats, doubles_rank_df, singles_rank_df, key_prefix=f"doubles_{index}")

    elif ranking_type == "Singles":
        rank_df, partner_stats = calculate_rankings(singles_matches_df)
        if rank_df.empty:
            st.info("No ranking data available for this view.")
        else:
            for index, row in rank_df.iterrows():
                display_ranking_card(row, players_df, singles_matches_df, partner_stats, doubles_rank_df, singles_rank_df, key_prefix=f"singles_{index}")

    elif ranking_type == "Nerd Stuff":
        if matches_df.empty or players_df.empty:
            st.info("No match data available to generate interesting stats.")
        else:
            rank_df, partner_stats = calculate_rankings(matches_df)
            st.header("Stats for Season Q1 2026 (Jan - Mar)")

            # Combined Table view in nerd view
            display_rankings_table(rank_df, "Combined")

            # Most Effective Partnership
            st.markdown("---")
            st.markdown("### 🤝 Most Effective Partnership")
            best_partner = None
            max_value = -1
            for player, partners in partner_stats.items():
                if player == "Visitor":
                    continue
                for partner, stats in partners.items():
                    if partner == "Visitor" or player < partner:  # Avoid double counting
                        win_rate = stats['wins'] / stats['matches'] if stats['matches'] > 0 else 0
                        avg_game_diff = stats['game_diff_sum'] / stats['matches'] if stats['matches'] > 0 else 0
                        score = win_rate + (avg_game_diff / 10)
                        if score > max_value:
                            max_value = score
                            best_partner = (player, partner, stats)

            if best_partner:
                p1, p2, stats = best_partner
                p1_styled = f"<span style='font-weight:bold; color:#fff500;'>{p1}</span>"
                p2_styled = f"<span style='font-weight:bold; color:#fff500;'>{p2}</span>"
                win_rate = (stats['wins'] / stats['matches'] * 100) if stats['matches'] > 0 else 0
                st.markdown(f"The most effective partnership is {p1_styled} and {p2_styled} with **{stats['wins']}** wins, **{stats['losses']}** losses, and a total game difference of **{stats['game_diff_sum']:.2f}** (win rate: {win_rate:.1f}%).", unsafe_allow_html=True)
            else:
                st.info("No doubles matches have been played to determine the most effective partnership.")

            st.markdown("---")

            # Best Player to Partner With
            st.markdown("### 🥇 Best Player to Partner With")
            player_stats = defaultdict(lambda: {'wins': 0, 'gd_sum': 0, 'partners': set()})
            for _, row in matches_df.iterrows():
                if row['match_type'] == 'Doubles':
                    t1 = [row['team1_player1'], row['team1_player2']]
                    t2 = [row['team2_player1'], row['team2_player2']]

                    match_gd_sum = 0
                    set_count = 0
                    for set_score in [row['set1'], row['set2'], row['set3']]:
                        if set_score and '-' in set_score:
                            try:
                                team1_games, team2_games = map(int, set_score.split('-'))
                                match_gd_sum += team1_games - team2_games
                                set_count += 1
                            except ValueError:
                                continue

                    if set_count > 0:
                        if row["winner"] == "Team 1":
                            for p in t1:
                                if p != "Visitor":
                                    player_stats[p]['wins'] += 1
                                    player_stats[p]['gd_sum'] += match_gd_sum
                                    for partner in t1:
                                        if partner != p and partner != "Visitor":
                                            player_stats[p]['partners'].add(partner)
                        elif row["winner"] == "Team 2":
                            for p in t2:
                                if p != "Visitor":
                                    player_stats[p]['wins'] += 1
                                    player_stats[p]['gd_sum'] += match_gd_sum
                                    for partner in t2:
                                        if partner != p and partner != "Visitor":
                                            player_stats[p]['partners'].add(partner)

            if player_stats:
                best_partner_candidate = None
                max_score = -1

                wins_list = [stats['wins'] for stats in player_stats.values()]
                gd_list = [stats['gd_sum'] for stats in player_stats.values()]
                partners_list = [len(stats['partners']) for stats in player_stats.values()]

                max_wins = max(wins_list) if wins_list else 1
                max_gd = max(gd_list) if gd_list else 1
                max_partners = max(partners_list) if partners_list else 1

                for player, stats in player_stats.items():
                    normalized_wins = stats['wins'] / max_wins
                    normalized_gd = stats['gd_sum'] / max_gd
                    normalized_partners = len(stats['partners']) / max_partners
                    composite_score = normalized_wins + normalized_gd + normalized_partners

                    if composite_score > max_score:
                        max_score = composite_score
                        best_partner_candidate = (player, stats)

                if best_partner_candidate:
                    player_name, stats = best_partner_candidate
                    player_styled = f"<span style='font-weight:bold; color:#fff500;'>{player_name}</span>"
                    st.markdown(f"The best player to partner with is {player_styled} based on their high number of wins, game difference sum, and variety of partners. They have:", unsafe_allow_html=True)
                    st.markdown(f"- **Total Wins**: {stats['wins']}")
                    st.markdown(f"- **Total Game Difference**: {stats['gd_sum']:.2f}")
                    st.markdown(f"- **Unique Partners Played With**: {len(stats['partners'])}")
                else:
                    st.info("Not enough data to determine the best player to partner with.")
            else:
                st.info("No doubles matches have been recorded yet.")

            st.markdown("---")

            # Most Frequent Player
            st.markdown("### 🏟️ Most Frequent Player")
            if not rank_df.empty:
                most_frequent_player = rank_df.sort_values(by="Matches", ascending=False).iloc[0]
                player_styled = f"<span style='font-weight:bold; color:#fff500;'>{most_frequent_player['Player']}</span>"
                st.markdown(f"{player_styled} has played the most matches, with a total of **{int(most_frequent_player['Matches'])}** matches played.", unsafe_allow_html=True)
            else:
                st.info("No match data available to determine the most frequent player.")

            st.markdown("---")

            # Player with highest Game Difference
            st.markdown("### 📈 Player with highest Game Difference")
            cumulative_game_diff = defaultdict(int)
            for _, row in matches_df.iterrows():
                t1 = [row['team1_player1'], row['team1_player2']] if row['match_type'] == 'Doubles' else [row['team1_player1']]
                t2 = [row['team2_player1'], row['team2_player2']] if row['match_type'] == 'Doubles' else [row['team2_player1']]
                for set_score in [row['set1'], row['set2'], row['set3']]:
                    if set_score and '-' in set_score:
                        try:
                            team1_games, team2_games = map(int, set_score.split('-'))
                            set_gd = team1_games - team2_games
                            for p in t1:
                                if p != "Visitor":
                                    cumulative_game_diff[p] += set_gd
                            for p in t2:
                                if p != "Visitor":
                                    cumulative_game_diff[p] -= set_gd
                        except ValueError:
                            continue

            if cumulative_game_diff:
                highest_gd_player, highest_gd_value = max(cumulative_game_diff.items(), key=lambda item: item[1])
                player_styled = f"<span style='font-weight:bold; color:#fff500;'>{highest_gd_player}</span>"
                st.markdown(f"{player_styled} has the highest cumulative game difference: <span style='font-weight:bold; color:#fff500;'>{highest_gd_value}</span>.", unsafe_allow_html=True)
            else:
                st.info("No match data available to calculate game difference.")

            st.markdown("---")

            # Player with the most wins
            st.markdown("### 👑 Player with the Most Wins")
            if not rank_df.empty:
                most_wins_player = rank_df.sort_values(by="Wins", ascending=False).iloc[0]
                player_styled = f"<span style='font-weight:bold; color:#fff500;'>{most_wins_player['Player']}</span>"
                st.markdown(f"{player_styled} holds the record for most wins with **{int(most_wins_player['Wins'])}** wins.", unsafe_allow_html=True)
            else:
                st.info("No match data available to determine the most wins.")

            st.markdown("---")

            # Player with the highest win percentage (minimum 5 matches)
            st.markdown("### 🔥 Highest Win Percentage (Min. 5 Matches)")
            eligible_players = rank_df[rank_df['Matches'] >= 5].sort_values(by="Win %", ascending=False)
            if not eligible_players.empty:
                highest_win_percent_player = eligible_players.iloc[0]
                player_styled = f"<span style='font-weight:bold; color:#fff500;'>{highest_win_percent_player['Player']}</span>"
                st.markdown(f"{player_styled} has the highest win percentage at **{highest_win_percent_player['Win %']:.2f}%**.", unsafe_allow_html=True)
            else:
                st.info("No players have played enough matches to calculate a meaningful win percentage.")

            st.markdown("---")
            st.markdown("### 🗓️ Community Activity: Last 7 Days")
            if 'matches_df' in st.session_state and not st.session_state.matches_df.empty:
                display_community_stats(st.session_state.matches_df)
            else:
                st.info("No recent match data available for community stats.")

            st.markdown("---")
            st.markdown("### 📊 Player Performance Overview")
            nerd_chart = create_nerd_stats_chart(rank_df)
            if nerd_chart:
                st.plotly_chart(nerd_chart, config={"responsive": True})
            else:
                st.info("Not enough data to generate the performance chart.")

            st.markdown("---")
            st.markdown("### 🤝 Partnership Performance Analyzer")
            doubles_players = []
            if partner_stats:
                doubles_players = sorted([p for p in partner_stats.keys() if p != "Visitor"])

            if not doubles_players:
                st.info("No doubles match data available to analyze partnerships.")
            else:
                selected_player_for_partners = st.selectbox(
                    "Select a player to see their partnership stats:",
                    doubles_players
                )
                if selected_player_for_partners:
                    partnership_chart = create_partnership_chart(selected_player_for_partners, partner_stats, players_df)
                    if partnership_chart:
                        st.plotly_chart(partnership_chart, config={"responsive": True})
                    else:
                        st.info(f"{selected_player_for_partners} has no partnership data to display.")


            # --- Head-to-Head Stats ---
           
            st.markdown("---")
            st.subheader("Head-to-Head Stats")
            col_ht1, col_ht2 = st.columns(2)
            with col_ht1:
                player_a = st.selectbox("Select Player A", [""] + available_players, key="ht_player_a")
            with col_ht2:
                player_b = st.selectbox("Select Player B", [""] + available_players, key="ht_player_b")
        
            if player_a and player_b and player_a != player_b:
                h2h_stats = calculate_head_to_head(player_a, player_b, st.session_state.matches_df)
                
                # Display stats in a table
                stats_data = []
                for match_type, stats in h2h_stats.items():
                    stats_data.append({
                        "Match Type": match_type,
                        "Total Matches": stats["matches"],
                        f"{player_a} Wins": stats["wins_a"],
                        f"{player_b} Wins": stats["wins_b"],
                        "Ties": stats["ties"]
                    })
                
                h2h_df = pd.DataFrame(stats_data)
                st.table(h2h_df)
                
                # Overall summary
                total_matches = sum(stats["matches"] for stats in h2h_stats.values())
                if total_matches > 0:
                    overall_wins_a = sum(stats["wins_a"] for stats in h2h_stats.values())
                    overall_wins_b = sum(stats["wins_b"] for stats in h2h_stats.values())
                    overall_ties = sum(stats["ties"] for stats in h2h_stats.values())
                    st.metric("Overall Record", f"{overall_wins_a}-{overall_wins_b}-{overall_ties}")
            else:
                st.info("Select two different players to view head-to-head stats.")




            

            st.markdown("---")
            with st.expander("Process being used for Rankings", expanded=False, icon="➡️"):
                st.markdown("""
                ### Ranking System Overview
                - **Points**: Players earn 3 points for a win, 1 point for a loss, and 1.5 points for a tie.
                - **Win Percentage**: Calculated as (Wins / Matches Played) * 100.
                - **Game Difference Average**: The average difference in games won vs. lost per match.
                - **Games Won**: Total games won across all sets.
                - **Ranking Criteria**: Players are ranked by Points (highest first), then by Win Percentage, Game Difference Average, Games Won, and finally alphabetically by name.
                - **Matches Included**: All matches, including those with a 'Visitor', contribute to AR players' stats, but 'Visitor' is excluded from rankings and insights.

                Detailed Ranking Logic at https://github.com/mahadevbk/ar2/blob/main/ar_ranking_logic.pdf
                """)

    elif ranking_type == "Table View":
        rank_df, _ = calculate_rankings(matches_df)
        display_rankings_table(rank_df, "Combined")
        display_rankings_table(doubles_rank_df, "Doubles")
        display_rankings_table(singles_rank_df, "Singles")

        st.markdown("---")
        st.subheader("Download Rankings as PDF")
        if st.button("Download All Rankings", key="download_rankings_pdf"):
            try:
                pdf_data = generate_pdf_reportlab(rank_df, doubles_rank_df, singles_rank_df)
                st.download_button(
                    label="Download PDF",
                    data=pdf_data,
                    file_name="AR_Tennis_League_Rankings.pdf",
                    mime="application/pdf",
                    key="download_pdf_button"
                )
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")

    else:  # Combined view
        rank_df, partner_stats = calculate_rankings(matches_df)
        if not rank_df.empty and len(rank_df) >= 3:
            top_3_players = rank_df.head(3)
            st.markdown("""
            <style>
            .podium-container {
                display: flex;
                flex-direction: row;
                justify-content: space-around;
                align-items: flex-end;
                width: 100%;
                margin: 20px 0;
                padding: 10px 0;
                height: 220px;
                border-bottom: 2px solid #fff500;
            }
            .podium-item {
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
                color: white;
                width: 32%;
            }
            .podium-item img {
                width: 90px;
                height: 90px;
                border-radius: 10%;
                border: 1px solid #fff500;
                transition: transform 0.2s;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4), 0 0 10px rgba(255, 245, 0, 0.6);
                margin-bottom: 10px;
                object-fit: contain;
                background-color: #404823;
            }
            .podium-name {
                font-weight: bold;
                font-size: 1.1em;
                color: #fff500;
            }
            .podium-rank {
                font-size: 1.5em;
                font-weight: bold;
                color: white;
            }
            .podium-item.rank-1 { order: 2; align-self: flex-start; }
            .podium-item.rank-2 { order: 1; }
            .podium-item.rank-3 { order: 3; }
            </style>
            """, unsafe_allow_html=True)

            p1 = top_3_players.iloc[0]
            p2 = top_3_players.iloc[1]
            p3 = top_3_players.iloc[2]

            podium_html = f"""
            <div class="podium-container">
                <div class="podium-item rank-2">
                    <img src="{p2['Profile']}" alt="{p2['Player']}">
                    <div class="podium-rank">🥈 {p2['Rank'].replace('🏆 ', '')}</div>
                    <div class="podium-name">{p2['Player']}</div>
                </div>
                <div class="podium-item rank-1">
                    <img src="{p1['Profile']}" alt="{p1['Player']}">
                    <div class="podium-rank">🥇 {p1['Rank'].replace('🏆 ', '')}</div>
                    <div class="podium-name">{p1['Player']}</div>
                </div>
                <div class="podium-item rank-3">
                    <img src="{p3['Profile']}" alt="{p3['Player']}">
                    <div class="podium-rank">🥉 {p3['Rank'].replace('🏆 ', '')}</div>
                    <div class="podium-name">{p3['Player']}</div>
                </div>
            </div>
            """
            st.markdown(podium_html, unsafe_allow_html=True)

        if rank_df.empty:
            st.info("No ranking data available for this view.")
        else:
            for index, row in rank_df.iterrows():
                display_ranking_card(row, players_df, matches_df, partner_stats, doubles_rank_df, singles_rank_df, key_prefix=f"combined_{index}")



























#---------------END OF TAB[0]-------------------------------------------------------------



with tabs[1]:
    st.header("Matches")
    # Check for duplicate match IDs
    if st.session_state.matches_df['match_id'].duplicated().any():
        st.warning("Duplicate match IDs detected in the database. Please remove duplicates in Supabase to enable editing.")
        duplicate_ids = st.session_state.matches_df[st.session_state.matches_df['match_id'].duplicated(keep=False)]['match_id'].tolist()
        st.write(f"Duplicate match IDs: {duplicate_ids}")
    
    
    with st.expander("➕ Post New Match Result", expanded=False, icon="➡️"):
        # Define available_players
        if "players_df" not in st.session_state or st.session_state.players_df.empty:
            st.warning("No players available. Please add players in the Player Profile tab.")
            available_players = []
        else:
            available_players = sorted([p for p in st.session_state.players_df["name"].dropna().tolist() if p != "Visitor"] + ["Visitor"])
        
        # Stop if no players are available
        if not available_players:
            st.stop()
        
        match_type = st.radio("Match Type", ["Doubles", "Singles"])
        
        # Players selection based on type
        if match_type == "Doubles":
            col1, col2 = st.columns(2)
            with col1:
                t1p1 = st.selectbox("Team 1 - Player 1 *", [""] + available_players, key="t1p1_doubles")
                t1p2 = st.selectbox("Team 1 - Player 2 *", [""] + available_players, key="t1p2_doubles")
            with col2:
                t2p1 = st.selectbox("Team 2 - Player 1 *", [""] + available_players, key="t2p1_doubles")
                t2p2 = st.selectbox("Team 2 - Player 2 *", [""] + available_players, key="t2p2_doubles")
            p1, p2 = "", ""
        else:  # Singles
            col1, col2 = st.columns(2)
            with col1:
                p1 = st.selectbox("Player 1 *", [""] + available_players, key="p1_singles")
            with col2:
                p2 = st.selectbox("Player 2 *", [""] + available_players, key="p2_singles")
            t1p1, t1p2, t2p1, t2p2 = p1, "", p2, ""
        
        # Date input
        date = st.date_input("Match Date *")
        
        # Score inputs
        set1 = st.selectbox("Set 1 Score *", [""] + tennis_scores(), key="set1")
        set2 = st.selectbox("Set 2 Score (optional for Singles, required for Doubles)", [""] + tennis_scores(), key="set2")
        set3 = st.selectbox("Set 3 Score (optional)", [""] + tennis_scores(), key="set3")
        if set2 == "":
            set2 = None
        if set3 == "":
            set3 = None
        
        # Winner selection
        if match_type == "Doubles":
            selected_players = [t1p1, t1p2, t2p1, t2p2]
            winner_options = ["Team 1", "Team 2", "Tie"]
        else:
            selected_players = [p1, p2]
            winner_options = ["Player 1", "Player 2", "Tie"]
        
        winner = st.selectbox("Winner *", winner_options, key="winner")
        
        # Map singles winner to team format for DB consistency
        if match_type == "Singles":
            if winner == "Player 1":
                winner = "Team 1"
            elif winner == "Player 2":
                winner = "Team 2"
        
        # Image upload (now mandatory)
        match_image = st.file_uploader("Upload Match Photo *", type=["jpg", "jpeg", "png"], key="match_image")
        
        col1, col2 = st.columns(2)
        with col1:
            with st.form(key=f"add_match_form_{st.session_state.form_key_suffix}"):
                submit = st.form_submit_button("Post Match")
                if submit:
                    current_time = time.time()
                    debounce_seconds = 10  # Adjust as needed based on typical upload time
                    
                    if current_time - st.session_state.last_match_submit_time < debounce_seconds:
                        st.warning("Please wait—your previous submission is still processing. Duplicates are prevented.")
                    else:
                        st.session_state.last_match_submit_time = current_time
                        
                        # Basic validation
                        if not match_image:
                            st.error("A match photo is required.")
                            valid = False
                        elif match_type == "Doubles":
                            if not all(selected_players) or not set1 or not set2:
                                st.error("For Doubles: All players, Set 1, Set 2, and a match photo are required. Set 3 is optional.")
                                valid = False
                            elif len(set([p for p in selected_players if p != ""])) != len([p for p in selected_players if p != ""]):
                                st.error("Please select different players for each position.")
                                valid = False
                            else:
                                valid = True
                        else:  # Singles
                            if not all(selected_players) or not set1:
                                st.error("For Singles: Both players, Set 1, and a match photo are required. Set 2 and Set 3 are optional.")
                                valid = False
                            elif p1 == p2:
                                st.error("Please select different players for singles.")
                                valid = False
                            else:
                                valid = True
                        
                        # Score-winner consistency validation
                        if valid:
                            team1_sets_won = 0
                            team2_sets_won = 0
                            sets = [set1, set2, set3]
                            valid_sets = [s for s in sets if s and s != ""]
                            for score in valid_sets:
                                try:
                                    if "Tie Break" in score:
                                        scores = [int(s) for s in re.findall(r'\d+', score)]
                                        if len(scores) != 2:
                                            st.error(f"Invalid tie break score: {score}. Please use formats like 'Tie Break 10-7'.")
                                            valid = False
                                            break
                                        t1, t2 = scores
                                    else:
                                        t1, t2 = map(int, score.split("-"))
                                    if t1 > t2:
                                        team1_sets_won += 1
                                    elif t2 > t1:
                                        team2_sets_won += 1
                                except (ValueError, TypeError) as e:
                                    st.error(f"Invalid score: {score}. Please use formats like '6-4' or 'Tie Break 10-7'.")
                                    valid = False
                                    break
                            
                            if valid:
                                if len(valid_sets) < 1:
                                    st.error("At least one set is required for all matches.")
                                    valid = False
                                elif match_type == "Doubles" and len(valid_sets) < 2:
                                    st.error("For Doubles: At least two sets are required (Set 1 and Set 2).")
                                    valid = False
                                elif len(valid_sets) >= 1:
                                    if team1_sets_won > team2_sets_won and winner != "Team 1":
                                        st.error(f"{'Team 1' if match_type == 'Doubles' else 'Player 1'} won more sets based on scores. Please select {'Team 1' if match_type == 'Doubles' else 'Player 1'} as the winner or correct the scores.")
                                        valid = False
                                    elif team2_sets_won > team1_sets_won and winner != "Team 2":
                                        st.error(f"{'Team 2' if match_type == 'Doubles' else 'Player 2'} won more sets based on scores. Please select {'Team 2' if match_type == 'Doubles' else 'Player 2'} as the winner or correct the scores.")
                                        valid = False
                                    elif team1_sets_won == team2_sets_won and winner != "Tie":
                                        st.error("Teams won an equal number of sets. Please select 'Tie' as the winner or correct the scores.")
                                        valid = False
                        
                        # Save match if valid
                        if valid:
                            try:
                                with st.spinner("Checking for duplicates and uploading match to Supabase..."):
                                    match_datetime = pd.to_datetime(date)
                                    match_id = generate_match_id(st.session_state.matches_df, match_datetime)
                                    image_url = upload_image_to_github(match_image, match_id, image_type="match")
                                    
                                    new_match = {
                                        "match_id": match_id,
                                        "date": date,
                                        "match_type": match_type,
                                        "team1_player1": t1p1,
                                        "team1_player2": t1p2 if match_type == "Doubles" else "",
                                        "team2_player1": t2p1,
                                        "team2_player2": t2p2 if match_type == "Doubles" else "",
                                        "set1": set1,
                                        "set2": set2,
                                        "set3": set3,
                                        "winner": winner,
                                        "match_image_url": image_url
                                    }
                                    
                                    # Check for duplicate (exclude match_id, date, winner, match_image_url from check)
                                    check_dict = {
                                        "match_type": match_type,
                                        "team1_player1": t1p1,
                                        "team1_player2": t1p2 if match_type == "Doubles" else "",
                                        "team2_player1": t2p1,
                                        "team2_player2": t2p2 if match_type == "Doubles" else "",
                                        "set1": set1,
                                        "set2": set2,
                                        "set3": set3
                                    }
                                    
                                    if is_duplicate_match(check_dict, st.session_state.matches_df):
                                        st.session_state.pending_match = new_match
                                        st.session_state.duplicate_flag = True
                                        st.rerun()
                                    else:
                                        # No duplicate: add and save
                                        st.session_state.matches_df = pd.concat([st.session_state.matches_df, pd.DataFrame([new_match])], ignore_index=True)
                                        save_matches(st.session_state.matches_df)
                                        st.success(f"Match {match_id} posted successfully!")
                                        st.balloons()
                                        st.session_state.form_key_suffix += 1
                                        st.rerun()
                            except Exception as e:
                                st.error(f"Failed to add match: {str(e)}")
                                if 'match_id' in locals() and match_id in st.session_state.matches_df["match_id"].values:
                                    st.session_state.matches_df = st.session_state.matches_df.drop(
                                        st.session_state.matches_df[st.session_state.matches_df["match_id"] == match_id].index
                                    )
                                st.rerun()
        
        # Handle duplicate resolution
        if st.session_state.get('duplicate_flag', False):
            st.warning("⚠️ This match (combination of players and scores) is already posted in the system.")
            
            col_choice1, col_choice2 = st.columns(2)
            with col_choice1:
                if st.button("Add as 2nd Match", key="add_duplicate"):
                    if st.session_state.get('pending_match'):
                        # Add the pending match
                        st.session_state.matches_df = pd.concat([st.session_state.matches_df, pd.DataFrame([st.session_state.pending_match])], ignore_index=True)
                        save_matches(st.session_state.matches_df)
                        st.success("Match added as second entry!")
                    st.session_state.duplicate_flag = False
                    st.session_state.pending_match = None
                    st.session_state.form_key_suffix += 1
                    st.rerun()
            
            with col_choice2:
                if st.button("Ignore", key="ignore_duplicate"):
                    st.info("Match entry ignored.")
                    st.session_state.duplicate_flag = False
                    st.session_state.pending_match = None
                    st.rerun()
            
            # Show details of the potential duplicate for reference
            if st.session_state.get('pending_match'):
                st.markdown("**Pending Match Details:**")
                st.json({k: v for k, v in st.session_state.pending_match.items() if k != "match_id"})
        
        st.markdown("*Required fields", unsafe_allow_html=True)
        
        st.markdown("---")
    
    st.markdown("---")
    st.subheader("Match History")

    # Create columns for the filters
    col1_filter, col2_filter = st.columns(2)
    with col1_filter:
        match_filter = st.radio("Filter by Type", ["All", "Singles", "Doubles"], horizontal=True, key="match_history_filter")
    with col2_filter:
        players = sorted([p for p in st.session_state.players_df["name"].tolist() if p != "Visitor"]) if "players_df" in st.session_state else []
        player_search = st.selectbox("Filter by Player", ["All Players"] + players, key="player_search_filter")

    # Start with a clean copy of the matches
    filtered_matches = st.session_state.matches_df.copy()

    # Apply type filter first
    if match_filter != "All":
        filtered_matches = filtered_matches[filtered_matches["match_type"] == match_filter]

    # Apply player search filter on the result
    if player_search != "All Players":
        filtered_matches = filtered_matches[
            (filtered_matches['team1_player1'] == player_search) |
            (filtered_matches['team1_player2'] == player_search) |
            (filtered_matches['team2_player1'] == player_search) |
            (filtered_matches['team2_player2'] == player_search)
        ]

    # Robust Date Handling and Sorting
    if not filtered_matches.empty:
        # Convert date column, turning errors into NaT (Not a Time)
        filtered_matches['date'] = pd.to_datetime(filtered_matches['date'], errors='coerce')

        # Keep only the rows with valid dates
        valid_matches = filtered_matches.dropna(subset=['date']).copy()
        
        # If some rows were dropped, inform the user
        if len(valid_matches) < len(filtered_matches):
            st.warning("Some match records were hidden due to missing or invalid date formats in the database.")
        
        if not valid_matches.empty:
            # Sort ascending to assign serial numbers correctly (oldest = #1)
            valid_matches = valid_matches.sort_values(by='date', ascending=True).reset_index(drop=True)
            valid_matches['serial_number'] = valid_matches.index + 1
            
            # Re-sort descending for display (newest first)
            display_matches = valid_matches.sort_values(by='date', ascending=False).reset_index(drop=True)
            
            # Add Match Type column
            players_df = st.session_state.get('players_df', pd.DataFrame())
            display_matches['Match Type'] = ''
            for idx, row in display_matches.iterrows():
                if row['match_type'] == 'Singles':
                    display_matches.at[idx, 'Match Type'] = 'Singles Match'
                else:  # Doubles
                    t1 = [p for p in [row['team1_player1'], row['team1_player2']] if p and p != "Visitor"]
                    t2 = [p for p in [row['team2_player1'], row['team2_player2']] if p and p != "Visitor"]
                    is_mixed_doubles = False
                    if len(t1) == 2 and len(t2) == 2:
                        try:
                            t1_genders = []
                            t2_genders = []
                            for p in t1:
                                if p in players_df['name'].values:
                                    gender = players_df[players_df['name'] == p]['gender'].iloc[0]
                                    t1_genders.append(gender if pd.notna(gender) else None)
                                else:
                                    t1_genders.append(None)
                            for p in t2:
                                if p in players_df['name'].values:
                                    gender = players_df[players_df['name'] == p]['gender'].iloc[0]
                                    t2_genders.append(gender if pd.notna(gender) else None)
                                else:
                                    t2_genders.append(None)
                            if (None not in t1_genders and None not in t2_genders and 
                                sorted(t1_genders) == ['F', 'M'] and sorted(t2_genders) == ['F', 'M']):
                                is_mixed_doubles = True
                        except KeyError as e:
                            st.warning(f"Gender column missing for match {row.get('match_id', 'unknown')}. Treating as regular doubles.")
                            is_mixed_doubles = False
                    display_matches.at[idx, 'Match Type'] = 'Mixed Doubles Match' if is_mixed_doubles else 'Doubles Match'
        else:
            display_matches = pd.DataFrame()
    else:
        display_matches = pd.DataFrame()

    # ────────────────────────────────────────────────
    #   Helper: format players line (your original logic)
    # ────────────────────────────────────────────────
    def format_match_players(row):
        verb, _ = get_match_verb_and_gda(row)
        if row["match_type"] == "Singles":
            p1_styled = f"<span style='font-weight:bold; color:#fff500;'>{row['team1_player1']}</span>"
            p2_styled = f"<span style='font-weight:bold; color:#fff500;'>{row['team2_player1']}</span>"
            if row["winner"] == "Tie":
                return f"{p1_styled} tied with {p2_styled}"
            elif row["winner"] == "Team 1":
                return f"{p1_styled} {verb} {p2_styled}"
            else:  # Team 2
                return f"{p2_styled} {verb} {p1_styled}"
        else:  # Doubles
            p1_styled = f"<span style='font-weight:bold; color:#fff500;'>{row['team1_player1']}</span>"
            p2_styled = f"<span style='font-weight:bold; color:#fff500;'>{row['team1_player2']}</span>"
            p3_styled = f"<span style='font-weight:bold; color:#fff500;'>{row['team2_player1']}</span>"
            p4_styled = f"<span style='font-weight:bold; color:#fff500;'>{row['team2_player2']}</span>"
            if row["winner"] == "Tie":
                return f"{p1_styled} & {p2_styled} tied with {p3_styled} & {p4_styled}"
            elif row["winner"] == "Team 1":
                return f"{p1_styled} & {p2_styled} {verb} {p3_styled} & {p4_styled}"
            else:  # Team 2
                return f"{p3_styled} & {p4_styled} {verb} {p1_styled} & {p2_styled}"

    # ────────────────────────────────────────────────
    #   IMPROVED: format scores + GDA (this is the main fix)
    # ────────────────────────────────────────────────
    def format_match_scores_and_date(row):
        winner = row.get('winner', '')
        team1_won_match = (winner == "Team 1")
        
        score_parts = []
        
        for s in [row['set1'], row['set2'], row['set3']]:
            if not s or str(s).strip() == "":
                continue
                
            if "Tie Break" in str(s):
                numbers = re.findall(r'\d+', str(s))
                if len(numbers) == 2:
                    tb1, tb2 = int(numbers[0]), int(numbers[1])
                    # Determine who actually won this tie-break set
                    tb_winner_team1 = tb1 > tb2
                    if tb_winner_team1:
                        score_parts.append(f"7-6({tb1}-{tb2})")
                    else:
                        score_parts.append(f"7-6({tb2}-{tb1})")
                else:
                    score_parts.append(str(s).strip())
                continue
            
            # Regular set
            if '-' in str(s):
                try:
                    t1_games, t2_games = map(int, str(s).split('-'))
                    # Show from perspective of match winner (or team1 if tie)
                    if team1_won_match or winner == "Tie":
                        score_parts.append(f"{t1_games}-{t2_games}")
                    else:
                        score_parts.append(f"{t2_games}-{t1_games}")
                except ValueError:
                    score_parts.append(str(s).strip())
            else:
                score_parts.append(str(s).strip())
        
        score_text = " | ".join(score_parts) if score_parts else "No score recorded"
        
        # ───── GDA calculation ─────
        team1_games = 0
        team2_games = 0
        set_count = 0
        
        for s in [row['set1'], row['set2'], row['set3']]:
            if not s or str(s).strip() == "":
                continue
            set_count += 1
            try:
                if "Tie Break" in str(s):
                    nums = re.findall(r'\d+', str(s))
                    if len(nums) == 2:
                        a, b = int(nums[0]), int(nums[1])
                        if a > b:
                            team1_games += 7
                            team2_games += 6
                        else:
                            team1_games += 6
                            team2_games += 7
                else:
                    g1, g2 = map(int, str(s).split('-'))
                    team1_games += g1
                    team2_games += g2
            except:
                pass  # skip invalid sets
        
        if set_count == 0:
            gda = 0.0
        else:
            diff = team1_games - team2_games
            if winner == "Team 2":
                diff = -diff
            elif winner == "Tie":
                diff = 0  # or keep signed — your choice
            gda = diff / set_count
        
        gda_colored = f"<span style='color:#00ff9d; font-weight:bold'>GDA: {gda:+.2f}</span>"
        
        # Date
        try:
            date_str = pd.to_datetime(row['date']).strftime('%A, %d %b')
        except:
            date_str = "Unknown Date"
        
        # Final HTML output
        return f"""
        <div style='font-family: monospace; white-space: pre;'>
            {score_text}  {gda_colored}<br>
            <span style='color:#aaa; font-size:0.95em'>{date_str}</span>
        </div>
        """

    # ────────────────────────────────────────────────
    #   Render loop — your original layout preserved
    # ────────────────────────────────────────────────
    if display_matches.empty:
        st.info("No matches found for the selected filters.")
    else:
        for idx, row in display_matches.iterrows():
            cols = st.columns([1, 1, 7, 1])
            with cols[0]:
                st.markdown(f"<span style='font-weight:bold; color:#fff500;'>{row['serial_number']}</span>", unsafe_allow_html=True)
            with cols[1]:
                match_image_url = row.get("match_image_url")
                if match_image_url:
                    try:
                        st.image(match_image_url, width=50, caption="")
                        card_key = f"download_match_card_{row['match_id']}_{idx}"
                        card_bytes = generate_match_card(pd.Series(row.to_dict()), match_image_url)
                        st.download_button(
                            label="💌",
                            data=card_bytes,
                            file_name=f"match_card_{row['match_id']}.jpg",
                            mime="image/jpeg",
                            key=card_key
                        )
                    except Exception as e:
                        st.error(f"Error displaying match image or generating card: {str(e)}")
            with cols[2]:
                st.markdown(f"{format_match_players(row)}", unsafe_allow_html=True)
                st.markdown(format_match_scores_and_date(row), unsafe_allow_html=True)
                match_type_code = f'<span style="color:#FF4F00"><b>{row["Match Type"]}</b></span>'
                st.markdown(match_type_code, unsafe_allow_html=True)
            with cols[3]:
                share_link = generate_whatsapp_link(row)   # assuming this function still exists
                st.markdown(f'<a href="{share_link}" target="_blank" style="text-decoration:none; color:#ffffff;"><img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg" alt="WhatsApp Share" style="width:30px;height:30px;"/></a>', unsafe_allow_html=True)
            
            st.markdown("<hr style='border-top: 1px solid #333333; margin: 10px 0;'>", unsafe_allow_html=True)

    # (your manage existing matches section remains unchanged below)
    # ... paste your original "Manage Existing Matches" code here if needed ...

    

    # Manage existing matches
        
    st.markdown("---")
    st.subheader("✏️ Manage Existing Matches")
    if 'edit_match_key' not in st.session_state:
        st.session_state.edit_match_key = 0
    
    if st.session_state.matches_df.empty:
        st.info("No matches available to manage.")
    else:
        # Ensure date is in datetime format and sort by date descending (latest first)
        matches_df = st.session_state.matches_df.copy()
        if 'date' in matches_df.columns:
            matches_df['date'] = pd.to_datetime(matches_df['date'], errors='coerce')
            matches_df = matches_df.sort_values(by='date', ascending=False).reset_index(drop=True)
        
        match_options = []
        for _, row in matches_df.iterrows():
            date_str = row['date'].strftime('%A, %d %b %Y') if pd.notnull(row['date']) else "Unknown Date"
            players = [p for p in [row['team1_player1'], row['team1_player2'], row['team2_player1'], row['team2_player2']] if p and p != ""]
            players_str = ", ".join(players) if players else "No players"
            winner = row['winner'] if row['winner'] else "Unknown"
            desc = f"Date: {date_str} | Match Type: {row['match_type']} | Players: {players_str} | Winner: {winner} | Match ID: {row['match_id']}"
            match_options.append(desc)
    
        selected_match = st.selectbox("Select a match to edit or delete", [""] + match_options, key=f"select_match_to_edit_{st.session_state.edit_match_key}")
        if selected_match:
            match_id = selected_match.split(" | Match ID: ")[-1]
            match_row = st.session_state.matches_df[st.session_state.matches_df["match_id"] == match_id].iloc[0]
            match_idx = st.session_state.matches_df[st.session_state.matches_df["match_id"] == match_id].index[0]
    
            with st.expander("Edit Match Details", expanded=True):
                date_edit = st.date_input(
                    "Match Date *",
                    value=pd.to_datetime(match_row["date"], errors="coerce").date() if pd.notnull(match_row["date"]) else datetime.date.today(),
                    key=f"edit_match_date_{match_id}"
                )
                match_type_edit = st.radio(
                    "Match Type",
                    ["Doubles", "Singles"],
                    index=0 if match_row["match_type"] == "Doubles" else 1,
                    key=f"edit_match_type_{match_id}"
                )
                
                # Initialize player variables to avoid NameError
                t1p1_edit = match_row.get("team1_player1", "")
                t1p2_edit = match_row.get("team1_player2", "")
                t2p1_edit = match_row.get("team2_player1", "")
                t2p2_edit = match_row.get("team2_player2", "")
                
                if match_type_edit == "Doubles":
                    col1, col2 = st.columns(2)
                    with col1:
                        t1p1_index = available_players.index(t1p1_edit) + 1 if t1p1_edit and t1p1_edit in available_players else 0
                        t1p1_edit = st.selectbox(
                            "Team 1 - Player 1 *",
                            [""] + available_players,
                            index=t1p1_index,
                            key=f"edit_t1p1_{match_id}"
                        )
                        t1p2_index = available_players.index(t1p2_edit) + 1 if t1p2_edit and t1p2_edit in available_players else 0
                        t1p2_edit = st.selectbox(
                            "Team 1 - Player 2 *",
                            [""] + available_players,
                            index=t1p2_index,
                            key=f"edit_t1p2_{match_id}"
                        )
                    with col2:
                        t2p1_index = available_players.index(t2p1_edit) + 1 if t2p1_edit and t2p1_edit in available_players else 0
                        t2p1_edit = st.selectbox(
                            "Team 2 - Player 1 *",
                            [""] + available_players,
                            index=t2p1_index,
                            key=f"edit_t2p1_{match_id}"
                        )
                        t2p2_index = available_players.index(t2p2_edit) + 1 if t2p2_edit and t2p2_edit in available_players else 0
                        t2p2_edit = st.selectbox(
                            "Team 2 - Player 2 *",
                            [""] + available_players,
                            index=t2p2_index,
                            key=f"edit_t2p2_{match_id}"
                        )
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        t1p1_index = available_players.index(t1p1_edit) + 1 if t1p1_edit and t1p1_edit in available_players else 0
                        t1p1_edit = st.selectbox(
                            "Player 1 *",
                            [""] + available_players,
                            index=t1p1_index,
                            key=f"edit_s1p1_{match_id}"
                        )
                    with col2:
                        t2p1_index = available_players.index(t2p1_edit) + 1 if t2p1_edit and t2p1_edit in available_players else 0
                        t2p1_edit = st.selectbox(
                            "Player 2 *",
                            [""] + available_players,
                            index=t2p1_index,
                            key=f"edit_s1p2_{match_id}"
                        )
                    # For singles, clear partner fields
                    t1p2_edit = ""
                    t2p2_edit = ""
    
                # Safe index for set scores
                def safe_index(lst, value):
                    try:
                        return lst.index(value) + 1 if value in lst else 0
                    except ValueError:
                        return 0
    
                set_scores = tennis_scores()
                set1_edit = st.selectbox(
                    "Set 1 Score *",
                    [""] + set_scores,
                    index=safe_index(set_scores, match_row["set1"]),
                    key=f"edit_set1_{match_id}"
                )
                set2_edit = st.selectbox(
                    "Set 2 Score (optional)",
                    [""] + set_scores,
                    index=safe_index(set_scores, match_row["set2"]),
                    key=f"edit_set2_{match_id}"
                )
                set3_edit = st.selectbox(
                    "Set 3 Score (optional)",
                    [""] + set_scores,
                    index=safe_index(set_scores, match_row["set3"]),
                    key=f"edit_set3_{match_id}"
                )
                winner_options = ["Team 1", "Team 2", "Tie"]
                winner_index = winner_options.index(match_row["winner"]) if match_row["winner"] in winner_options else 0
                winner_edit = st.radio(
                    "Winner *",
                    winner_options,
                    index=winner_index,
                    key=f"edit_winner_{match_id}"
                )
                uploaded_image_edit = st.file_uploader(
                    "Update Match Image (optional)",
                    type=["jpg", "jpeg", "png", "gif", "bmp", "webp"],
                    key=f"edit_match_image_{match_id}"
                )
                st.markdown("*Required fields", unsafe_allow_html=True)
    
                col_save, col_delete = st.columns(2)
                with col_save:
                    if st.button("Save Changes", key=f"save_match_changes_{match_id}"):
                        valid = True
                        if not date_edit or not set1_edit or not winner_edit:
                            st.error("Please fill in all required fields: Match Date, Set 1 Score, and Winner.")
                            valid = False
                        if valid:
                            if match_type_edit == "Doubles":
                                if not all([t1p1_edit, t1p2_edit, t2p1_edit, t2p2_edit]):
                                    st.error("All player fields are required for doubles.")
                                    valid = False
                                elif len(set([t1p1_edit, t1p2_edit, t2p1_edit, t2p2_edit])) != 4:
                                    st.error("Please select different players for each position.")
                                    valid = False
                            else:
                                if not t1p1_edit or not t2p1_edit:
                                    st.error("Both player fields are required for singles.")
                                    valid = False
                                elif t1p1_edit == t2p1_edit:
                                    st.error("Please select different players for singles.")
                                    valid = False
    
                        if valid:
                            team1_sets_won = 0
                            team2_sets_won = 0
                            sets = [set1_edit, set2_edit, set3_edit]
                            valid_sets = [s for s in sets if s and s != ""]
                            for score in valid_sets:
                                try:
                                    if "Tie Break" in score:
                                        scores = [int(s) for s in re.findall(r'\d+', score)]
                                        if len(scores) != 2:
                                            st.error(f"Invalid tie break score: {score}. Please use formats like 'Tie Break 10-7'.")
                                            valid = False
                                            break
                                        t1, t2 = scores
                                    else:
                                        t1, t2 = map(int, score.split("-"))
                                    if t1 > t2:
                                        team1_sets_won += 1
                                    elif t2 > t1:
                                        team2_sets_won += 1
                                except (ValueError, TypeError) as e:
                                    st.error(f"Invalid score: {score}. Please use formats like '6-4' or 'Tie Break 10-7'.")
                                    valid = False
                                    break
    
                            if valid:
                                if len(valid_sets) == 1 and winner_edit != "Tie":
                                    st.error("A match with only one set should be a tie or have additional sets.")
                                    valid = False
                                elif len(valid_sets) >= 2:
                                    if team1_sets_won > team2_sets_won and winner_edit != "Team 1":
                                        st.error("Team 1 won more sets based on scores. Please select Team 1 as the winner or correct the scores.")
                                        valid = False
                                    elif team2_sets_won > team1_sets_won and winner_edit != "Team 2":
                                        st.error("Team 2 won more sets based on scores. Please select Team 2 as the winner or correct the scores.")
                                        valid = False
                                    elif team1_sets_won == team2_sets_won and winner_edit != "Tie":
                                        st.error("Teams won an equal number of sets. Please select 'Tie' as the winner or correct the scores.")
                                        valid = False
    
                        if valid:
                            try:
                                image_url_edit = match_row["match_image_url"]
                                if uploaded_image_edit:
                                    image_url_edit = upload_image_to_github(uploaded_image_edit, match_id, image_type="match")
                                updated_match = {
                                    "match_id": match_id,
                                    "date": pd.to_datetime(date_edit).isoformat(),
                                    "match_type": match_type_edit,
                                    "team1_player1": t1p1_edit,
                                    "team1_player2": t1p2_edit if match_type_edit == "Doubles" else None,
                                    "team2_player1": t2p1_edit,
                                    "team2_player2": t2p2_edit if match_type_edit == "Doubles" else None,
                                    "set1": set1_edit,
                                    "set2": set2_edit if set2_edit else None,
                                    "set3": set3_edit if set3_edit else None,
                                    "winner": winner_edit,
                                    "match_image_url": image_url_edit
                                }
                                st.session_state.matches_df.loc[match_idx] = updated_match
                                save_matches(st.session_state.matches_df)
                                load_matches()
                                st.success("Match updated successfully.")
                                st.session_state.edit_match_key += 1
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to save match: {str(e)}")
                                st.session_state.edit_match_key += 1
                                st.rerun()
    
                with col_delete:
                    if st.button("🗑️ Delete This Match", key=f"delete_match_{match_id}"):
                        try:
                            delete_match_from_db(match_id)
                            load_matches()
                            st.success("Match deleted successfully.")
                            st.session_state.edit_match_key += 1
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete match: {str(e)}")
                            st.session_state.edit_match_key += 1
                            st.rerun()
























#----------------END OF TAB[1]-----------------------------------------------------------





with tabs[2]:
    st.header("Player Profile")
    with st.expander("Add, Edit or Remove Player", expanded=False, icon="➡️"):
        st.markdown("##### Add New Player")
        new_player = st.text_input("Player Name *", key="new_player_input").strip()
        new_gender = st.radio("Gender *", ["M", "F"], index=None, key="new_player_gender", horizontal=True)
        st.markdown("*Required fields", unsafe_allow_html=True)
        if st.button("Add Player", key="add_player_button"):
            if not new_player:
                st.warning("Please enter a player name.")
            elif new_gender is None:
                st.warning("Please select a gender.")
            elif new_player.lower() == "visitor":
                st.warning("The name 'Visitor' is reserved and cannot be added.")
            elif new_player in st.session_state.players_df["name"].tolist():
                st.warning(f"{new_player} already exists.")
            else:
                new_player_data = {
                    "name": new_player,
                    "profile_image_url": "",
                    "birthday": "",
                    "gender": new_gender
                }
                st.session_state.players_df = pd.concat([st.session_state.players_df, pd.DataFrame([new_player_data])], ignore_index=True)
                save_players(st.session_state.players_df)
                load_players()
                st.success(f"{new_player} added.")
                st.rerun()
        
        st.markdown("---")
        st.markdown("##### Edit or Remove Existing Player")
        if 'players_df' in st.session_state and not st.session_state.players_df.empty:
            players = sorted([p for p in st.session_state.players_df["name"].dropna().tolist() if p != "Visitor"]) if "name" in st.session_state.players_df.columns else []
            if not players:
                st.info("No players available. Add a new player to begin.")
            else:
                selected_player = st.selectbox("Select Player", [""] + players, key="manage_player_select")
                if selected_player:
                    player_data = st.session_state.players_df[st.session_state.players_df["name"] == selected_player].iloc[0]
                    current_image = player_data.get("profile_image_url", "")
                    current_birthday = player_data.get("birthday", "")
                    current_gender = player_data.get("gender", "M")  # Default to "M" if gender is missing
                    st.markdown(f"**Current Profile for {selected_player}**")
                    if current_image:
                        st.image(current_image, width=100)
                    else:
                        st.write("No profile image set.")
                    
                    with st.expander("Edit Player Details", expanded=True):
                        new_name = st.text_input("Player Name *", value=player_data["name"], key=f"name_edit_{selected_player}")
                        # Birthday inputs (day and month)
                        default_day = 1
                        default_month = 1
                        if current_birthday and isinstance(current_birthday, str) and re.match(r'^\d{2}-\d{2}$', current_birthday):
                            try:
                                day_str, month_str = current_birthday.split("-")
                                default_day = int(day_str)
                                default_month = int(month_str)
                            except (ValueError, IndexError):
                                pass
                        birthday_day = st.number_input("Birthday Day", min_value=1, max_value=31, value=default_day, key=f"birthday_day_{selected_player}")
                        birthday_month = st.number_input("Birthday Month", min_value=1, max_value=12, value=default_month, key=f"birthday_month_{selected_player}")
                        # Gender selector
                        gender_edit = st.radio("Gender *", ["M", "F"], index=0 if current_gender == "M" else 1, key=f"gender_edit_{selected_player}", horizontal=True)
                        profile_image = st.file_uploader("Upload New Profile Image (optional)", type=["jpg", "jpeg", "png", "gif", "bmp", "webp"], key=f"profile_image_upload_{selected_player}")
                        st.markdown("*Required fields", unsafe_allow_html=True)
                        
                        col_save, col_delete = st.columns(2)
                        with col_save:
                            if st.button("Save Profile Changes", key=f"save_profile_changes_{selected_player}"):
                                if not new_name.strip():
                                    st.error("Player name is required.")
                                elif new_name.lower() == "visitor":
                                    st.error("The name 'Visitor' is reserved and cannot be used.")
                                elif new_name != selected_player and new_name in st.session_state.players_df["name"].tolist():
                                    st.error(f"{new_name} already exists. Choose a different name.")
                                else:
                                    image_url = current_image
                                    if profile_image:
                                        image_url = upload_image_to_github(
                                            profile_image,
                                            re.sub(r'[^a-zA-Z0-9]', '_', new_name.lower()),
                                            image_type="profile"
                                        )
                                    updated_player = {
                                        "name": new_name,
                                        "profile_image_url": image_url,
                                        "birthday": f"{birthday_day:02d}-{birthday_month:02d}",
                                        "gender": gender_edit
                                    }
                                    try:
                                        # Update the specific row in the DataFrame using index
                                        player_index = st.session_state.players_df[st.session_state.players_df["name"] == selected_player].index[0]
                                        st.session_state.players_df.iloc[player_index] = pd.Series(updated_player)
                                        # Ensure no null names in DataFrame
                                        st.session_state.players_df["name"] = st.session_state.players_df["name"].fillna("")
                                        st.session_state.players_df = st.session_state.players_df[st.session_state.players_df["name"] != ""]
                                        save_players(st.session_state.players_df)
                                        load_players()
                                        st.success(f"Profile for {new_name} updated.")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to save player: {str(e)}")
                        
                        with col_delete:
                            delete_password = st.text_input("Admin Password to Delete", type="password", key=f"delete_password_{selected_player}")
                            if st.button("🗑️ Remove Player", key=f"remove_player_{selected_player}"):
                                try:
                                    admin_password = st.secrets["admin"]["password"]
                                except KeyError:
                                    st.error("Admin password not configured in secrets. Contact the administrator.")
                                    admin_password = None
                                
                                if selected_player.lower() == "visitor":
                                    st.warning("The 'Visitor' player cannot be removed.")
                                elif admin_password is None:
                                    st.error("Deletion aborted due to missing admin password configuration.")
                                elif delete_password != admin_password:
                                    st.error("Incorrect admin password. Deletion aborted.")
                                else:
                                    try:
                                        # Replace player with "Visitor" in matches
                                        matches_mask = (
                                            (st.session_state.matches_df["team1_player1"] == selected_player) |
                                            (st.session_state.matches_df["team1_player2"] == selected_player) |
                                            (st.session_state.matches_df["team2_player1"] == selected_player) |
                                            (st.session_state.matches_df["team2_player2"] == selected_player)
                                        )
                                        if matches_mask.any():
                                            st.session_state.matches_df.loc[matches_mask, "team1_player1"] = st.session_state.matches_df.loc[matches_mask, "team1_player1"].replace(selected_player, "Visitor")
                                            st.session_state.matches_df.loc[matches_mask, "team1_player2"] = st.session_state.matches_df.loc[matches_mask, "team1_player2"].replace(selected_player, "Visitor")
                                            st.session_state.matches_df.loc[matches_mask, "team2_player1"] = st.session_state.matches_df.loc[matches_mask, "team2_player1"].replace(selected_player, "Visitor")
                                            st.session_state.matches_df.loc[matches_mask, "team2_player2"] = st.session_state.matches_df.loc[matches_mask, "team2_player2"].replace(selected_player, "Visitor")
                                            save_matches(st.session_state.matches_df)
                                            load_matches()
                                            st.info(f"Replaced {selected_player} with 'Visitor' in associated matches.")
                                        delete_player_from_db(selected_player)
                                        st.session_state.players_df = st.session_state.players_df[st.session_state.players_df["name"] != selected_player].reset_index(drop=True)
                                        save_players(st.session_state.players_df)
                                        load_players()
                                        st.success(f"{selected_player} removed successfully.")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to delete player: {str(e)}")
        else:
            st.info("No players available to edit.")
    
    st.markdown("---")
    st.header("Player Insights")

    # --- CSS for Badges List ---
    badges_css = """
    <style>
    .badges-list-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .badge-item {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .badge-item span.badge {
        background: #fff500;
        color: #031827;
        padding: 2px 6px;
        border-radius: 6px;
        font-size: 14px;
    }
    .badge-item span.description {
        color: #bbbbbb;
        font-size: 14px;
    }
    @media (max-width: 600px) {
        .badge-item span.badge {
            font-size: 12px;
        }
        .badge-item span.description {
            font-size: 12px;
        }
    }
    </style>
    """
    st.markdown(badges_css, unsafe_allow_html=True)

    # --- Badge Explanations ---
    badge_explanations = {
        "🎯 Tie-break Monster": "Dominates tie-breaks with the most wins (clutch factor >70% in 3+ clutch matches)",
        "🔥 Hot Streak": "Achieved a winning streak of 5 or more matches",
        "📈 Consistent Performer": "Reliable performance with low variation in game differences (consistency index <2 over 5+ matches)",
        "💪 Ironman": "Played the most matches without missing a session",
        "🔄 Comeback Kid": "Won 3 or more matches after losing the first set",
        "🚀 Most Improved": "Recent win rate (last 10 matches) is 20%+ higher than overall career win rate",
        "🥇 Game Hog": "Won the highest total number of games across all matches"
    }

    # --- Player Insights ---
    rank_df_combined, partner_stats_combined = calculate_rankings(st.session_state.matches_df)
    if players:
        display_player_insights(players, st.session_state.players_df, st.session_state.matches_df, rank_df_combined, partner_stats_combined, key_prefix="profile_")
    else:
        st.info("No players available for insights. Please add players above.")

    # --- Debugging Output ---
    st.markdown("---")
    st.header("Explanation of Badges")
    # --- All Badges Expander ---
    with st.expander("View All Badges", expanded=False, icon="➡️"):
        badges_list_html = "<div class='badges-list-container'>"
        for badge, description in badge_explanations.items():
            badges_list_html += (
                f"<div class='badge-item'>"
                f"<span class='badge'>{badge}</span>"
                f"<span class='description'>{description}</span>"
                f"</div>"
            )
        badges_list_html += "</div>"
        st.markdown(badges_list_html, unsafe_allow_html=True)

    st.markdown("---")
    st.header("Detailed explanation of Player insights")
    st.markdown("https://github.com/mahadevbk/ar2/blob/main/Player%20insights.pdf")



















#-----------------------end of TAB 2 ----------------------------------------------------------------

with tabs[3]:
    st.header("Court Locations")
    
    # Icon URL (use a free tennis court icon; you can host it or use an external link)
    court_icon_url = "https://img.icons8.com/color/48/000000/tennis.png"  # Example from Icons8; replace if needed
    
    # Arabian Ranches courts (as a list of dicts for name and URL)
    ar_courts = [
        {"name": "Alvorado 1", "url": "https://maps.google.com/?q=25.041792,55.259258"},
        {"name": "Alvorado 2", "url": "https://maps.google.com/?q=25.041792,55.259258"},
        {"name": "Palmera 2", "url": "https://maps.app.goo.gl/CHimjtqQeCfU1d3W6"},
        {"name": "Palmera 4", "url": "https://maps.app.goo.gl/4nn1VzqMpgVkiZGN6"},
        {"name": "Saheel", "url": "https://maps.app.goo.gl/a7qSvtHCtfgvJoxJ8"},
        {"name": "Hattan", "url": "https://maps.app.goo.gl/fjGpeNzncyG1o34c7"},
        {"name": "MLC Mirador La Colleccion", "url": "https://maps.app.goo.gl/n14VSDAVFZ1P1qEr6"},
        {"name": "Al Mahra", "url": "https://maps.app.goo.gl/zVivadvUsD6yyL2Y9"},
        {"name": "Mirador", "url": "https://maps.app.goo.gl/kVPVsJQ3FtMWxyKP8"},
        {"name": "Reem 1", "url": "https://maps.app.goo.gl/qKswqmb9Lqsni5RD7"},
        {"name": "Reem 2", "url": "https://maps.app.goo.gl/oFaUFQ9DRDMsVbMu5"},
        {"name": "Reem 3", "url": "https://maps.app.goo.gl/o8z9pHo8tSqTbEL39"},
        {"name": "Alma", "url": "https://maps.app.goo.gl/BZNfScABbzb3osJ18"},
    ]
    
    # Mira & Mira Oasis courts
    mira_courts = [
        {"name": "Mira 2", "url": "https://maps.app.goo.gl/JeVmwiuRboCnzhnb9"},
        {"name": "Mira 4", "url": "https://maps.app.goo.gl/e1Vqv5MJXB1eusv6A"},
        {"name": "Mira 5 A", "url": "https://maps.app.goo.gl/rWBj5JEUdw4LqJZb6"},
        {"name": "Mira 5 B", "url": "https://maps.app.goo.gl/rWBj5JEUdw4LqJZb6"},
        {"name": "Mira Oasis 1", "url": "https://maps.app.goo.gl/F9VYsFBwUCzvdJ2t8"},
        {"name": "Mira Oasis 2", "url": "https://maps.app.goo.gl/ZNJteRu8aYVUy8sd9"},
        {"name": "Mira Oasis 3 A", "url": "https://maps.app.goo.gl/ouXQGUxYSZSfaW1z9"},
        {"name": "Mira Oasis 3 B", "url": "https://maps.app.goo.gl/ouXQGUxYSZSfaW1z9"},
        {"name": "Mira Oasis 3 C", "url": "https://maps.app.goo.gl/kf7A9K7DoYm4PEPu8"},
    ]

    # Mudon, Arabian Ranches 2, Dubai Hills and other courts
    other_courts = [
        {"name": "Mudon Main courts", "url": "https://maps.app.goo.gl/AZ8WJ1mnnwMgNxhz7?g_st=aw"},
        {"name": "Mudon Arabella", "url": "https://maps.app.goo.gl/iudbB5YqrGKyHNqM6?g_st=aw"},
        {"name": "Mudon Arabella 3", "url": "https://maps.app.goo.gl/o46ERJCq8LKg1Cz59?g_st=aw"},
        {"name": "AR2 Rosa", "url": "https://maps.app.goo.gl/at1EKgatfMmvAg7g8?g_st=aw"},
        {"name": "AR2 Palma", "url": "https://maps.app.goo.gl/oKxXvbXKYe3JgJco8?g_st=aw"},
        {"name": "AR 2 Fitness First", "url": "https://maps.app.goo.gl/iZGipHv8KdfW82dW9?g_st=aw"},
        {"name": "Dubai Hills Maple", "url": "https://maps.app.goo.gl/rypmwnSGbGeknykv6?g_st=aw"},
    ]
    


    
    # Function to display courts in a grid
    def display_courts(section_title, courts_list):
        st.subheader(section_title)
        num_cols = 3 if len(courts_list) > 6 else 2  # Responsive: 2-3 columns based on list length
        for i in range(0, len(courts_list), num_cols):
            cols = st.columns(num_cols)
            for j, court in enumerate(courts_list[i:i+num_cols]):
                with cols[j]:
                    st.markdown(f"""
                    <div class="court-card">
                        <img src="{court_icon_url}" class="court-icon" alt="Tennis Court Icon">
                        <h4>{court['name']}</h4>
                        <a href="{court['url']}" target="_blank">View on Map</a>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Display sections
    with st.expander("Arabian Ranches Tennis Courts", expanded=False, icon="➡️"):
        display_courts("", ar_courts)  # No extra title inside expander
    with st.expander("Mira & Mira Oasis Tennis Courts", expanded=False, icon="➡️"):
        display_courts("", mira_courts)
    with st.expander("Mudon, AR2 & Other Tennis Courts", expanded=False, icon="➡️"):
        display_courts("", other_courts)  


        
#-----TAB 4 WITH THUMBNAILS INSIDE BOOKING BOX AND WHATSAPP SHARE WITH PROPER FORMATTING--------------------------------------------






with tabs[4]:
    # --- MATCH UP EXPANDER ---
    with st.expander("Match up", expanded=False, icon="➡️"):
        
        match_type = st.radio("Select Match Type", ["Doubles", "Singles"], horizontal=True)

        if match_type == "Doubles":
            t1p1 = st.selectbox("Team 1 - Player 1", [""] + available_players, key="matchup_doubles_t1p1")
            t1p2 = st.selectbox("Team 1 - Player 2", [""] + available_players, key="matchup_doubles_t1p2")
            t2p1 = st.selectbox("Team 2 - Player 1", [""] + available_players, key="matchup_doubles_t2p1")
            t2p2 = st.selectbox("Team 2 - Player 2", [""] + available_players, key="matchup_doubles_t2p2")

            if st.button("Match up", key="btn_matchup_doubles"):
                st.subheader("Match Odds")
                players = [t1p1, t1p2, t2p1, t2p2]
                doubles_rank_df, _ = calculate_rankings(
                    st.session_state.matches_df[st.session_state.matches_df['match_type']=="Doubles"]
                )
                if all(p in doubles_rank_df["Player"].values for p in players if p):
                    pairing_text, team1_odds, team2_odds = suggest_balanced_pairing(players, doubles_rank_df)
                    if pairing_text:
                        st.markdown(pairing_text, unsafe_allow_html=True)
                        st.write(f"Team 1: {team1_odds:.1f}% | Team 2: {team2_odds:.1f}%")
                    else:
                        st.info("No odds available for this combination.")
                else:
                    st.info("No odds available (one or more players have no doubles match history).")
        else:  # Singles
            p1 = st.selectbox("Player 1", [""] + available_players, key="matchup_singles_p1")
            p2 = st.selectbox("Player 2", [""] + available_players, key="matchup_singles_p2")

            if st.button("Match up", key="btn_matchup_singles"):
                st.subheader("Match Odds")
                if p1 and p2:
                    singles_rank_df, _ = calculate_rankings(
                        st.session_state.matches_df[st.session_state.matches_df['match_type']=="Singles"]
                    )
                    if p1 in singles_rank_df["Player"].values and p2 in singles_rank_df["Player"].values:
                        odds1, odds2 = suggest_singles_odds([p1, p2], singles_rank_df)
                        st.write(f"Odds → {p1}: {odds1:.1f}% | {p2}: {odds2:.1f}%")
                    else:
                        st.info("No odds available (one or both players have no singles match history).")
                else:
                    st.warning("Please select both players.")

    # --- EXISTING BOOKING MANAGEMENT ---
    load_bookings()

    with st.expander("Add New Booking", expanded=False, icon="➡️"):
        st.subheader("Add New Booking")
        match_type = st.radio("Match Type", ["Doubles", "Singles"], index=0, key=f"new_booking_match_type_{st.session_state.form_key_suffix}")
        
        with st.form(key=f"add_booking_form_{st.session_state.form_key_suffix}"):
            date = st.date_input("Booking Date *", key=f"new_booking_date_{st.session_state.form_key_suffix}")
            hours = []
            hours.append(datetime.strptime("6:00", "%H:%M").strftime("%I:%M %p").lstrip('0'))  # 6:00 AM
            hours.append(datetime.strptime("6:30", "%H:%M").strftime("%I:%M %p").lstrip('0'))  # 6:30 AM
            hours.append(datetime.strptime("7:30", "%H:%M").strftime("%I:%M %p").lstrip('0'))  # 7:30 AM
            for h in range(7, 22):  # From 7 AM to 9 PM
                hours.append(datetime.strptime(f"{h:02d}:00", "%H:%M").strftime("%I:%M %p").lstrip('0'))
            time = st.selectbox("Booking Time *", hours, key=f"new_booking_time_{st.session_state.form_key_suffix}")
            
            if match_type == "Doubles":
                col1, col2 = st.columns(2)
                with col1:
                    p1 = st.selectbox("Player 1 (optional)", [""] + available_players, key=f"new_booking_t1p1_{st.session_state.form_key_suffix}")
                    p2 = st.selectbox("Player 2 (optional)", [""] + available_players, key=f"new_booking_t1p2_{st.session_state.form_key_suffix}")
                with col2:
                    p3 = st.selectbox("Player 3 (optional)", [""] + available_players, key=f"new_booking_t2p1_{st.session_state.form_key_suffix}")
                    p4 = st.selectbox("Player 4 (optional)", [""] + available_players, key=f"new_booking_t2p2_{st.session_state.form_key_suffix}")
            else:
                p1 = st.selectbox("Player 1 (optional)", [""] + available_players, key=f"new_booking_s1p1_{st.session_state.form_key_suffix}")
                p3 = st.selectbox("Player 2 (optional)", [""] + available_players, key=f"new_booking_s1p2_{st.session_state.form_key_suffix}")
                p2 = ""
                p4 = ""
            
            standby = st.selectbox("Standby Player (optional)", [""] + available_players, key=f"new_booking_standby_{st.session_state.form_key_suffix}")
            court = st.selectbox("Court Name *", [""] + court_names, key=f"court_{st.session_state.form_key_suffix}")
            screenshot = st.file_uploader("Booking Screenshot (optional)", type=["jpg", "jpeg", "png", "gif", "bmp", "webp"], key=f"screenshot_{st.session_state.form_key_suffix}")
            st.markdown("*Required fields", unsafe_allow_html=True)
            
            submit = st.form_submit_button("Add Booking")
            if submit:
                if not court:
                    st.error("Court name is required.")
                elif not date or not time:
                    st.error("Booking date and time are required.")
                else:
                    selected_players = [p for p in [p1, p2, p3, p4, standby] if p]
                    if match_type == "Doubles" and len(set(selected_players)) != len(selected_players):
                        st.error("Please select different players for each position.")
                    else:
                        booking_id = str(uuid.uuid4())
                        screenshot_url = upload_image_to_github(screenshot, booking_id, image_type="booking") if screenshot else None
                        try:
                            time_24hr = datetime.strptime(time, "%I:%M %p").strftime("%H:%M:%S")
                        except ValueError:
                            st.error("Invalid time format. Please select a valid time.")
                            st.rerun()
                        new_booking = {
                            "booking_id": booking_id,
                            "date": date.isoformat(),
                            "time": time_24hr,
                            "match_type": match_type,
                            "court_name": court,
                            "player1": p1 if p1 else None,
                            "player2": p2 if p2 else None,
                            "player3": p3 if p3 else None,
                            "player4": p4 if p4 else None,
                            "standby_player": standby if standby else None,
                            "screenshot_url": screenshot_url
                        }
                        st.session_state.bookings_df = pd.concat([st.session_state.bookings_df, pd.DataFrame([new_booking])], ignore_index=True)
                        try:
                            expected_columns = ['booking_id', 'date', 'time', 'match_type', 'court_name', 'player1', 'player2', 'player3', 'player4', 'standby_player', 'screenshot_url']
                            bookings_to_save = st.session_state.bookings_df[expected_columns].copy()
                            for col in ['player1', 'player2', 'player3', 'player4', 'standby_player', 'screenshot_url']:
                                bookings_to_save[col] = bookings_to_save[col].replace("", None)
                            save_bookings(bookings_to_save)
                            load_bookings()
                            st.success("Booking added successfully.")
                            st.session_state.form_key_suffix += 1
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to save booking: {str(e)}")
                            st.rerun()  


    st.markdown("---")
    st.subheader("📅 Upcoming Bookings")
    bookings_df = st.session_state.bookings_df.copy()
    court_url_mapping = {court["name"]: court["url"] for court in ar_courts + mira_courts}
    if bookings_df.empty:
        st.info("No upcoming bookings found.")
    else:
        if 'standby_player' not in bookings_df.columns:
            bookings_df['standby_player'] = ""
        if 'standby' in bookings_df.columns:
            bookings_df = bookings_df.drop(columns=['standby'])
        if 'players' in bookings_df.columns:
            bookings_df = bookings_df.drop(columns=['players'])
        
        # Create datetime column with explicit timezone handling
        bookings_df['datetime'] = pd.to_datetime(
            bookings_df['date'].astype(str) + ' ' + bookings_df['time'],
            errors='coerce',
            format='%Y-%m-%d %H:%M:%S'
        ).dt.tz_localize('Asia/Dubai')
        
        # Debug: Display datetime values
        # st.write("Debug - Bookings datetime:", bookings_df[['booking_id', 'date', 'time', 'datetime']])
        
        upcoming_bookings = bookings_df[
            (bookings_df['datetime'].notna()) & 
            (bookings_df['datetime'] >= pd.Timestamp.now(tz='Asia/Dubai'))
        ].sort_values('datetime')
        
        if upcoming_bookings.empty:
            st.info("No upcoming bookings found.")
        else:
            try:
                doubles_matches_df = st.session_state.matches_df[st.session_state.matches_df['match_type'] == 'Doubles']
                singles_matches_df = st.session_state.matches_df[st.session_state.matches_df['match_type'] == 'Singles']
                doubles_rank_df, _ = calculate_rankings(doubles_matches_df)
                singles_rank_df, _ = calculate_rankings(singles_matches_df)
            except Exception as e:
                doubles_rank_df = pd.DataFrame()
                singles_rank_df = pd.DataFrame()
                st.warning(f"Unable to load rankings for pairing odds: {str(e)}")
            
            for _, row in upcoming_bookings.iterrows():
                players = [p for p in [row['player1'], row['player2'], row['player3'], row['player4']] if p]
                players_str = ", ".join([f"<span style='font-weight:bold; color:#fff500;'>{p}</span>" for p in players]) if players else "No players specified"
                standby_str = f"<span style='font-weight:bold; color:#fff500;'>{row['standby_player']}</span>" if row['standby_player'] else "None"
                date_str = pd.to_datetime(row['date']).strftime('%A, %d %b')
                time_value = str(row['time']).strip()
            
                time_ampm = ""
                if time_value and time_value not in ["NaT", "nan", "None"]:
                    try:
                        dt_obj = datetime.strptime(time_value, "%H:%M:%S")
                        time_ampm = dt_obj.strftime("%I:%M %p").lstrip('0')
                    except ValueError:
                        try:
                            dt_obj = datetime.strptime(time_value, "%H:%M")
                            time_ampm = dt_obj.strftime("%I:%M %p").lstrip('0')
                        except ValueError:
                            time_ampm = "Invalid Time"
                
                court_url = court_url_mapping.get(row['court_name'], "#")
                court_name_html = f"<a href='{court_url}' target='_blank' style='font-weight:bold; color:#fff500; text-decoration:none;'>{row['court_name']}</a>"
            
                pairing_suggestion = ""
                plain_suggestion = ""
                try:
                    if row['match_type'] == "Doubles" and len(players) == 4:
                        rank_df = doubles_rank_df
                        unranked = [p for p in players if p not in rank_df["Player"].values]
                        if unranked:
                            styled_unranked = ", ".join([f"<span style='font-weight:bold; color:#fff500;'>{p}</span>" for p in unranked])
                            message = f"Players {styled_unranked} are unranked, therefore no pairing odds available."
                            pairing_suggestion = f"<div><strong style='color:white;'>Pairing Odds:</strong> {message}</div>"
                            plain_suggestion = f"Players {', '.join(unranked)} are unranked, therefore no pairing odds available."
                        else:
                            all_pairings = []
                            player_list = list(players)
                            seen_pairings = set()
                            for team1 in combinations(player_list, 2):
                                team1_set = frozenset(team1)
                                team2 = tuple(p for p in player_list if p not in team1)
                                team2_set = frozenset(team2)
                                pairing_key = frozenset([team1_set, team2_set])
                                if pairing_key in seen_pairings:
                                    continue
                                seen_pairings.add(pairing_key)
                                team1_score = sum(_calculate_performance_score(rank_df[rank_df['Player'] == p].iloc[0], rank_df) for p in team1)
                                team2_score = sum(_calculate_performance_score(rank_df[rank_df['Player'] == p].iloc[0], rank_df) for p in team2)
                                diff = abs(team1_score - team2_score)
                                odds_team1 = (team1_score / (team1_score + team2_score)) * 100 if team1_score + team2_score > 0 else 50
                                odds_team2 = 100 - odds_team1
                                team1_str = ", ".join([f"<span style='font-weight:bold; color:#fff500;'>{p}</span>" for p in team1])
                                team2_str = ", ".join([f"<span style='font-weight:bold; color:#fff500;'>{p}</span>" for p in team2])
                                pairing_str = f"{team1_str} vs {team2_str}"
                                plain_pairing_str = f"{', '.join(team1)} vs {', '.join(team2)}"
                                all_pairings.append({
                                    'pairing': pairing_str,
                                    'plain_pairing': plain_pairing_str,
                                    'team1_odds': odds_team1,
                                    'team2_odds': odds_team2,
                                    'diff': diff
                                })
                            all_pairings.sort(key=lambda x: x['diff'])
                            pairing_suggestion = "<div><strong style='color:white;'>Pairing Combos and Odds:</strong></div>"
                            plain_suggestion = "*Pairing Combos and Odds:* | "
                            for idx, pairing in enumerate(all_pairings[:3], 1):
                                pairing_suggestion += (
                                    f"<div>Option {idx}: {pairing['pairing']} "
                                    f"(<span style='font-weight:bold; color:#fff500;'>{pairing['team1_odds']:.1f}%</span> vs "
                                    f"<span style='font-weight:bold; color:#fff500;'>{pairing['team2_odds']:.1f}%</span>)</div>"
                                )
                                plain_suggestion += (
                                    f"Option {idx}: {pairing['plain_pairing']} ({pairing['team1_odds']:.1f}% vs {pairing['team2_odds']:.1f}%) | "
                                )
                            plain_suggestion = plain_suggestion.rstrip(" | ")
                    elif row['match_type'] == "Doubles" and len(players) < 4:
                        pairing_suggestion = "<div><strong style='color:white;'>Pairing Odds:</strong> Not enough players for pairing odds</div>"
                        plain_suggestion = "Not enough players for pairing odds"
                    elif row['match_type'] == "Singles" and len(players) == 2:
                        rank_df = singles_rank_df
                        unranked = [p for p in players if p not in rank_df["Player"].values]
                        if unranked:
                            styled_unranked = ", ".join([f"<span style='font-weight:bold; color:#fff500;'>{p}</span>" for p in unranked])
                            message = f"Players {styled_unranked} are unranked, therefore no odds available."
                            pairing_suggestion = f"<div><strong style='color:white;'>Odds:</strong> {message}</div>"
                            plain_suggestion = f"Players {', '.join(unranked)} are unranked, therefore no odds available."
                        else:
                            p1_odds, p2_odds = suggest_singles_odds(players, singles_rank_df)
                            if p1_odds is not None:
                                p1_styled = f"<span style='font-weight:bold; color:#fff500;'>{players[0]}</span>"
                                p2_styled = f"<span style='font-weight:bold; color:#fff500;'>{players[1]}</span>"
                                pairing_suggestion = (
                                    f"<div><strong style='color:white;'>Odds:</strong> "
                                    f"{p1_styled} ({p1_odds:.1f}%) vs {p2_styled} ({p2_odds:.1f}%)</div>"
                                )
                                plain_suggestion = f"Odds: {players[0]} ({p1_odds:.1f}%) vs {players[1]} ({p2_odds:.1f}%)"
                except Exception as e:
                    pairing_suggestion = f"<div><strong style='color:white;'>Pairing Odds:</strong> Error calculating: {e}</div>"
                    plain_suggestion = f"Error calculating odds: {str(e)}"
                
                # Generate ICS for calendar
                ics_content, ics_error = generate_ics_for_booking(row, plain_suggestion)
                calendar_link = ""
                if ics_content:
                    encoded_ics = urllib.parse.quote(ics_content)
                    calendar_link = f'data:text/calendar;charset=utf8,{encoded_ics}'
                else:
                    calendar_link = "#"
                    st.warning(f"Calendar add failed for booking {row['booking_id']}: {ics_error}")
                
                weekday = pd.to_datetime(row['date']).strftime('%a')
                date_part = pd.to_datetime(row['date']).strftime('%d %b')
                full_date = f"{weekday}, {date_part}, {time_ampm}"
                court_name = row['court_name']
                players_list = ", ".join([f"{i+1}. *{p}*" for i, p in enumerate(players)]) if players else "No players"
                standby_text = f" | STD. BY: *{row['standby_player']}*" if row['standby_player'] else ""
                
                share_text = f"*Game Booking:* Date: *{full_date}* | Court: *{court_name}* | Players: {players_list}{standby_text} | {plain_suggestion} | Court location: {court_url}"
                encoded_text = urllib.parse.quote(share_text)
                whatsapp_link = f"https://api.whatsapp.com/send/?text={encoded_text}&type=custom_url&app_absent=0"
                
                booking_text = f"""
                <div class="booking-row" style='background-color: rgba(255, 255, 255, 0.1); padding: 10px; border-radius: 8px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
                    <div><strong>Date:</strong> <span style='font-weight:bold; color:#fff500;'>{date_str}</span></div>
                    <div><strong>Court:</strong> {court_name_html}</div>
                    <div><strong>Time:</strong> <span style='font-weight:bold; color:#fff500;'>{time_ampm}</span></div>
                    <div><strong>Match Type:</strong> <span style='font-weight:bold; color:#fff500;'>{row['match_type']}</span></div>
                    <div><strong>Players:</strong> {players_str}</div>
                    <div><strong>Standby Player:</strong> {standby_str}</div>
                    {pairing_suggestion}
                    <div style="margin-top: 10px; display: flex; align-items: center; gap: 10px;">
                        <a href="{whatsapp_link}" class="whatsapp-share" target="_blank">
                            <img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg" alt="WhatsApp" style="width: 30px; height: 30px;">
                        </a>
                        <a href="{calendar_link}" class="calendar-share" download="tennis-booking-{row['booking_id']}.ics" target="_blank">
                            <img src="https://img.icons8.com/color/48/000000/calendar.png" alt="Add to Calendar" style="width: 30px; height: 30px;">
                        </a>
                    </div>
                """
                
                visuals_html = '<div style="display: flex; flex-direction: row; align-items: center; margin-top: 10px;">'
                screenshot_url = row["screenshot_url"] if row["screenshot_url"] and isinstance(row["screenshot_url"], str) else None
                if screenshot_url:
                    visuals_html += f'<a href="{screenshot_url}" target="_blank"><img src="{screenshot_url}" style="width:120px; margin-right:20px; cursor:pointer;" title="Click to view full-size"></a>'
                visuals_html += '<div style="display: flex; flex-direction: row; align-items: center; flex-wrap: nowrap;">'
                booking_players = [row['player1'], row['player2'], row['player3'], row['player4'], row.get('standby_player', '')]
                players_df = st.session_state.players_df
                image_urls = []
                placeholder_initials = []
                for player_name in booking_players:
                    if player_name and isinstance(player_name, str) and player_name.strip() and player_name != "Visitor":
                        player_data = players_df[players_df["name"] == player_name]
                        if not player_data.empty:
                            img_url = player_data.iloc[0].get("profile_image_url")
                            if img_url and isinstance(img_url, str) and img_url.strip():
                                image_urls.append((player_name, img_url))
                            else:
                                placeholder_initials.append((player_name, player_name[0].upper()))
                for player_name, img_url in image_urls:
                    visuals_html += f'<img src="{img_url}" class="profile-image" style="width: 50px; height: 50px; margin-right: 8px;" title="{player_name}">'
                for player_name, initial in placeholder_initials:
                    visuals_html += f'<div title="{player_name}" style="width: 50px; height: 50px; margin-right: 8px; border-radius: 50%; background-color: #07314f; border: 2px solid #fff500; display: flex; align-items: center; justify-content: center; font-size: 22px; color: #fff500; font-weight: bold;">{initial}</div>'
                visuals_html += '</div></div>'
                booking_text += visuals_html + '</div>'
                
                try:
                    st.markdown(booking_text, unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Failed to render HTML for booking {row['booking_id']}: {str(e)}")
                    st.markdown(f"""
                    **Court:** {court_name_html}  
                    **Date:** {date_str}  
                    **Time:** {time_ampm}  
                    **Match Type:** {row['match_type']}  
                    **Players:** {', '.join(players) if players else 'No players'}  
                    **Standby Player:** {row.get('standby_player', 'None')}  
                    {pairing_suggestion.replace('<div><strong style="color:white;">', '**').replace('</strong>', '**').replace('</div>', '').replace('<span style="font-weight:bold; color:#fff500;">', '').replace('</span>', '')}
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <a href="{whatsapp_link}" class="whatsapp-share" target="_blank">
                            <img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg" alt="WhatsApp" style="width:30px; height:30px;">
                        </a>
                        <a href="{calendar_link}" class="calendar-share" download="tennis-booking-{row['booking_id']}.ics" target="_blank">
                            <img src="https://img.icons8.com/color/48/000000/calendar.png" alt="Add to Calendar" style="width:30px; height:30px;">
                        </a>
                    </div>
                    """, unsafe_allow_html=True)
                    if screenshot_url:
                        st.markdown(f"""
                        <a href="{screenshot_url}" target="_blank">
                            <img src="{screenshot_url}" style="width:120px; cursor:pointer;" title="Click to view full-size">
                        </a>
                        """, unsafe_allow_html=True)
                    if image_urls or placeholder_initials:
                        cols = st.columns(len(image_urls) + len(placeholder_initials))
                        col_idx = 0
                        for player_name, img_url in image_urls:
                            with cols[col_idx]:
                                st.image(img_url, width=50, caption=player_name)
                            col_idx += 1
                        for player_name, initial in placeholder_initials:
                            with cols[col_idx]:
                                st.markdown(f"""
                                <div style='width: 50px; height: 50px; border-radius: 50%; background-color: #07314f; border: 2px solid #fff500; display: flex; align-items: center; justify-content: center; font-size: 22px; color: #fff500; font-weight: bold;'>{initial}</div>
                                <div style='text-align: center;'>{player_name}</div>
                                """, unsafe_allow_html=True)
                            col_idx += 1
    
    st.markdown("---")
    
   
    
    
    
    
    
    #------------new Calendar feature -------------------------------




    

    
    # Insert this new section right after the "Upcoming Bookings" subheader and before the existing bookings_df processing


    
    
    st.markdown("---")
    st.subheader("👥 Player Availability")
    st.markdown("""
    *Players can indicate their availability for the next 10 days. This helps in scheduling matches more efficiently.*
    """)
    
    # Define next 10 days
    from datetime import datetime, timedelta
    today = datetime.now().date()
    next_10_days = [today + timedelta(days=i) for i in range(1, 11)]
    day_options = [d.strftime("%A, %d %b") for d in next_10_days]
    date_options = [d.isoformat() for d in next_10_days]
    
    # Load availability from Supabase
    availability_table_name = "availability"
    expected_columns = ["id", "player_name", "date", "comment"]
    
    if 'availability_df' not in st.session_state:
        try:
            response = supabase.table(availability_table_name).select("*").execute()
            df = pd.DataFrame(response.data)
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = ""
            # Ensure id is int if present
            if 'id' in df.columns:
                df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0).astype(int)
            # Drop any old columns like 'time_slot' if present
            if 'time_slot' in df.columns:
                df = df.drop(columns=['time_slot'])
            st.session_state.availability_df = df
        except Exception as e:
            st.error(f"Error loading availability: {str(e)}")
            st.session_state.availability_df = pd.DataFrame(columns=expected_columns)
    
    def save_availability(availability_df):
        try:
            if len(availability_df) == 0:
                return  # Skip upsert if no data
            # Select only expected columns
            availability_df_to_save = availability_df[expected_columns].copy()
            # Replace NaN with None for JSON compliance
            availability_df_to_save = availability_df_to_save.where(pd.notna(availability_df_to_save), None)
            # Ensure id is int
            if 'id' in availability_df_to_save.columns:
                availability_df_to_save['id'] = pd.to_numeric(availability_df_to_save['id'], errors='coerce').fillna(0).astype(int)
            supabase.table(availability_table_name).upsert(availability_df_to_save.to_dict("records")).execute()
            st.success("Availability updated successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error saving availability: {str(e)}")
    
    def delete_availability(player_name, date_str):
        try:
            # Delete all entries for player and date
            supabase.table(availability_table_name).delete().eq("player_name", player_name).eq("date", date_str).execute()
            st.session_state.availability_df = st.session_state.availability_df[
                ~((st.session_state.availability_df["player_name"] == player_name) & 
                  (st.session_state.availability_df["date"] == date_str))
            ].reset_index(drop=True)
            save_availability(st.session_state.availability_df)
        except Exception as e:
            st.error(f"Error deleting availability: {str(e)}")
    
    # Add/Update Availability Form (simplified: one day at a time)
    with st.expander("Add/Update Your Availability", expanded=False, icon="📅"):
        selected_player = st.selectbox("Select Player", [""] + available_players, key="avail_player")
        selected_day_label = st.selectbox("Select Day", [""] + day_options, key="avail_day")
        if selected_player and selected_day_label:
            day_date = next_10_days[day_options.index(selected_day_label)]
            comment_key = f"comment_{selected_day_label.replace(', ', '_')}"
            comment = st.text_area(
                f"Comment for {selected_day_label} (e.g., 'free from 7pm onwards')",
                key=comment_key,
                help="Describe your availability for this day"
            )
            # Show current availability for this day
            current_row = st.session_state.availability_df[
                (st.session_state.availability_df["player_name"] == selected_player) &
                (st.session_state.availability_df["date"] == day_date.isoformat())
            ]
            if not current_row.empty:
                current_comment = current_row.iloc[0].get("comment", "")
                if current_comment:
                    st.info(f"Current comment for {selected_day_label}: {current_comment}")
            
            col_update, col_clear = st.columns(2)
            with col_update:
                if st.button(f"Update Availability for {selected_day_label}", key=f"update_{selected_day_label.replace(', ', '_')}"):
                    if not comment.strip():
                        st.warning("Please add a comment to update availability.")
                    else:
                        # Remove old entries for this player/day
                        st.session_state.availability_df = st.session_state.availability_df[
                            ~((st.session_state.availability_df["player_name"] == selected_player) &
                              (st.session_state.availability_df["date"] == day_date.isoformat()))
                        ].reset_index(drop=True)
                        
                        # Get next id
                        next_id = st.session_state.availability_df['id'].max() + 1 if not st.session_state.availability_df.empty else 1
                        
                        # Add new entry
                        new_entry = {
                            "id": next_id,
                            "player_name": selected_player,
                            "date": day_date.isoformat(),
                            "comment": comment.strip()
                        }
                        st.session_state.availability_df = pd.concat([
                            st.session_state.availability_df,
                            pd.DataFrame([new_entry])
                        ], ignore_index=True)
                        
                        save_availability(st.session_state.availability_df)
            
            with col_clear:
                if st.button(f"Clear Availability for {selected_day_label}", key=f"clear_{selected_day_label.replace(', ', '_')}"):
                    delete_availability(selected_player, day_date.isoformat())
    
    # Display Availability Overview (Enhanced)
    st.markdown("---")
    st.subheader("Upcoming Availability Overview")
    
    if st.session_state.availability_df.empty:
        st.info("No availability entries yet. Add some using the form above!")
    else:
        # Filter to next 10 days
        recent_avail = st.session_state.availability_df[st.session_state.availability_df['date'].isin(date_options)]
        day_grouped = recent_avail.groupby("date")
        
        # Custom CSS for day cards (add to your existing <style> block)
        st.markdown("""
        <style>
        .availability-day-card {
            background: linear-gradient(to bottom, #031827, #07314f);
            border: 1px solid #fff500;
            border-radius: 12px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3), 0 0 10px rgba(255, 245, 0, 0.2);
            transition: transform 0.2s, box-shadow 0.2s;
            text-align: left;
        }
        .availability-day-card:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 12px rgba(255, 245, 0, 0.4);
        }
        .day-header {
            color: #fff500;
            font-weight: bold;
            font-size: 1.2em;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }
        .player-item {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            padding: 5px;
            background: rgba(255, 245, 0, 0.1);
            border-radius: 6px;
            border-left: 3px solid #fff500;
        }
        .player-name {
            font-weight: bold;
            color: #fff500;
            margin-right: 8px;
            min-width: 80px;
        }
        .player-comment {
            color: #ffffff;
            font-size: 0.9em;
            flex-grow: 1;
            white-space: pre-line;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Grid: 3 columns for desktop, stack on mobile
        for i in range(0, len(date_options), 3):
            cols = st.columns(3)
            for j, date_str in enumerate(date_options[i:i+3]):
                with cols[j]:
                    day_data = day_grouped.get_group(date_str) if date_str in day_grouped.groups else pd.DataFrame()
                    if day_data.empty:
                        continue
                    
                    day_label = next_10_days[date_options.index(date_str)].strftime("%A, %d %b")
                    player_comments = {}
                    for _, row in day_data.iterrows():
                        player = row['player_name']
                        if player not in player_comments:
                            player_comments[player] = row.get('comment', '')
                    
                    if player_comments:
                        # Build HTML card content
                        players_html = ""
                        for player, comment in sorted(player_comments.items()):
                            # For title, replace newlines with ' | ' for better tooltip display
                            title_attr = comment.replace('\n', ' | ')
                            players_html += f"""
                            <div class="player-item">
                                <span class="player-name">👤 {player}:</span>
                                <span class="player-comment" title="{title_attr}">{comment}</span>
                            </div>
                            """
                        
                        card_html = f"""
                        <div class="availability-day-card">
                            <div class="day-header">📅 {day_label}</div>
                            {players_html}
                        </div>
                        """
                        st.html(card_html)
    
    # Manage Existing Availability (optional)
    with st.expander("Manage All Availability", expanded=False, icon="⚙️"):
        if not st.session_state.availability_df.empty:
            st.dataframe(st.session_state.availability_df, width="stretch")
            
            selected_to_delete = st.multiselect("Select entries to delete (by ID)", 
                                              st.session_state.availability_df["id"].tolist(),
                                              key="delete_avail")
            if st.button("Delete Selected"):
                for entry_id in selected_to_delete:
                    # Delete single entry by id
                    try:
                        supabase.table(availability_table_name).delete().eq("id", entry_id).execute()
                        st.session_state.availability_df = st.session_state.availability_df[
                            st.session_state.availability_df["id"] != entry_id
                        ].reset_index(drop=True)
                    except Exception as e:
                        st.error(f"Error deleting {entry_id}: {str(e)}")
                save_availability(st.session_state.availability_df)
        else:
            st.info("No availability to manage.")
    
    st.markdown("---")
    # Continue with the existing bookings_df processing below this point...



    



    

        

    






    #----------end of new calendar
    
    st.subheader("✏️ Manage Existing Booking")
    if 'edit_booking_key' not in st.session_state:
        st.session_state.edit_booking_key = 0
    unique_key = f"select_booking_to_edit_{st.session_state.edit_booking_key}"

    if bookings_df.empty:
        st.info("No bookings available to manage.")
    else:
        duplicate_ids = bookings_df[bookings_df.duplicated(subset=['booking_id'], keep=False)]['booking_id'].unique()
        if len(duplicate_ids) > 0:
            st.warning(f"Found duplicate booking_id values: {duplicate_ids.tolist()}. Please remove duplicates in Supabase before editing.")
        else:
            booking_options = []

            def format_time_safe(time_str):
                if not time_str or str(time_str).lower() in ["nat", "nan", "none"]:
                    return "Unknown Time"
                t = str(time_str).strip()
                for fmt in ["%H:%M", "%H:%M:%S"]:
                    try:
                        return datetime.strptime(t, fmt).strftime("%I:%M %p").lstrip('0')
                    except ValueError:
                        continue
                return "Unknown Time"

            for _, row in bookings_df.iterrows():
                date_str = pd.to_datetime(row['date'], errors="coerce").strftime('%A, %d %b') if row['date'] else "Unknown Date"
                time_ampm = format_time_safe(row['time'])
                players = [p for p in [row['player1'], row['player2'], row['player3'], row['player4']] if p]
                players_str = ", ".join(players) if players else "No players"
                standby_str = row.get('standby_player', "None")
                desc = f"Court: {row['court_name']} | Date: {date_str} | Time: {time_ampm} | Match Type: {row['match_type']} | Players: {players_str} | Standby: {standby_str}"
                booking_options.append(f"{desc} | Booking ID: {row['booking_id']}")

            selected_booking = st.selectbox("Select a booking to edit or delete", [""] + booking_options, key=unique_key)
            if selected_booking:
                booking_id = selected_booking.split(" | Booking ID: ")[-1]
                booking_row = bookings_df[bookings_df["booking_id"] == booking_id].iloc[0]
                booking_idx = bookings_df[bookings_df["booking_id"] == booking_id].index[0]

                with st.expander("Edit Booking Details", expanded=True):
                    date_edit = st.date_input(
                        "Booking Date *",
                        value=pd.to_datetime(booking_row["date"], errors="coerce").date(),
                        key=f"edit_booking_date_{booking_id}"
                    )

                    current_time_ampm = format_time_safe(booking_row["time"])
                    hours = []
                    hours.append(datetime.strptime("6:00", "%H:%M").strftime("%I:%M %p").lstrip('0'))  # 6:00 AM
                    hours.append(datetime.strptime("6:30", "%H:%M").strftime("%I:%M %p").lstrip('0'))  # 6:30 AM
                    hours.append(datetime.strptime("7:30", "%H:%M").strftime("%I:%M %p").lstrip('0'))  # 7:30 AM
                    for h in range(7, 22):  # From 7 AM to 9 PM
                        hours.append(datetime.strptime(f"{h:02d}:00", "%H:%M").strftime("%I:%M %p").lstrip('0'))
                    time_index = hours.index(current_time_ampm) if current_time_ampm in hours else 0
                    time_edit = st.selectbox("Booking Time *", hours, index=time_index, key=f"edit_booking_time_{booking_id}")
                    match_type_edit = st.radio("Match Type", ["Doubles", "Singles"],
                                               index=0 if booking_row["match_type"] == "Doubles" else 1,
                                               key=f"edit_booking_match_type_{booking_id}")

                    if match_type_edit == "Doubles":
                        col1, col2 = st.columns(2)
                        with col1:
                            p1_edit = st.selectbox("Player 1 (optional)", [""] + available_players,
                                                   index=available_players.index(booking_row["player1"]) + 1 if booking_row["player1"] in available_players else 0,
                                                   key=f"edit_t1p1_{booking_id}")
                            p2_edit = st.selectbox("Player 2 (optional)", [""] + available_players,
                                                   index=available_players.index(booking_row["player2"]) + 1 if booking_row["player2"] in available_players else 0,
                                                   key=f"edit_t1p2_{booking_id}")
                        with col2:
                            p3_edit = st.selectbox("Player 3 (optional)", [""] + available_players,
                                                   index=available_players.index(booking_row["player3"]) + 1 if booking_row["player3"] in available_players else 0,
                                                   key=f"edit_t2p1_{booking_id}")
                            p4_edit = st.selectbox("Player 4 (optional)", [""] + available_players,
                                                   index=available_players.index(booking_row["player4"]) + 1 if booking_row["player4"] in available_players else 0,
                                                   key=f"edit_t2p2_{booking_id}")
                    else:
                        p1_edit = st.selectbox("Player 1 (optional)", [""] + available_players,
                                               index=available_players.index(booking_row["player1"]) + 1 if booking_row["player1"] in available_players else 0,
                                               key=f"edit_s1p1_{booking_id}")
                        p3_edit = st.selectbox("Player 2 (optional)", [""] + available_players,
                                               index=available_players.index(booking_row["player3"]) + 1 if booking_row["player3"] in available_players else 0,
                                               key=f"edit_s1p2_{booking_id}")
                        p2_edit = ""
                        p4_edit = ""

                    standby_initial_index = 0
                    if "standby_player" in booking_row and booking_row["standby_player"] and booking_row["standby_player"] in available_players:
                        standby_initial_index = available_players.index(booking_row["standby_player"]) + 1

                    standby_edit = st.selectbox("Standby Player (optional)", [""] + available_players,
                                                index=standby_initial_index, key=f"edit_standby_{booking_id}")
                    court_edit = st.selectbox("Court Name *", [""] + court_names,
                                              index=court_names.index(booking_row["court_name"]) + 1 if booking_row["court_name"] in court_names else 0,
                                              key=f"edit_court_{booking_id}")
                    screenshot_edit = st.file_uploader("Update Booking Screenshot (optional)",
                                                       type=["jpg", "jpeg", "png", "gif", "bmp", "webp"],
                                                       key=f"edit_screenshot_{booking_id}")
                    st.markdown("*Required fields", unsafe_allow_html=True)

                    col_save, col_delete = st.columns(2)
                    with col_save:
                        if st.button("Save Changes", key=f"save_booking_changes_{booking_id}"):
                            if not court_edit:
                                st.error("Court name is required.")
                            elif not date_edit or not time_edit:
                                st.error("Booking date and time are required.")
                            else:
                                players_edit = [p for p in [p1_edit, p2_edit, p3_edit, p4_edit] if p]
                                if len(set(players_edit)) != len(players_edit):
                                    st.error("Please select different players for each position.")
                                else:
                                    screenshot_url_edit = booking_row["screenshot_url"]
                                    if screenshot_edit:
                                        screenshot_url_edit = upload_image_to_github(screenshot_edit, booking_id, image_type="booking")

                                    time_24hr_edit = datetime.strptime(time_edit, "%I:%M %p").strftime("%H:%M:%S")
                                    updated_booking = {
                                        "booking_id": booking_id,
                                        "date": date_edit.isoformat(),
                                        "time": time_24hr_edit,
                                        "match_type": match_type_edit,
                                        "court_name": court_edit,
                                        "player1": p1_edit if p1_edit else None,
                                        "player2": p2_edit if p2_edit else None,
                                        "player3": p3_edit if p3_edit else None,
                                        "player4": p4_edit if p4_edit else None,
                                        "standby_player": standby_edit if standby_edit else None,
                                        "screenshot_url": screenshot_url_edit if screenshot_url_edit else None
                                    }
                                    try:
                                        st.session_state.bookings_df.loc[booking_idx] = {**updated_booking, "date": date_edit.isoformat()}
                                        expected_columns = ['booking_id', 'date', 'time', 'match_type', 'court_name',
                                                            'player1', 'player2', 'player3', 'player4', 'standby_player', 'screenshot_url']
                                        bookings_to_save = st.session_state.bookings_df[expected_columns].copy()
                                        for col in ['player1', 'player2', 'player3', 'player4', 'standby_player', 'screenshot_url']:
                                            bookings_to_save[col] = bookings_to_save[col].replace("", None)
                                        bookings_to_save = bookings_to_save.drop_duplicates(subset=['booking_id'], keep='last')
                                        save_bookings(bookings_to_save)
                                        load_bookings()
                                        st.success("Booking updated successfully.")
                                        st.session_state.edit_booking_key += 1
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to save booking: {str(e)}")
                                        st.session_state.edit_booking_key += 1
                                        st.rerun()
                    with col_delete:
                        if st.button("🗑️ Delete This Booking", key=f"delete_booking_{booking_id}"):
                            try:
                                delete_booking_from_db(booking_id)
                                load_bookings()
                                st.success("Booking deleted.")
                                st.session_state.edit_booking_key += 1
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to delete booking: {str(e)}")
                                st.session_state.edit_booking_key += 1
                                st.rerun()
    st.markdown("---")
    st.markdown("Odds Calculation Logic process uploaded at https://github.com/mahadevbk/ar2/blob/main/ar%20odds%20prediction%20system.pdf")




# ...START OF TAB 5 HALL OF FAME -------------------------------------------------------------------------
with tabs[5]:
    #st.header("Hall of Fame")
    display_hall_of_fame()




#--MINI TOURNEY -----------------------
with tabs[6]:
    st.header("Mini Tournaments Organiser")
    st.info("Tournament Ograniser is moved to https://tournament-organiser.streamlit.app/")
    st.info("App may be dormant and need to be 'woken up'.")




#----MINI TOURNEY--------------------------------------------------------------------------------------------







with tabs[7]:
    st.header("📊 Analyze League Data with Google Gemini")
    st.markdown("""
    Get instant insights, charts, and answers about your tennis league.

    Click below to:
    1. Download the latest `matches.csv`
    2. Open **Google Gemini** in a new tab
    3. Upload the CSV and ask questions like:
       - "Who has the most wins?"
       - "Show a chart of player points over time"
       - "Which players have the best win percentage?"
       - "Suggest balanced teams for next week"
    """)

    if not st.session_state.matches_df.empty:
        # Prepare CSV data
        matches_csv_bytes = st.session_state.matches_df.to_csv(index=False).encode('utf-8')
        current_time = datetime.now().strftime("%Y%m%d-%H%M")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.download_button(
                label="📥 Download matches.csv",
                data=matches_csv_bytes,
                file_name=f"mmd-matches-{current_time}.csv",
                mime="text/csv",
                key=f"gemini_csv_download_{uuid.uuid4().hex}",
                help="Download the latest match data to upload to Gemini"
            )

        with col2:
            st.markdown("""
            <a href="https://gemini.google.com/app" target="_blank">
                <button style="
                    background-color: #fff500;
                    color: #031827;
                    padding: 14px 20px;
                    border: none;
                    border-radius: 10px;
                    font-size: 16px;
                    font-weight: bold;
                    cursor: pointer;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                    width: 100%;
                    margin-top: 0;
                ">
                    🚀 Open Google Gemini 
                </button>
            </a>
            """, unsafe_allow_html=True)

        st.info("""
        **How to use:**
        1. Click **Download matches.csv**
        2. Click **Open Google Gemini**
        3. In Gemini, click the 📎 (paperclip) icon → Upload the CSV
        4. Ask any question about the league!
        """)

        st.success("Gemini is excellent at tennis stats — it will even generate beautiful charts automatically! 🎾📈")
    else:
        st.warning("No match data available yet. Add some matches first!")













#------------------END OF MMD AI AGENT ------------------------------------------------------------------







#st.markdown("---")

# Backup Download Button
st.markdown("---")
st.subheader("Data Backup")

# Reset ZIP buffer on each run to avoid stale data
zip_buffer = io.BytesIO()
try:
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # --- Matches CSV ---
        if not st.session_state.matches_df.empty:
            matches_csv = st.session_state.matches_df.to_csv(index=False)
            zip_file.writestr("matches.csv", matches_csv)
        else:
            zip_file.writestr("matches.csv", "No data")

        # --- Players CSV ---
        if not st.session_state.players_df.empty:
            players_csv = st.session_state.players_df.to_csv(index=False)
            zip_file.writestr("players.csv", players_csv)
        else:
            zip_file.writestr("players.csv", "No data")

        # --- Bookings CSV ---
        if not st.session_state.bookings_df.empty:
            bookings_csv = st.session_state.bookings_df.to_csv(index=False)
            zip_file.writestr("bookings.csv", bookings_csv)
        else:
            zip_file.writestr("bookings.csv", "No data")

        # Create directories in ZIP if needed (for images)
        # --- Profile Images ---
        zip_file.writestr("profile_images/.keep", "")  # Placeholder dir
        for _, row in st.session_state.players_df.iterrows():
            url = row.get("profile_image_url")
            if url and url.strip():
                try:
                    r = requests.get(url, timeout=10)
                    if r.status_code == 200:
                        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', str(row["name"]))
                        zip_file.writestr(f"profile_images/{safe_name}.jpg", r.content)
                except Exception as e:
                    st.warning(f"Could not include profile image for {row.get('name')}: {e}")

        # --- Match Images ---
        zip_file.writestr("match_images/.keep", "")  # Placeholder dir
        for _, row in st.session_state.matches_df.iterrows():
            url = row.get("match_image_url")
            if url and url.strip():
                try:
                    r = requests.get(url, timeout=10)
                    if r.status_code == 200:
                        match_id = row.get("match_id", str(uuid.uuid4()))
                        zip_file.writestr(f"match_images/{match_id}.jpg", r.content)
                except Exception as e:
                    st.warning(f"Could not include match image for {row.get('match_id')}: {e}")

    zip_buffer.seek(0)
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    
    # Use a unique key to prevent caching issues
    #backup_key = f"backup_download_{current_time}_{random.randint(1, 1000)}"
    backup_key = f"backup_download_{current_time}_{uuid.uuid4().hex}"
    st.download_button(
        label="Download Backup ZIP",
        data=zip_buffer.getvalue(),  # Use getvalue() to avoid buffer issues
        file_name=f"mmd-tennis-data-{current_time}.zip",
        mime="application/zip",
        key=backup_key
    )
    st.success("Backup ready for download!")
    
except Exception as e:
    st.error(f"Backup generation failed: {str(e)}. Check dataframes for issues.")
    st.info("Try adding some data or refreshing the app.")

st.markdown("""
<div style='background-color: #0d5384; padding: 1rem; border-left: 5px solid #fff500; border-radius: 0.5rem; color: white;'>
Built with ❤️ using <a href='https://streamlit.io/' style='color: #ccff00;'>Streamlit</a> — free and open source.
<a href='https://devs-scripts.streamlit.app/' style='color: #ccff00;'>Other Scripts by dev</a> on Streamlit.
</div>
""", unsafe_allow_html=True)
