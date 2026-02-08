import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import re
import io
import uuid
import time
import base64
import urllib.parse
import zipfile
import random
import os
from datetime import datetime, timedelta
from collections import defaultdict
from itertools import combinations
from dateutil import parser
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from supabase import create_client, Client

# Optional imports
try:
    from xai_sdk import Client as XAIClient
    from xai_sdk.chat import user, system
except ImportError:
    pass

# --- Configuration & Setup ---
st.set_page_config(page_title="MMD Mira Mixed Doubles Tennis League", layout="wide")
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# --- Custom CSS ---
st.markdown("""
<style>
.stApp {
  background: linear-gradient(to bottom, #041136, #21000a);
  background-attachment: scroll;
}
@media print {
  html, body { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
  body { background: linear-gradient(to bottom, #21000a, #041136) !important; height: 100vh; margin: 0; padding: 0; }
  header, .stToolbar { display: none; }
}
[data-testid="stHeader"] { background: linear-gradient(to top, #041136 , #21000a) !important; }
.profile-image {
    width: 80px; height: 80px; object-fit: cover; border: 2px solid #fff500;
    border-radius: 50%; margin-right: 15px; vertical-align: middle;
    transition: transform 0.2s; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4), 0 0 10px rgba(255, 245, 0, 0.6);
}
.profile-image:hover { transform: scale(1.1); }
.birthday-banner {
    background: linear-gradient(45deg, #FFFF00, #EEE8AA); color: #950606; padding: 15px;
    border-radius: 10px; text-align: center; font-size: 1.2em; font-weight: bold;
    margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    display: flex; justify-content: center; align-items: center;
}
.whatsapp-share, .calendar-share {
    background-color: #25D366; color: white !important; padding: 5px 10px; border-radius: 5px; 
    text-decoration: none; font-weight: bold; display: inline-flex; align-items: center;
    font-size: 0.8em; border: none; cursor: pointer; margin-top: 5px;
}
.whatsapp-share img { width: 18px; vertical-align: middle; margin-right: 5px; filter: brightness(0) invert(1); }
.court-card {
    background: linear-gradient(to bottom, #031827, #07314f); border: 1px solid #fff500;
    border-radius: 10px; padding: 15px; margin: 10px 0; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s, box-shadow 0.2s; text-align: center;
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
h1 { font-size: 24px !important; }
h2 { font-size: 22px !important; }
h3 { font-size: 16px !important; }
.rankings-table-container {
    width: 100%; margin-top: 0px !important; padding: 5px;
}
.ranking-row {
    display: block; padding: 15px; margin-bottom: 15px; border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.01) 100%);
    overflow: visible; transition: transform 0.2s;
}
.ranking-row:hover { transform: translateY(-2px); border-color: rgba(255, 245, 0, 0.5); }
.rank-profile-player-group { display: flex; align-items: center; margin-bottom: 15px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; }
.rank-col { font-size: 2em; font-weight: bold; color: #fff500; margin-right: 15px; min-width: 40px; text-align: center; }
.player-col { font-size: 1.4em; font-weight: bold; color: #ffffff; flex-grow: 1; }
.badge { background: rgba(255, 215, 0, 0.2); color: #ffd700; padding: 2px 6px; border-radius: 4px; font-size: 0.6em; margin-right: 5px; border: 1px solid rgba(255, 215, 0, 0.4); vertical-align: middle; }
.stat-box { flex: 1; min-width: 100px; text-align: center; padding: 5px; }
.stat-label { font-size: 0.75em; color: #aaa; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }
.stat-value { font-size: 1.1em; color: #fff; font-weight: bold; }
.stat-highlight { color: #fff500; }
[data-testid="stMetric"] > div:nth-of-type(1) { color: #FF7518 !important; }
.block-container { display: flex; flex-wrap: wrap; justify-content: center; }
[data-testid="stHorizontalBlock"] { flex: 1 1 100% !important; margin: 10px 0; }
[data-testid="stExpander"] i, [data-testid="stExpander"] span.icon { font-family: 'Material Icons' !important; }
</style>
""", unsafe_allow_html=True)

# --- Supabase Initialization ---
try:
    supabase_url = st.secrets["supabase"]["supabase_url"]
    supabase_key = st.secrets["supabase"]["supabase_key"]
    supabase: Client = create_client(supabase_url, supabase_key)
except KeyError:
    st.error("Supabase secrets not found.")
    st.stop()

# --- Constants ---
PLAYERS_TABLE = "players"
MATCHES_TABLE = "matches"
BOOKINGS_TABLE = "bookings"
HOF_TABLE = "hall_of_fame"
AVAILABILITY_TABLE = "availability"

# --- Session State Init ---
if 'players_df' not in st.session_state:
    st.session_state.players_df = pd.DataFrame(columns=["name", "profile_image_url", "birthday", "gender"])
if 'matches_df' not in st.session_state:
    st.session_state.matches_df = pd.DataFrame(columns=["match_id", "date", "match_type", "team1_player1", "team1_player2", "team2_player1", "team2_player2", "set1", "set2", "set3", "winner", "match_image_url"])
if 'bookings_df' not in st.session_state:
    st.session_state.bookings_df = pd.DataFrame(columns=["booking_id", "date", "time", "match_type", "court_name", "player1", "player2", "player3", "player4", "screenshot_url"])
if 'form_key_suffix' not in st.session_state:
    st.session_state.form_key_suffix = 0
if 'last_match_submit_time' not in st.session_state:
    st.session_state.last_match_submit_time = 0

# --- Helper Functions ---

@st.cache_data(ttl=60)
def fetch_data(table_name):
    try:
        response = supabase.table(table_name).select("*").execute()
        return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"Error fetching {table_name}: {e}")
        return pd.DataFrame()

def load_players():
    df = fetch_data(PLAYERS_TABLE)
    if not df.empty:
        for col in ["name", "profile_image_url", "birthday", "gender"]:
            if col not in df.columns: df[col] = ""
        df['name'] = df['name'].str.upper().str.strip()
    st.session_state.players_df = df

def save_players(players_df):
    try:
        df_save = players_df.copy()
        df_save['name'] = df_save['name'].str.upper().str.strip()
        df_save = df_save.where(pd.notna(df_save), None)
        df_save = df_save.drop_duplicates(subset=['name'], keep='last')
        supabase.table(PLAYERS_TABLE).upsert(df_save.to_dict("records")).execute()
        fetch_data.clear() # Clear cache on write
    except Exception as e:
        st.error(f"Error saving players: {e}")

def delete_player_from_db(player_name):
    supabase.table(PLAYERS_TABLE).delete().eq("name", player_name).execute()
    fetch_data.clear()

def load_matches():
    df = fetch_data(MATCHES_TABLE)
    expected_cols = ["match_id", "date", "match_type", "team1_player1", "team1_player2", 
                     "team2_player1", "team2_player2", "set1", "set2", "set3", "winner", "match_image_url"]
    for col in expected_cols:
        if col not in df.columns: df[col] = ""
    
    if not df.empty:
        # Vectorized date conversion
        df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce').dt.tz_localize(None)
        df['date'] = df['date'].fillna(pd.Timestamp('1970-01-01'))
        
    st.session_state.matches_df = df

def save_matches(matches_df):
    try:
        df_save = matches_df.copy()
        df_save = df_save.where(pd.notna(df_save), None)
        # Vectorized string format
        df_save['date'] = pd.to_datetime(df_save['date'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
        df_save.loc[df_save['date'].isna(), 'date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        df_save = df_save[df_save["match_id"].notnull() & (df_save["match_id"] != "")]
        df_save = df_save.drop_duplicates(subset=['match_id'], keep='last')
        
        if not df_save.empty:
            supabase.table(MATCHES_TABLE).upsert(df_save.to_dict("records")).execute()
            fetch_data.clear()
            st.success("Match saved successfully.")
    except Exception as e:
        st.error(f"Error saving matches: {e}")

def delete_match_from_db(match_id):
    supabase.table(MATCHES_TABLE).delete().eq("match_id", match_id).execute()
    st.session_state.matches_df = st.session_state.matches_df[st.session_state.matches_df["match_id"] != match_id]
    fetch_data.clear()

def upload_image_to_github(file, file_name, image_type="match"):
    if not file: return ""
    try:
        token = st.secrets["github"]["token"]
        repo = st.secrets["github"]["repo"]
        branch = st.secrets["github"]["branch"]
    except KeyError:
        st.error("GitHub secrets missing.")
        return ""

    path_map = {
        "profile": "assets/profile_images",
        "match": "assets/match_images",
        "booking": "assets/bookings_images"
    }
    folder = path_map.get(image_type, "assets/others")
    path_in_repo = f"{folder}/{file_name}.jpg"

    try:
        content_bytes = file.getvalue()
        img = Image.open(io.BytesIO(content_bytes))
        img = ImageOps.exif_transpose(img)
        
        # Resize logic
        if image_type == "match" and (img.width > 1200 or img.height > 1200):
            img.thumbnail((1200, 1200), Image.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format=img.format or "JPEG", quality=85)
        content_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        api_url = f"https://api.github.com/repos/{repo}/contents/{path_in_repo}"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
        
        # Check existing
        sha = None
        resp = requests.get(api_url, headers=headers)
        if resp.status_code == 200:
            sha = resp.json().get('sha')
            
        payload = {
            "message": f"feat: Upload {image_type} {file_name}",
            "branch": branch,
            "content": content_b64
        }
        if sha: payload["sha"] = sha
        
        requests.put(api_url, headers=headers, json=payload).raise_for_status()
        
        # URL encode the path to handle spaces safely in the return URL
        encoded_path = urllib.parse.quote(path_in_repo)
        return f"https://raw.githubusercontent.com/{repo}/{branch}/{encoded_path}"
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return ""

# --- Business Logic ---

def tennis_scores():
    scores = ["6-0", "6-1", "6-2", "6-3", "6-4", "7-5", "7-6", "0-6", "1-6", "2-6", "3-6", "4-6", "5-7", "6-7"]
    for i in range(10): scores.extend([f"Tie Break 7-{i}", f"Tie Break {i}-7"])
    for i in range(6): scores.extend([f"Tie Break 10-{i}", f"Tie Break {i}-10"])
    return scores

def generate_match_id(matches_df, match_datetime):
    year = match_datetime.year
    month = match_datetime.month
    quarter = f"Q{(month-1)//3 + 1}"
    
    # Filter using vectorized operations if dataframe is populated
    if not matches_df.empty:
        # Ensure date column is datetime
        dates = pd.to_datetime(matches_df['date'], errors='coerce')
        mask = (dates.dt.year == year) & ((dates.dt.month-1)//3 + 1 == (month-1)//3 + 1)
        serial = mask.sum() + 1
    else:
        serial = 1
        
    while True:
        new_id = f"MMD{quarter}{year}-{serial:02d}"
        if matches_df.empty or new_id not in matches_df['match_id'].values:
            return new_id
        serial += 1

# Helper functions for defaultdict to allow pickling
def get_player_stats_template():
    return {'wins': 0, 'losses': 0, 'matches': 0, 'games_won': 0, 'gd_sum': 0, 
            'clutch_wins': 0, 'clutch_matches': 0, 'gd_list': []}

def get_partner_stats_inner_template():
    return {'wins': 0, 'losses': 0, 'ties': 0, 'matches': 0, 'game_diff_sum': 0}

def get_partner_stats_template():
    return defaultdict(get_partner_stats_inner_template)

@st.cache_data(show_spinner=False)
def calculate_rankings(matches_to_rank):
    # Initialize containers
    scores = defaultdict(float)
    stats = defaultdict(get_player_stats_template)
    partner_stats = defaultdict(get_partner_stats_template)
    current_streaks = defaultdict(int)

    players_df = st.session_state.players_df
    # Pre-fetch genders map for O(1) lookup
    gender_map = pd.Series(players_df.gender.values, index=players_df.name).to_dict() if not players_df.empty else {}

    # Sort matches by date to ensure accurate streaks
    if not matches_to_rank.empty:
        matches_to_rank = matches_to_rank.sort_values('date')

    # Optimize iteration
    for row in matches_to_rank.itertuples(index=False):
        match_type = row.match_type
        # Extract players, filtering None/Empty/Visitor
        t1 = [p for p in [row.team1_player1, row.team1_player2] if p and p != "Visitor"]
        t2 = [p for p in [row.team2_player1, row.team2_player2] if p and p != "Visitor"]
        
        if not t1 or not t2: continue

        # Mixed doubles check
        is_mixed = False
        if match_type == 'Doubles' and len(t1) == 2 and len(t2) == 2:
            g1 = sorted([gender_map.get(p) for p in t1])
            g2 = sorted([gender_map.get(p) for p in t2])
            if g1 == ['F', 'M'] and g2 == ['F', 'M']:
                is_mixed = True

        # Process Sets
        match_gd = 0
        is_clutch = False
        sets = [row.set1, row.set2, row.set3]
        
        # Determine winner/loser
        winner_code = row.winner # "Team 1", "Team 2", "Tie"

        for s in sets:
            if not s: continue
            s_str = str(s)
            t1_g, t2_g = 0, 0
            
            if "Tie Break" in s_str:
                is_clutch = True
                try:
                    nums = [int(x) for x in re.findall(r'\d+', s_str)]
                    if len(nums) == 2:
                        # Tie break games logic (normalized to 7-6 or 6-7)
                        if nums[0] > nums[1]: t1_g, t2_g = 7, 6
                        else: t1_g, t2_g = 6, 7
                except: continue
            elif '-' in s_str:
                try:
                    parts = s_str.split('-')
                    t1_g, t2_g = int(parts[0]), int(parts[1])
                except: continue
            
            diff = t1_g - t2_g
            match_gd += diff
            
            for p in t1:
                stats[p]['games_won'] += t1_g
                stats[p]['gd_sum'] += diff
                stats[p]['gd_list'].append(diff)
            for p in t2:
                stats[p]['games_won'] += t2_g
                stats[p]['gd_sum'] -= diff
                stats[p]['gd_list'].append(-diff)

        if row.set3 and str(row.set3).strip(): is_clutch = True

        # Scoring
        w_points = 3 if is_mixed else 2
        
        def update_stats(players, outcome, gd):
            for p in players:
                stats[p]['matches'] += 1
                if is_clutch: stats[p]['clutch_matches'] += 1
                
                if outcome == 'win':
                    scores[p] += w_points
                    stats[p]['wins'] += 1
                    if is_clutch: stats[p]['clutch_wins'] += 1
                    # Streak Logic
                    if current_streaks[p] < 0: current_streaks[p] = 0
                    current_streaks[p] += 1

                elif outcome == 'loss':
                    scores[p] += 1
                    stats[p]['losses'] += 1
                    # Streak Logic
                    if current_streaks[p] > 0: current_streaks[p] = 0
                    current_streaks[p] -= 1
                else: # tie
                    scores[p] += 1.5
                    current_streaks[p] = 0

        if winner_code == "Team 1":
            update_stats(t1, 'win', match_gd)
            update_stats(t2, 'loss', -match_gd)
        elif winner_code == "Team 2":
            update_stats(t2, 'win', -match_gd)
            update_stats(t1, 'loss', match_gd)
        else:
            update_stats(t1, 'tie', match_gd)
            update_stats(t2, 'tie', -match_gd)

        # Partner stats
        if match_type == 'Doubles':
            for team, outcome_code, gd_val in [(t1, 1, match_gd), (t2, 2, -match_gd)]:
                if len(team) < 2: continue
                p1, p2 = team[0], team[1]
                # Canonical ordering for key not needed if we store symmetric keys in partner_stats
                # Current logic stores p1->p2 and p2->p1
                for a, b in [(p1, p2), (p2, p1)]:
                    ps = partner_stats[a][b]
                    ps['matches'] += 1
                    ps['game_diff_sum'] += gd_val
                    if winner_code == "Tie": ps['ties'] += 1
                    elif (winner_code == "Team 1" and outcome_code == 1) or (winner_code == "Team 2" and outcome_code == 2):
                        ps['wins'] += 1
                    else:
                        ps['losses'] += 1

    # Build DataFrame
    rank_data = []
    # Pre-fetch image urls
    img_map = pd.Series(players_df.profile_image_url.values, index=players_df.name).to_dict() if not players_df.empty else {}
    max_matches_played = max([s['matches'] for s in stats.values()]) if stats else 0
    
    for p, s in stats.items():
        matches_played = s['matches']
        if matches_played == 0: continue
        
        win_pct = (s['wins'] / matches_played) * 100
        gd_avg = s['gd_sum'] / matches_played
        clutch_pct = (s['clutch_wins'] / s['clutch_matches'] * 100) if s['clutch_matches'] > 0 else 0
        consistency = np.std(s['gd_list']) if s['gd_list'] else 0
        
        # Badges
        badges = []
        if clutch_pct > 70 and s['clutch_matches'] >= 3: badges.append("🎯 Tie-break Monster")
        if consistency < 2 and matches_played >= 5: badges.append("📈 Consistent Performer")
        if current_streaks[p] >= 3: badges.append("🔥 On Fire")
        if win_pct == 100 and matches_played >= 3: badges.append("🦄 Unbeatable")
        if matches_played == max_matches_played and max_matches_played >= 5: badges.append("🐝 Busy Bee")
        
        rank_data.append({
            "Player": p, "Points": scores[p], "Win %": round(win_pct, 2),
            "Matches": matches_played, "Wins": s['wins'], "Losses": s['losses'],
            "Games Won": s['games_won'], "Game Diff Avg": round(gd_avg, 2),
            "Cumulative Game Diff": s['gd_sum'], "Clutch Factor": round(clutch_pct, 1),
            "Consistency Index": round(consistency, 2), "Badges": badges,
            "Profile": img_map.get(p, ""), "Recent Trend": "" # Trend calc is separate/expensive
        })
        
    df = pd.DataFrame(rank_data)
    if not df.empty:
        df = df.sort_values(by=["Points", "Win %", "Game Diff Avg", "Games Won"], ascending=[False, False, False, False]).reset_index(drop=True)
        df["Rank"] = [f"🏆 {i+1}" for i in df.index]
        
    return df, partner_stats

def get_player_trend(player, matches, max_matches=5):
    # Vectorized filter is hard because player can be in 4 cols.
    # But since we need sorted limits, simple filtering is OK.
    mask = (matches['team1_player1'] == player) | (matches['team1_player2'] == player) | \
           (matches['team2_player1'] == player) | (matches['team2_player2'] == player)
    
    # Ensure dates
    if not pd.api.types.is_datetime64_any_dtype(matches['date']):
         matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
         
    pm = matches[mask].sort_values(by='date', ascending=False).head(max_matches)
    trend = []
    for row in pm.itertuples(index=False):
        w = row.winner
        is_t1 = player in [row.team1_player1, row.team1_player2]
        is_t2 = player in [row.team2_player1, row.team2_player2]
        
        if w == "Tie": res = 'T'
        elif (w == "Team 1" and is_t1) or (w == "Team 2" and is_t2): res = 'W'
        else: res = 'L'
        trend.append(res)
    return ' '.join(trend) if trend else "No matches"

def get_birthday_banner(players_df):
    if players_df.empty: return
    today = datetime.now()
    birthdays = []
    
    for row in players_df.itertuples(index=False):
        if row.birthday:
            try:
                dob = pd.to_datetime(row.birthday)
                if dob.month == today.month and dob.day == today.day:
                    birthdays.append(row.name)
            except: pass
            
    if birthdays:
        names = ", ".join(birthdays)
        st.markdown(f"""
        <div class="birthday-banner">
            🎂 Happy Birthday to {names}! 🥳
        </div>
        """, unsafe_allow_html=True)

def load_bookings():
    df = fetch_data(BOOKINGS_TABLE)
    cols = ['booking_id', 'date', 'time', 'match_type', 'court_name', 'player1', 'player2', 'player3', 'player4', 'standby_player', 'screenshot_url']
    for c in cols: 
        if c not in df.columns: df[c] = None
        
    if not df.empty:
        # Cleanup expired with robust format handling
        try:
             df['dt_combo'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), format='%Y-%m-%d %H:%M', errors='coerce')
        except:
             df['dt_combo'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')

        # Safe timezone handling
        if isinstance(df['dt_combo'].dtype, pd.DatetimeTZDtype):
             df['dt_combo'] = df['dt_combo'].dt.tz_convert('Asia/Dubai')
        else:
             df['dt_combo'] = df['dt_combo'].dt.tz_localize('Asia/Dubai', ambiguous='infer')
             
        cutoff = pd.Timestamp.now(tz='Asia/Dubai') - timedelta(hours=4)
        
        expired = df[df['dt_combo'] < cutoff]
        if not expired.empty:
            for bid in expired['booking_id']:
                supabase.table(BOOKINGS_TABLE).delete().eq("booking_id", bid).execute()
            df = df[df['dt_combo'] >= cutoff]
        
        # Format for display
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df = df.fillna("")
        
    st.session_state.bookings_df = df[cols]

def save_bookings(df):
    try:
        df_save = df.copy().where(pd.notna(df), None)
        supabase.table(BOOKINGS_TABLE).upsert(df_save.to_dict("records")).execute()
        fetch_data.clear()
    except Exception as e:
        st.error(f"Save bookings error: {e}")

# --- Initial Load ---
load_players()
load_matches()
load_bookings()

# --- Main Layout ---
st.image("https://raw.githubusercontent.com/mahadevbk/mmd/main/mmdheaderQ12026.png", use_container_width=True)
get_birthday_banner(st.session_state.players_df)

tab_names = ["Rankings", "Matches", "Player Profile", "Maps", "Bookings", "Hall of Fame", "Mini Tourney", "MMD AI"]
tabs = st.tabs(tab_names)

# --- Tab 1: Rankings ---
with tabs[0]:
    st.header(f"Rankings as of {datetime.now().strftime('%d %b %Y')}")
    ranking_type = st.radio("View", ["Combined", "Doubles", "Singles", "Table View"], horizontal=True)
    
    matches_df = st.session_state.matches_df
    if not matches_df.empty:
        if ranking_type == "Doubles":
            m_sub = matches_df[matches_df.match_type == "Doubles"]
            rank_df, partner_stats_global = calculate_rankings(m_sub)
        elif ranking_type == "Singles":
            m_sub = matches_df[matches_df.match_type == "Singles"]
            rank_df, partner_stats_global = calculate_rankings(m_sub)
        else:
            rank_df, partner_stats_global = calculate_rankings(matches_df)
            
        if ranking_type == "Table View":
            st.dataframe(
                rank_df, 
                hide_index=True, 
                use_container_width=True,
                column_config={
                    "Profile": st.column_config.ImageColumn("Profile"),
                }
            )
        else:
            # Podium for Top 3
            if len(rank_df) >= 3:
                top3 = rank_df.head(3).to_dict('records')
                c1, c2, c3 = st.columns([1,1,1])
                
                # Visual order: 2nd, 1st, 3rd
                podium_order = [(c1, top3[1]), (c2, top3[0]), (c3, top3[2])]
                
                for col, player in podium_order:
                    with col:
                        # Construct badges text if needed or just show summary
                        st.markdown(f"""
                        <div style="text-align: center; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 15px; border: 1px solid rgba(255,215,0,0.3); box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
                            <div style="font-size: 2.5em; margin-bottom: 10px;">{player['Rank'].split()[0]}</div>
                            <img src="{player['Profile'] or 'https://via.placeholder.com/100?text=Player'}" style="width: 100px; height: 100px; border-radius: 50%; object-fit: cover; border: 3px solid gold; box-shadow: 0 0 15px rgba(255,215,0,0.5);">
                            <h3 style="margin: 10px 0; color: #fff500; font-size: 1.2em;">{player['Player']}</h3>
                            <div style="color: white; font-weight: bold; font-size: 1.1em;">{player['Points']} pts</div>
                            <div style="color: #aaa; font-size: 0.9em;">{player['Win %']}% Win Rate</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)

            # Detailed List (HTML based)
            st.markdown("<div class='rankings-table-container'>", unsafe_allow_html=True)
            for row in rank_df.to_dict('records'):
                badges_str = " ".join([f"<span class='badge'>{b}</span>" for b in row['Badges']]) if row['Badges'] else ""
                trend_str = get_player_trend(row['Player'], matches_df)
                
                row_html = f"""
                <div class="ranking-row">
                    <div class="rank-profile-player-group">
                        <div class="rank-col">{row['Rank']}</div>
                        <img src="{row['Profile'] or 'https://via.placeholder.com/80?text=Player'}" class="profile-image">
                        <div class="player-col">
                            {row['Player']}
                            <div style="margin-top: 5px;">{badges_str}</div>
                        </div>
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 5px; justify-content: space-between;">
                        <div class="stat-box"><div class="stat-label">Points</div><div class="stat-value stat-highlight">{row['Points']}</div></div>
                        <div class="stat-box"><div class="stat-label">Win %</div><div class="stat-value">{row['Win %']}%</div></div>
                        <div class="stat-box"><div class="stat-label">Record</div><div class="stat-value">{row['Wins']}W - {row['Losses']}L</div></div>
                        <div class="stat-box"><div class="stat-label">Games</div><div class="stat-value">{row['Games Won']}</div></div>
                        <div class="stat-box"><div class="stat-label">Game Diff</div><div class="stat-value">{row['Game Diff Avg']}</div></div>
                        <div class="stat-box"><div class="stat-label">Trend</div><div class="stat-value" style="font-family: monospace; letter-spacing: 2px;">{trend_str}</div></div>
                    </div>
                </div>
                """
                st.markdown(row_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No matches recorded yet.")
        partner_stats_global = {}

# --- Tab 2: Matches ---
with tabs[1]:
    st.header("Matches")
    with st.expander("➕ Post Match Result"):
        if not st.session_state.players_df.empty:
            names = sorted([n for n in st.session_state.players_df['name'] if n != 'Visitor'])
            
            with st.form("match_form"):
                mtype = st.radio("Type", ["Doubles", "Singles"])
                c1, c2 = st.columns(2)
                if mtype == "Doubles":
                    t1p1 = c1.selectbox("T1 P1", [""]+names)
                    t1p2 = c1.selectbox("T1 P2", [""]+names)
                    t2p1 = c2.selectbox("T2 P1", [""]+names)
                    t2p2 = c2.selectbox("T2 P2", [""]+names)
                else:
                    t1p1 = c1.selectbox("P1", [""]+names)
                    t2p1 = c2.selectbox("P2", [""]+names)
                    t1p2, t2p2 = None, None
                
                date = st.date_input("Date")
                scores = tennis_scores()
                s1 = st.selectbox("Set 1", [""]+scores)
                s2 = st.selectbox("Set 2", [""]+scores)
                s3 = st.selectbox("Set 3", [""]+scores)
                winner = st.selectbox("Winner", ["Team 1", "Team 2", "Tie"])
                img = st.file_uploader("Image", type=["jpg","png"])
                
                if st.form_submit_button("Submit"):
                    mid = generate_match_id(st.session_state.matches_df, pd.to_datetime(date))
                    url = upload_image_to_github(img, mid) if img else ""
                    
                    new_match = {
                        "match_id": mid, "date": date.isoformat(), "match_type": mtype,
                        "team1_player1": t1p1, "team1_player2": t1p2,
                        "team2_player1": t2p1, "team2_player2": t2p2,
                        "set1": s1, "set2": s2, "set3": s3, "winner": winner,
                        "match_image_url": url
                    }
                    st.session_state.matches_df = pd.concat([st.session_state.matches_df, pd.DataFrame([new_match])], ignore_index=True)
                    save_matches(st.session_state.matches_df)
                    st.success("Saved!")
                    st.rerun()

    # Match History
    st.subheader("History")
    m_hist = st.session_state.matches_df.copy()
    if not m_hist.empty:
        m_hist['date'] = pd.to_datetime(m_hist['date'])
        m_hist = m_hist.sort_values('date', ascending=False)
        for row in m_hist.itertuples():
            # Construct text for sharing
            res_txt = f"{row.team1_player1}/{row.team1_player2} vs {row.team2_player1}/{row.team2_player2} ({row.set1}, {row.set2})"
            share_text = urllib.parse.quote(f"MMD Match Result ({row.date.strftime('%Y-%m-%d')}):\n{res_txt}\nWinner: {row.winner}")
            
            # Image HTML if available
            img_html = ""
            if row.match_image_url:
                img_html = f"""
                <div style="text-align: center; margin: 10px 0;">
                    <img src="{row.match_image_url}" style="max-height: 200px; max-width: 100%; border-radius: 8px; border: 1px solid rgba(255,255,255,0.2);" />
                </div>
                """

            # Match Card HTML
            match_html = f"""
            <div class="ranking-row" style="padding: 20px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px;">
                     <span style="font-size:1.1em; color:#fff500; font-weight: bold;">{row.date.strftime('%d %b %Y')}</span>
                     <span style="color:#aaa; text-transform: uppercase; font-size: 0.8em; letter-spacing: 1px;">{row.match_type}</span>
                </div>
                <div style="margin: 15px 0; font-size: 1.3em; text-align: center;">
                     <span style="color: {'#fff500' if row.winner == 'Team 1' else 'white'}">{row.team1_player1} & {row.team1_player2}</span>
                     <br><span style="color: #FF7518; font-size: 0.8em;">VS</span><br>
                     <span style="color: {'#fff500' if row.winner == 'Team 2' else 'white'}">{row.team2_player1} & {row.team2_player2}</span>
                </div>
                <div style="text-align: center; margin-bottom:15px; font-weight: bold; font-size: 1.1em; background: rgba(0,0,0,0.3); padding: 5px; border-radius: 5px;">
                     {row.set1} {f"| {row.set2}" if row.set2 else ""} {f"| {row.set3}" if row.set3 else ""}
                </div>
                {img_html}
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                    <a href="https://wa.me/?text={share_text}" target="_blank" class="whatsapp-share">
                        <img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg" /> Share Result
                    </a>
                    {f'<a href="{row.match_image_url}" target="_blank" style="color: #aaa; text-decoration: none; font-size: 0.9em;">View Full Photo 📷</a>' if row.match_image_url else ''}
                </div>
            </div>
            """
            st.markdown(match_html, unsafe_allow_html=True)
    else:
        st.info("No matches recorded yet.")

# --- Tab 3: Player Profile ---
with tabs[2]:
    st.header("Player Profile")
    all_names = sorted(st.session_state.players_df['name'].unique()) if not st.session_state.players_df.empty else []
    
    col_sel, col_add = st.columns([2, 1])
    with col_sel:
        selected_player = st.selectbox("Select Player", [""] + all_names)
    
    if selected_player:
        # Get Player Data
        p_row = st.session_state.players_df[st.session_state.players_df['name'] == selected_player].iloc[0]
        rank_info = rank_df[rank_df['Player'] == selected_player].iloc[0] if 'rank_df' in locals() and not rank_df.empty and selected_player in rank_df['Player'].values else None
        
        # Display Header
        hc1, hc2 = st.columns([1, 3])
        with hc1:
            if p_row['profile_image_url']:
                st.image(p_row['profile_image_url'], width=150, caption=selected_player)
            else:
                st.markdown(f"**{selected_player}** (No Image)")
        
        with hc2:
            if rank_info is not None:
                st.subheader(f"Rank: {rank_info['Rank']}")
                st.write(f"Points: {rank_info['Points']} | Win Rate: {rank_info['Win %']}%")
                if rank_info['Badges']:
                    st.write(f"Badges: {', '.join(rank_info['Badges'])}")
            
            if p_row['birthday']:
                st.caption(f"Birthday: {p_row['birthday']}")

        st.divider()
        
        # Partner Stats
        if 'partner_stats_global' in locals() and selected_player in partner_stats_global:
            st.subheader("Partner Analysis")
            my_partners = partner_stats_global[selected_player]
            if my_partners:
                best_partner = max(my_partners.items(), key=lambda x: (x[1]['wins']/(x[1]['matches'] or 1), x[1]['matches']))
                st.write(f"🤝 **Best Partner:** {best_partner[0]} ({best_partner[1]['wins']}W - {best_partner[1]['losses']}L)")
        
        # Player History
        st.subheader("Recent Matches")
        p_history = st.session_state.matches_df[
            (st.session_state.matches_df['team1_player1'] == selected_player) | 
            (st.session_state.matches_df['team1_player2'] == selected_player) |
            (st.session_state.matches_df['team2_player1'] == selected_player) |
            (st.session_state.matches_df['team2_player2'] == selected_player)
        ].sort_values('date', ascending=False).head(5)
        
        if not p_history.empty:
            for r in p_history.itertuples():
                res = "Won" if (r.winner == "Team 1" and selected_player in [r.team1_player1, r.team1_player2]) or \
                               (r.winner == "Team 2" and selected_player in [r.team2_player1, r.team2_player2]) else "Lost"
                if r.winner == "Tie": res = "Tie"
                color = "green" if res == "Won" else "red"
                st.markdown(f":{color}[{res}] on {pd.to_datetime(r.date).strftime('%Y-%m-%d')} ({r.match_type})")

    with st.expander("Manage Players (Add/Edit)"):
        p_name = st.text_input("Name (New or Existing)")
        gender = st.radio("Gender", ["M", "F"], horizontal=True)
        p_img_file = st.file_uploader("Profile Photo", type=["jpg", "png"])
        p_dob = st.date_input("Birthday", value=None)
        
        if st.button("Save Player"):
            img_url = ""
            if p_img_file:
                img_url = upload_image_to_github(p_img_file, p_name.upper().strip(), "profile")
            
            new_p = {
                "name": p_name, "gender": gender, 
                "profile_image_url": img_url if img_url else None,
                "birthday": p_dob.strftime("%Y-%m-%d") if p_dob else None
            }
            # Remove existing if update
            st.session_state.players_df = st.session_state.players_df[st.session_state.players_df['name'] != p_name.upper().strip()]
            st.session_state.players_df = pd.concat([st.session_state.players_df, pd.DataFrame([new_p])], ignore_index=True)
            save_players(st.session_state.players_df)
            st.success("Player Saved")
            st.rerun()

# --- Tab 4: Maps ---
with tabs[3]:
    st.header("Locations")
    courts = [
        {"name": "Alvorado 1", "url": "https://maps.google.com/?q=25.041792,55.259258"},
        {"name": "Palmera 2", "url": "https://maps.app.goo.gl/CHimjtqQeCfU1d3W6"},
        {"name": "Mira 4", "url": "https://maps.google.com/?q=25.035,55.265"},
        {"name": "Mirador 1", "url": "https://maps.google.com/?q=25.038,55.260"},
        {"name": "Reem 2", "url": "https://maps.google.com/?q=25.045,55.255"},
        {"name": "Reem 3", "url": "https://maps.google.com/?q=25.048,55.252"},
    ]
    cols = st.columns(3)
    for i, c in enumerate(courts):
        with cols[i%3]:
            st.markdown(f"<div class='court-card'><h4>{c['name']}</h4><a href='{c['url']}' target='_blank'>Map</a></div>", unsafe_allow_html=True)

# --- Tab 5: Bookings ---
with tabs[4]:
    st.header("Bookings")
    
    # Booking Form
    with st.expander("🎾 Book a Court", expanded=True):
        with st.form("booking_form"):
            bc1, bc2 = st.columns(2)
            court_opts = [c['name'] for c in courts]
            b_court = bc1.selectbox("Court", court_opts)
            b_date = bc1.date_input("Date", min_value=datetime.now())
            b_time = bc2.selectbox("Time", [f"{h}:00" for h in range(6, 23)])
            b_type = bc2.radio("Type", ["Singles", "Doubles", "Training"], horizontal=True)
            
            p_opts = [""] + sorted(st.session_state.players_df['name'].unique())
            bp1 = st.selectbox("Player 1", p_opts)
            bp2 = st.selectbox("Player 2", p_opts)
            bp3 = st.selectbox("Player 3", p_opts)
            bp4 = st.selectbox("Player 4", p_opts)
            
            b_img = st.file_uploader("Booking Screenshot", type=["jpg", "png"])
            
            if st.form_submit_button("Book Court"):
                # Basic collision check
                existing = st.session_state.bookings_df
                is_taken = False
                if not existing.empty:
                    conflict = existing[
                        (existing['date'] == b_date.strftime('%Y-%m-%d')) & 
                        (existing['time'] == b_time) & 
                        (existing['court_name'] == b_court)
                    ]
                    if not conflict.empty:
                        is_taken = True
                
                if is_taken:
                    st.error("Court is already booked for this slot!")
                else:
                    bid = str(uuid.uuid4())
                    url = upload_image_to_github(b_img, bid, "booking") if b_img else ""
                    new_b = {
                        "booking_id": bid, "date": b_date.strftime('%Y-%m-%d'), "time": b_time,
                        "court_name": b_court, "match_type": b_type,
                        "player1": bp1, "player2": bp2, "player3": bp3, "player4": bp4,
                        "screenshot_url": url
                    }
                    st.session_state.bookings_df = pd.concat([st.session_state.bookings_df, pd.DataFrame([new_b])], ignore_index=True)
                    save_bookings(st.session_state.bookings_df)
                    st.success("Booking Confirmed!")
                    st.rerun()

    # Upcoming Bookings
    st.subheader("Upcoming Games")
    if not st.session_state.bookings_df.empty:
        # Sort by date/time
        df_b = st.session_state.bookings_df.copy()
        df_b['dt_sort'] = pd.to_datetime(df_b['date'] + ' ' + df_b['time'])
        df_b = df_b.sort_values('dt_sort')
        
        for row in df_b.itertuples():
            with st.container():
                st.info(f"📅 {row.date} @ {row.time} | 🏟️ {row.court_name} | {row.match_type}")
                players = [p for p in [row.player1, row.player2, row.player3, row.player4] if p]
                st.write(f"Players: {', '.join(players)}")
                if row.screenshot_url:
                    st.image(row.screenshot_url, width=200)
    else:
        st.write("No upcoming bookings.")

# --- Tab 6: Hall of Fame ---
with tabs[5]:
    st.header("Hall of Fame")
    hof = fetch_data(HOF_TABLE)
    if not hof.empty:
        st.dataframe(hof, use_container_width=True)
    else:
        st.write("No Hall of Fame records found.")

# --- Tab 7: Mini Tourney ---
with tabs[6]:
    st.info("Tournament Organiser moved to external app.")

# --- Tab 8: AI ---
with tabs[7]:
    st.header("AI Analysis")
    if not st.session_state.matches_df.empty:
        csv = st.session_state.matches_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Matches CSV", csv, "matches.csv", "text/csv")
        st.markdown("[Open Gemini](https://gemini.google.com/app)")

# --- Backup ---
st.markdown("---")
if st.button("Generate Backup"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("players.csv", st.session_state.players_df.to_csv(index=False))
        z.writestr("matches.csv", st.session_state.matches_df.to_csv(index=False))
        z.writestr("bookings.csv", st.session_state.bookings_df.to_csv(index=False))
    st.download_button("Download ZIP", buf.getvalue(), "backup.zip", "application/zip")
