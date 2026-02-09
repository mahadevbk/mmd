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
    border-radius: 15px; margin-right: 15px; vertical-align: middle;
    transition: transform 0.2s; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4), 0 0 15px rgba(255, 245, 0, 0.6);
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
#HOF_TABLE = "hall_of_fame"
hall_of_fame_table_name="hall_of_fame"
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

def format_set_score(s):
    if not s: return ""
    s = str(s)
    if "Tie Break" in s:
        # Extract numbers
        try:
            # Expected format "Tie Break 7-5"
            parts = s.replace("Tie Break", "").strip().split('-')
            if len(parts) == 2:
                p1, p2 = int(parts[0]), int(parts[1])
                # Check if super tie break (usually to 10)
                if p1 >= 10 or p2 >= 10:
                    # Treat as super tie break
                    return f"1-0 ({s})" 
                else:
                    if p1 > p2:
                        return f"7-6 ({s})"
                    else:
                        return f"6-7 ({s})"
        except:
            pass
        return s 
    return s

def flip_score(s):
    """Reverses the score representation for display when Team 2 wins."""
    if not s: return ""
    s = str(s)
    if "Tie Break" in s:
        try:
            parts = s.replace("Tie Break", "").strip().split('-')
            if len(parts) == 2:
                return f"Tie Break {parts[1]}-{parts[0]}"
        except:
            return s
    elif '-' in s:
        try:
            parts = s.split('-')
            if len(parts) == 2:
                return f"{parts[1]}-{parts[0]}"
        except:
            return s
    return s

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
    scores = defaultdict(float)
    stats = defaultdict(get_player_stats_template)
    partner_stats = defaultdict(get_partner_stats_template)
    current_streaks = defaultdict(int)
    
    # Track performance breakdown
    perf_breakdown = defaultdict(lambda: {'singles_w': 0, 'singles_m': 0, 'doubles_w': 0, 'doubles_m': 0})

    # Prepare gender mapping from the players dataframe
    players_df = st.session_state.players_df
    gender_map = pd.Series(players_df.gender.values, index=players_df.name).to_dict() if not players_df.empty else {}

    if not matches_to_rank.empty:
        matches_to_rank = matches_to_rank.sort_values('date')

    for row in matches_to_rank.itertuples(index=False):
        match_type = row.match_type
        
        # 1. Identify valid players (Excluding Visitors from gender checks and ranking points)
        t1 = [p for p in [row.team1_player1, row.team1_player2] if p and str(p).strip() and str(p).upper() != "VISITOR"]
        t2 = [p for p in [row.team2_player1, row.team2_player2] if p and str(p).strip() and str(p).upper() != "VISITOR"]
        
        if not t1 or not t2: 
            continue

        # 2. AUTOMATED MIXED DOUBLES DETECTION
        # Criteria: M+F vs M+F strictly. If a Visitor is in the match, t1 or t2 length 
        # will be less than 2 (since we filtered them above), so it stays 'is_mixed = False'.
        is_mixed = False
        if match_type in ['Doubles', 'Mixed Doubles'] and len(t1) == 2 and len(t2) == 2:
            g1 = sorted([gender_map.get(p, 'U') for p in t1]) # 'U' for Unknown
            g2 = sorted([gender_map.get(p, 'U') for p in t2])
            if g1 == ['F', 'M'] and g2 == ['F', 'M']:
                is_mixed = True

        match_gd = 0
        is_clutch = False
        sets = [row.set1, row.set2, row.set3]
        winner_code = row.winner

        # 3. Game and Set Logic
        for s in sets:
            if not s or str(s).lower() == 'nan': continue
            s_str = str(s)
            t1_g, t2_g = 0, 0
            
            if "Tie Break" in s_str:
                is_clutch = True
                nums = [int(x) for x in re.findall(r'\d+', s_str)]
                if len(nums) >= 2:
                    if nums[0] > nums[1]: t1_g, t2_g = 7, 6
                    else: t1_g, t2_g = 6, 7
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

        if row.set3 and str(row.set3).strip() and str(row.set3).lower() != 'nan': 
            is_clutch = True
        
        # 4. POINT SYSTEM ASSIGNMENT
        # Winners: 3 pts (Mixed) or 2 pts (Singles/Standard Doubles)
        # Losers: Always 1 pt
        w_points = 3 if is_mixed else 2
        
        def update_player_metrics(players, outcome):
            for p in players:
                stats[p]['matches'] += 1
                if is_clutch: stats[p]['clutch_matches'] += 1
                
                # Logic for performance breakdown tabs
                if match_type == 'Singles': 
                    perf_breakdown[p]['singles_m'] += 1
                else: 
                    perf_breakdown[p]['doubles_m'] += 1

                if outcome == 'win':
                    scores[p] += w_points
                    stats[p]['wins'] += 1
                    if is_clutch: stats[p]['clutch_wins'] += 1
                    if match_type == 'Singles': perf_breakdown[p]['singles_w'] += 1
                    else: perf_breakdown[p]['doubles_w'] += 1
                    
                    if current_streaks[p] < 0: current_streaks[p] = 0
                    current_streaks[p] += 1
                elif outcome == 'loss':
                    scores[p] += 1
                    stats[p]['losses'] += 1
                    if current_streaks[p] > 0: current_streaks[p] = 0
                    current_streaks[p] -= 1
                else: # Tie logic
                    scores[p] += 1.5
                    current_streaks[p] = 0

        if winner_code == "Team 1":
            update_player_metrics(t1, 'win')
            update_player_metrics(t2, 'loss')
        elif winner_code == "Team 2":
            update_player_metrics(t2, 'win')
            update_player_metrics(t1, 'loss')
        else:
            update_player_metrics(t1, 'tie')
            update_player_metrics(t2, 'tie')

        # 5. PARTNER STATS (Tracks performance of specific pairings)
        if match_type in ['Doubles', 'Mixed Doubles']:
            for team, code, gd_val in [(t1, 1, match_gd), (t2, 2, -match_gd)]:
                if len(team) < 2: continue
                p1, p2 = team[0], team[1]
                for a, b in [(p1, p2), (p2, p1)]:
                    ps = partner_stats[a][b]
                    ps['matches'] += 1
                    ps['game_diff_sum'] += gd_val
                    if winner_code == "Tie": ps['ties'] += 1
                    elif (winner_code == "Team 1" and code == 1) or (winner_code == "Team 2" and code == 2):
                        ps['wins'] += 1
                    else: ps['losses'] += 1

    # 6. RANKING DATA AGGREGATION
    rank_data = []
    img_map = pd.Series(players_df.profile_image_url.values, index=players_df.name).to_dict() if not players_df.empty else {}
    
    for p, s in stats.items():
        m_played = s['matches']
        if m_played == 0: continue
        
        clutch_pct = (s['clutch_wins'] / s['clutch_matches'] * 100) if s['clutch_matches'] > 0 else 0
        consistency = np.std(s['gd_list']) if s['gd_list'] else 0
        
        pb = perf_breakdown[p]
        s_perf = (pb['singles_w'] / pb['singles_m'] * 100) if pb['singles_m'] > 0 else 0
        d_perf = (pb['doubles_w'] / pb['doubles_m'] * 100) if pb['doubles_m'] > 0 else 0

        badges = []
        if clutch_pct > 70 and s['clutch_matches'] >= 3: badges.append("🎯 Clutch")
        if consistency < 2.5 and m_played >= 5: badges.append("📉 Steady")
        if current_streaks[p] >= 3: badges.append("🔥 Hot")
        
        rank_data.append({
            "Player": p, 
            "Points": scores[p], 
            "Win %": round((s['wins']/m_played)*100, 1),
            "Matches": m_played, 
            "Wins": s['wins'], 
            "Losses": s['losses'],
            "Games Won": s['games_won'], 
            "Game Diff Avg": round(s['gd_sum']/m_played, 2),
            "Clutch Factor": round(clutch_pct, 1), 
            "Consistency Index": round(consistency, 2),
            "Singles Perf": round(s_perf, 1), 
            "Doubles Perf": round(d_perf, 1),
            "Badges": badges, 
            "Profile": img_map.get(p, "")
        })
        
    df = pd.DataFrame(rank_data)
    if not df.empty:
        # Sort by primary Points, then secondary Win %
        df = df.sort_values(by=["Points", "Win %"], ascending=[False, False]).reset_index(drop=True)
        df["Rank"] = [f"🏆 {i+1}" for i in df.index]
        
    return df, partner_stats


def detect_match_category(row, gender_map):
    """
    Automated detection: 
    Mixed Doubles = (M+F) vs (M+F) with NO Visitors.
    """
    if row.match_type == "Singles":
        return "Singles"
    
    # Identify players
    p1, p2 = str(row.team1_player1), str(row.team1_player2)
    p3, p4 = str(row.team2_player1), str(row.team2_player2)
    
    # 1. Check for Visitors (Manual override to standard Doubles)
    all_players = [p1.upper(), p2.upper(), p3.upper(), p4.upper()]
    if "VISITOR" in all_players or "NONE" in all_players or "" in all_players:
        return "Doubles"

    # 2. Get Genders from the players_rows.csv map
    g1, g2 = gender_map.get(p1), gender_map.get(p2)
    g3, g4 = gender_map.get(p3), gender_map.get(p4)

    # 3. Check for Mixed (One M, One F per team)
    team1_is_mixed = set([g1, g2]) == {'M', 'F'}
    team2_is_mixed = set([g3, g4]) == {'M', 'F'}

    if team1_is_mixed and team2_is_mixed:
        return "Mixed Doubles"
    
    return "Doubles"



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

# --- Helper to plot trend ---
@st.cache_data(ttl=300)
def plot_player_performance(player_name, matches_df):
    if matches_df.empty: return None
    
    # Filter matches involving the player
    mask = (matches_df['team1_player1'] == player_name) | (matches_df['team1_player2'] == player_name) | \
            (matches_df['team2_player1'] == player_name) | (matches_df['team2_player2'] == player_name)
    df = matches_df[mask].copy()
    
    if df.empty: return None
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    history = []
    cum_gd = 0
    matches_count = 0
    
    for row in df.itertuples():
        # Determine side and sets
        is_t1 = player_name in [row.team1_player1, row.team1_player2]
        
        match_gd = 0
        sets = [row.set1, row.set2, row.set3]
        for s in sets:
            if not s: continue
            s_str = str(s)
            t1_g, t2_g = 0, 0
            if "Tie Break" in s_str: 
                nums = re.findall(r'\d+', s_str)
                if len(nums) >= 2:
                    g1, g2 = int(nums[0]), int(nums[1])
                    if g1 > g2: t1_g, t2_g = 1, 0
                    else: t1_g, t2_g = 0, 1
            elif '-' in s_str:
                try:
                    p = s_str.split('-')
                    t1_g, t2_g = int(p[0]), int(p[1])
                except: pass
            
            if is_t1: match_gd += (t1_g - t2_g)
            else: match_gd += (t2_g - t1_g)
        
        cum_gd += match_gd
        matches_count += 1
        
        # Determine Result label
        w = row.winner
        res = "Tie"
        if w == "Team 1": res = "Win" if is_t1 else "Loss"
        elif w == "Team 2": res = "Win" if not is_t1 else "Loss"
        
        history.append({
            "Date": row.date,
            "Match": f"Match {matches_count}",
            "Cumulative Game Diff": cum_gd,
            "Result": res,
            "Opponents": f"{row.team2_player1}/{row.team2_player2}" if is_t1 else f"{row.team1_player1}/{row.team1_player2}"
        })
        
    fig = px.line(history, x="Match", y="Cumulative Game Diff", 
                    hover_data=["Date", "Result", "Opponents"],
                    title=f"Performance Trend - {player_name}",
                    markers=True)
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def get_birthday_banner(players_df):
    if players_df.empty:
        return
    
    today = datetime.now()
    today_str = today.strftime("%d-%m") # Matches your CSV format like '15-10'
    
    # Check for players whose birthday matches today
    # We strip leading zeros to handle both '05-11' and '5-11'
    birthday_people = []
    for _, row in players_df.iterrows():
        if pd.notna(row['birthday']) and str(row['birthday']).strip() != "":
            # Normalize strings by removing leading zeros from both day and month
            # This ensures '5-6' matches '05-06'
            normalized_bday = "-".join([part.lstrip('0') for part in str(row['birthday']).split('-')])
            normalized_today = "-".join([part.lstrip('0') for part in today_str.split('-')])
            
            if normalized_bday == normalized_today:
                birthday_people.append(row['name'])

    if birthday_people:
        names = " & ".join(birthday_people)
        st.markdown(f"""
            <div style="
                background: linear-gradient(90deg, #fff500, #ff0055);
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 25px;
                animation: pulse 2s infinite;
                box-shadow: 0 4px 15px rgba(255, 245, 0, 0.4);
            ">
                <h2 style="color: white; margin: 0; font-size: 1.5em;">🎂 Happy Birthday, {names}! 🥳</h2>
                <p style="color: white; margin: 5px 0 0 0; opacity: 0.9;">Wishing you a great day on and off the court!</p>
            </div>
            <style>
            @keyframes pulse {{
                0% {{ transform: scale(1); }}
                50% {{ transform: scale(1.02); }}
                100% {{ transform: scale(1); }}
            }}
            </style>
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


#----------------------HALL OF FAME FUNCTION ---------------------------------------------




def display_hall_of_fame():
    """
    Fetches and displays detailed Hall of Fame data from Supabase.
    This version uses min-height to allow cards to dynamically resize.
    """
    st.header("🏆 Hall of Fame")

    def season_to_date(season_str):
        if not season_str:
            return datetime(1900, 1, 1)
        match = re.match(r'Q(\d) (\d{4})', season_str)
        if match:
            q = int(match.group(1))
            year = int(match.group(2))
            month = (q - 1) * 3 + 3  # Q1:3, Q2:6, Q3:9, Q4:12
            return datetime(year, month, 1)
        return datetime(1900, 1, 1)

    try:
        response = supabase.table(hall_of_fame_table_name).select("*").order("Season", desc=True).order("Rank", desc=False).execute()
        hof_data = response.data

        if not hof_data:
            st.info("The Hall of Fame is still empty. Add some top players from past seasons!")
            return

        # Using a set for faster unique lookups
        seasons = sorted(list(set(p['Season'] for p in hof_data)), key=season_to_date, reverse=True)

        for season in seasons:
            st.subheader(f"🏅 Season: {season}")
            
            season_players = [p for p in hof_data if p['Season'] == season]

            cols = st.columns(len(season_players) if len(season_players) <= 3 else 3)
            col_index = 0

            for player in season_players:
                with cols[col_index]:
                    # --- Robust Data Conversion & Handling ---
                    try:
                        rank = int(player.get('Rank', 0))
                        rank_emoji = '🥇' if rank == 1 else '🥈' if rank == 2 else '🥉'
                    except (ValueError, TypeError):
                        rank = player.get('Rank', 'N/A')
                        rank_emoji = '🏆'

                    try:
                        points_display = f"{float(player.get('Points', 0)):.2f}"
                    except (ValueError, TypeError):
                        points_display = player.get('Points', 'N/A')
                    try:
                        Games_won_display = f"{float(player.get('Games_won', 0)):.2f}"
                    except (ValueError, TypeError):
                        Games_won_display = player.get('Games_won', 'N/A')
                    try:
                        cumulative_GD_display = f"{float(player.get('cumulative_GD', 0)):.2f}"
                    except (ValueError, TypeError):
                        cumulative_GD_display = player.get('cumulative_GD', 'N/A')

                        
                    try:
                        gda_display = f"{float(player.get('GDA', 0)):.2f}"
                    except (ValueError, TypeError):
                        gda_display = player.get('GDA', 'N/A')

                    try:
                        win_rate_display = f"{float(player.get('WinRate', 0)):.1f}%"
                    except (ValueError, TypeError):
                        win_rate_display = f"{player.get('WinRate', 'N/A')}%"

                    matches_played = player.get('Matches', 'N/A')
                    performance_score = player.get('Performance_score', 'N/A')
                    profile_image = player.get('profile_image', '')
                    player_name = player.get('Player', 'N/A')

                    # --- Display Card ---
                    st.markdown(
                        f"""
                        <div class="court-card" style="text-align: center; padding: 15px; min-height: 390px; display: flex; flex-direction: column; justify-content: space-between;">
                            <div>
                                <img src="{profile_image}" class="profile-image" style="width:120px; height:120px; border-radius: 10%; border: 3px solid #fff500;">
                                <p style="font-size: 1.5em; font-weight: bold; color: #fff500; margin-top: 10px;">{player_name}</p>
                                <p style="font-size: 1.5em; margin-top: -10px; font-weight: bold;">
                                    {rank_emoji} Rank <span style="font-weight: bold; color: #FFFF00;">{rank}</span>
                                </p>
                            </div>
                            <div style="text-align: left; font-size: 0.95em; padding: 0 10px;">
                                <p><strong>Data for the Season:</strong></p>
                                <p><strong>Points won:</strong> <span style="font-weight: bold; color: #FFFF00;">{points_display}</span></p>
                                <p><strong>Games Won:</strong> <span style="font-weight: bold; color: #FFFF00;">{Games_won_display}</span></p>
                                <p><strong>Win Rate:</strong> <span style="font-weight: bold; color: #FFFF00;">{win_rate_display}</span></p>
                                <p><strong>Matches Played:</strong> <span style="font-weight: bold; color: #FFFF00;">{matches_played}</span></p>
                                <p><strong>Game Differential Avg:</strong> <span style="font-weight: bold; color: #FFFF00;">{gda_display}</span></p>
                                <p><strong>Cumulative Game Differential:</strong> <span style="font-weight: bold; color: #FFFF00;">{cumulative_GD_display}</span></p>
                                <p><strong>Performance Score:</strong> <span style="font-weight: bold; color: #FFFF00;">{performance_score}</span></p>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                col_index = (col_index + 1) % 3

            st.markdown("<hr>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please double-check your Supabase table name and column names for any typos.")





# --- Initial Load ---
load_players()
load_matches()
load_bookings()

# --- Global Data Pre-calculation ---
# Calculate rankings once so they are available for all tabs
rank_df = pd.DataFrame()
partner_stats_global = {}
if not st.session_state.matches_df.empty:
    rank_df, partner_stats_global = calculate_rankings(st.session_state.matches_df)

# --- Main Layout ---
st.image("https://raw.githubusercontent.com/mahadevbk/mmd/main/mmdheaderQ12026.png", width="stretch")
get_birthday_banner(st.session_state.players_df)

tab_names = ["Rankings", "Matches", "Player Profile", "Court Locations", "Bookings", "Hall of Fame", "Mini Tourney", "MMD AI"]
tabs = st.tabs(tab_names)

# --- Tab 1: Rankings ---
with tabs[0]:
    st.header(f"Rankings as of {datetime.now().strftime('%d %b %Y')}")
    ranking_view = st.radio("View", ["Combined", "Doubles", "Singles", "Table View"], horizontal=True, key="rank_view_radio")
    
    # Determine which dataframe to use for display
    display_rank_df = rank_df
    if not st.session_state.matches_df.empty:
        if ranking_view == "Doubles":
            m_sub = st.session_state.matches_df[st.session_state.matches_df.match_type == "Doubles"]
            display_rank_df, _ = calculate_rankings(m_sub)
        elif ranking_view == "Singles":
            m_sub = st.session_state.matches_df[st.session_state.matches_df.match_type == "Singles"]
            display_rank_df, _ = calculate_rankings(m_sub)
            
        if ranking_view == "Table View":
            st.dataframe(
                display_rank_df, 
                hide_index=True, 
                width="stretch",
                column_config={
                    "Profile": st.column_config.ImageColumn("Profile"),
                }
            )
        else:
            # Podium for Top 3
            if len(display_rank_df) >= 3:
                top3 = display_rank_df.head(3).to_dict('records')
                
                podium_items = [
                    {"player": top3[1], "margin": "40px"}, # Rank 2
                    {"player": top3[0], "margin": "0px"},  # Rank 1
                    {"player": top3[2], "margin": "40px"}  # Rank 3
                ]
                
                cols_html = ""
                for item in podium_items:
                    p = item['player']
                    cols_html += f"""
<div style="flex: 1; margin-top: {item['margin']}; min-width: 0; display: flex; flex-direction: column;">
    <div style="flex-grow: 1; text-align: center; padding: 10px 2px; background: rgba(255,255,255,0.05); border-radius: 12px; border: 1px solid rgba(255,215,0,0.3); box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
        <div style="font-size: 1.2em; margin-bottom: 5px; color: #FFD700; font-weight: bold;">{p['Rank']}</div>
        <div style="display: flex; justify-content: center; margin-bottom: 5px;">
            <img src="{p['Profile'] or 'https://via.placeholder.com/100?text=Player'}" style="width: clamp(50px, 20vw, 90px); height: clamp(50px, 20vw, 90px); border-radius: 15px; object-fit: cover; border: 2px solid #fff500; box-shadow: 0 0 15px rgba(255,245,0,0.6);">
        </div>
        <div style="margin: 5px 0; color: #fff500; font-size: 0.9em; font-weight: bold; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; padding: 0 2px;">{p['Player']}</div>
        <div style="color: white; font-weight: bold; font-size: 0.8em;">{p['Points']} pts</div>
        <div style="color: #aaa; font-size: 0.7em;">{p['Win %']}% Win</div>
    </div>
</div>"""
                
                st.markdown(f"""
<div style="display: flex; flex-direction: row; flex-wrap: nowrap; justify-content: center; align-items: flex-start; gap: 8px; margin-bottom: 25px; width: 100%;">
{cols_html}
</div>""", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)

            # Detailed List (HTML based)
            st.markdown("<div class='rankings-table-container'>", unsafe_allow_html=True)
            for row in display_rank_df.to_dict('records'):
                badges_str = " ".join([f"<span class='badge'>{b}</span>" for b in row['Badges']]) if row['Badges'] else ""
                trend_str = get_player_trend(row['Player'], st.session_state.matches_df)
                
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














# --- Tab 2: Matches ---
# --- Tab 2: Matches ---
with tabs[1]:
    st.header("Matches")
    
    # 1. Custom CSS for Cards and Images
    st.markdown("""
        <style>
        .match-score-container {
            text-align: center;
            background: rgba(0,0,0,0.3);
            padding: 8px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .gda-label {
            font-size: 0.85em;
            color: #CCFF00; /* Optic Yellow */
            font-weight: bold;
            margin-top: 4px;
            border-top: 1px solid rgba(255,255,255,0.1);
            padding-top: 4px;
        }
        .player-name-bold {
            color: #CCFF00; /* Optic Yellow */
            font-weight: bold;
            text-transform: uppercase;
        }
        .status-text-grey {
            color: #888888; /* Grey */
            font-size: 0.9em;
            font-weight: normal;
        }
        .match-img-wrapper {
            width: 100%;
            display: flex;
            justify-content: center;
            background: transparent;
        }
        .match-img-content {
            width: 100%;
            max-height: 550px;
            object-fit: contain;
            display: block;
        }
        </style>
    """, unsafe_allow_html=True)

    # 2. Helper Logic for Category Detection
    gender_map = dict(zip(st.session_state.players_df['name'], st.session_state.players_df['gender']))
    
    def auto_detect_category(row):
        # row can be a namedtuple from itertuples or a Series from iloc
        m_type = getattr(row, 'match_type', 'Doubles')
        if m_type == "Singles": return "Singles"
        
        p1 = str(getattr(row, 'team1_player1', ''))
        p2 = str(getattr(row, 'team1_player2', ''))
        p3 = str(getattr(row, 'team2_player1', ''))
        p4 = str(getattr(row, 'team2_player2', ''))
        
        p_list = [p.upper() for p in [p1, p2, p3, p4]]
        if "VISITOR" in p_list or "" in p_list or "NONE" in p_list:
            return "Doubles"
            
        g1 = sorted([gender_map.get(p1, 'U'), gender_map.get(p2, 'U')])
        g2 = sorted([gender_map.get(p3, 'U'), gender_map.get(p4, 'U')])
        
        if g1 == ['F', 'M'] and g2 == ['F', 'M']:
            return "Mixed Doubles"
        return "Doubles"

    # --- Match Forms ---
    if not st.session_state.players_df.empty:
        names = sorted([n for n in st.session_state.players_df['name'] if n.upper() != 'VISITOR'])
        
        # A. POST MATCH FORM
        with st.expander("➕ Post Match Result", expanded=False, icon="➡️"):
            with st.form("match_form"):
                mtype = st.radio("Type", ["Singles", "Doubles"])
                c1, c2 = st.columns(2)
                if mtype == "Doubles":
                    t1p1 = c1.selectbox("T1 P1", [""]+names)
                    t1p2 = c1.selectbox("T1 P2", ["", "Visitor"]+names)
                    t2p1 = c2.selectbox("T2 P1", [""]+names)
                    t2p2 = c2.selectbox("T2 P2", ["", "Visitor"]+names)
                else:
                    t1p1 = c1.selectbox("P1", [""]+names); t2p1 = c2.selectbox("P2", [""]+names)
                    t1p2, t2p2 = None, None
                
                date = st.date_input("Date")
                s1 = st.selectbox("Set 1", [""]+tennis_scores()); s2 = st.selectbox("Set 2", [""]+tennis_scores()); s3 = st.selectbox("Set 3", [""]+tennis_scores())
                winner = st.selectbox("Winner", ["Team 1", "Team 2", "Tie"])
                img = st.file_uploader("Upload Image", type=["jpg","png"])
                
                if st.form_submit_button("Submit"):
                    mid = generate_match_id(st.session_state.matches_df, pd.to_datetime(date))
                    url = upload_image_to_github(img, mid) if img else ""
                    new_match = {
                        "match_id": mid, "date": date.isoformat(), "match_type": mtype,
                        "team1_player1": t1p1, "team1_player2": t1p2, "team2_player1": t2p1, "team2_player2": t2p2,
                        "set1": s1, "set2": s2, "set3": s3, "winner": winner, "match_image_url": url
                    }
                    st.session_state.matches_df = pd.concat([st.session_state.matches_df, pd.DataFrame([new_match])], ignore_index=True)
                    save_matches(st.session_state.matches_df)
                    st.success("Saved!"); st.rerun()

        # B. EDIT MATCH FORM (Restored)
        with st.expander("✏️ Edit Match Result", expanded=False, icon="➡️"):
            if not st.session_state.matches_df.empty:
                m_df = st.session_state.matches_df.sort_values('date', ascending=False)
                match_options = {f"{r.date[:10]} | {r.team1_player1} vs {r.team2_player1}": r.match_id for r in m_df.itertuples()}
                sel_label = st.selectbox("Select Match to Edit", list(match_options.keys()))
                
                if sel_label:
                    mid_edit = match_options[sel_label]
                    row_edit = m_df[m_df['match_id'] == mid_edit].iloc[0]
                    
                    with st.form("edit_match_form"):
                        em_type = st.radio("Type", ["Singles", "Doubles"], index=0 if row_edit.match_type=="Singles" else 1)
                        ec1, ec2 = st.columns(2)
                        
                        def get_idx(val, opt): 
                            try: return opt.index(val) 
                            except: return 0

                        if em_type == "Doubles":
                            et1p1 = ec1.selectbox("T1 P1", [""]+names, index=get_idx(row_edit.team1_player1, [""]+names))
                            et1p2 = ec1.selectbox("T1 P2", ["", "Visitor"]+names, index=get_idx(row_edit.team1_player2, ["", "Visitor"]+names))
                            et2p1 = ec2.selectbox("T2 P1", [""]+names, index=get_idx(row_edit.team2_player1, [""]+names))
                            et2p2 = ec2.selectbox("T2 P2", ["", "Visitor"]+names, index=get_idx(row_edit.team2_player2, ["", "Visitor"]+names))
                        else:
                            et1p1 = ec1.selectbox("P1", [""]+names, index=get_idx(row_edit.team1_player1, [""]+names))
                            et2p1 = ec2.selectbox("P2", [""]+names, index=get_idx(row_edit.team2_player1, [""]+names))
                            et1p2, et2p2 = None, None
                        
                        edate = st.date_input("Date", value=pd.to_datetime(row_edit.date))
                        es1 = st.selectbox("Set 1", [""]+tennis_scores(), index=get_idx(row_edit.set1, [""]+tennis_scores()))
                        es2 = st.selectbox("Set 2", [""]+tennis_scores(), index=get_idx(row_edit.set2, [""]+tennis_scores()))
                        es3 = st.selectbox("Set 3", [""]+tennis_scores(), index=get_idx(row_edit.set3, [""]+tennis_scores()))
                        ewinner = st.selectbox("Winner", ["Team 1", "Team 2", "Tie"], index=get_idx(row_edit.winner, ["Team 1", "Team 2", "Tie"]))
                        
                        if st.form_submit_button("Update"):
                            upd = {
                                "match_id": mid_edit, "date": edate.isoformat(), "match_type": em_type,
                                "team1_player1": et1p1, "team1_player2": et1p2, "team2_player1": et2p1, "team2_player2": et2p2,
                                "set1": es1, "set2": es2, "set3": es3, "winner": ewinner, "match_image_url": row_edit.match_image_url
                            }
                            st.session_state.matches_df = st.session_state.matches_df[st.session_state.matches_df['match_id'] != mid_edit]
                            st.session_state.matches_df = pd.concat([st.session_state.matches_df, pd.DataFrame([upd])], ignore_index=True)
                            save_matches(st.session_state.matches_df); st.success("Updated!"); st.rerun()

    # --- Match History ---
    st.subheader("History")
    m_hist = st.session_state.matches_df.copy()
    if not m_hist.empty:
        m_hist['date'] = pd.to_datetime(m_hist['date'])
        m_hist = m_hist.sort_values('date', ascending=False)
        
        for row in m_hist.itertuples():
            display_type = auto_detect_category(row)
            t1_total, t2_total, sets_count = 0, 0, 0
            display_scores = []
            for s in [row.set1, row.set2, row.set3]:
                if s and str(s).strip() and str(s).lower() != 'nan':
                    nums = re.findall(r'\d+', str(s))
                    if len(nums) >= 2:
                        g1, g2 = int(nums[0]), int(nums[1])
                        t1_total, t2_total, sets_count = t1_total+g1, t2_total+g2, sets_count+1
                        display_scores.append(f"{g1}-{g2}" if not ("Tie Break" in str(s)) else f"7-6 (TB {g1}-{g2})")

            match_gda = round(abs(t1_total - t2_total) / sets_count, 2) if sets_count > 0 else 0
            
            def fmt_team(p1, p2):
                p1s, p2s = str(p1), str(p2)
                if display_type in ["Doubles", "Mixed Doubles"]:
                    return f"<span class='player-name-bold'>{p1s} / {p2s}</span>"
                return f"<span class='player-name-bold'>{p1s}</span>"

            t1_h, t2_h = fmt_team(row.team1_player1, row.team1_player2), fmt_team(row.team2_player1, row.team2_player2)
            status_txt = "defeated" if row.winner != "Tie" else "tied with"
            winner_h = t1_h if row.winner == "Team 1" else t2_h if row.winner == "Team 2" else t1_h
            loser_h = t2_h if row.winner == "Team 1" else t1_h if row.winner == "Team 2" else t2_h
            headline = f"{winner_h} <span class='status-text-grey'>{status_txt}</span> {loser_h}"

            img_html = f'<div class="match-img-wrapper"><img src="{row.match_image_url}" class="match-img-content"></div>' if row.match_image_url else ""
            
            card_html = f"""
                <div style="background: rgba(255,255,255,0.05); border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 25px; overflow: hidden;">
                    {img_html}
                    <div style="padding: 15px;">
                        <div style="font-size: 0.85em; color: #888; margin-bottom: 8px;">{row.date.strftime('%d %b %Y')} | {display_type}</div>
                        <div style="font-size: 1.1em; text-align: center; margin: 10px 0;">{headline}</div>
                        <div class="match-score-container">
                            <div style="font-size: 1.2em; font-weight: bold; color: #FF7518;">{" | ".join(display_scores)}</div>
                            <div class="gda-label">Game Diff Avg: +{match_gda}</div>
                        </div>
                    </div>
                </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
    else:
        st.info("No matches recorded.")









# --- Tab 3: Player Profile ---
with tabs[2]:
    st.header("Player Profile")

    # CSS for badge styling
    st.markdown("""
        <style>
        .badge {
            background: #fff500; color: black; padding: 2px 8px; 
            border-radius: 10px; font-size: 0.75em; font-weight: bold; margin-left: 5px;
        }
        .stat-box {
            background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; 
            border-left: 4px solid #fff500; margin-bottom: 10px;
        }
        .metric-label { font-size: 0.7em; color: #aaa; text-transform: uppercase; }
        .metric-value { font-size: 1.1em; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

    # Birthday Helper
    def parse_bd(val):
        if pd.isna(val) or not str(val).strip(): return pd.NaT
        s = str(val).strip()
        if len(s) <= 5: s += "-2024"
        return pd.to_datetime(s, dayfirst=True, errors='coerce')

    # --- Manage Profiles ---
    with st.expander("⚙️ Manage Player Profiles", expanded=False, icon="➡️"):
        mp_action = st.radio("Action", ["Add New", "Edit Existing"], horizontal=True)
        with st.form("player_form"):
            if mp_action == "Add New":
                n, img, dob, g = st.text_input("Name"), st.text_input("Img URL"), st.date_input("Bday", value=None), st.selectbox("Gender", ["M", "F"])
                orig_name = None
            else:
                names = sorted(st.session_state.players_df['name'].unique())
                sel = st.selectbox("Select Player", names)
                curr = st.session_state.players_df[st.session_state.players_df['name'] == sel].iloc[0] if sel else None
                n = st.text_input("Name", value=curr['name'] if curr is not None else "")
                img = st.text_input("Img URL", value=curr['profile_image_url'] if curr is not None else "")
                dob = st.date_input("Bday", value=parse_bd(curr['birthday']) if curr is not None else None)
                g = st.selectbox("Gender", ["M", "F"], index=0 if curr is not None and curr['gender'] == "M" else 1)
                orig_name = sel
            
            if st.form_submit_button("Save"):
                if n:
                    new_e = {"name": n.upper().strip(), "profile_image_url": img, "birthday": dob.strftime("%d/%m/%Y") if dob else None, "gender": g}
                    if orig_name: st.session_state.players_df = st.session_state.players_df[st.session_state.players_df['name'] != orig_name]
                    st.session_state.players_df = pd.concat([st.session_state.players_df, pd.DataFrame([new_e])], ignore_index=True)
                    save_players(st.session_state.players_df)
                    st.success("Saved!"); st.rerun()

    st.divider()
    sort_opt = st.radio("Sort By", ["Alphabetical", "Birthday"], horizontal=True)

    # --- Process Display ---
    disp = st.session_state.players_df.copy()
    disp['dt_birthday'] = disp['birthday'].apply(parse_bd)
    if sort_opt == "Birthday":
        disp = disp.dropna(subset=['dt_birthday']).sort_values(['dt_birthday'])
    else:
        disp = disp.sort_values("name")

    for idx, row in disp.iterrows():
        p_name = row['name']
        p_stats = rank_df[rank_df['Player'] == p_name] if not rank_df.empty else pd.DataFrame()
        has_stats = not p_stats.empty
        s = p_stats.iloc[0] if has_stats else {}

        with st.container():
            c1, c2 = st.columns([1, 3])
            with c1:
                img = row['profile_image_url'] or "https://via.placeholder.com/150"
                bday_str = f"🎂 {row['dt_birthday'].strftime('%d %b')}" if pd.notna(row['dt_birthday']) else ""
                st.markdown(f"""
                    <div style="text-align: center;">
                        <img src="{img}" style="width: 120px; height: 120px; object-fit: cover; border-radius: 15px; border: 3px solid #fff500;">
                        <div style="margin-top: 10px; font-weight: bold; font-size: 1.2em;">{p_name}</div>
                        <div style="color: #ffd700; font-size: 0.85em;">{bday_str}</div>
                    </div>
                """, unsafe_allow_html=True)

            with c2:
                if has_stats:
                    badges_html = "".join([f"<span class='badge'>{b}</span>" for b in s.get('Badges', [])])
                    st.markdown(f"""
                    <div class="stat-box">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 5px;">
                            <span style="color: #fff500; font-weight: bold; font-size: 1.1em;">Rank: {s.get('Rank', 'N/A')}</span>
                            <div>{badges_html}</div>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; text-align: center;">
                            <div><div class="metric-label">Games Won</div><div class="metric-value">{s.get('Games Won', 0)}</div></div>
                            <div><div class="metric-label">GD Avg</div><div class="metric-value">{s.get('Game Diff Avg', 0)}</div></div>
                            <div><div class="metric-label">Clutch</div><div class="metric-value">{s.get('Clutch Factor', 0)}%</div></div>
                            <div><div class="metric-label">Consistency</div><div class="metric-value">{s.get('Consistency Index', 0)}</div></div>
                            <div><div class="metric-label">Doubles Perf</div><div class="metric-value" style="color: #00ff00;">{s.get('Doubles Perf', 0)}%</div></div>
                            <div><div class="metric-label">Singles Perf</div><div class="metric-value" style="color: #00bfff;">{s.get('Singles Perf', 0)}%</div></div>
                            <div><div class="metric-label">Win %</div><div class="metric-value">{s.get('Win %', 0)}%</div></div>
                            <div><div class="metric-label">Record</div><div class="metric-value">{s.get('Wins', 0)}W-{s.get('Losses', 0)}L</div></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("Details & Partners", expanded=False, icon="➡️"):
                        t1, t2 = st.tabs(["Trends", "Partners"])
                        with t1:
                            fig = plot_player_performance(p_name, st.session_state.matches_df)
                            if fig: st.plotly_chart(fig, use_container_width=True, key=f"p_{idx}")
                        with t2:
                            if p_name in partner_stats_global:
                                parts = partner_stats_global[p_name]
                                p_list = [{"Partner": n, "Win%": round((d['wins']/d['matches'])*100,1), "GD": d['game_diff_sum']} for n, d in parts.items() if d['matches'] > 0]
                                if p_list: st.dataframe(pd.DataFrame(p_list).sort_values("Win%", ascending=False), hide_index=True)
                else:
                    st.info("No match data yet.")
        st.divider()



# --- Tab 4: Maps ---
with tabs[3]:
    st.header("Court Locations")
    court_icon_url = "https://img.icons8.com/color/48/000000/tennis.png"
    
    known_urls = {
        "Alvorado 1": "https://maps.google.com/?q=25.041792,55.259258",
        "Alvorado 2": "https://maps.google.com/?q=25.041792,55.259258",
        "Palmera 2": "https://maps.app.goo.gl/CHimjtqQeCfU1d3W6",
        "Palmera 4": "https://maps.app.goo.gl/4nn1VzqMpgVkiZGN6",
        "Saheel": "https://maps.app.goo.gl/a7qSvtHCtfgvJoxJ8",
        "Hattan": "https://maps.app.goo.gl/fjGpeNzncyG1o34c7",
        "MLC Mirador La Colleccion": "https://maps.app.goo.gl/n14VSDAVFZ1P1qEr6",
        "Al Mahra": "https://maps.app.goo.gl/zVivadvUsD6yyL2Y9",
        "Mirador": "https://maps.app.goo.gl/kVPVsJQ3FtMWxyKP8",
        "Reem 1": "https://maps.app.goo.gl/qKswqmb9Lqsni5RD7",
        "Reem 2": "https://maps.app.goo.gl/oFaUFQ9DRDMsVbMu5",
        "Reem 3": "https://maps.app.goo.gl/o8z9pHo8tSqTbEL39",
        "Alma": "https://maps.app.goo.gl/BZNfScABbzb3osJ18",
        "Mira 2": "https://maps.app.goo.gl/JeVmwiuRboCnzhnb9",
        "Mira 4": "https://maps.app.goo.gl/e1Vqv5MJXB1eusv6A",
        "Mira 5 A": "https://maps.app.goo.gl/rWBj5JEUdw4LqJZb6",
        "Mira 5 B": "https://maps.app.goo.gl/rWBj5JEUdw4LqJZb6",
        "Mira Oasis 1": "https://maps.app.goo.gl/F9VYsFBwUCzvdJ2t8",
        "Mira Oasis 2": "https://maps.app.goo.gl/ZNJteRu8aYVUy8sd9",
        "Mira Oasis 3 A": "https://maps.app.goo.gl/ouXQGUxYSZSfaW1z9",
        "Mira Oasis 3 B": "https://maps.app.goo.gl/ouXQGUxYSZSfaW1z9",
        "Mira Oasis 3 C": "https://maps.app.goo.gl/kf7A9K7DoYm4PEPu8",
        "Mudon Main courts": "https://maps.app.goo.gl/AZ8WJ1mnnwMgNxhz7?g_st=aw",
        "Mudon Arabella": "https://maps.app.goo.gl/iudbB5YqrGKyHNqM6?g_st=aw",
        "Mudon Arabella 3": "https://maps.app.goo.gl/o46ERJCq8LKg1Cz59?g_st=aw",
        "AR2 Rosa": "https://maps.app.goo.gl/at1EKgatfMmvAg7g8?g_st=aw",
        "AR2 Palma": "https://maps.app.goo.gl/oKxXvbXKYe3JgJco8?g_st=aw",
        "AR 2 Fitness First": "https://maps.app.goo.gl/iZGipHv8KdfW82dW9?g_st=aw",
        "Dubai Hills Maple": "https://maps.app.goo.gl/rypmwnSGbGeknykv6?g_st=aw",
    }
    
    court_names = [
        "Alvorado 1","Alvorado 2", "Palmera 2", "Palmera 4", "Saheel", "Hattan",
        "MLC Mirador La Colleccion", "Al Mahra", "Mirador", "Reem 1", "Reem 2",
        "Reem 3", "Alma", "Mira 2", "Mira 4", "Mira 5 A", "Mira 5 B", "Mira Oasis 1",
        "Mira Oasis 2", "Mira Oasis 3 A","Mira Oasis 3 B", "Mira Oasis 3 C","Mudon Main courts",
        "Mudon Arabella","Mudon Arabella 3","AR2 Rosa","AR2 Palma","AR 2 Fitness First","Dubai Hills Maple"
    ]
    
    ar_courts = []
    mira_courts = []
    other_courts = []
    
    for name in court_names:
        url = known_urls.get(name, f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(name + ' tennis court Dubai')}")
        item = {"name": name, "url": url}
        if any(x in name for x in ["Mira", "Mira Oasis"]):
            mira_courts.append(item)
        elif any(x in name for x in ["Alvorado", "Palmera", "Saheel", "Hattan", "MLC", "Mirador", "Al Mahra", "Reem", "Alma"]):
            ar_courts.append(item)
        else:
            other_courts.append(item)
    
    def display_courts(section_title, courts_list):
        if not courts_list: return
        # Create responsive grid: 3 cols if plenty of items, else 2 (or just stick to 3 for consistency)
        num_cols = 3 
        
        # Batch items into rows
        for i in range(0, len(courts_list), num_cols):
            cols = st.columns(num_cols)
            batch = courts_list[i:i+num_cols]
            for j, court in enumerate(batch):
                with cols[j]:
                    st.markdown(f"""
                    <div class="court-card">
                        <img src="{court_icon_url}" style="width: 40px; margin-bottom: 5px;">
                        <h4>{court['name']}</h4>
                        <a href="{court['url']}" target="_blank">View on Map</a>
                    </div>
                    """, unsafe_allow_html=True)
    
    with st.expander("Arabian Ranches Tennis Courts", expanded=False, icon="➡️"):
        display_courts("", ar_courts)
    with st.expander("Mira & Mira Oasis Tennis Courts", expanded=False, icon="➡️"):
        display_courts("", mira_courts)
    with st.expander("Mudon, AR2 & Other Tennis Courts", expanded=False, icon="➡️"):
        display_courts("", other_courts)

# --- Tab 5: Bookings ---
with tabs[4]:
    st.header("Court Bookings")
    
    with st.expander("📅 Book a Court", expanded=False, icon="➡️"):
        with st.form("booking_form"):
            b_date = st.date_input("Date")
            b_time = st.time_input("Time")
            b_court = st.selectbox("Court", ["Mira 4 Court 1", "Mira 4 Court 2", "Mira 1", "Town Square"])
            b_type = st.selectbox("Type", ["Match", "Practice", "Coaching"])
            bp1 = st.selectbox("Player 1", [""]+sorted(st.session_state.players_df['name'].tolist()))
            bp2 = st.selectbox("Player 2", [""]+sorted(st.session_state.players_df['name'].tolist()))
            
            if st.form_submit_button("Book Court"):
                new_booking = {
                    "booking_id": str(uuid.uuid4()),
                    "date": b_date.strftime("%Y-%m-%d"),
                    "time": b_time.strftime("%H:%M"),
                    "match_type": b_type,
                    "court_name": b_court,
                    "player1": bp1, "player2": bp2
                }
                st.session_state.bookings_df = pd.concat([st.session_state.bookings_df, pd.DataFrame([new_booking])], ignore_index=True)
                save_bookings(st.session_state.bookings_df)
                st.success("Booking Added!")
    
    if not st.session_state.bookings_df.empty:
        st.subheader("Upcoming Bookings")
        st.dataframe(st.session_state.bookings_df[['date', 'time', 'court_name', 'player1', 'player2', 'match_type']], hide_index=True, width="stretch")
    else:
        st.info("No upcoming bookings.")

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
