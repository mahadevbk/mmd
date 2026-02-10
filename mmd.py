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
import textwrap

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
.mobile-card {
    background: linear-gradient(135deg, #071a3d 0%, #0c0014 100%);
    border: 1px solid rgba(255, 245, 0, 0.2);
    border-radius: 15px;
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.5);
}
.rank-badge {
    background: #fff500;
    color: #041136;
    font-weight: bold;
    border-radius: 5px;
    padding: 2px 8px;
    font-size: 14px;
}
.trend-dot {
    height: 10px; width: 10px; border-radius: 50%; display: inline-block; margin-right: 3px;
}
.dot-w { background-color: #00ff88; box-shadow: 0 0 5px #00ff88; }
.dot-l { background-color: #ff4b4b; }
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
    if not file: 
        return ""
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
        # 1. Read the file into bytes
        content_bytes = file.getvalue()
        if not content_bytes:
            return ""

        # 2. Open and transpose image (fixes rotation issues from phones)
        img = Image.open(io.BytesIO(content_bytes))
        img = ImageOps.exif_transpose(img)
        
        # 3. Ensure image is in RGB mode (required for JPEG saving)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        
        # 4. Resize logic for match photos to prevent GitHub API timeouts
        if image_type == "match" and (img.width > 1200 or img.height > 1200):
            img.thumbnail((1200, 1200), Image.LANCZOS)
        
        # 5. Save to buffer and encode
        buffer = io.BytesIO()
        # Explicitly use JPEG to ensure the buffer is written correctly
        img.save(buffer, format="JPEG", quality=85)
        
        # Get the byte value and check if it's empty before decoding
        image_data = buffer.getvalue()
        if not image_data:
            raise ValueError("The image buffer is empty. Save operation failed.")
            
        content_b64 = base64.b64encode(image_data).decode("utf-8")
        
        # 6. GitHub API communication
        api_url = f"https://api.github.com/repos/{repo}/contents/{path_in_repo}"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
        
        # Check if file already exists to get its SHA (required for updates)
        sha = None
        resp = requests.get(api_url, headers=headers)
        if resp.status_code == 200:
            sha = resp.json().get('sha')
            
        payload = {
            "message": f"feat: Upload {image_type} {file_name}",
            "branch": branch,
            "content": content_b64
        }
        if sha: 
            payload["sha"] = sha
        
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

def create_radar_chart(player_data):
    """Generates a small radar chart for player stats."""
    categories = ['Win %', 'Clutch', 'Consistency', 'GDA', 'Exp']
    
    # Normalize stats for visual balance (0-100 scale)
    # Consistency: Lower is better, so we invert it (0 index = 100 score)
    consistency_score = max(0, 100 - (player_data.get('Consistency Index', 0) * 10))
    
    # GDA: Assume +3.0 is a perfect score
    gda_score = min(100, max(0, (player_data.get('Game Diff Avg', 0) + 1) * 25))
    
    values = [
        player_data.get('Win %', 0),
        player_data.get('Clutch Factor', 0),
        consistency_score,
        gda_score,
        min(100, (player_data.get('Matches', 0) / 15) * 100) # Experience cap at 15 matches
    ]
    
    # Close the polygon by repeating the first value
    values += values[:1]
    categories += categories[:1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(255, 245, 0, 0.3)',
        line=dict(color='#fff500', width=2),
        hoverinfo='r+theta'
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=False, range=[0, 100]),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.1)", 
                linecolor="rgba(255,255,255,0.1)",
                tickfont=dict(size=9, color="#aaa")
            )
        ),
        showlegend=False,
        margin=dict(l=25, r=25, t=10, b=10),
        height=140, # Compact height for mobile cards
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig



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
        is_mixed = False
        if match_type in ['Doubles', 'Mixed Doubles'] and len(t1) == 2 and len(t2) == 2:
            g1 = sorted([gender_map.get(p, 'U') for p in t1]) 
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
        w_points = 3 if is_mixed else 2
        
        def update_player_metrics(players, outcome):
            for p in players:
                stats[p]['matches'] += 1
                if is_clutch: stats[p]['clutch_matches'] += 1
                
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

        # 5. PARTNER STATS
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

        # --- Recent Trend Logic (Fixes KeyError) ---
        p_gd_list = s['gd_list'][-5:]
        trend_html = ""
        for gd in p_gd_list:
            dot_class = "dot-w" if gd > 0 else "dot-l"
            trend_html += f'<span class="trend-dot {dot_class}"></span>'

        badges = []
        if clutch_pct > 70 and s['clutch_matches'] >= 3: badges.append("🎯 Clutch")
        if consistency < 2.5 and m_played >= 5: badges.append("📉 Steady")
        if current_streaks[p] >= 3: badges.append("🔥 Hot")
        
        rank_data.append({
            "Player": p, 
            "Points": scores[p], 
            "Win %": round((s['wins']/m_played)*100, 1),
            "Recent Trend": trend_html, # Now explicitly included
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
        # 5-tier sorting logic
        df = df.sort_values(
            by=["Points", "Win %", "Game Diff Avg", "Games Won", "Player"], 
            ascending=[False, False, False, False, True] 
        ).reset_index(drop=True)
        
        df["Rank"] = [f"🏆 {i+1}" for i in df.index]
        
    return df, partner_stats


# ==============================================================================
# START: NEW COMPLEX ODDS CALCULATION FUNCTIONS
# ==============================================================================

def _calculate_performance_score(player_stats, full_dataset):
    """
    Calculates a weighted performance score for a player based on normalized stats.
    """
    # Define weights for each component
    w_wp = 0.50  # Win Percentage
    w_agd = 0.35 # Average Game Difference
    w_ef = 0.15  # Experience Factor (Matches Played)

    # --- 1. Normalize Win Percentage (WP) ---
    max_wp = full_dataset['Win %'].max()
    wp_norm = player_stats['Win %'] / max_wp if max_wp > 0 else 0

    # --- 2. Normalize Average Game Difference (AGD) ---
    max_agd = full_dataset['Game Diff Avg'].max()
    min_agd = full_dataset['Game Diff Avg'].min()
    if max_agd == min_agd:
        agd_norm = 0.5 # Avoid division by zero if all values are the same
    else:
        agd_norm = (player_stats['Game Diff Avg'] - min_agd) / (max_agd - min_agd)

    # --- 3. Normalize Experience Factor (EF) ---
    max_matches = full_dataset['Matches'].max()
    ef_norm = player_stats['Matches'] / max_matches if max_matches > 0 else 0

    # --- 4. Calculate Final Performance Score ---
    performance_score = (w_wp * wp_norm) + (w_agd * agd_norm) + (w_ef * ef_norm)
    
    return performance_score

def calculate_enhanced_doubles_odds(players, doubles_rank_df):
    """
    Calculates balanced teams and odds for a doubles match using a multi-factor Performance Score.
    """
    if len(players) != 4 or "" in players or doubles_rank_df.empty:
        return ("Please select four players with doubles match history.", None, None)

    player_scores = {}
    for player in players:
        player_data = doubles_rank_df[doubles_rank_df["Player"] == player]
        if not player_data.empty:
            # Calculate performance score for this player
            player_scores[player] = _calculate_performance_score(player_data.iloc[0], doubles_rank_df)
        else:
            # Player has no doubles history, assign a baseline score (e.g., 0)
            player_scores[player] = 0

    # Find the most balanced pairing based on the new Performance Score
    min_diff = float('inf')
    best_pairing = None
    
    for team1_combo in combinations(players, 2):
        team2_combo = tuple(p for p in players if p not in team1_combo)
        
        team1_score = sum(player_scores.get(p, 0) for p in team1_combo)
        team2_score = sum(player_scores.get(p, 0) for p in team2_combo)
        
        diff = abs(team1_score - team2_score)
        
        if diff < min_diff:
            min_diff = diff
            best_pairing = (team1_combo, team2_combo)

    if not best_pairing:
        return ("Could not determine a balanced pairing.", None, None)

    team1, team2 = best_pairing
    team1_total_score = sum(player_scores.get(p, 0) for p in team1)
    team2_total_score = sum(player_scores.get(p, 0) for p in team2)
    total_match_score = team1_total_score + team2_total_score

    team1_odds = (team1_total_score / total_match_score) * 100 if total_match_score > 0 else 50.0
    team2_odds = (team2_total_score / total_match_score) * 100 if total_match_score > 0 else 50.0

    # Styled output
    t1p1_styled = f"<span style='font-weight:bold; color:#fff500;'>{team1[0]}</span>"
    t1p2_styled = f"<span style='font-weight:bold; color:#fff500;'>{team1[1]}</span>"
    t2p1_styled = f"<span style='font-weight:bold; color:#fff500;'>{team2[0]}</span>"
    t2p2_styled = f"<span style='font-weight:bold; color:#fff500;'>{team2[1]}</span>"
    pairing_text = f"Team 1: {t1p1_styled} & {t1p2_styled} vs Team 2: {t2p1_styled} & {t2p2_styled}"
    
    return (pairing_text, team1_odds, team2_odds)

def calculate_enhanced_singles_odds(players, singles_rank_df):
    """
    Calculates odds for a singles match using a multi-factor Performance Score.
    """
    if len(players) != 2 or "" in players or singles_rank_df.empty:
        return (None, None)

    player_scores = {}
    for player in players:
        player_data = singles_rank_df[singles_rank_df["Player"] == player]
        if not player_data.empty:
            player_scores[player] = _calculate_performance_score(player_data.iloc[0], singles_rank_df)
        else:
            player_scores[player] = 0

    p1_score = player_scores.get(players[0], 0)
    p2_score = player_scores.get(players[1], 0)
    total_score = p1_score + p2_score

    p1_odds = (p1_score / total_score) * 100 if total_score > 0 else 50.0
    p2_odds = (p2_score / total_score) * 100 if total_score > 0 else 50.0

    return (p1_odds, p2_odds)

# ==============================================================================
# UPDATED: Original functions now call the new enhanced versions
# ==============================================================================

def suggest_balanced_pairing(players, doubles_rank_df):
    """Suggests balanced doubles teams. This function now calls the enhanced odds calculation."""
    if len(players) != 4 or "" in players:
        return ("Please select all four players for a doubles match.", None, None)
    
    return calculate_enhanced_doubles_odds(players, doubles_rank_df)

def suggest_singles_odds(players, singles_rank_df):
    """Calculates winning odds for a singles match. This function now calls the enhanced odds calculation."""
    if len(players) != 2 or "" in players:
        return (None, None)
        
    return calculate_enhanced_singles_odds(players, singles_rank_df)

# ==============================================================================
# END: NEW COMPLEX ODDS CALCULATION FUNCTIONS
# ==============================================================================


def generate_ics_for_booking(row, plain_suggestion=""):
    try:
        # Create a summary for the calendar event
        summary = f"Tennis: {row['match_type']} at {row['court_name']}"
        
        # Combine date and time (Time is usually HH:MM:SS from your DB)
        dt_str = f"{row['date']} {row['time']}"
        dt_start = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        dt_end = dt_start + timedelta(hours=1.5)  # Matches usually last 1.5 hours
        
        ics_format = "%Y%MT%H%M%S"
        
        # We include the 'plain_suggestion' (the odds/pairings) in the description
        description = f"MMD Tennis Match\\n{plain_suggestion}"
        
        ics_content = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//MMD Tennis League//EN
BEGIN:VEVENT
SUMMARY:{summary}
DTSTART:{dt_start.strftime(ics_format)}
DTEND:{dt_end.strftime(ics_format)}
LOCATION:{row['court_name']}
DESCRIPTION:{description}
END:VEVENT
END:VCALENDAR"""
        return ics_content, None
    except Exception as e:
        return None, str(e)



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

tab_names = ["Rankings", "Matches", "Player Profile", "Court Locations", "Bookings", "Hall of Fame", "Mini Tourney", "AI Data"]
tabs = st.tabs(tab_names)



# --- Tab 1: Rankings ---
# --- Tab 1: Rankings ---
with tabs[0]:
    st.header(f"Rankings as of {datetime.now().strftime('%d %b %Y')}")
    
    # 1. Filter Selection
    ranking_view = st.radio(
        "View", 
        ["Combined", "Doubles", "Singles", "Table View"], 
        horizontal=True, 
        key="rank_view_radio"
    )
    
    # 2. Determine which Dataframe to use
    display_rank_df = rank_df.copy() if not rank_df.empty else pd.DataFrame()

    if not st.session_state.matches_df.empty:
        if ranking_view == "Doubles":
            m_sub = st.session_state.matches_df[st.session_state.matches_df.match_type == "Doubles"]
            display_rank_df, _ = calculate_rankings(m_sub)
        elif ranking_view == "Singles":
            m_sub = st.session_state.matches_df[st.session_state.matches_df.match_type == "Singles"]
            display_rank_df, _ = calculate_rankings(m_sub)
    
    # 3. Handle Empty Data
    if display_rank_df.empty:
        st.info("No matches recorded for this category yet.")
    
    # 4. Table View (Classic)
    elif ranking_view == "Table View":
        st.dataframe(
            display_rank_df, 
            hide_index=True, 
            width=None, # Auto width
            column_config={
                "Profile": st.column_config.ImageColumn("Profile"),
                "Win %": st.column_config.ProgressColumn("Win %", format="%.1f%%", min_value=0, max_value=100),
            }
        )

    # 5. Mobile Card View (Graphical)
    else:
        # --- A. Podium for Top 3 ---
        if len(display_rank_df) >= 3:
            top3 = display_rank_df.head(3).to_dict('records')
            podium_items = [
                {"p": top3[1], "m": "40px"}, # Rank 2
                {"p": top3[0], "m": "0px"},  # Rank 1
                {"p": top3[2], "m": "40px"}  # Rank 3
            ]
            
            # Podium HTML
            cols_html = "".join([f"""
                <div style="flex: 1; margin-top: {i["m"]}; min-width: 0; display: flex; flex-direction: column;">
                    <div style="flex-grow: 1; text-align: center; padding: 10px 2px; background: rgba(255,255,255,0.05); border-radius: 12px; border: 1px solid rgba(255,215,0,0.3); box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
                        <div style="font-size: 1.2em; margin-bottom: 5px; color: #FFD700; font-weight: bold;">{i["p"]["Rank"]}</div>
                        <div style="display: flex; justify-content: center; margin-bottom: 5px;">
                            <img src="{i["p"]["Profile"] or "https://via.placeholder.com/100?text=Player"}" style="width: clamp(50px, 20vw, 80px); height: clamp(50px, 20vw, 80px); border-radius: 15px; object-fit: cover; border: 2px solid #fff500; box-shadow: 0 0 15px rgba(255,245,0,0.6);">
                        </div>
                        <div style="margin: 5px 0; color: #fff500; font-size: 0.9em; font-weight: bold; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; padding: 0 2px;">{i["p"]["Player"]}</div>
                        <div style="color: white; font-weight: bold; font-size: 0.8em;">{i["p"]["Points"]} pts</div>
                        <div style="color: #aaa; font-size: 0.7em;">{i["p"]["Win %"]}% Win</div>
                    </div>
                </div>""" for i in podium_items])
            
            st.markdown(f'<div style="display: flex; flex-direction: row; flex-wrap: nowrap; justify-content: center; align-items: flex-start; gap: 8px; margin-bottom: 25px; width: 100%;">{cols_html}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- B. Detailed Player Cards ---
        # Note: We iterate over display_rank_df (the filtered list)
        
        
       
       
       
        
        # --- B. Detailed Player Cards ---
        for idx, row in display_rank_df.iterrows():
            with st.container(border=True):
                # Data Prep
                profile_pic = row['Profile'] if row['Profile'] else 'https://via.placeholder.com/100'
                trend = row.get('Recent Trend', '')
                badges_html = ' '.join([f'<span title="{b}" style="font-size:16px; margin-left: 5px;">{b.split()[0]}</span>' for b in row.get('Badges', [])])
                
                # Header remains the same
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 5px;">
                    <div style="display: flex; align-items: center;">
                        <img src="{profile_pic}" 
                             style="width: 110px; height: 110px; border-radius: 12px; margin-right: 15px; 
                                    object-fit: contain; background: transparent; border: 3px solid #CCFF00; 
                                    box-shadow: 0 0 15px rgba(204, 255, 0, 0.5);">
                        <div>
                            <div style="font-size: 22px; font-weight: bold; color: white; line-height: 1.1;">{row['Player']}</div>
                            <div style="font-size: 13px; color: #00ff88; margin-top: 5px; font-weight: 500;">{trend}</div>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="background: #CCFF00; color: #041136; font-weight: bold; border-radius: 6px; padding: 4px 10px; font-size: 16px; display: inline-block;">
                            #{row['Rank']}
                        </div>
                        <div style="color: #ccc; font-size: 13px; margin-top: 6px; font-weight: bold;">{row['Points']} PTS</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # --- NEW RATIO: 1.8 for chart, 1 for stats ---
                col_chart, col_stats = st.columns([1.8, 1])
                
                with col_chart:
                    fig = create_radar_chart(row)
                    # This removes the "dead air" around the graph to make it appear much larger
                    fig.update_layout(
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=250, # Fixed height to keep card size stable
                        autosize=True
                    )
                    st.plotly_chart(fig, config={'displayModeBar': False}, width="stretch", key=f"radar_{row['Player']}_{idx}")
                    
                with col_stats:
                    # Stats column slightly tightened to make room for the graph
                    stats_html = f"""
                        <div style="text-align: right; padding-right: 2px; margin-top: 10px;">
                            <div style="margin-bottom: 8px;">
                                <div style="font-size: 9px; color: #888; letter-spacing: 1px;">WIN RATE</div>
                                <div style="font-size: 22px; font-weight: bold; color: #CCFF00;">{row['Win %']}%</div>
                            </div>
                            <div style="display: flex; justify-content: flex-end; gap: 10px; margin-bottom: 8px;">
                                <div>
                                    <div style="font-size: 8px; color: #888;">MATCHES</div>
                                    <div style="font-size: 14px; font-weight: bold; color: #eee;">{row['Matches']}</div>
                                </div>
                                <div>
                                    <div style="font-size: 8px; color: #888;">W/L</div>
                                    <div style="font-size: 14px; font-weight: bold; color: #eee;">{row['Wins']}/{row['Losses']}</div>
                                </div>
                            </div>
                            <div style="margin-bottom: 8px;">
                                <div style="font-size: 9px; color: #888; letter-spacing: 1px;">AVG GDA</div>
                                <div style="font-size: 16px; font-weight: bold; color: #eee;">{row['Game Diff Avg']}</div>
                            </div>
                            <div style="display: flex; justify-content: flex-end; gap: 8px; margin-bottom: 8px;">
                                <div>
                                    <div style="font-size: 8px; color: #888;">CLUTCH</div>
                                    <div style="font-size: 12px; font-weight: bold; color: #00ff88;">{row['Clutch Factor']}%</div>
                                </div>
                                <div>
                                    <div style="font-size: 8px; color: #888;">CONSISTENCY</div>
                                    <div style="font-size: 12px; font-weight: bold; color: #ff4b4b;">{row['Consistency Index']}</div>
                                </div>
                            </div>
                            <div style="margin-top: 5px;">{badges_html}</div>
                        </div>
                    """
                    st.markdown(stats_html, unsafe_allow_html=True)










# --- Tab 2: Matches ---# --- Tab 2: Matches ---
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

    # 2. Gender Mapping for Detection
    gender_map = dict(zip(st.session_state.players_df['name'], st.session_state.players_df['gender']))
    
    def auto_detect_category(row):
        m_type = getattr(row, 'match_type', 'Doubles')
        if m_type == "Singles": return "Singles"
        
        # Access players safely (works for NamedTuple and Series)
        p1 = str(getattr(row, 'team1_player1', ''))
        p2 = str(getattr(row, 'team1_player2', ''))
        p3 = str(getattr(row, 'team2_player1', ''))
        p4 = str(getattr(row, 'team2_player2', ''))
        
        p_list = [p.upper() for p in [p1, p2, p3, p4]]
        # Check for Visitors or empty slots
        if "VISITOR" in p_list or "" in p_list or "NONE" in p_list or "NAN" in p_list:
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
                
        with st.expander("➕ Post New Match Result", expanded=False, icon="➡️"):
            # 1. Setup Player Lists
            if "players_df" not in st.session_state or st.session_state.players_df.empty:
                st.warning("No players available. Please add players in the Player Profile tab.")
                st.stop()
            
            permanent_names = sorted([p for p in st.session_state.players_df["name"].dropna().tolist() if p != "Visitor"])
            
            # 2. Match Type Selection
            m_type = st.radio("Match Type", ["Doubles", "Singles"], index=0, horizontal=True)
            m_date = st.date_input("Match Date *", datetime.now())
        
            # 3. Dynamic Player Selection Layout
            col1, col2 = st.columns(2)
            if m_type == "Doubles":
                # Include Visitor for Doubles
                doubles_options = [""] + permanent_names + ["Visitor"]
                with col1:
                    st.markdown("**Team 1**")
                    t1p1 = st.selectbox("Player 1 *", doubles_options, key="d_t1p1")
                    t1p2 = st.selectbox("Player 2 *", doubles_options, key="d_t1p2")
                with col2:
                    st.markdown("**Team 2**")
                    t2p1 = st.selectbox("Player 1 *", doubles_options, key="d_t2p1")
                    t2p2 = st.selectbox("Player 2 *", doubles_options, key="d_t2p2")
            else:
                # No Visitor for Singles
                singles_options = [""] + permanent_names
                with col1:
                    st.markdown("**Player 1 (Team 1)**")
                    t1p1 = st.selectbox("Select Name *", singles_options, key="s_t1p1")
                    t1p2 = ""
                with col2:
                    st.markdown("**Player 2 (Team 2)**")
                    t2p1 = st.selectbox("Select Name *", singles_options, key="s_t2p1")
                    t2p2 = ""
        
            st.markdown("---")
            
            # 4. Scores and Winner Selection
            sc1, sc2, sc3 = st.columns(3)
            s1 = sc1.selectbox("Set 1 Score *", [""] + tennis_scores(), key="match_s1")
            s2 = sc2.selectbox("Set 2 Score", [""] + tennis_scores(), key="match_s2")
            s3 = sc3.selectbox("Set 3 Score", [""] + tennis_scores(), key="match_s3")
            
            winner_selection = st.radio("Select Winner *", ["Team 1", "Team 2", "Tie"], horizontal=True)
            match_img = st.file_uploader("Upload Match Photo *", type=["jpg", "jpeg", "png"])
        
            # 5. Form Submission & Validation Logic
            if st.button("🚀 Post Match Result"):
                valid = True
                error_msg = ""
                
                # --- A. Basic Field Validation ---
                selected_players = [p for p in [t1p1, t1p2, t2p1, t2p2] if p != ""]
                visitor_count = sum(1 for p in selected_players if p == "Visitor")
                
                if not s1:
                    st.error("Set 1 score is required.")
                    valid = False
                elif not match_img:
                    st.error("A match photo is required.")
                    valid = False
                elif m_type == "Doubles" and (len(selected_players) < 4 or not s2):
                    st.error("Doubles requires 4 players and at least 2 sets.")
                    valid = False
                elif m_type == "Doubles" and visitor_count > 1:
                    st.error("Invalid: Only ONE Visitor allowed in Doubles.")
                    valid = False
                elif m_type == "Singles" and len(selected_players) < 2:
                    st.error("Singles requires 2 players.")
                    valid = False
                
                # --- B. Winner vs Score Cross-Check Logic ---
                if valid:
                    t1_sets = 0
                    t2_sets = 0
                    active_sets = [s for s in [s1, s2, s3] if s]
                    
                    for score_str in active_sets:
                        try:
                            # Handle "Tie Break 10-8" or "6-4"
                            nums = [int(n) for n in re.findall(r'\d+', score_str)]
                            if len(nums) >= 2:
                                if nums[0] > nums[1]: t1_sets += 1
                                elif nums[1] > nums[0]: t2_sets += 1
                        except:
                            continue
                    
                    # Determine Mathematical Winner
                    math_winner = "Tie"
                    if t1_sets > t2_sets: math_winner = "Team 1"
                    elif t2_sets > t1_sets: math_winner = "Team 2"
                    
                    # --- C. Final Cross-Check Flag ---
                    if winner_selection != math_winner:
                        valid = False
                        if math_winner == "Tie":
                            st.error(f"❌ Score Mismatch: Sets are split ({t1_sets}-{t2_sets}). You must select 'Tie' as the winner.")
                        else:
                            st.error(f"❌ Score Mismatch: Based on the scores, {math_winner} won {max(t1_sets, t2_sets)} sets. Please correct the winner selection or the scores.")
        
                # --- D. Final Execution ---
                if valid:
                    with st.spinner("Uploading and saving..."):
                        mid = generate_match_id(st.session_state.matches_df, datetime.combine(m_date, datetime.min.time()))
                        img_url = upload_image_to_github(match_img, mid, "match")
                        
                        new_match = {
                            "match_id": mid,
                            "date": m_date.strftime('%Y-%m-%d'),
                            "match_type": m_type,
                            "team1_player1": t1p1, "team1_player2": t1p2,
                            "team2_player1": t2p1, "team2_player2": t2p2,
                            "set1": s1, "set2": s2 if s2 else "", "set3": s3 if s3 else "",
                            "winner": winner_selection, "match_image_url": img_url
                        }
                        
                        st.session_state.matches_df = pd.concat([st.session_state.matches_df, pd.DataFrame([new_match])], ignore_index=True)
                        save_matches(st.session_state.matches_df)
                        st.success("Match verified and saved!")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()     


        # B. EDIT MATCH FORM (Fixed AttributeError & Timestamp error)
        with st.expander("✏️ Edit Match Result", expanded=False, icon="➡️"):
            if not st.session_state.matches_df.empty:
                m_df = st.session_state.matches_df.copy()
                m_df['date'] = pd.to_datetime(m_df['date'])
                m_df = m_df.sort_values('date', ascending=False)
                
                # FIXED: Corrected r vs row and added str conversion for date
                match_options = {
                    f"{str(r.date)[:10]} | {r.team1_player1} vs {r.team2_player1}": r.match_id 
                    for r in m_df.itertuples()
                }
                
                sel_label = st.selectbox("Select Match to Edit", list(match_options.keys()))
                
                if sel_label:
                    mid_edit = match_options[sel_label]
                    edit_data = st.session_state.matches_df[st.session_state.matches_df['match_id'] == mid_edit].iloc[0]
                    
                    with st.form("edit_match_form"):
                        em_type = st.radio("Type", ["Singles", "Doubles"], index=0 if edit_data['match_type']=="Singles" else 1)
                        ec1, ec2 = st.columns(2)
                        
                        def get_idx(val, opt): 
                            try: return opt.index(val) 
                            except: return 0

                        if em_type == "Doubles":
                            et1p1 = ec1.selectbox("T1 P1", [""]+names, index=get_idx(edit_data['team1_player1'], [""]+names))
                            et1p2 = ec1.selectbox("T1 P2", ["", "Visitor"]+names, index=get_idx(edit_data['team1_player2'], ["", "Visitor"]+names))
                            et2p1 = ec2.selectbox("T2 P1", [""]+names, index=get_idx(edit_data['team2_player1'], [""]+names))
                            et2p2 = ec2.selectbox("T2 P2", ["", "Visitor"]+names, index=get_idx(edit_data['team2_player2'], ["", "Visitor"]+names))
                        else:
                            et1p1 = ec1.selectbox("P1", [""]+names, index=get_idx(edit_data['team1_player1'], [""]+names))
                            et2p1 = ec2.selectbox("P2", [""]+names, index=get_idx(edit_data['team2_player1'], [""]+names))
                            et1p2, et2p2 = None, None
                        
                        edate = st.date_input("Date", value=pd.to_datetime(edit_data['date']))
                        es1 = st.selectbox("Set 1", [""]+tennis_scores(), index=get_idx(edit_data['set1'], [""]+tennis_scores()))
                        es2 = st.selectbox("Set 2", [""]+tennis_scores(), index=get_idx(edit_data['set2'], [""]+tennis_scores()))
                        es3 = st.selectbox("Set 3", [""]+tennis_scores(), index=get_idx(edit_data['set3'], [""]+tennis_scores()))
                        ewinner = st.selectbox("Winner", ["Team 1", "Team 2", "Tie"], index=get_idx(edit_data['winner'], ["Team 1", "Team 2", "Tie"]))
                        
                        if st.form_submit_button("Update"):
                            upd = {
                                "match_id": mid_edit, "date": edate.isoformat(), "match_type": em_type,
                                "team1_player1": et1p1, "team1_player2": et1p2, "team2_player1": et2p1, "team2_player2": et2p2,
                                "set1": es1, "set2": es2, "set3": es3, "winner": ewinner, "match_image_url": edit_data['match_image_url']
                            }
                            st.session_state.matches_df = st.session_state.matches_df[st.session_state.matches_df['match_id'] != mid_edit]
                            st.session_state.matches_df = pd.concat([st.session_state.matches_df, pd.DataFrame([upd])], ignore_index=True)
                            save_matches(st.session_state.matches_df); st.success("Updated!"); st.rerun()


    # --- Match History ---
    st.subheader("Match Records")
    m_hist = st.session_state.matches_df.copy()

    if not m_hist.empty:
        # 1. ADD PLAYER FILTER
        filter_names = sorted(st.session_state.players_df['name'].unique())
        selected_player = st.selectbox("Filter by Player", ["All Players"] + filter_names)

        m_hist['date'] = pd.to_datetime(m_hist['date'])
        m_hist = m_hist.sort_values('date', ascending=False)

        # 2. APPLY FILTER LOGIC
        if selected_player != "All Players":
            m_hist = m_hist[
                (m_hist['team1_player1'] == selected_player) |
                (m_hist['team1_player2'] == selected_player) |
                (m_hist['team2_player1'] == selected_player) |
                (m_hist['team2_player2'] == selected_player)
            ]

        if m_hist.empty:
            st.info(f"No matches found for {selected_player}.")
        else:
            for row in m_hist.itertuples():
                display_type = auto_detect_category(row)
                t1_total, t2_total, sets_count = 0, 0, 0
                display_scores = []
                
                # Score parsing logic
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
    












# --- Tab 3: Player Profiles ---
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
        if pd.isna(val) or not str(val).strip() or str(val) == "None":
            return None  # Change from pd.NaT to None
        s = str(val).strip()
        if len(s) <= 5:
            s += "-2024"
        try:
            return pd.to_datetime(s, dayfirst=True, errors='coerce').to_pydatetime().date()
        except:
            return None

    # --- Manage Profiles ---
    #with st.expander("⚙️ Manage Player Profiles", expanded=False, icon="➡️"):
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
                        <div style="
                            width: 120px; 
                            height: 120px; 
                            background-color: #262626; 
                            border-radius: 15px; 
                            border: 3px solid #fff500; 
                            display: flex; 
                            justify-content: center; 
                            align-items: center; 
                            overflow: hidden; 
                            margin: 0 auto;
                        ">
                            <img src="{img}" style="
                                max-width: 100%; 
                                max-height: 100%; 
                                object-fit: contain;
                            ">
                        </div>
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
                            if fig: st.plotly_chart(fig, width="stretch", key=f"p_{idx}")
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

    available_players = sorted(st.session_state.players_df['name'].tolist()) if not st.session_state.players_df.empty else []
    

    # --- MATCH UP EXPANDER ---
    with st.expander("Match up", expanded=False, icon="➡️"):
        
        available_players = sorted(st.session_state.players_df['name'].tolist()) if not st.session_state.players_df.empty else []
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
    st.header("Mini Tournaments Organiser & Highlights")


    # --- 1. Fetch Photos Dynamically from GitHub ---
    @st.cache_data(ttl=3600)
    def get_tournament_photos():
        owner = "mahadevbk"
        repo = "mmd"
        path = "assets/minitourney"
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        
        try:
            # Note: Ensure 'requests' is imported at the top of your file
            response = requests.get(api_url)
            if response.status_code == 200:
                files = response.json()
                image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.gif')
                photo_urls = [
                    file['download_url'] for file in files 
                    if file['name'].lower().endswith(image_extensions)
                ]
                return photo_urls
        except Exception as e:
            st.error(f"Error fetching photos: {e}")
        return []

    photos = get_tournament_photos()

    # --- 2. Carousel / Slideshow UI ---
    if photos:
        if 'carousel_index' not in st.session_state:
            st.session_state.carousel_index = 0

        # We wrap the carousel in a specific div class "tourney-container"
        st.markdown('<div class="tourney-container">', unsafe_allow_html=True)
        
        # Navigation Buttons
        col_prev, col_mid, col_next = st.columns([1, 4, 1])
        
        with col_prev:
            st.write("###") # Vertical alignment
            if st.button("⬅️ Prev", key="prev_btn"):
                st.session_state.carousel_index = (st.session_state.carousel_index - 1) % len(photos)

        with col_next:
            st.write("###") # Vertical alignment
            if st.button("Next ➡️", key="next_btn"):
                st.session_state.carousel_index = (st.session_state.carousel_index + 1) % len(photos)

        with col_mid:
            current_img = photos[st.session_state.carousel_index]
            # Use a container to target THIS image only
            tourney_img_placeholder = st.empty()
            tourney_img_placeholder.image(current_img, width="stretch")
            st.caption(f"📸 Tournament Highlight {st.session_state.carousel_index + 1} of {len(photos)}")

        st.markdown('</div>', unsafe_allow_html=True)

        # --- 3. Scoped CSS (Targets ONLY this tab's container) ---
        st.markdown(
            f"""
            <style>
                /* Targets only images inside the tourney-container */
                .tourney-container [data-testid="stImage"] img {{
                    height: 800px !important;
                    width: auto !important;
                    object-fit: contain !important;
                    margin-left: auto;
                    margin-right: auto;
                    display: block;
                    border-radius: 10px;
                }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("No photos found in the tournament gallery.")

    st.markdown("---")
    st.info("Tournament Organiser is moved to https://tournament-organiser.streamlit.app/")
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


st.markdown("----")
st.info("Built with ❤️ using [Streamlit](https://streamlit.io/) — free and open source. [Other Scripts by dev](https://devs-scripts.streamlit.app/) on Streamlit.")
