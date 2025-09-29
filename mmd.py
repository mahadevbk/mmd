# dev's scratch pad
# tab[0] 2378
#tab [1] 2878
#tab [2] 3451
#tab [3] 3650
#tab [4] 3731
#tab [5] 4282
#  
# 
#
#
#
#
import streamlit as st
import pandas as pd
import uuid
from datetime import datetime, timedelta
from collections import defaultdict
from supabase import create_client, Client
import re
import urllib.parse
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io  # Added to fix 'name io is not defined' error
from itertools import combinations
from dateutil import parser
import plotly.graph_objects as go # Added for the new chart
import plotly.express as px # added for animated chart
import random
from fpdf import FPDF
import zipfile
import io
from datetime import datetime
import urllib.parse
import requests
import random
import numpy as np
import uuid
import base64
import time
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"


# Set the page title
st.set_page_config(page_title="MMD Mira Mixed Doubles Tennis League")

# Custom CSS for a scenic background
st.markdown("""
<style>
.stApp {
  background: linear-gradient(to bottom, #041136, #21000a);
  background-attachment: scroll;
}

/* Styles for printing */
@media print {
  /* This forces browsers to print background colors and images */
  html, body {
    -webkit-print-color-adjust: exact !important;
    print-color-adjust: exact !important;
  }
  
  /* Ensure the body takes up the full page */
  body {
    background: linear-gradient(to bottom, #21000a, #041136) !important;
    height: 100vh;
    margin: 0;
    padding: 0;
  }
  
  /* Hide the Streamlit header and toolbar when printing */
  header, .stToolbar {
    display: none;
  }
}

[data-testid="stHeader"] {
  background: linear-gradient(to top, #041136 , #21000a) !important;
}

.profile-image {
    width: 100px;
    height: 100px;
    object-fit: cover;
    border: 1px solid #fff500;
    border-radius: 10%;
    margin-right: 10px;
    vertical-align: middle;
    transition: transform 0.2s;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4), 0 0 10px rgba(255, 245, 0, 0.6);
}
.profile-image:hover {
    transform: scale(1.1);
}

/* Birthday Banner Styling */
.birthday-banner {
    background: linear-gradient(45deg, #FFFF00, #EEE8AA);
    color: #950606;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 1.2em;
    font-weight: bold;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    display: flex;
    justify-content: center;
    align-items: center;
}
.whatsapp-share img {
    width: 24px;
    vertical-align: middle;
    margin-right: 5px;
}
.whatsapp-share {
    background-color: #25D366;
    color: white !important;
    padding: 5px 10px;
    border-radius: 5px;
    text-decoration: none;
    font-weight: bold;
    margin-left: 15px;
    display: inline-flex;
    align-items: center;
    font-size: 0.8em;
    border: none;
}
.whatsapp-share:hover {
    opacity: 0.9;
}

/* Card styling for court locations */
.court-card {
    background: linear-gradient(to bottom, #031827, #07314f);
    border: 1px solid #fff500;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s, box-shadow 0.2s;
    text-align: center;
}
.court-card:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 12px rgba(255, 245, 0, 0.3);
}
.court-card h4 {
    color: #fff500;
    margin-bottom: 10px;
}
.court-card a {
    background-color: #fff500;
    color: #031827;
    padding: 8px 16px;
    border-radius: 5px;
    text-decoration: none;
    font-weight: bold;
    display: inline-block;
    margin-top: 10px;
    transition: background-color 0.2s;
}
.court-card a:hover {
    background-color: #ffd700;
}
.court-icon {
    width: 50px;
    height: 50px;
    margin-bottom: 10px;
}

@import url('https://fonts.googleapis.com/css2?family=Offside&display=swap');
html, body, [class*="st-"], h1, h2, h3, h4, h5, h6 {
    font-family: 'Offside', sans-serif !important;
}

/* ✅ Header & subheader resize to ~125% of tab font size (14px → 17–18px) */
h1 {
    font-size: 24px !important;
}
h2 {
    font-size: 22px !important;
}
h3 {
    font-size: 16px !important;
}

/* Rankings table container */
.rankings-table-container {
    width: 100%;
    background: #ffffff;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-top: 0px !important;
    padding: 10px;
}
.rankings-table-scroll {
    max-height: 500px;
    overflow-y: auto;
}

.ranking-header-row {
    display: none;
}
.ranking-row {
    display: block;
    padding: 10px;
    margin-bottom: 10px;
    border: 1px solid #696969;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    background-color: rgba(255, 255, 255, 0.05);
    overflow: visible;
}
.ranking-row:last-child {
    margin-bottom: 0;
}

.rank-col, .profile-col, .player-col, .points-col, .win-percent-col, .matches-col, .wins-col, .losses-col, .games-won-col, .game-diff-avg-col, .cumulative-game-diff-col, .trend-col, .birthday-col, .partners-col, .best-partner-col {
    width: 100%;
    text-align: left;
    padding: 2px 0;
    font-size: 1em;
    margin-bottom: 5px;
    word-break: break-word;
}
.rank-col {
    display: inline-block;
    white-space: nowrap;
    font-size: 1.3em;
    font-weight: bold;
    margin-right: 5px;
    color: #fff500;
}
.profile-col {
    text-align: left;
    margin-bottom: 10px;
    display: inline-block;
    vertical-align: middle;
}
.player-col {
    font-size: 1.3em;
    font-weight: bold;
    display: inline-block;
    flex-grow: 1;
    vertical-align: middle;
}

.rank-profile-player-group {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
}
.rank-profile-player-group .rank-col {
    width: auto;
    margin-right: 10px;
}
.rank-profile-player-group .profile-col {
     width: auto;
     margin-right: 10px;
}

.points-col::before { content: "Points: "; font-weight: bold; color: #bbbbbb; }
.win-percent-col::before { content: "Win %: "; font-weight: bold; color: #bbbbbb; }
.matches-col::before { content: "Matches: "; font-weight: bold; color: #bbbbbb; }
.wins-col::before { content: "Wins: "; font-weight: bold; color: #bbbbbb; }
.losses-col::before { content: "Losses: "; font-weight: bold; color: #bbbbbb; }
.games-won-col::before { content: "Games Won: "; font-weight: bold; color: #bbbbbb; }
.game-diff-avg-col::before { content: "Game Diff Avg: "; font-weight: bold; color: #bbbbbb; }
.cumulative-game-diff-col::before { content: "Cumulative Game Diff.: "; font-weight: bold; color: #bbbbbb; }
.trend-col::before { content: "Recent Trend: "; font-weight: bold; color: #bbbbbb; }
.birthday-col::before { content: "Birthday: "; font-weight: bold; color: #bbbbbb; }

.points-col, .win-percent-col, .matches-col, .wins-col, .losses-col, .games-won-col, .game-diff-avg-col, .cumulative-game-diff-col, .trend-col, .birthday-col, .partners-col, .best-partner-col {
    color: #fff500;
}

div.st-emotion-cache-1jm692n {
    margin-bottom: 0px !important;
    padding-bottom: 0px !important;
}
div.st-emotion-cache-1jm692n h3 {
    margin-bottom: 0px !important;
    padding-bottom: 0px !important;
    line-height: 1 !important;
}

.rankings-table-container > div {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
.rankings-table-container > .rankings-table-scroll {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

.stTabs [data-baseweb="tab-list"] {
    flex-wrap: nowrap;
    overflow-x: auto;
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    flex: 1 0 auto;
    padding: 10px 0;
    font-size: 14px;
    text-align: center;
    margin: 2px;
}
/* Style the value inside the st.metric component */
[data-testid="stMetric"] > div:nth-of-type(1) {
    color: #FF7518 !important; /* Optic Yellow #fff500 */
}
/* Prevent columns from stacking vertically on small screens */
.block-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
}

[data-testid="stHorizontalBlock"] {
    flex: 1 1 45% !important; /* Each column ~45% width */
    min-width: 400px;          /* Prevent too narrow columns */
    max-width: 700px;          /* Keep nice size */
    margin: 10px;
}
.calendar-share {
    background-color: #4285F4; /* Blue for calendar */
    color: white !important;
    padding: 5px 10px;
    border-radius: 5px;
    text-decoration: none;
    font-weight: bold;
    margin-left: 15px;
    display: inline-flex;
    align-items: center;
    font-size: 0.8em;
    border: none;
}
.calendar-share:hover {
    opacity: 0.9;
}

[data-testid="stExpander"] i,
[data-testid="stExpander"] span.icon {
    font-family: 'Material Icons' !important;
}
</style>



""", unsafe_allow_html=True)

# Supabase setup
supabase_url = st.secrets["supabase"]["supabase_url"]
supabase_key = st.secrets["supabase"]["supabase_key"]
supabase: Client = create_client(supabase_url, supabase_key)

# Table names
players_table_name = "players"
matches_table_name = "matches"
bookings_table_name = "bookings"
hall_of_fame_table_name="hall_of_fame"

# --- Session state initialization ---
if 'players_df' not in st.session_state:
    st.session_state.players_df = pd.DataFrame(columns=["name", "profile_image_url", "birthday"])
if 'matches_df' not in st.session_state:
    st.session_state.matches_df = pd.DataFrame(columns=["match_id", "date", "match_type", "team1_player1", "team1_player2", "team2_player1", "team2_player2", "set1", "set2", "set3", "winner", "match_image_url"])
if 'form_key_suffix' not in st.session_state:
    st.session_state.form_key_suffix = 0

if 'bookings_df' not in st.session_state:
    st.session_state.bookings_df = pd.DataFrame(columns=["booking_id", "date", "time", "match_type", "court_name", "player1", "player2", "player3", "player4", "screenshot_url"])

if 'last_match_submit_time' not in st.session_state:
    st.session_state.last_match_submit_time = 0
  
if 'image_urls' not in st.session_state:
    st.session_state.image_urls = {}  



# --- Functions ---
def load_players():
    try:
        response = supabase.table(players_table_name).select("name, profile_image_url, birthday, gender").execute()
        df = pd.DataFrame(response.data)
        expected_columns = ["name", "profile_image_url", "birthday", "gender"]
        for col in expected_columns:
            if col not in df.columns:
                df[col] = ""  # Default to empty string for missing columns
        st.session_state.players_df = df
    except Exception as e:
        st.error(f"Error loading players: {str(e)}")
        st.session_state.players_df = pd.DataFrame(columns=["name", "profile_image_url", "birthday", "gender"])





def save_players(players_df):
    try:
        expected_columns = ["name", "profile_image_url", "birthday", "gender"]
        players_df_to_save = players_df[expected_columns].copy()
        
        # Replace NaN with None for JSON compliance before saving
        players_df_to_save = players_df_to_save.where(pd.notna(players_df_to_save), None)
        
        # Remove duplicates based on 'name', keeping the last entry
        players_df_to_save = players_df_to_save.drop_duplicates(subset=['name'], keep='last')
        
        supabase.table(players_table_name).upsert(players_df_to_save.to_dict("records")).execute()
    except Exception as e:
        st.error(f"Error saving players: {str(e)}")




      
def delete_player_from_db(player_name):
    try:
        supabase.table(players_table_name).delete().eq("name", player_name).execute()
    except Exception as e:
        st.error(f"Error deleting player from database: {str(e)}")

def generate_pdf_reportlab(rank_df_combined, rank_df_doubles, rank_df_singles):
    # Format the current date
    current_date = datetime.now().strftime("%d/%m/%Y")
    
    # Buffer to store PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), leftMargin=0.5*inch, rightMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name='Title',
        fontName='Helvetica-Bold',
        fontSize=24,
        alignment=1,  # Center
        spaceAfter=12
    )
    subtitle_style = ParagraphStyle(
        name='Subtitle',
        fontName='Helvetica-Bold',
        fontSize=14,
        alignment=1,  # Center
        spaceAfter=12
    )
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.yellow),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ])
    
    # Function to format DataFrame for table
    def df_to_table(df, ranking_type):
        if df.empty:
            return [Paragraph(f"{ranking_type} Rankings as of {current_date}", subtitle_style), Paragraph(f"No data available for {ranking_type.lower()} rankings.", styles['Normal'])]
        
        # Format the DataFrame
        display_df = df[["Rank", "Player", "Points", "Win %", "Matches", "Wins", "Losses", "Games Won", "Game Diff Avg", "Recent Trend"]].copy()
        display_df["Points"] = display_df["Points"].map("{:.1f}".format)
        display_df["Win %"] = display_df["Win %"].map("{:.1f}%".format)
        display_df["Game Diff Avg"] = display_df["Game Diff Avg"].map("{:.2f}".format)
        display_df["Matches"] = display_df["Matches"].astype(int)
        display_df["Wins"] = display_df["Wins"].astype(int)
        display_df["Losses"] = display_df["Losses"].astype(int)
        display_df["Games Won"] = display_df["Games Won"].astype(int)
        
        # Table data
        headers = ["Rank", "Player", "Points", "Win %", "Matches", "Wins", "Losses", "Games Won", "Game Diff Avg", "Recent Trend"]
        data = [headers] + display_df.values.tolist()
        
        # Create table
        col_widths = [0.6*inch, 1.5*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1*inch, 1.2*inch]
        table = Table(data, colWidths=col_widths, repeatRows=1)
        table.setStyle(table_style)
        
        return [Paragraph(f"{ranking_type} Rankings as of {current_date}", subtitle_style), table]
    
    # Add main heading
    elements.append(Paragraph("AR Tennis League", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Add tables
    elements.extend(df_to_table(rank_df_combined, "Combined"))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(PageBreak())
    
    elements.extend(df_to_table(rank_df_doubles, "Doubles"))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(PageBreak())
    
    elements.extend(df_to_table(rank_df_singles, "Singles"))
    
    # Build PDF
    doc.build(elements)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data
  

def load_matches():
    try:
        response = supabase.table(matches_table_name).select("*").execute()
        df = pd.DataFrame(response.data)
        expected_columns = ["match_id", "date", "match_type", "team1_player1", "team1_player2", 
                           "team2_player1", "team2_player2", "set1", "set2", "set3", "winner", 
                           "match_image_url"]
        for col in expected_columns:
            if col not in df.columns:
                df[col] = ""

        # Store raw date for display
        df['raw_date'] = df['date']

        # Convert dates, use far-past fallback for invalid/NaT
        df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce').dt.tz_localize(None)
        
        # Log invalid dates for debugging
        invalid_dates = df[df['date'].isna()]['raw_date'].unique()
        if len(invalid_dates) > 0:
            st.warning(f"Found {len(invalid_dates)} matches with invalid or missing dates: {invalid_dates.tolist()}. Using fallback date.")
        
        # Set fallback date (e.g., 1970-01-01) for NaT to keep records
        df['date'] = df['date'].fillna(pd.Timestamp('1970-01-01'))
        
        st.session_state.matches_df = df
    except Exception as e:
        st.error(f"Error loading matches: {str(e)}")
        st.session_state.matches_df = pd.DataFrame(columns=expected_columns + ["raw_date"])



def save_matches(matches_df):
    """
    Saves matches to Supabase after validating and standardizing date formats to 'YYYY-MM-DD HH:MM:SS'.
    Returns True if save is successful, False otherwise.
    """
    try:
        expected_columns = ["match_id", "date", "match_type", "team1_player1", "team1_player2", 
                           "team2_player1", "team2_player2", "set1", "set2", "set3", 
                           "winner", "match_image_url"]
        matches_df_to_save = matches_df[expected_columns].copy()
        
        # Replace NaN with None for JSON compliance
        matches_df_to_save = matches_df_to_save.where(pd.notna(matches_df_to_save), None)
        
        # Validate and standardize dates to 'YYYY-MM-DD HH:MM:SS'
        matches_df_to_save['date'] = matches_df_to_save['date'].apply(
            lambda d: pd.to_datetime(d, errors='coerce').strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(pd.to_datetime(d, errors='coerce')) else None
        )
        
        # Log and handle records with invalid/missing dates
        invalid_date_rows = matches_df_to_save[matches_df_to_save['date'].isna()]
        if not invalid_date_rows.empty:
            st.warning(f"Found {len(invalid_date_rows)} matches with invalid or missing dates. Setting to current date and time: {invalid_date_rows['match_id'].tolist()}")
            matches_df_to_save.loc[matches_df_to_save['date'].isna(), 'date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Ensure no null or empty match IDs
        matches_df_to_save = matches_df_to_save[matches_df_to_save["match_id"].notnull() & (matches_df_to_save["match_id"] != "")]
        
        # Remove duplicates based on 'match_id', keeping the last entry
        matches_df_to_save = matches_df_to_save.drop_duplicates(subset=['match_id'], keep='last')
        
        if matches_df_to_save.empty:
            st.warning("No valid matches to save (all match IDs were null or empty).")
            return False
        
        # Upsert to Supabase
        supabase.table(matches_table_name).upsert(matches_df_to_save.to_dict("records")).execute()
        st.success("Match saved successfully. Refreshing page...")
        return True
    except Exception as e:
        st.error(f"Error saving matches: {str(e)}")
        return False








def delete_match_from_db(match_id):
    try:
        supabase.table(matches_table_name).delete().eq("match_id", match_id).execute()
        # Remove the match from session state
        st.session_state.matches_df = st.session_state.matches_df[st.session_state.matches_df["match_id"] != match_id].reset_index(drop=True)
        save_matches(st.session_state.matches_df)  # Save to ensure consistency
    except Exception as e:
        st.error(f"Error deleting match from database: {str(e)}")






def upload_image_to_github(file, file_name, image_type="match"):
    """
    Uploads a file to a specified folder in a GitHub repository and returns its public URL.
    Handles both new uploads and updates by fetching sha if the file exists.
    """
    if not file:
        return ""

    # --- Load credentials from Streamlit secrets ---
    try:
        token = st.secrets["github"]["token"]
        repo_full_name = st.secrets["github"]["repo"]
        branch = st.secrets["github"]["branch"]
    except KeyError:
        st.error("GitHub credentials are not set in st.secrets. Please check your secrets.toml file.")
        return ""

    # --- Determine the path in the repository ---
    if image_type == "profile":
        path_in_repo = f"assets/profile_images/{file_name}.jpg"
    elif image_type == "match":
        path_in_repo = f"assets/match_images/{file_name}.jpg"
    elif image_type == "booking":
        path_in_repo = f"assets/bookings_images/{file_name}.jpg"
    else:
        path_in_repo = f"assets/others/{file_name}.jpg"

    # --- Prepare the API URL and headers ---
    api_url = f"https://api.github.com/repos/{repo_full_name}/contents/{path_in_repo}"

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # --- Check if the file exists and get sha (for updates) ---
    sha = None
    get_response = requests.get(api_url, headers=headers)
    if get_response.status_code == 200:
        sha = get_response.json().get('sha')
        st.info(f"File {path_in_repo} already exists. Updating with sha: {sha}")
    elif get_response.status_code == 404:
        st.info(f"File {path_in_repo} does not exist. Creating new file.")
    else:
        st.error(f"GitHub GET Error: Status {get_response.status_code}. Response: {get_response.text}")
        return ""

    # --- Prepare the file content ---
    content_bytes = file.getvalue()
    content_base64 = base64.b64encode(content_bytes).decode("utf-8")

    payload = {
        "message": f"feat: Upload {image_type} image {file_name}",
        "branch": branch,
        "content": content_base64,
    }
    if sha:
        payload["sha"] = sha  # Include sha only for updates

    # --- Make the API call to upload/update the file ---
    try:
        response = requests.put(api_url, headers=headers, json=payload)
        response.raise_for_status()  # Raises exception for bad status codes
        st.success(f"Image '{file_name}.jpg' uploaded/updated to GitHub successfully!")
        return f"https://raw.githubusercontent.com/{repo_full_name}/{branch}/{path_in_repo}"
    except requests.exceptions.HTTPError as e:
        st.error(f"GitHub API Error: Failed to upload/update image. Status Code: {e.response.status_code}")
        st.error(f"Response: {e.response.text}")
        return ""
    except Exception as e:
        st.error(f"An unexpected error occurred during image upload: {str(e)}")
        return ""




        
def tennis_scores():
    scores = ["6-0", "6-1", "6-2", "6-3", "6-4", "7-5", "7-6", "0-6", "1-6", "2-6", "3-6", "4-6", "5-7", "6-7"]
    
    # Add winning super tie-break scores (e.g., 10-0 to 10-9)
    for i in range(10):
        scores.append(f"Tie Break 7-{i}")
        
    # Add losing super tie-break scores (e.g., 0-10 to 9-10)
    for i in range(10):
        scores.append(f"Tie Break {i}-7")
        
    # Add winning standard tie-break scores (e.g., 7-0 to 7-5)
    for i in range(6): # Scores from 0 to 5
        scores.append(f"Tie Break 10-{i}")
        
    # Add losing standard tie-break scores (e.g., 0-7 to 5-7)
    for i in range(6): # Scores from 0 to 5
        scores.append(f"Tie Break {i}-10")
        
    return scores



def download_image(url):
    """Download image bytes from a public Supabase URL."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.content
    except Exception as e:
        st.error(f"Failed to download {url}: {e}")
    return None


def get_quarter(month):
    if 1 <= month <= 3:
        return "Q1"
    elif 4 <= month <= 6:
        return "Q2"
    elif 7 <= month <= 9:
        return "Q3"
    else:
        return "Q4"

def generate_match_id(matches_df, match_datetime):
    year = match_datetime.year
    quarter = get_quarter(match_datetime.month)
    if not matches_df.empty and 'date' in matches_df.columns:
        matches_df['date'] = pd.to_datetime(matches_df['date'], errors='coerce')
        filtered_matches = matches_df[
            (matches_df['date'].dt.year == year) &
            (matches_df['date'].apply(lambda d: get_quarter(d.month) == quarter))
        ]
        serial_number = len(filtered_matches) + 1
        new_id = f"MMD{quarter}{year}-{serial_number:02d}"
        # Ensure the ID is unique
        while new_id in matches_df['match_id'].values:
            serial_number += 1
            new_id = f"MMD{quarter}{year}-{serial_number:02d}"
    else:
        serial_number = 1
        new_id = f"MMD{quarter}{year}-{serial_number:02d}"
    return new_id




def get_player_trend(player, matches, max_matches=5):
    player_matches = matches[
        (matches['team1_player1'] == player) |
        (matches['team1_player2'] == player) |
        (matches['team2_player1'] == player) |
        (matches['team2_player2'] == player)
    ].copy()
    
    # Handle NaT by treating as oldest (or skip if preferred)
    player_matches['date'] = pd.to_datetime(player_matches['date'], errors='coerce')
    player_matches['sort_date'] = player_matches['date'].fillna(pd.Timestamp('1970-01-01'))
    player_matches = player_matches.sort_values(by='sort_date', ascending=False)
    
    trend = []
    for _, row in player_matches.head(max_matches).iterrows():
        if row['match_type'] == 'Doubles':
            team1 = [row['team1_player1'], row['team1_player2']]
            team2 = [row['team2_player1'], row['team2_player2']]
        else:
            team1 = [row['team1_player1']]
            team2 = [row['team2_player1']]
        if player in team1 and row['winner'] == 'Team 1':
            trend.append('W')
        elif player in team2 and row['winner'] == 'Team 2':
            trend.append('W')
        elif row['winner'] != 'Tie':
            trend.append('L')
    return ' '.join(trend) if trend else 'No recent matches'

def calculate_rankings(matches_to_rank):
    scores = defaultdict(float)
    wins = defaultdict(int)
    losses = defaultdict(int)
    matches_played = defaultdict(int)
    singles_matches = defaultdict(int)
    doubles_matches = defaultdict(int)
    games_won = defaultdict(int)
    cumulative_game_diff = defaultdict(int)
    partner_stats = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'ties': 0, 'matches': 0, 'game_diff_sum': 0}))
    clutch_wins = defaultdict(int)
    clutch_matches = defaultdict(int)
    game_diffs = defaultdict(list)

    players_df = st.session_state.players_df

    for idx, row in matches_to_rank.iterrows():
        match_type = row['match_type']
        if match_type == 'Doubles':
            t1 = [p for p in [row['team1_player1'], row['team1_player2']] if p and p != "Visitor"]
            t2 = [p for p in [row['team2_player1'], row['team2_player2']] if p and p != "Visitor"]
        else:
            t1 = [p for p in [row['team1_player1']] if p and p != "Visitor"]
            t2 = [p for p in [row['team2_player1']] if p and p != "Visitor"]

        match_gd_sum = 0
        is_clutch_match = False

        is_mixed_doubles = False
        if match_type == 'Doubles' and len(t1) == 2 and len(t2) == 2:
            try:
                t1_genders = [players_df[players_df['name'] == p]['gender'].iloc[0] if p in players_df['name'].values else None for p in t1]
                t2_genders = [players_df[players_df['name'] == p]['gender'].iloc[0] if p in players_df['name'].values else None for p in t2]
                if (None not in t1_genders and None not in t2_genders and 
                    sorted(t1_genders) == ['F', 'M'] and sorted(t2_genders) == ['F', 'M']):
                    is_mixed_doubles = True
            except KeyError as e:
                st.warning(f"Gender column missing for match {row.get('match_id', 'unknown')}: {str(e)}")

        for set_score in [row['set1'], row['set2'], row['set3']]:
            if not set_score or ('-' not in str(set_score) and 'Tie Break' not in str(set_score)):
                continue
            try:
                team1_games, team2_games = 0, 0
                is_tie_break = "Tie Break" in str(set_score)
                if is_tie_break:
                    is_clutch_match = True
                    tie_break_scores = [int(s) for s in re.findall(r'\d+', str(set_score))]
                    if len(tie_break_scores) != 2:
                        continue
                    team1_games, team2_games = tie_break_scores
                    team1_games = 7 if team1_games > team2_games else 6
                    team2_games = 6 if team1_games > team2_games else 7
                else:
                    team1_games, team2_games = map(int, str(set_score).split('-'))

                team1_set_diff = team1_games - team2_games
                team2_set_diff = team2_games - team1_games
                match_gd_sum += team1_set_diff

                for p in t1:
                    games_won[p] += team1_games
                    cumulative_game_diff[p] += team1_set_diff
                    game_diffs[p].append(team1_set_diff)
                for p in t2:
                    games_won[p] += team2_games
                    cumulative_game_diff[p] += team2_set_diff
                    game_diffs[p].append(team2_set_diff)
            except (ValueError, TypeError) as e:
                st.warning(f"Skipping invalid set score {set_score} in match {row.get('match_id', 'unknown')}: {str(e)}")
                continue

        if row['set3'] and str(row['set3']).strip():
            is_clutch_match = True

        if row["winner"] == "Team 1":
            for p in t1:
                scores[p] += 3 if is_mixed_doubles else 2
                wins[p] += 1
                matches_played[p] += 1
                if is_clutch_match:
                    clutch_wins[p] += 1
                    clutch_matches[p] += 1
                if match_type == 'Doubles':
                    doubles_matches[p] += 1
                else:
                    singles_matches[p] += 1
            for p in t2:
                scores[p] += 1
                losses[p] += 1
                matches_played[p] += 1
                if is_clutch_match:
                    clutch_matches[p] += 1
                if match_type == 'Doubles':
                    doubles_matches[p] += 1
                else:
                    singles_matches[p] += 1
        elif row["winner"] == "Team 2":
            for p in t2:
                scores[p] += 3 if is_mixed_doubles else 2
                wins[p] += 1
                matches_played[p] += 1
                if is_clutch_match:
                    clutch_wins[p] += 1
                    clutch_matches[p] += 1
                if match_type == 'Doubles':
                    doubles_matches[p] += 1
                else:
                    singles_matches[p] += 1
            for p in t1:
                scores[p] += 1
                losses[p] += 1
                matches_played[p] += 1
                if is_clutch_match:
                    clutch_matches[p] += 1
                if match_type == 'Doubles':
                    doubles_matches[p] += 1
                else:
                    singles_matches[p] += 1
        elif row["winner"] == "Tie":
            for p in t1 + t2:
                scores[p] += 1.5
                matches_played[p] += 1
                if is_clutch_match:
                    clutch_matches[p] += 1
                if match_type == 'Doubles':
                    doubles_matches[p] += 1
                else:
                    singles_matches[p] += 1

        if match_type == 'Doubles':
            for p1 in t1:
                for p2 in t1:
                    if p1 != p2:
                        partner_stats[p1][p2]['matches'] += 1
                        partner_stats[p1][p2]['game_diff_sum'] += match_gd_sum
                        if row["winner"] == "Team 1":
                            partner_stats[p1][p2]['wins'] += 1
                        elif row["winner"] == "Team 2":
                            partner_stats[p1][p2]['losses'] += 1
                        else:
                            partner_stats[p1][p2]['ties'] += 1
            for p1 in t2:
                for p2 in t2:
                    if p1 != p2:
                        partner_stats[p1][p2]['matches'] += 1
                        partner_stats[p1][p2]['game_diff_sum'] += match_gd_sum if row["winner"] == "Team 2" else -match_gd_sum
                        if row["winner"] == "Team 2":
                            partner_stats[p1][p2]['wins'] += 1
                        elif row["winner"] == "Team 1":
                            partner_stats[p1][p2]['losses'] += 1
                        else:
                            partner_stats[p1][p2]['ties'] += 1

    rank_data = []
    for player in scores:
        if player == "Visitor":
            continue
        win_percentage = (wins[player] / matches_played[player] * 100) if matches_played[player] > 0 else 0
        game_diff_avg = cumulative_game_diff[player] / matches_played[player] if matches_played[player] > 0 else 0
        profile_image = players_df.loc[players_df["name"] == player, "profile_image_url"].iloc[0] if player in players_df["name"].values else ""
        player_trend = get_player_trend(player, matches_to_rank)

        clutch_factor = (clutch_wins[player] / clutch_matches[player] * 100) if clutch_matches[player] > 0 else 0
        consistency_index = np.std(game_diffs[player]) if game_diffs[player] else 0

        badges = []
        if clutch_factor > 70 and clutch_matches[player] >= 3:
            badges.append("🎯 Tie-break Monster")
        if wins[player] >= 5 and player_trend.startswith("W W W W W"):
            badges.append("🔥 Hot Streak")
        if consistency_index < 2 and matches_played[player] >= 5:
            badges.append("📈 Consistent Performer")
        if matches_played[player] == max(matches_played.values()):
            badges.append("💪 Ironman")

        player_matches = matches_to_rank[
            (matches_to_rank['team1_player1'] == player) |
            (matches_to_rank['team1_player2'] == player) |
            (matches_to_rank['team2_player1'] == player) |
            (matches_to_rank['team2_player2'] == player)
        ]
        comeback_wins = 0
        for _, r in player_matches.iterrows():
            sets = [r['set1'], r['set2'], r['set3']]
            valid_sets = [s for s in sets if s]
            if len(valid_sets) >= 2:
                try:
                    g1, g2 = map(int, valid_sets[0].split('-'))
                except:
                    continue
                first_set_winner = "Team 1" if g1 > g2 else "Team 2"
                if (player in [r['team1_player1'], r['team1_player2']] and first_set_winner == "Team 2" and r['winner'] == "Team 1") or \
                   (player in [r['team2_player1'], r['team2_player2']] and first_set_winner == "Team 1" and r['winner'] == "Team 2"):
                    comeback_wins += 1
        if comeback_wins >= 3:
            badges.append("🔄 Comeback Kid")

        recent_matches = player_matches.sort_values(by="date", ascending=False).head(10)
        recent_wins = 0
        for _, r in recent_matches.iterrows():
            if (player in [r['team1_player1'], r['team1_player2']] and r['winner'] == "Team 1") or \
               (player in [r['team2_player1'], r['team2_player2']] and r['winner'] == "Team 2"):
                recent_wins += 1
        recent_win_rate = recent_wins / len(recent_matches) * 100 if not recent_matches.empty else 0
        if recent_win_rate - win_percentage >= 20:
            badges.append("🚀 Most Improved")

        if games_won[player] == max(games_won.values()):
            badges.append("🥇 Game Hog")

        rank_data.append({
            "Rank": 0, "Profile": profile_image, "Player": player, "Points": scores[player],
            "Win %": round(win_percentage, 2), "Matches": matches_played[player],
            "Doubles Matches": doubles_matches[player], "Singles Matches": singles_matches[player],
            "Wins": wins[player], "Losses": losses[player], "Games Won": games_won[player],
            "Game Diff Avg": round(game_diff_avg, 2), "Cumulative Game Diff": cumulative_game_diff[player],
            "Recent Trend": player_trend,
            "Clutch Factor": round(clutch_factor, 1),
            "Consistency Index": round(consistency_index, 2),
            "Badges": badges
        })

    rank_df = pd.DataFrame(rank_data)
    if not rank_df.empty:
        rank_df = rank_df.sort_values(
            by=["Points", "Win %", "Game Diff Avg", "Games Won", "Player"],
            ascending=[False, False, False, False, True]
        ).reset_index(drop=True)
        rank_df["Rank"] = [f"🏆 {i}" for i in range(1, len(rank_df) + 1)]

    return rank_df, partner_stats







#------------------- Update the display_player_insights  and calculate rankings function --------------------------------




def calculate_rankings(matches_to_rank):
    scores = defaultdict(float)
    wins = defaultdict(int)
    losses = defaultdict(int)
    matches_played = defaultdict(int)
    singles_matches = defaultdict(int)
    doubles_matches = defaultdict(int)
    games_won = defaultdict(int)
    cumulative_game_diff = defaultdict(int)
    partner_stats = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'ties': 0, 'matches': 0, 'game_diff_sum': 0}))

    # --- NEW: clutch + consistency tracking ---
    clutch_wins = defaultdict(int)
    clutch_matches = defaultdict(int)
    game_diffs = defaultdict(list)

    # Get players dataframe for gender information
    players_df = st.session_state.players_df

    for idx, row in matches_to_rank.iterrows():
        match_type = row['match_type']

        if match_type == 'Doubles':
            t1 = [p for p in [row['team1_player1'], row['team1_player2']] if p and p != "Visitor"]
            t2 = [p for p in [row['team2_player1'], row['team2_player2']] if p and p != "Visitor"]
        else:
            t1 = [p for p in [row['team1_player1']] if p and p != "Visitor"]
            t2 = [p for p in [row['team2_player1']] if p and p != "Visitor"]

        match_gd_sum = 0
        is_clutch_match = False

        # Check if the match is a mixed doubles game
        is_mixed_doubles = False
        if match_type == 'Doubles' and len(t1) == 2 and len(t2) == 2:
            try:
                # Get genders for team1 and team2 players
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

                # Check if both teams have one 'M' and one 'F' and no None values
                if (None not in t1_genders and None not in t2_genders and 
                    sorted(t1_genders) == ['F', 'M'] and sorted(t2_genders) == ['F', 'M']):
                    is_mixed_doubles = True
            except KeyError as e:
                st.warning(f"Gender column missing or inaccessible for match {row.get('match_id', 'unknown')}: {str(e)}. Treating as non-mixed doubles.")
                is_mixed_doubles = False

        for set_score in [row['set1'], row['set2'], row['set3']]:
            if not set_score or ('-' not in str(set_score) and 'Tie Break' not in str(set_score)):
                continue

            try:
                team1_games, team2_games = 0, 0
                is_tie_break = "Tie Break" in str(set_score)

                if is_tie_break:
                    is_clutch_match = True
                    tie_break_scores = [int(s) for s in re.findall(r'\d+', str(set_score))]
                    if len(tie_break_scores) != 2:
                        continue
                    team1_games, team2_games = tie_break_scores
                    # Normalize tie-break games for consistency
                    team1_games = 7 if team1_games > team2_games else 6
                    team2_games = 6 if team1_games > team2_games else 7
                else:
                    team1_games, team2_games = map(int, str(set_score).split('-'))

                team1_set_diff = team1_games - team2_games
                team2_set_diff = team2_games - team1_games
                match_gd_sum += team1_set_diff

                for p in t1:
                    games_won[p] += team1_games
                    cumulative_game_diff[p] += team1_set_diff
                    game_diffs[p].append(team1_set_diff)
                for p in t2:
                    games_won[p] += team2_games
                    cumulative_game_diff[p] += team2_set_diff
                    game_diffs[p].append(team2_set_diff)

            except (ValueError, TypeError) as e:
                st.warning(f"Skipping invalid set score {set_score} in match {row.get('match_id', 'unknown')}: {str(e)}")
                continue

        if row['set3'] and str(row['set3']).strip():
            is_clutch_match = True

        # --- results ---
        if row["winner"] == "Team 1":
            for p in t1:
                # Award 3 points for winners in mixed doubles, 2 points otherwise
                scores[p] += 3 if is_mixed_doubles else 2
                wins[p] += 1
                matches_played[p] += 1
                if is_clutch_match:
                    clutch_wins[p] += 1
                    clutch_matches[p] += 1
                if match_type == 'Doubles':
                    doubles_matches[p] += 1
                else:
                    singles_matches[p] += 1
            for p in t2:
                # Losers always get 1 point
                scores[p] += 1
                losses[p] += 1
                matches_played[p] += 1
                if is_clutch_match:
                    clutch_matches[p] += 1
                if match_type == 'Doubles':
                    doubles_matches[p] += 1
                else:
                    singles_matches[p] += 1
        elif row["winner"] == "Team 2":
            for p in t2:
                # Award 3 points for winners in mixed doubles, 2 points otherwise
                scores[p] += 3 if is_mixed_doubles else 2
                wins[p] += 1
                matches_played[p] += 1
                if is_clutch_match:
                    clutch_wins[p] += 1
                    clutch_matches[p] += 1
                if match_type == 'Doubles':
                    doubles_matches[p] += 1
                else:
                    singles_matches[p] += 1
            for p in t1:
                # Losers always get 1 point
                scores[p] += 1
                losses[p] += 1
                matches_played[p] += 1
                if is_clutch_match:
                    clutch_matches[p] += 1
                if match_type == 'Doubles':
                    doubles_matches[p] += 1
                else:
                    singles_matches[p] += 1
        elif row["winner"] == "Tie":
            for p in t1 + t2:
                # All players get 1.5 points for a tie
                scores[p] += 1.5
                matches_played[p] += 1
                if is_clutch_match:
                    clutch_matches[p] += 1
                if match_type == 'Doubles':
                    doubles_matches[p] += 1
                else:
                    singles_matches[p] += 1

        # partner stats
        if match_type == 'Doubles':
            for p1 in t1:
                for p2 in t1:
                    if p1 != p2:
                        partner_stats[p1][p2]['matches'] += 1
                        partner_stats[p1][p2]['game_diff_sum'] += match_gd_sum
                        if row["winner"] == "Team 1":
                            partner_stats[p1][p2]['wins'] += 1
                        elif row["winner"] == "Team 2":
                            partner_stats[p1][p2]['losses'] += 1
                        else:
                            partner_stats[p1][p2]['ties'] += 1
            for p1 in t2:
                for p2 in t2:
                    if p1 != p2:
                        partner_stats[p1][p2]['matches'] += 1
                        partner_stats[p1][p2]['game_diff_sum'] += match_gd_sum if row["winner"] == "Team 2" else -match_gd_sum
                        if row["winner"] == "Team 2":
                            partner_stats[p1][p2]['wins'] += 1
                        elif row["winner"] == "Team 1":
                            partner_stats[p1][p2]['losses'] += 1
                        else:
                            partner_stats[p1][p2]['ties'] += 1

    # --- build rank dataframe ---
    rank_data = []
    for player in scores:
        if player == "Visitor":
            continue
        win_percentage = (wins[player] / matches_played[player] * 100) if matches_played[player] > 0 else 0
        game_diff_avg = cumulative_game_diff[player] / matches_played[player] if matches_played[player] > 0 else 0
        profile_image = players_df.loc[players_df["name"] == player, "profile_image_url"].iloc[0] if player in players_df["name"].values else ""
        player_trend = get_player_trend(player, matches_to_rank)

        clutch_factor = (clutch_wins[player] / clutch_matches[player] * 100) if clutch_matches[player] > 0 else 0
        consistency_index = np.std(game_diffs[player]) if game_diffs[player] else 0

        # --- BADGES ---
        badges = []
        if clutch_factor > 70 and clutch_matches[player] >= 3:
            badges.append("🎯 Tie-break Monster")
        if wins[player] >= 5 and player_trend.startswith("W W W W W"):
            badges.append("🔥 Hot Streak")
        if consistency_index < 2 and matches_played[player] >= 5:
            badges.append("📈 Consistent Performer")
        if matches_played[player] == max(matches_played.values()):
            badges.append("💪 Ironman")

        # NEW: Comeback Kid
        player_matches = matches_to_rank[
            (matches_to_rank['team1_player1'] == player) |
            (matches_to_rank['team1_player2'] == player) |
            (matches_to_rank['team2_player1'] == player) |
            (matches_to_rank['team2_player2'] == player)
        ]
        comeback_wins = 0
        for _, r in player_matches.iterrows():
            sets = [r['set1'], r['set2'], r['set3']]
            valid_sets = [s for s in sets if s]
            if len(valid_sets) >= 2:
                try:
                    g1, g2 = map(int, valid_sets[0].split('-'))
                except:
                    continue
                first_set_winner = "Team 1" if g1 > g2 else "Team 2"
                if (player in [r['team1_player1'], r['team1_player2']] and first_set_winner == "Team 2" and r['winner'] == "Team 1") or \
                   (player in [r['team2_player1'], r['team2_player2']] and first_set_winner == "Team 1" and r['winner'] == "Team 2"):
                    comeback_wins += 1
        if comeback_wins >= 3:
            badges.append("🔄 Comeback Kid")

        # NEW: Most Improved (last 10 vs career)
        recent_matches = player_matches.sort_values(by="date", ascending=False).head(10)
        recent_wins = 0
        for _, r in recent_matches.iterrows():
            if (player in [r['team1_player1'], r['team1_player2']] and r['winner'] == "Team 1") or \
               (player in [r['team2_player1'], r['team2_player2']] and r['winner'] == "Team 2"):
                recent_wins += 1
        recent_win_rate = recent_wins / len(recent_matches) * 100 if not recent_matches.empty else 0
        if recent_win_rate - win_percentage >= 20:
            badges.append("🚀 Most Improved")

        # NEW: Game Hog
        if games_won[player] == max(games_won.values()):
            badges.append("🥇 Game Hog")

        rank_data.append({
            "Rank": 0, "Profile": profile_image, "Player": player, "Points": scores[player],
            "Win %": round(win_percentage, 2), "Matches": matches_played[player],
            "Doubles Matches": doubles_matches[player], "Singles Matches": singles_matches[player],
            "Wins": wins[player], "Losses": losses[player], "Games Won": games_won[player],
            "Game Diff Avg": round(game_diff_avg, 2), "Cumulative Game Diff": cumulative_game_diff[player],
            "Recent Trend": player_trend,
            "Clutch Factor": round(clutch_factor, 1),
            "Consistency Index": round(consistency_index, 2),
            "Badges": badges
        })

    rank_df = pd.DataFrame(rank_data)
    if not rank_df.empty:
        rank_df = rank_df.sort_values(
            by=["Points", "Win %", "Game Diff Avg", "Games Won", "Player"],
            ascending=[False, False, False, False, True]
        ).reset_index(drop=True)
        rank_df["Rank"] = [f"🏆 {i}" for i in range(1, len(rank_df) + 1)]

    return rank_df, partner_stats








#------CHART FOR PLAYER TO INSIGHTS -------------------------------------


def create_trend_chart(trend):
    if not trend or trend == 'No recent matches':
        return None

    results = trend.split()  # Keep original order (newest to oldest)

    fig = go.Figure()

    x = list(range(1, len(results) + 1))
    y = [1 if r == 'W' else 0 for r in results]
    colors = ['#00FF00' if r == 'W' else '#FF0000' for r in results]  # Green for W, Red for L

    fig.add_trace(go.Bar(
        x=x,
        y=y,
        marker_color=colors,
        width=0.4
    ))

    fig.update_layout(
        height=150,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fff500'),
        xaxis=dict(
            title='Recent Matches (Newest to Oldest)',
            tickmode='array',
            tickvals=x,
            ticktext=[f"M{i}" for i in x],
            title_font=dict(color='#fff500'),
            tickfont=dict(color='#fff500')
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        showlegend=False
    )

    return fig







# -----------------------------Updated display_player_insights function to ensure performance score ----------------------------







def display_player_insights(selected_players, players_df, matches_df, doubles_rank_df, singles_rank_df, key_prefix=""):
    import pandas as pd
    from collections import defaultdict
    if isinstance(selected_players, str):
        selected_players = [selected_players] if selected_players else []
    selected_players = [p for p in selected_players if p != "Visitor"]
    if not selected_players:
        st.info("No players selected or available for insights.")
        return

    # --- Birthday View ---
    view_option = st.radio("Select View", ["Player Insights", "Birthdays"], horizontal=True, key=f"{key_prefix}view_selector")
    if view_option == "Birthdays":
        birthday_data = []
        for player in selected_players:
            player_info = players_df[players_df["name"] == player].iloc[0] if player in players_df["name"].values else None
            if player_info is None:
                continue
            birthday = player_info.get("birthday", "")
            profile_image = player_info.get("profile_image_url", "")
            if birthday and re.match(r'^\d{2}-\d{2}$', birthday):
                try:
                    day, month = map(int, birthday.split("-"))
                    # Use non-leap year (2001) to avoid Feb 29 ambiguity
                    birthday_dt = datetime(year=2001, month=month, day=day)
                    birthday_data.append({
                        "Player": player,
                        "Birthday": birthday_dt.strftime("%b %d"),
                        "SortDate": birthday_dt,
                        "Profile": profile_image
                    })
                except ValueError:
                    continue  # Skip invalid dates like Feb 29
        if not birthday_data:
            st.info("No valid birthday data available for selected players.")
            return
        birthday_df = pd.DataFrame(birthday_data).sort_values(by="SortDate").reset_index(drop=True)
        st.markdown('<div class="rankings-table-container">', unsafe_allow_html=True)
        for _, row in birthday_df.iterrows():
            profile_html = f'<a href="{row["Profile"]}" target="_blank"><img src="{row["Profile"]}" class="profile-image" alt="Profile"></a>' if row["Profile"] else ''
            st.markdown(f"""
            <div class="ranking-row">
                <div class="rank-profile-player-group">
                    <div class="profile-col">{profile_html}</div>
                    <div class="player-col"><span style='font-weight:bold; color:#fff500;'>{row['Player']}</span></div>
                </div>
                <div class="birthday-col"><span style='font-weight:bold; color:#fff500;'>{row['Birthday']}</span></div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # --- Player Insights View ---
    # Use rank_df_combined for filtering active players
    rank_df_combined, partner_stats = calculate_rankings(matches_df)
    active_players = [
        p for p in selected_players
        if p in rank_df_combined["Player"].values and rank_df_combined[rank_df_combined["Player"] == p].iloc[0]["Matches"] > 0
    ]
    if not active_players:
        st.info("No players with matches played are available for insights.")
        return

    # Ensure doubles_rank_df and singles_rank_df are DataFrames
    if not isinstance(doubles_rank_df, pd.DataFrame):
        doubles_rank_df = pd.DataFrame(columns=["Rank", "Profile", "Player", "Points", "Win %", "Matches", "Doubles Matches", "Singles Matches", "Wins", "Losses", "Games Won", "Game Diff Avg", "Cumulative Game Diff", "Recent Trend", "Clutch Factor", "Consistency Index", "Badges"])
    if not isinstance(singles_rank_df, pd.DataFrame):
        singles_rank_df = pd.DataFrame(columns=["Rank", "Profile", "Player", "Points", "Win %", "Matches", "Doubles Matches", "Singles Matches", "Wins", "Losses", "Games Won", "Game Diff Avg", "Cumulative Game Diff", "Recent Trend", "Clutch Factor", "Consistency Index", "Badges"])

    # CSS for tooltips
    tooltip_css = """
    <style>
    .badge-container {
        position: relative;
        display: inline-block;
        margin-right: 4px;
    }
    .badge-tooltip {
        visibility: hidden;
        width: 200px;
        background-color: rgba(255, 255, 255, 0.9);
        color: #031827;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .badge-container:hover .badge-tooltip {
        visibility: visible;
        opacity: 1;
    }
    @media (max-width: 600px) {
        .badge-tooltip {
            width: 150px;
            font-size: 10px;
        }
    }
    </style>
    """
    st.markdown(tooltip_css, unsafe_allow_html=True)

    for idx, player in enumerate(sorted(active_players)):
        player_info = players_df[players_df["name"] == player].iloc[0]
        player_data = rank_df_combined[rank_df_combined["Player"] == player].iloc[0]

        # --- Data Calculation & Formatting ---
        profile_image = player_info.get("profile_image_url", "")
        wins, losses = int(player_data["Wins"]), int(player_data["Losses"])
        trend = get_player_trend(player, matches_df)

        # --- Performance Score Calculation ---
        if not doubles_rank_df.empty and 'Player' in doubles_rank_df.columns and player in doubles_rank_df['Player'].values:
            doubles_perf_score = _calculate_performance_score(doubles_rank_df[doubles_rank_df['Player'] == player].iloc[0], doubles_rank_df)
        else:
            doubles_perf_score = 0.0

        if not singles_rank_df.empty and 'Player' in singles_rank_df.columns and player in singles_rank_df['Player'].values:
            singles_perf_score = _calculate_performance_score(singles_rank_df[singles_rank_df['Player'] == player].iloc[0], singles_rank_df)
        else:
            singles_perf_score = 0.0

        rank_value = player_data['Rank']
        rank_display = re.sub(r'[^0-9]', '', str(rank_value))

        birthday_str = ""
        raw_birthday = player_info.get("birthday")
        if raw_birthday and isinstance(raw_birthday, str) and re.match(r'^\d{2}-\d{2}$', raw_birthday):
            try:
                day, month = map(int, raw_birthday.split("-"))
                # Use non-leap year (2001) to avoid Feb 29 ambiguity
                bday_obj = datetime(year=2001, month=month, day=day)
                birthday_str = bday_obj.strftime("%d %b")
            except ValueError:
                birthday_str = ""

        # --- Partner Calculation Logic (Full History) ---
        partners_list_str = "No doubles matches played."
        best_partner_str = "N/A"
        if player in partner_stats and partner_stats[player]:
            partners_list_items = [
                f'<li><b>{p}</b>: {item["wins"]}W - {item["losses"]}L ({item["matches"]} played)</li>'
                for p, item in partner_stats[player].items() if p != "Visitor"
            ]
            partners_list_str = f"<ul>{''.join(partners_list_items)}</ul>"

            sorted_partners = sorted(
                [(p, item) for p, item in partner_stats[player].items() if p != "Visitor"],
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

        # --- Badges HTML with Hover Tooltips ---
        badge_explanations = {
            "🎯 Tie-break Monster": "Dominates tie-breaks with the most wins (clutch factor >70% in 3+ clutch matches)",
            "🔥 Hot Streak": "Achieved a winning streak of 5 or more matches",
            "📈 Consistent Performer": "Reliable performance with low variation in game differences (consistency index <2 over 5+ matches)",
            "💪 Ironman": "Played the most matches without missing a session",
            "🔄 Comeback Kid": "Won 3 or more matches after losing the first set",
            "🚀 Most Improved": "Recent win rate (last 10 matches) is 20%+ higher than overall career win rate",
            "🥇 Game Hog": "Won the highest total number of games across all matches"
        }
        badges = player_data["Badges"]
        badges_html = ""
        if badges:
            badges_html = (
                "<span class='badges-col' style='display: block; margin-top: 6px;'>"
                "<span style='font-weight:bold; color:#bbbbbb;'>Badges: </span><br>"
                + " ".join([
                    f"<span class='badge-container'>"
                    f"<span style='background:#fff500; color:#031827; padding:2px 6px; border-radius:6px;'>{b}</span>"
                    f"<span class='badge-tooltip'>{badge_explanations.get(b, 'No description available')}</span>"
                    f"</span>"
                    for b in badges
                ])
                + "</span>"
            )

        # --- Unique key to avoid StreamlitDuplicateElementKey ---
        unique_id = f"{key_prefix}_{idx}"

        # --- Updated Card Layout ---
        st.markdown("---")

        header_html = f"""
        <div style="margin-bottom: 15px;">
            <h2 style="color: #fff500; margin-bottom: 5px; font-size: 2.0em; font-weight: bold;">{player}</h2>
            <span style="font-weight: bold; color: #bbbbbb; font-size: 1.1em;">
                Rank: <span style="color: #fff500;">#{rank_display}</span>
            </span>
            {f' | <span style="font-weight:bold; color:#bbbbbb; font-size: 1.1em;">🎂 Birthday: <span style="color: #fff500;">{birthday_str}</span></span>' if birthday_str else ''}
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
                st.plotly_chart(win_loss_chart, config={"responsive": True}, key=f"{unique_id}_win_loss")

            st.markdown("##### Recent Trend")
            trend_chart = create_trend_chart(trend)
            if trend_chart:
                st.plotly_chart(trend_chart, config={"responsive": True}, key=f"{unique_id}_trend")
            else:
                st.markdown("No recent matches")

            st.markdown(f"<div class='trend-col'>{trend}</div>", unsafe_allow_html=True)

        with col2:  # Right column for stats
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Points", f"{player_data['Points']:.1f}")
            m_col2.metric("Win Rate", f"{player_data['Win %']:.1f}%")
            m_col3.metric("Matches", f"{int(player_data['Matches'])}")

            clutch_factor = player_data["Clutch Factor"]
            consistency_index = player_data["Consistency Index"]

            st.markdown(f"""
            <div style="line-height: 2;">
                <span class="games-won-col" style="display: block;">Games Won: {int(player_data['Games Won'])}</span>
                <span class="game-diff-avg-col" style="display: block;">Game Diff Avg: {player_data['Game Diff Avg']:.2f}</span>
                <span class="cumulative-game-diff-col" style="display: block;">Cumulative Game Diff: {int(player_data['Cumulative Game Diff'])}</span>
                <span class="performance-score-col" style="display: block;">
                    <span style='font-weight:bold; color:#bbbbbb;'>Performance Score: </span>
                    <span style='font-weight:bold; color:#fff500;'>Doubles: {doubles_perf_score:.1f} ({int(player_data["Doubles Matches"])}), Singles: {singles_perf_score:.1f} ({int(player_data["Singles Matches"])})</span>
                </span>
                <span class="clutch-col" style="display: block;">
                    <span style='font-weight:bold; color:#bbbbbb;'>Clutch Factor: </span>
                    <span style='font-weight:bold; color:#fff500;'>{clutch_factor:.1f}%</span>
                </span>
                <span class="consistency-col" style="display: block;">
                    <span style='font-weight:bold; color:#bbbbbb;'>Consistency Index: </span>
                    <span style='font-weight:bold; color:#fff500;'>{consistency_index:.2f}</span>
                </span>
                <span class="best-partner-col" style="display: block;">
                    <span style='font-weight:bold; color:#bbbbbb;'>Most Effective Partner (All Time): </span>{best_partner_str}
                </span>
                {badges_html}
            </div>
            """, unsafe_allow_html=True)

            with st.expander("View Partner Stats", expanded=False, icon="➡️"):
                st.markdown(partners_list_str, unsafe_allow_html=True)













#------------------- END OF display_player_insights  and calculate rankings function --------------------------------



def display_community_stats(matches_df):
    """
    Calculates and displays interesting community stats for the last 7 days.
    """
    # Ensure the 'date' column is in datetime format and remove timezone
    #matches_df['date'] = pd.to_datetime(matches_df['date'], errors='coerce').dt.tz_localize(None)
    matches_df['date'] = pd.to_datetime(matches_df['date'], utc=True, errors='coerce').dt.tz_localize(None)

    # Get the date 7 days ago from today
    seven_days_ago = datetime.now() - pd.Timedelta(days=7)

    # Filter matches from the last 7 days
    recent_matches = matches_df[matches_df['date'] >= seven_days_ago]

    if recent_matches.empty:
        st.info("No matches played in the last 7 days.")
        return

    # 1. Number of matches played in the last 7 days
    num_matches = len(recent_matches)
    st.metric("Matches Played", num_matches)

    # 2. Number of active players in the last 7 days
    player_columns = ['team1_player1', 'team1_player2', 'team2_player1', 'team2_player2']
    active_players = pd.unique(recent_matches[player_columns].values.ravel('K'))
    # Remove any potential 'None' or empty values
    active_players = [player for player in active_players if pd.notna(player) and player != '']
    num_active_players = len(active_players)
    st.metric("Active Players", num_active_players)

    # 4. Other interesting item: Top 5 players with the most wins in the last 7 days
    st.markdown("##### Top 5 Winners (Last 7 Days)")
    winners = []
    for index, row in recent_matches.iterrows():
        if row['winner'] == 'Team 1':
            winners.extend([row['team1_player1'], row['team1_player2']])
        elif row['winner'] == 'Team 2':
            winners.extend([row['team2_player1'], row['team2_player2']])

    winners = [w for w in winners if pd.notna(w) and w != '']
    if winners:
        win_counts = pd.Series(winners).value_counts().nlargest(5)
        st.table(win_counts)
    else:
        st.info("No wins recorded in the last 7 days.")




# Chart --------------

def create_nerd_stats_chart(rank_df):
    """Creates a styled, stacked bar chart for player performance."""
    if rank_df is None or rank_df.empty:
        return None

    # Sort players from highest to lowest rank (which is the default order of rank_df)
    df = rank_df.copy()

    # Define colors
    optic_yellow = '#fff500'
    bright_orange = '#FFA500'
    # Updated color palette for higher contrast
    bar_colors = ['#1E90FF', '#FFD700', '#9A5BE2']  # Dodger Blue, Gold, and a vibrant Purple

    fig = go.Figure()

    # Add traces for the stacked bars as per the user's request
    fig.add_trace(go.Bar(
        x=df['Player'],
        y=df['Matches'],
        name='Matches Played',
        marker_color=bar_colors[0]
    ))
    fig.add_trace(go.Bar(
        x=df['Player'],
        y=df['Wins'],
        name='Matches Won',
        marker_color=bar_colors[1]
    ))
    fig.add_trace(go.Bar(
        x=df['Player'],
        y=df['Points'],
        name='Points',
        marker_color=bar_colors[2]
    ))

    # Update the layout for custom styling
    fig.update_layout(
        barmode='stack',
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area
        font=dict(color=optic_yellow),  # Set default font color for the chart
        xaxis=dict(
            title=dict(
                text='Players (Ranked Highest to Lowest)',
                font=dict(color=optic_yellow)
            ),
            tickfont=dict(color=optic_yellow),
            showgrid=False,
            linecolor=bright_orange,
            linewidth=2,
            mirror=True
        ),
        yaxis=dict(
            title=dict(
                text='Stacked Value (Points + Wins + Matches)',
                font=dict(color=optic_yellow)
            ),
            tickfont=dict(color=optic_yellow),
            gridcolor='rgba(255, 165, 0, 0.2)',  # Faint orange grid lines
            linecolor=bright_orange,
            linewidth=2,
            mirror=True
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=optic_yellow)
        ),
        margin=dict(t=60, b=10, l=10, r=10)  # Adjust top margin for legend
    )

    return fig



# --------------------------------------

def create_partnership_chart(player_name, partner_stats, players_df):
    """Creates a horizontal bar chart showing a player's performance with different partners."""
    if player_name not in partner_stats or not partner_stats[player_name]:
        return None

    partners_data = partner_stats[player_name]
    
    # Exclude "Visitor" and prepare data for DataFrame
    chart_data = []
    for partner, stats in partners_data.items():
        if partner == "Visitor":
            continue
        win_percentage = (stats['wins'] / stats['matches'] * 100) if stats['matches'] > 0 else 0
        chart_data.append({
            'Partner': partner,
            'Win %': win_percentage,
            'Matches Played': stats['matches'],
            'Wins': stats['wins'],
            'Losses': stats['losses']
        })

    if not chart_data:
        return None

    df = pd.DataFrame(chart_data).sort_values(by='Win %', ascending=True)

    # Define colors
    optic_yellow = '#fff500'
    
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df['Partner'],
        x=df['Win %'],
        orientation='h',
        text=df.apply(lambda row: f"{row['Wins']}W - {row['Losses']}L ({row['Matches Played']} Matches)", axis=1),
        textposition='auto',
        marker=dict(
            color=df['Win %'],
            colorscale='Viridis',
            colorbar=dict(title='Win %')
        )
    ))

    # --- THIS SECTION IS CORRECTED ---
    fig.update_layout(
        title=f'Partnership Performance for: {player_name}',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=optic_yellow),
        xaxis=dict(
            title=dict(text='Win Percentage (%)', font=dict(color=optic_yellow)),
            tickfont=dict(color=optic_yellow),
            showgrid=True,
            gridcolor='rgba(255, 165, 0, 0.2)'
        ),
        yaxis=dict(
            title=dict(text='Partner', font=dict(color=optic_yellow)),
            tickfont=dict(color=optic_yellow),
            showgrid=False
        ),
        margin=dict(l=100, r=20, t=60, b=40)
    )

    return fig



  #-----------------------------------------------------------------------------------

def load_bookings():
    try:
        response = supabase.table("bookings").select("*").execute()
        df = pd.DataFrame(response.data)
        
        expected_columns = ['booking_id', 'date', 'time', 'match_type', 'court_name',
                            'player1', 'player2', 'player3', 'player4',
                            'standby_player', 'screenshot_url']
        for col in expected_columns:
            if col not in df:
                df[col] = None

        if not df.empty:
            # FIX: Create a timezone-aware datetime column directly from the source strings.
            # This avoids the combine() error and the timezone comparison error.
            #datetime_str = df['date'].astype(str) + ' ' + df['time'].astype(str)
            #df['booking_datetime'] = pd.to_datetime(datetime_str, errors='coerce')
            datetime_str = df['date'].astype(str) + ' ' + df['time'].astype(str)
            df['booking_datetime'] = pd.to_datetime(
                datetime_str,
                format="%Y-%m-%d %H:%M:%S",   # match your data format
                errors='coerce'
            )

          
            # Localize the naive datetime to the correct timezone ('Asia/Dubai')
            # This is the crucial step to fix the "Invalid comparison" error
            df['booking_datetime'] = df['booking_datetime'].dt.tz_localize('Asia/Dubai', ambiguous='infer')
            
            # Now, the cutoff and the booking_datetime column are both timezone-aware
            cutoff = pd.Timestamp.now(tz='Asia/Dubai') - timedelta(hours=4)

            # Filter for expired bookings
            expired = df[df['booking_datetime'].notnull() & (df['booking_datetime'] < cutoff)]

            # Delete expired bookings from Supabase
            for _, row in expired.iterrows():
                try:
                    supabase.table("bookings").delete().eq("booking_id", row['booking_id']).execute()
                except Exception as e:
                    st.error(f"Failed to delete expired booking {row['booking_id']}: {e}")

            # Keep only valid, non-expired bookings
            df = df[df['booking_datetime'].isnull() | (df['booking_datetime'] >= cutoff)]

        # Final cleaning for display and session state
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d').fillna("")
        df['time'] = df['time'].fillna("")
        
        for col in ['player1', 'player2', 'player3', 'player4', 'standby_player', 'screenshot_url']:
            df[col] = df[col].fillna("")

        st.session_state.bookings_df = df.reindex(columns=expected_columns)

    except Exception as e:
        st.error(f"Failed to load bookings: {str(e)}")
        # Initialize with an empty DataFrame on failure
        st.session_state.bookings_df = pd.DataFrame(columns=expected_columns)



def save_bookings(df):
    try:
        df_to_save = df.copy()
        
        if 'date' in df_to_save.columns:
            # Convert to datetime (no timezone conversion)
            df_to_save['date'] = pd.to_datetime(df_to_save['date'], errors='coerce')
            df_to_save = df_to_save.dropna(subset=['date'])
            
            # Format datetime as "YYYY-MM-DD HH:MM:SS" to include seconds
            df_to_save['date'] = df_to_save['date'].dt.strftime("%Y-%m-%d %H:%M:%S")

        # Remove duplicates
        duplicates = df_to_save[df_to_save.duplicated(subset=['booking_id'], keep=False)]
        if not duplicates.empty:
            st.warning(f"Found duplicate booking_id values: {duplicates['booking_id'].tolist()}")
            df_to_save = df_to_save.drop_duplicates(subset=['booking_id'], keep='last')

        # Replace NaN with None for JSON compliance
        df_to_save = df_to_save.where(pd.notna(df_to_save), None)

        # Upsert to Supabase
        supabase.table(bookings_table_name).upsert(df_to_save.to_dict("records")).execute()

    except Exception as e:
        st.error(f"Error saving bookings: {str(e)}")






      
def create_backup_zip(players_df, matches_df, bookings_df):
    """Create a zip file with CSV tables + images from Supabase URLs."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        # --- CSVs ---
        zf.writestr("players.csv", players_df.to_csv(index=False))
        zf.writestr("matches.csv", matches_df.to_csv(index=False))
        zf.writestr("bookings.csv", bookings_df.to_csv(index=False))

        # --- Profile images ---
        for _, row in players_df.iterrows():
            url = row.get("profile_image_url")
            if url:
                img_data = download_image(url)
                if img_data:
                    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', row["name"])  # sanitize filename
                    filename = f"profile_images/{safe_name}.jpg"
                    zf.writestr(filename, img_data)

        # --- Match images ---
        for _, row in matches_df.iterrows():
            url = row.get("match_image_url")
            if url:
                img_data = download_image(url)
                if img_data:
                    match_id = row.get("match_id", str(uuid.uuid4()))
                    filename = f"match_images/{match_id}.jpg"
                    zf.writestr(filename, img_data)

    buffer.seek(0)
    return buffer



def generate_booking_id(bookings_df, booking_date):
    year = booking_date.year
    quarter = get_quarter(booking_date.month)
    if not bookings_df.empty and 'date' in bookings_df.columns:
        bookings_df['date'] = pd.to_datetime(bookings_df['date'], errors='coerce')
        filtered_bookings = bookings_df[
            (bookings_df['date'].dt.year == year) &
            (bookings_df['date'].apply(lambda d: get_quarter(d.month) == quarter))
        ]
        serial_number = len(filtered_bookings) + 1
        new_id = f"BK{quarter}{year}-{serial_number:02d}"
        while new_id in bookings_df['booking_id'].values:
            serial_number += 1
            new_id = f"BK{quarter}{year}-{serial_number:02d}"
    else:
        serial_number = 1
        new_id = f"BK{quarter}{year}-{serial_number:02d}"
    return new_id


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


def delete_booking_from_db(booking_id):
    try:
        supabase.table(bookings_table_name).delete().eq("booking_id", booking_id).execute()
        st.session_state.bookings_df = st.session_state.bookings_df[st.session_state.bookings_df["booking_id"] != booking_id].reset_index(drop=True)
        save_bookings(st.session_state.bookings_df)
    except Exception as e:
        st.error(f"Error deleting booking from database: {str(e)}")

def display_rankings_table(rank_df, title):
    if rank_df.empty:
        st.info(f"No {title} ranking data available.")
        return
    display_df = rank_df[["Rank", "Player", "Points", "Win %", "Matches", "Wins", "Losses", "Games Won", "Game Diff Avg", "Recent Trend"]].copy()
    display_df["Points"] = display_df["Points"].map("{:.1f}".format)
    display_df["Win %"] = display_df["Win %"].map("{:.1f}%".format)
    display_df["Game Diff Avg"] = display_df["Game Diff Avg"].map("{:.2f}".format)
    display_df["Matches"] = display_df["Matches"].astype(int)
    display_df["Wins"] = display_df["Wins"].astype(int)
    display_df["Losses"] = display_df["Losses"].astype(int)
    display_df["Games Won"] = display_df["Games Won"].astype(int)
    st.subheader(f"{title} Rankings")
    st.dataframe(display_df, hide_index=True, height=300)

def display_match_table(df, title):
    if df.empty:
        st.info(f"No {title} match data available.")
        return

    table_df = df.copy()

    # Create a formatted Match column
    def format_match_info(row):
        scores = [s for s in [row['set1'], row['set2'], row['set3']] if s]
        scores_str = ", ".join(scores)

        if row['match_type'] == 'Doubles':
            players = f"{row['team1_player1']} & {row['team1_player2']} vs. {row['team2_player1']} & {row['team2_player2']}"
        else:
            players = f"{row['team1_player1']} vs. {row['team2_player1']}"

        # Handle tie and winner cases with "tied with" for ties
        if row['winner'] == "Tie":
            if row['match_type'] == 'Doubles':
                return f"{row['team1_player1']} & {row['team1_player2']} tied with {row['team2_player1']} & {row['team2_player2']} ({scores_str})"
            else:
                return f"{row['team1_player1']} tied with {row['team2_player1']} ({scores_str})"
        elif row['winner'] == "Team 1":
            return f"{row['team1_player1']} {'& ' + row['team1_player2'] if row['match_type']=='Doubles' else ''} def. {row['team2_player1']} {'& ' + row['team2_player2'] if row['match_type']=='Doubles' else ''} ({scores_str})"
        elif row['winner'] == "Team 2":
            return f"{row['team2_player1']} {'& ' + row['team2_player2'] if row['match_type']=='Doubles' else ''} def. {row['team1_player1']} {'& ' + row['team1_player2'] if row['match_type']=='Doubles' else ''} ({scores_str})"
        else:
            return f"{players} ({scores_str})"

    table_df['Match Details'] = table_df.apply(format_match_info, axis=1)

    # Select and rename columns for display
    display_df = table_df[['date', 'Match Details', 'match_image_url']].copy()
    display_df.rename(columns={
        'date': 'Date',
        'match_image_url': 'Image URL'
    }, inplace=True)

    # Format the date column as dd MMM yy
    display_df['Date'] = pd.to_datetime(display_df['Date']).dt.strftime('%d %b %y')

    st.dataframe(display_df, height=300)


# --- Updated generate_whatsapp_link Function ---
def generate_whatsapp_link(row):
    """
    Generate a WhatsApp share link with match details, including GDA-based verb and GDA value.
    """
    # Get verb and GDA
    verb, gda = get_match_verb_and_gda(row)
    
    # Build side labels
    if row["match_type"] == "Singles":
        t1 = f"{row['team1_player1']}"
        t2 = f"{row['team2_player1']}"
    else:  # Doubles
        t1 = f"{row['team1_player1']} & {row['team1_player2']}"
        t2 = f"{row['team2_player1']} & {row['team2_player2']}"
    
    # Scores and GDA
    scores_list = []
    for s in [row['set1'], row['set2'], row['set3']]:
        if s:
            if "Tie Break" in s:
                tie_break_scores = s.replace("Tie Break", "").strip().split('-')
                if int(tie_break_scores[0]) > int(tie_break_scores[1]):
                    scores_list.append(f'*7-6({tie_break_scores[0]}:{tie_break_scores[1]})*')
                else:
                    scores_list.append(f'*6-7({tie_break_scores[0]}:{tie_break_scores[1]})*')
            else:
                scores_list.append(f'*{s.replace("-", ":")}*')
    
    scores_str = " ".join(scores_list)
    gda_text = f" | GDA: {gda:.2f}"
    
    # Check if the date is valid before formatting
    if pd.notna(row['date']):
        date_str = pd.to_datetime(row['date']).strftime('%A, %d %b')
    else:
        date_str = "Unknown Date"  # Fallback text
    
    # Headline text: use "tied with" for ties and GDA-based verb for wins
    if row["winner"] == "Tie":
        headline = f"*{t1} tied with {t2}*"
    elif row["winner"] == "Team 1":
        headline = f"*{t1} {verb} {t2}*"
    else:  # Team 2
        headline = f"*{t2} {verb} {t1}*"
    
    # Construct share text
    share_text = f"{headline}\nSet scores {scores_str}{gda_text}\nDate: *{date_str}*"
    #if row["match_image_url"]:
    #    share_text += f"\nImage: {row['match_image_url']}"
    
    # Encode text for WhatsApp URL
    encoded_text = urllib.parse.quote(share_text)
    return f"https://api.whatsapp.com/send/?text={encoded_text}&type=custom_url&app_absent=0"


# Birthday Functions added



def check_birthdays(players_df):
    todays_birthdays = []
    today = datetime.now().date()
    for _, row in players_df.iterrows():
        birthday = row.get("birthday")
        if birthday and re.match(r'^\d{2}-\d{2}$', birthday):
            try:
                day, month = map(int, birthday.split("-"))
                birthday_date = datetime(today.year, month, day).date()
                if birthday_date == today:
                    todays_birthdays.append(row["name"])  # Append only name (string)
            except ValueError:
                continue  # Skip invalid dates
    return todays_birthdays


def display_birthday_message(birthday_players):
    """Displays a prominent birthday banner for each player in the list with a WhatsApp share button."""
    for player_name in birthday_players:  # Changed to single variable
        message = f"Happy Birthday {player_name}!"
        whatsapp_message = f"*{message}* 🎂🎈"
        encoded_message = urllib.parse.quote(whatsapp_message)
        whatsapp_link = f"https://api.whatsapp.com/send/?text={encoded_message}&type=custom_url&app_absent=0"
        st.markdown(
            f"""
            <div class="birthday-banner">
                {message} Have a smashing year on the court! 🎾
                <a href="{whatsapp_link}" target="_blank" class="whatsapp-share">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg" alt="Share on WhatsApp">
                    Share
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )




def create_win_loss_donut(wins, losses):
    if wins == 0 and losses == 0:
        return None
    fig = go.Figure(data=[go.Pie(labels=['Wins', 'Losses'],
                                 values=[wins, losses],
                                 hole=.6,
                                 marker_colors=['#00a86b', '#ff4136'],
                                 textinfo='none',
                                 hoverinfo='label+value')])
    fig.update_layout(
        showlegend=False,
        height=120,
        width=120,
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_trend_sparkline(trend_string):
    trend_map = {'W': 1, 'L': -1, 'T': 0}
    trend_values = [trend_map[result] for result in trend_string.split() if result in trend_map]
    if not trend_values:
        return None

    fig = go.Figure(go.Scatter(
        y=trend_values,
        mode='lines+markers',
        line=dict(color='#fff500', width=3),
        marker=dict(color='#fff500', size=6, symbol='circle'),
        hoverinfo='none'
    ))
    fig.update_layout(
        showlegend=False,
        height=50,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5])
    )
    return fig

def create_match_type_bar_chart(singles_count, doubles_count):
    if singles_count == 0 and doubles_count == 0:
        return None
    fig = go.Figure(data=[
        go.Bar(name='Singles', x=['Singles'], y=[singles_count], marker_color='#3498db', text=singles_count, textposition='auto'),
        go.Bar(name='Doubles', x=['Doubles'], y=[doubles_count], marker_color='#9b59b6', text=doubles_count, textposition='auto')
    ])
    fig.update_layout(
        showlegend=False,
        height=150,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        xaxis=dict(tickfont=dict(color='white')),
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )
    return fig



def get_match_verb_and_gda(row):
    """
    Calculate the Game Difference Average (GDA) for a match and select an appropriate verb.
    Returns a tuple of (verb, gda).
    """
    match_0to1 = ["squeaked past", "barely beat", "got lucky against", "got lucky against","survived (barely)" ]
    match_1to2_9 = ["with luck & gusto, def."," defeated","edged past","squeaked past", "barely beat", "got lucky against", "got lucky against","survived (barely)" ]
    match_3to4_9 = ["outplayed", "dominated", "got the better of", "vanquished", "trounced"]
    match_5to6 = ["thrashed", "crushed", "beat the hell out of", "smashed", "obliterated", 
                        "demolished", "routed", "destroyed","deadpooled"]
    
    game_diffs = []
    for set_score in [row['set1'], row['set2'], row['set3']]:
        if not set_score or ('-' not in str(set_score) and 'Tie Break' not in str(set_score)):
            continue
        try:
            if "Tie Break" in str(set_score):
                # For tie-break or super tie-break, always GD = 1 (equivalent to 7-6)
                game_diff = 1
            else:
                # Handle regular sets
                team1_games, team2_games = map(int, str(set_score).split('-'))
                game_diff = abs(team1_games - team2_games)
            game_diffs.append(game_diff)
        except (ValueError, TypeError):
            continue
    
    # Calculate GDA: sum(GD) / num_sets
    gda = sum(game_diffs) / len(game_diffs) if game_diffs else 0
    
    # Select verb based on GDA
    if 0 <= gda <= 1:
        verb = random.choice(match_0to1)
    elif 1.1 <= gda <= 2.9:
        verb = random.choice(match_1to2_9)
    elif 3 <= gda <= 4.9:
        verb = random.choice(match_3to4_9)
    elif 5 <= gda <= 6:
        verb = random.choice(match_5to6)
    
    else:  # GDA > 6 or GDA < 0
        verb = random.choice(match_5to6)  # Treat as easy match for simplicity
    return verb, gda

#------------------  ADD BOOKINGS TO CALENDAR -----------------------------


def generate_ics_for_booking(row, plain_suggestion):
    """
    Generates ICS content for a booking to add to calendar, using UTC time for DTSTART/DTEND.
    """
    try:
        if pd.isna(row['datetime']) or row['datetime'] is None:
            return None, "Invalid date/time for this booking."
        
        # Use Asia/Dubai time directly from row['datetime']
        dt_start = row['datetime']
        dt_end = row['datetime'] + pd.Timedelta(hours=2)
        dt_stamp = pd.Timestamp.now(tz='UTC')
        
        # Convert to UTC for ICS
        dt_start_utc = dt_start.tz_convert('UTC')
        dt_end_utc = dt_end.tz_convert('UTC')
        
        # Format datetimes for ICS (YYYYMMDDTHHMMSSZ for UTC)
        dtstart_str = dt_start_utc.strftime('%Y%m%dT%H%M%SZ')
        dtend_str = dt_end_utc.strftime('%Y%m%dT%H%M%SZ')
        dtstamp_str = dt_stamp.strftime('%Y%m%dT%H%M%SZ')
        
        uid = f"{row['booking_id']}@ar-tennis.com"
        summary = f"Tennis {row['match_type']} Booking at {row['court_name']}"
        
        players_str = ', '.join([p for p in [row['player1'], row['player2'], row['player3'], row['player4']] if p])
        standby_str = row.get('standby_player', 'None')
        date_str = pd.to_datetime(row['date']).strftime('%A, %d %b')
        time_ampm = dt_start.strftime('%I:%M %p').lstrip('0')  # Local Asia/Dubai time, e.g., 6:00 PM
        court_url = court_url_mapping.get(row['court_name'], "#")
        
        description = f"""Date: {date_str}
Time: {time_ampm}
Players: {players_str}
Standby: {standby_str}
Pairing Odds: {plain_suggestion}
Court Map: {court_url}""".replace('\n', '\\n')
        
        location = row['court_name']
        
        ics_content = f"""BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:{uid}
DTSTAMP:{dtstamp_str}
DTSTART:{dtstart_str}
DTEND:{dtend_str}
SUMMARY:{summary}
DESCRIPTION:{description}
LOCATION:{location}
END:VEVENT
END:VCALENDAR"""
        
        return ics_content, None
    except Exception as e:
        return None, f"Error generating ICS: {str(e)}"


# ----------------------GENERATE MATCH CARD ---------------------------------------





def generate_match_card(row, image_url):
    # Download the image
    response = requests.get(image_url)
    if response.status_code != 200:
        raise ValueError("Failed to download match image")
    img = Image.open(io.BytesIO(response.content))
    
    # Handle EXIF orientation to prevent unintended rotation
    img = ImageOps.exif_transpose(img)
    
    # Resize proportionally to height 1200
    base_height = 1200
    h_percent = base_height / float(img.size[1])
    new_width = int(float(img.size[0]) * float(h_percent))
    img = img.resize((new_width, base_height), Image.LANCZOS)
    
    # Apply rounded corners to the image
    radius = 20  # Corner radius
    mask = Image.new('L', (new_width, base_height), 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.rounded_rectangle((0, 0, new_width, base_height), radius=radius, fill=255)
    
    img_rgba = img.convert('RGBA')
    rounded_img = Image.new('RGBA', (new_width, base_height), (0, 0, 0, 0))
    rounded_img.paste(img_rgba, (0, 0), mask)
    
    # --- MODIFICATION: Create Polaroid canvas with gradient background ---
    border_sides = 30
    border_bottom = 150  # Space for text
    new_img_width = new_width + 2 * border_sides
    new_img_height = base_height + border_sides + border_bottom
    
    # Gradient colors (from your app's CSS)
    top_color = (7, 49, 79)    # #07314f
    bottom_color = (3, 24, 39) # #031827
    
    # Create the gradient background
    gradient_background = Image.new('RGBA', (new_img_width, new_img_height))
    draw = ImageDraw.Draw(gradient_background)
    for y in range(new_img_height):
        # Interpolate color
        r = int(top_color[0] + (bottom_color[0] - top_color[0]) * y / new_img_height)
        g = int(top_color[1] + (bottom_color[1] - top_color[1]) * y / new_img_height)
        b = int(top_color[2] + (bottom_color[2] - top_color[2]) * y / new_img_height)
        draw.line([(0, y), (new_img_width, y)], fill=(r, g, b, 255))

    polaroid_img = gradient_background

    # --- MODIFICATION: Create optic yellow shadow for the image ---
    shadow_offset_img = 5
    shadow_size = (new_width + 10, base_height + 10)
    shadow = Image.new('RGBA', shadow_size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    # Use optic yellow for the fill color
    optic_yellow_color = (204, 255, 0, 128)
    shadow_draw.rounded_rectangle((0, 0, new_width, base_height), radius=radius, fill=optic_yellow_color)
    shadow = shadow.filter(ImageFilter.GaussianBlur(8)) # Increased blur for better effect
    polaroid_img.paste(shadow, (border_sides + shadow_offset_img, border_sides + shadow_offset_img), shadow)
    
    # Paste the rounded image
    polaroid_img.paste(rounded_img, (border_sides, border_sides), rounded_img)
    
    # --- (Rest of the function remains the same for logic) ---

    # Prepare teams
    match_type = row['match_type']
    if match_type == 'Doubles':
        team1 = f"{row['team1_player1']} & {row['team1_player2']}"
        team2 = f"{row['team2_player1']} & {row['team2_player2']}"
    else:
        team1 = row['team1_player1']
        team2 = row['team2_player1']
    
    # Compute sets and GDA
    sets = [s for s in [row['set1'], row['set2'], row['set3']] if s]
    set_text = ", ".join(sets)
    
    match_gd_sum = 0
    num_sets = 0
    for set_score in sets:
        if not set_score:
            continue
        is_tie_break = "Tie Break" in str(set_score)
        if is_tie_break:
            tie_break_scores = [int(s) for s in re.findall(r'\d+', str(set_score))]
            if len(tie_break_scores) != 2:
                continue
            team1_games, team2_games = tie_break_scores
            team1_set_diff = team1_games - team2_games
        else:
            try:
                team1_games, team2_games = map(int, str(set_score).split('-'))
                team1_set_diff = team1_games - team2_games
            except ValueError:
                continue
        match_gd_sum += team1_set_diff
        num_sets += 1
    
    gda = match_gd_sum / num_sets if num_sets > 0 else 0.0
    
    # Adjust GDA sign based on winner
    winner = row['winner']
    if winner == 'Team 2':
        gda = -gda
    elif winner == 'Tie':
        gda = 0.0
    
    abs_gda = abs(gda)
    
    # Define verb lists
    strong_verbs = ["obliterated", "crushed", "smashed", "annihilated", "destroyed"]
    solid_verbs = ["dominated", "overpowered", "outplayed", "thrashed"]
    decent_verbs = ["defeated", "beat", "conquered", "toppled"]
    close_verbs = ["overcame", "outlasted", "surpassed", "bested"]
    tight_verbs = ["edged out", "nipped", "pipped", "squeaked by"]
    tie_verbs = ["tied with", "matched", "drew with"]
    
    # Select random verb
    if winner == 'Tie':
        verb = random.choice(tie_verbs)
    else:
        if abs_gda > 4.0: verb = random.choice(strong_verbs)
        elif abs_gda > 3.0: verb = random.choice(solid_verbs)
        elif abs_gda > 2.0: verb = random.choice(decent_verbs)
        elif abs_gda > 1.0: verb = random.choice(close_verbs)
        else: verb = random.choice(tight_verbs)
        
    # This string is now only used for length calculation, not for drawing
    if winner == 'Tie': players_text = f"{team1} {verb} {team2}"
    elif winner == 'Team 1': players_text = f"{team1} {verb} {team2}"
    else: players_text = f"{team2} {verb} {team1}"
        
    if len(players_text) > 50:
        players_text = players_text[:47] + "..."
    
    date_str = pd.to_datetime(row['date']).strftime('%d %b %y')
    
    # =========================================================================
    # --- START: MODIFIED TEXT DRAWING SECTION ---
    # =========================================================================
    
    draw = ImageDraw.Draw(polaroid_img)
    try:
        font = ImageFont.truetype("CoveredByYourGrace-Regular.ttf", 50)
    except IOError:
        font = ImageFont.load_default() # Simplified fallback
    
    # Font scaling logic...
    max_text_width = new_width * 0.8
    players_bbox = draw.textbbox((0, 0), players_text, font=font)
    if (players_bbox[2] - players_bbox[0]) > max_text_width:
        scale_factor = max_text_width / (players_bbox[2] - players_bbox[0])
        font_size = int(50 * scale_factor)
        try:
            font = ImageFont.truetype("CoveredByYourGrace-Regular.ttf", font_size)
        except IOError:
            font = ImageFont.truetype("arial.ttf", font_size)

    # Text positions
    text_area_top = base_height + border_sides
    x_center = new_img_width / 2
    y_positions = [text_area_top + 30, text_area_top + 80, text_area_top + 130]
    
    # --- Define text and shadow colors ---
    optic_yellow = (204, 255, 0, 255)
    medium_grey = (128, 128, 128, 255)
    shadow_fill = (53, 66, 0, 255)
    shadow_offset = 2
    
    # Helper function for drawing text with shadow
    def draw_text_with_shadow(draw_surface, pos, text, font, fill_color, anchor="lm"):
        x, y = pos
        # Draw shadow
        draw_surface.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=shadow_fill, anchor=anchor)
        # Draw main text
        draw_surface.text((x, y), text, font=font, fill=fill_color, anchor=anchor)

    # --- Draw Line 1: Player Names (Yellow) and Verb (Grey) ---
    y1 = y_positions[0]
    if winner == 'Tie':
        part1, part2, part3 = team1, f' {verb} ', team2
    elif winner == 'Team 1':
        part1, part2, part3 = team1, f' {verb} ', team2
    else: # Team 2 won
        part1, part2, part3 = team2, f' {verb} ', team1

    width1 = draw.textlength(part1, font=font)
    width2 = draw.textlength(part2, font=font)
    total_width = width1 + width2 + draw.textlength(part3, font=font)
    
    current_x = x_center - (total_width / 2)

    draw_text_with_shadow(draw, (current_x, y1), part1, font, optic_yellow, anchor="lm")
    current_x += width1
    draw_text_with_shadow(draw, (current_x, y1), part2, font, medium_grey, anchor="lm")
    current_x += width2
    draw_text_with_shadow(draw, (current_x, y1), part3, font, optic_yellow, anchor="lm")

    # --- Draw Line 2: Set Scores (Yellow) ---
    y2 = y_positions[1]
    # For single-color lines, we can still center it directly
    draw.text((x_center + shadow_offset, y2 + shadow_offset), set_text, font=font, fill=shadow_fill, anchor="mm")
    draw.text((x_center, y2), set_text, font=font, fill=optic_yellow, anchor="mm")

    # --- Draw Line 3: GDA & Date (Labels Grey, Values Yellow) ---
    y3 = y_positions[2]
    gda_label, gda_value = "GDA: ", f"{gda:.2f}"
    date_label, date_value = " | Date: ", date_str
    
    gda_label_w = draw.textlength(gda_label, font=font)
    gda_value_w = draw.textlength(gda_value, font=font)
    date_label_w = draw.textlength(date_label, font=font)
    total_width_line3 = gda_label_w + gda_value_w + date_label_w + draw.textlength(date_value, font=font)

    current_x = x_center - (total_width_line3 / 2)

    draw_text_with_shadow(draw, (current_x, y3), gda_label, font, medium_grey, anchor="lm")
    current_x += gda_label_w
    draw_text_with_shadow(draw, (current_x, y3), gda_value, font, optic_yellow, anchor="lm")
    current_x += gda_value_w
    draw_text_with_shadow(draw, (current_x, y3), date_label, font, medium_grey, anchor="lm")
    current_x += date_label_w
    draw_text_with_shadow(draw, (current_x, y3), date_value, font, optic_yellow, anchor="lm")
    
    # =========================================================================
    # --- END: MODIFIED TEXT DRAWING SECTION ---
    # =========================================================================
    
    polaroid_img = polaroid_img.convert('RGB')
    
    buf = io.BytesIO()
    polaroid_img.save(buf, format='JPEG')
    buf.seek(0)
    return buf.getvalue()













#----------------------HALL OF FAME FUNCTION ---------------------------------------------




def display_hall_of_fame():
    """
    Fetches and displays detailed Hall of Fame data from Supabase.
    This version uses min-height to allow cards to dynamically resize.
    """
    st.header("🏆 Hall of Fame")

    try:
        response = supabase.table(hall_of_fame_table_name).select("*").order("Season", desc=True).order("Rank", desc=False).execute()
        hof_data = response.data

        if not hof_data:
            st.info("The Hall of Fame is still empty. Add some top players from past seasons!")
            return

        # Using a set for faster unique lookups
        seasons = sorted(list(set(p['Season'] for p in hof_data)), reverse=True)

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

if not matches.empty and ("match_id" not in matches.columns or matches["match_id"].isnull().any()):
    matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
    for i in matches.index:
        if pd.isna(matches.at[i, "match_id"]):
            match_date_for_id = matches.at[i, "date"] if pd.notna(matches.at[i, "date"]) else datetime.now()
            matches.at[i, "match_id"] = generate_match_id(matches, match_date_for_id)
    save_matches(matches)

st.image("https://raw.githubusercontent.com/mahadevbk/mmd/main/mmdheader.png", width='stretch')

tab_names = ["Rankings", "Matches", "Player Profile", "Maps", "Bookings","Hall of Fame","Mini Tourney"]

tabs = st.tabs(tab_names)


#-------------START OF TABS -----------------------------------------------------------------







with tabs[0]:
    st.header(f"Rankings as of {datetime.now().strftime('%d %b %Y')}")
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
                <span class="games-won-col" style="display: block;">Games Won: {int(player_data['Games Won'])}</span>
                <span class="game-diff-avg-col" style="display: block;">Game Diff Avg: {player_data['Game Diff Avg']:.2f}</span>
                <span class="cumulative-game-diff-col" style="display: block;">Cumulative Game Diff: {int(player_data['Cumulative Game Diff'])}</span>
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
            st.header("Stats for Season Q3 2025 (Jul - Aug)")

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
                border-radius: 50%;
                border: 1px solid #fff500;
                transition: transform 0.2s;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4), 0 0 10px rgba(255, 245, 0, 0.6);
                margin-bottom: 10px;
                object-fit: cover;
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
                                with st.spinner("Uploading match to Supabase..."):
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
                                    st.session_state.matches_df = pd.concat([st.session_state.matches_df, pd.DataFrame([new_match])], ignore_index=True)
                                    save_matches(st.session_state.matches_df)
                                    st.success(f"Match {match_id} posted successfully!")
                                    st.session_state.form_key_suffix += 1
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Failed to add match: {str(e)}")
                                if match_id in st.session_state.matches_df["match_id"].values:
                                    st.session_state.matches_df = st.session_state.matches_df.drop(
                                        st.session_state.matches_df[st.session_state.matches_df["match_id"] == match_id].index
                                    )
                                st.rerun()
        
        st.markdown("*Required fields", unsafe_allow_html=True)
    
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
                        display_matches.at[idx, 'Match Type'] = 'Doubles Match'
        else:
            display_matches = pd.DataFrame()
    else:
        display_matches = pd.DataFrame()

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

    def format_match_scores_and_date(row):
        score_parts_plain = []
        for s in [row['set1'], row['set2'], row['set3']]:
            if s:
                if "Tie Break" in s:
                    tie_break_scores = s.replace("Tie Break", "").strip().split('-')
                    if len(tie_break_scores) == 2 and tie_break_scores[0].isdigit() and tie_break_scores[1].isdigit():
                        if int(tie_break_scores[0]) > int(tie_break_scores[1]):
                            score_parts_plain.append(f"7-6({s})")
                        else:
                            score_parts_plain.append(f"6-7({s})")
                    else:
                        score_parts_plain.append(s)
                else:
                    score_parts_plain.append(s)

        score_text = ", ".join(score_parts_plain)
        _, gda = get_match_verb_and_gda(row)
        gda_text = f"GDA: {gda:.2f}"
        score_parts_html = [f"<span style='font-weight:bold; color:#fff500;'>{s}</span>" for s in score_parts_plain]
        score_html = ", ".join(score_parts_html)
        gda_html = f"<span style='font-weight:bold; color:#fff500;'>{gda_text}</span>"
        
        if pd.notna(row['date']):
            date_str = row['date'].strftime('%A, %d %b')
        else:
            date_str = "Invalid Date"
            
        return f"<div style='font-family: monospace; white-space: pre;'>{score_html} | {gda_html}<br>{date_str}</div>"

    def create_whatsapp_share_link(row):
        verb, gda = get_match_verb_and_gda(row)
        scores = ", ".join([s for s in [row['set1'], row['set2'], row['set3']] if s])
        if row['match_type'] == "Singles":
            players_text = f"{row['team1_player1']} vs {row['team2_player1']}"
        else:
            players_text = f"{row['team1_player1']} & {row['team1_player2']} vs {row['team2_player1']} & {row['team2_player2']}"
        date_str = row['date'].strftime('%A, %d %b %Y') if pd.notna(row['date']) else "Unknown Date"
        message = (
            f"{row['Match Type']} on {date_str}\n"
            f"{players_text}\n"
            f"Result: {row['winner'].replace('Team 1', row['team1_player1'] + (' & ' + row['team1_player2'] if row['team1_player2'] else '')).replace('Team 2', row['team2_player1'] + (' & ' + row['team2_player2'] if row['team2_player2'] else ''))}\n"
            f"Scores: {scores}\n"
            f"GDA: {gda:.2f}"
        )
        encoded_message = urllib.parse.quote(message)
        return f"https://wa.me/?text={encoded_message}"

    # Updated match history display
    #if display_matches.empty:
    #    st.info("No matches found for the selected filters.")
    #else:
    #    # Ensure serial_number is present
    #    if 'serial_number' not in display_matches.columns:
    #        display_matches['serial_number'] = range(1, len(display_matches) + 1)
    #    
    #    for idx, row in display_matches.iterrows():
    #        with st.container():
    #            # Create columns for layout
    #            col1, col2, col3 = st.columns([1, 3, 2])
    #            
    #            with col1:
    #                st.markdown(f"**Match #{row['serial_number']}**")
    #                st.markdown(f"**{row['Match Type']}**")
    #            
    #            with col2:
    #                st.markdown(format_match_players(row), unsafe_allow_html=True)
    #                st.markdown(format_match_scores_and_date(row), unsafe_allow_html=True)
    #            
    #            with col3:
    #                if pd.notna(row.get('match_image_url')) and row['match_image_url']:
    #                    st.image(row['match_image_url'], width=150)
    #                share_link = create_whatsapp_share_link(row)
    #                st.markdown(f'<a href="{share_link}" target="_blank">Share on WhatsApp</a>', unsafe_allow_html=True)
    #            
    #            st.markdown("---")

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
                        # Add the match card download button without caching
                        card_key = f"download_match_card_{row['match_id']}_{idx}"
                        card_bytes = generate_match_card(pd.Series(row.to_dict()), match_image_url)
                        st.download_button(
                            label="📇",
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
            with cols[3]:
                share_link = generate_whatsapp_link(row)
                st.markdown(f'<a href="{share_link}" target="_blank" style="text-decoration:none; color:#ffffff;"><img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg" alt="WhatsApp Share" style="width:30px;height:30px;"/></a>', unsafe_allow_html=True)
            st.markdown("<hr style='border-top: 1px solid #333333; margin: 10px 0;'>", unsafe_allow_html=True)
            st.markdown(f"**{row['Match Type']}**")
            


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
                if match_type_edit == "Doubles":
                    col1, col2 = st.columns(2)
                    with col1:
                        t1p1_edit = st.selectbox(
                            "Team 1 - Player 1 *",
                            [""] + available_players,
                            index=available_players.index(match_row["team1_player1"]) + 1 if match_row["team1_player1"] in available_players else 0,
                            key=f"edit_t1p1_{match_id}"
                        )
                        t1p2_edit = st.selectbox(
                            "Team 1 - Player 2 *",
                            [""] + available_players,
                            index=available_players.index(match_row["team1_player2"]) + 1 if match_row["team1_player2"] in available_players else 0,
                            key=f"edit_t1p2_{match_id}"
                        )
                    with col2:
                        t1p2_edit = st.selectbox(
                            "Team 2 - Player 1 *",
                            [""] + available_players,
                            index=available_players.index(match_row["team2_player1"]) + 1 if match_row["team2_player1"] in available_players else 0,
                            key=f"edit_t2p1_{match_id}"
                        )
                        t2p2_edit = st.selectbox(
                            "Team 2 - Player 2 *",
                            [""] + available_players,
                            index=available_players.index(match_row["team2_player2"]) + 1 if match_row["team2_player2"] in available_players else 0,
                            key=f"edit_t2p2_{match_id}"
                        )
                else:
                    t1p1_edit = st.selectbox(
                        "Player 1 *",
                        [""] + available_players,
                        index=available_players.index(match_row["team1_player1"]) + 1 if match_row["team1_player1"] in available_players else 0,
                        key=f"edit_s1p1_{match_id}"
                    )
                    t2p1_edit = st.selectbox(
                        "Player 2 *",
                        [""] + available_players,
                        index=available_players.index(match_row["team2_player1"]) + 1 if match_row["team2_player1"] in available_players else 0,
                        key=f"edit_s1p2_{match_id}"
                    )
                    t1p2_edit = ""
                    t2p2_edit = ""

                set1_edit = st.selectbox(
                    "Set 1 Score *",
                    [""] + tennis_scores(),
                    index=tennis_scores().index(match_row["set1"]) + 1 if match_row["set1"] in tennis_scores() else 0,
                    key=f"edit_set1_{match_id}"
                )
                set2_edit = st.selectbox(
                    "Set 2 Score (optional)",
                    [""] + tennis_scores(),
                    index=tennis_scores().index(match_row["set2"]) + 1 if match_row["set2"] in tennis_scores() else 0,
                    key=f"edit_set2_{match_id}"
                )
                set3_edit = st.selectbox(
                    "Set 3 Score (optional)",
                    [""] + tennis_scores(),
                    index=tennis_scores().index(match_row["set3"]) + 1 if match_row["set3"] in tennis_scores() else 0,
                    key=f"edit_set3_{match_id}"
                )
                winner_edit = st.radio(
                    "Winner *",
                    ["Team 1", "Team 2", "Tie"],
                    index=["Team 1", "Team 2", "Tie"].index(match_row["winner"]) if match_row["winner"] in ["Team 1", "Team 2", "Tie"] else 0,
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
                            plain_suggestion = "\n*Pairing Combos and Odds:*\n"
                            for idx, pairing in enumerate(all_pairings[:3], 1):
                                pairing_suggestion += (
                                    f"<div>Option {idx}: {pairing['pairing']} "
                                    f"(<span style='font-weight:bold; color:#fff500;'>{pairing['team1_odds']:.1f}%</span> vs "
                                    f"<span style='font-weight:bold; color:#fff500;'>{pairing['team2_odds']:.1f}%</span>)</div>"
                                )
                                plain_suggestion += (
                                    f"Option {idx}: {pairing['plain_pairing']} ({pairing['team1_odds']:.1f}% vs {pairing['team2_odds']:.1f}%)\n"
                                )
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
                players_list = "\n".join([f"{i+1}. *{p}*" for i, p in enumerate(players)]) if players else "No players"
                standby_text = f"\nSTD. BY: *{row['standby_player']}*" if row['standby_player'] else ""
                
                share_text = f"*Game Booking:*\nDate: *{full_date}*\nCourt: *{court_name}*\nPlayers:\n{players_list}{standby_text}\n{plain_suggestion}\nCourt location: {court_url}"
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
    st.markdown("<small><i>Assignments of teams and courts are done randomly.</i></small>", unsafe_allow_html=True)

    # Input fields
    st.subheader("Tournament Setup")
    tournament_name = st.text_input("Enter Tournament Name")
    num_teams = st.number_input("Enter number of teams", min_value=2, step=1)
    num_courts = st.number_input("Enter number of courts", min_value=1, step=1)
    enter_names = st.radio("Do you want to enter team names?", ("No", "Yes"))
    enter_court_names = st.radio("Do you want to enter court names?", ("No", "Yes"))

    # Collect team names early, depending on radio selection
    team_names = []
    if num_teams and enter_names == "Yes":
        st.subheader("Enter Team Names")
        if num_teams <= 8:
            cols = st.columns(2)
        elif num_teams <= 16:
            cols = st.columns(3)
        else:
            cols = st.columns(4)

        for i in range(num_teams):
            col = cols[i % len(cols)]
            with col:
                name = st.text_input(f"Team {i+1} Name", key=f"team_{i}")
                team_names.append(name if name else f"Team {i+1}")
    else:
        team_names = [f"Team {i+1}" for i in range(num_teams)]

    # Optional court names
    court_names = []
    if num_courts and enter_court_names == "Yes":
        st.subheader("Enter Court Names")
        for i in range(num_courts):
            key = f"court_name_{i}"
            if key not in st.session_state:
                st.session_state[key] = f"Court {i+1}"
            name = st.text_input(f"Court {i+1} Name", key=key)
            court_names.append(name)
    else:
        court_names = [f"Court {i+1}" for i in range(num_courts)]

    # Optional tournament rules input
    rules = st.text_area("Enter Tournament Rules (optional, supports rich text)")

    if num_teams % 2 != 0:
        st.warning("Number of teams is odd. Consider adding one more team for even distribution.")

    if st.button("Organise Tournament"):
        random.shuffle(team_names)

        base = len(team_names) // num_courts
        extras = len(team_names) % num_courts

        courts = []
        idx = 0
        for i in range(num_courts):
            num = base + (1 if i < extras else 0)
            if num % 2 != 0:
                if i < num_courts - 1:
                    num += 1
            court_teams = team_names[idx:idx+num]
            courts.append((court_names[i], court_teams))
            idx += num

        st.markdown("---")
        st.subheader("Court Assignments")

        # Dynamic court layout with styled boxes matching ar.py colors
        primary_color = "#07314f"  # From ar.py gradient
        accent_color = "#fff500"   # Optic yellow from ar.py

        if len(courts) <= 4:
            num_cols = len(courts)
        elif len(courts) <= 8:
            num_cols = 4
        elif len(courts) <= 12:
            num_cols = 3
        else:
            num_cols = 2

        for i in range(0, len(courts), num_cols):
            row = st.columns(num_cols)
            for j, court in enumerate(courts[i:i + num_cols]):
                with row[j]:
                    court_name, teams = court
                    st.markdown(
                        f"""
                        <div style='border: 2px solid {accent_color}; border-radius: 12px; padding: 15px; margin: 10px 0; background-color: {primary_color}; color: white;'>
                            <img src='court.png' width='100%' style='border-radius: 8px;' />
                            <h4 style='text-align:center; color:{accent_color};'>{court_name}</h4>
                            <ul>{''.join(f'<li><b style="color:{accent_color};">{team}</b></li>' for team in teams)}</ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        if rules:
            st.subheader("Tournament Rules")
            st.markdown(rules, unsafe_allow_html=True)

        # PDF Generation
        def generate_pdf(tournament_name, courts, rules):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, tournament_name, ln=True, align="C")
            pdf.ln(10)

            pdf.set_font("Arial", '', 12)
            for court_name, teams in courts:
                pdf.set_text_color(7, 49, 79)  # RGB for #07314f
                pdf.cell(0, 10, court_name, ln=True)
                pdf.set_text_color(0, 0, 0)
                for team in teams:
                    pdf.cell(10)
                    pdf.cell(0, 10, f"- {team}", ln=True)
                pdf.ln(2)

            if rules:
                pdf.ln(5)
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "Tournament Rules", ln=True)
                pdf.set_font("Arial", '', 11)
                for line in rules.splitlines():
                    pdf.multi_cell(0, 8, line)

            return pdf.output(dest='S').encode('latin-1')

        pdf_bytes = generate_pdf(tournament_name, courts, rules)
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name=f"{tournament_name or 'tournament'}.pdf",
            mime='application/pdf'
        )
        #----MINI TOURNEY-------


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
    backup_key = f"backup_download_{current_time}_{random.randint(1, 1000)}"
    st.download_button(
        label="Download Backup ZIP",
        data=zip_buffer.getvalue(),  # Use getvalue() to avoid buffer issues
        file_name=f"ar-tennis-data-{current_time}.zip",
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
