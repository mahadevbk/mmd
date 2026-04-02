import time
import streamlit as st
from supabase import create_client, Client
from datetime import datetime, timedelta, timezone
import pandas as pd
import zipfile
import io
import random
from postgrest.exceptions import APIError 
# NEW IMPORT: For browser-side storage execution
from streamlit_javascript import st_javascript

# Set page configuration to wide mode by default
st.set_page_config(
    page_title="Mira Court Booking",
    page_icon="🎾",
    layout="wide",
)

# --- DATABASE SETUP (SUPABASE) ---
@st.cache_resource
def init_supabase():
    url: str = st.secrets["SUPABASE_URL"]
    key: str = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase: Client = init_supabase()

# Constants
sub_community_list = [
    "Mira 1", "Mira 2", "Mira 3", "Mira 4", "Mira 5",
    "Mira Oasis 1", "Mira Oasis 2", "Mira Oasis 3"
]

courts = ["Mira 2", "Mira 4", "Mira 5A", "Mira 5B", "Mira Oasis 1", "Mira Oasis 2", "Mira Oasis 3A", "Mira Oasis 3B", "Mira Oasis 3C"]

def get_start_hours_for_date(date_str):
    """Returns the list of start hours for a given date.
    Before or on 2026-03-22: 7 AM to 12 AM (start hours 7 to 23).
    After 2026-03-22: 7 AM to 10 PM (start hours 7 to 21).
    """
    if date_str <= "2026-03-22":
        return list(range(7, 24))
    return list(range(7, 22))

# --- HELPER FUNCTIONS ---

def get_utc_plus_4():
    return datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=4)

def get_today():
    return get_utc_plus_4().date()

def get_next_14_days():
    today = get_today()
    return [today + timedelta(days=i) for i in range(15)]

def run_query(query_method):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return query_method.execute()
        except APIError as e:
            if e.code == "23505":
                raise e
            if attempt == max_retries - 1:
                raise e
            time.sleep((0.5 * (2 ** attempt)) + random.uniform(0, 0.2))
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"⚠️ Connection Error: {str(e)}")
                return None
            time.sleep((0.5 * (2 ** attempt)) + random.uniform(0, 0.2))

def add_log(event_type, details):
    timestamp = get_utc_plus_4().isoformat()
    # If fingerprint is available in session state, append it to details for locking
    fp = st.session_state.get('client_fp')
    if fp and fp != 0 and fp != 'unknown':
        details = f"⟦FP:{fp}⟧ {details}"
    try:
        supabase.table("logs").insert({
            "timestamp": timestamp,
            "event_type": event_type,
            "details": details
        }).execute()
    except:
        pass 

def get_global_reset_ts():
    """Returns the timestamp of the latest Global Reset event."""
    try:
        response = supabase.table("logs").select("timestamp").eq("event_type", "Global Reset").order("timestamp", desc=True).limit(1).execute()
        if response and response.data:
            return response.data[0]['timestamp']
    except:
        pass
    return "1970-01-01T00:00:00"

def check_device_lock(current_villa, current_sub):
    """Checks if current fingerprint is already associated with a different villa in logs."""
    fp = st.session_state.get('client_fp')
    
    if not fp or fp == 0 or fp == 'unknown': return None
    
    now = get_utc_plus_4()
    global_reset = get_global_reset_ts()
    # Fingerprint lock is persistent (90 days)
    fp_cutoff = max((now - timedelta(days=90)).isoformat(), global_reset)

    response = run_query(
        supabase.table("logs")
        .select("timestamp, event_type, details")
        .ilike("details", f"%⟦FP:{fp}⟧%")
        .order("timestamp", desc=True)
        .limit(100)
    )
    
    if not response or not response.data: return None

    for log in response.data:
        ts = log['timestamp']
        event = log['event_type']
        details = log['details']
        
        if ts < fp_cutoff: continue
        if event == "Global Reset": break
        if event == "Lock Reset" and f"⟦FP:{fp}⟧" in details: break
            
        try:
            # Extract villa info from details
            msg = details.split("⟧", 1)[-1].strip()
            log_sub, log_villa = None, None
            
            # Common patterns in logs
            if " Villa " in msg:
                # e.g. "Mira 1 Villa 101 booked..." or "New login for Mira 1 - Villa 101"
                if " - Villa " in msg:
                    parts = msg.split(" - Villa ")
                    log_sub = parts[0].split("for ")[-1].strip()
                    log_villa = parts[1].split(" ")[0].strip()
                else:
                    parts = msg.split(" Villa ")
                    log_sub = parts[0].split("for ")[-1].strip()
                    log_villa = parts[1].split(" ")[0].strip()
            
            if log_sub in sub_community_list and log_villa:
                # Special group exception: Mira 1 villas 229, 231, 233 are interchangeable
                mira1_group = ["229", "231", "233"]
                if log_sub == "Mira 1" and current_sub == "Mira 1" and log_villa in mira1_group and current_villa in mira1_group:
                    continue
                
                if log_villa != current_villa or log_sub != current_sub:
                    return f"{log_sub} - {log_villa}"
        except: continue
    return None

def get_bookings_for_day_with_details(date_str):
    response = run_query(supabase.table("bookings").select("court, start_hour, sub_community, villa").eq("date", date_str))
    if not response or not response.data: return {}
    return {(row['court'], row['start_hour']): f"{row['sub_community']} - {row['villa']}" for row in response.data}

def abbreviate_community(full_name):
    if full_name.startswith("Mira Oasis"):
        num = full_name.split()[-1]
        return f"MO{num}"
    elif full_name.startswith("Mira"):
        num = full_name.split()[-1]
        return f"M{num}"
    return full_name

def color_cell(val):
    if val == "Available":
        return "background-color: #d4edda; color: #155724; font-weight: bold;"
    elif val == "—":
        return "background-color: #e9ecef; color: #e9ecef; border: none;"
    else:
        return "background-color: #f8d7da; color: #721c24; font-weight: bold;"

def get_active_bookings_count(villa, sub_community):
    today_str = get_today().strftime('%Y-%m-%d')
    now_hour = get_utc_plus_4().hour
    
    # Active bookings are individual per villa (Limit 6)
    q_future = supabase.table("bookings").select("id", count="exact")
    q_future = q_future.eq("villa", villa).eq("sub_community", sub_community)
    
    res_future = run_query(q_future.gt("date", today_str))
    count_future = res_future.count if res_future and res_future.count is not None else 0
    
    # Count today's active bookings (ongoing or later)
    q_today = supabase.table("bookings").select("id", count="exact")
    q_today = q_today.eq("villa", villa).eq("sub_community", sub_community)
    
    res_today = run_query(q_today.eq("date", today_str).gte("start_hour", now_hour))
    count_today = res_today.count if res_today and res_today.count is not None else 0
    
    return count_future + count_today

def get_daily_bookings_count(villa, sub_community, date_str):
    mira1_group = ["229", "231", "233"]
    is_mira1_group = (sub_community == "Mira 1" and villa in mira1_group)

    if is_mira1_group:
        # Rule: Only ONE villa from the group can book per day.
        # Check if anyone ELSE in the group has a booking.
        other_villas = [v for v in mira1_group if v != villa]
        res_others = run_query(supabase.table("bookings").select("id", count="exact").eq("sub_community", "Mira 1").in_("villa", other_villas).eq("date", date_str))
        others_count = res_others.count if res_others and res_others.count is not None else 0
        
        if others_count > 0:
            # If someone else booked, this villa's daily limit is effectively exceeded (return 2 or more)
            return 99 
        
        # If no one else booked, check this villa's own daily count (Limit 2)
        res_self = run_query(supabase.table("bookings").select("id", count="exact").eq("sub_community", "Mira 1").eq("villa", villa).eq("date", date_str))
        return res_self.count if res_self and res_self.count is not None else 0
    else:
        query = supabase.table("bookings").select("id", count="exact")
        query = query.eq("villa", villa).eq("sub_community", sub_community)
        response = run_query(query.eq("date", date_str))
        if response is None or response.count is None: return 99
        return response.count

def is_slot_booked(court, date_str, start_hour):
    response = run_query(
        supabase.table("bookings").select("id")\
        .eq("court", court)\
        .eq("date", date_str)\
        .eq("start_hour", start_hour)
    )
    if not response or not response.data: return False
    return len(response.data) > 0

def is_slot_in_past(date_str, start_hour):
    now = get_utc_plus_4()
    today_str = now.strftime('%Y-%m-%d')
    if date_str < today_str: return True
    if date_str > today_str: return False
    if start_hour < now.hour: return True
    if start_hour == now.hour and now.minute > 0: return True
    return False

def book_slot(villa, sub_community, court, date_str, start_hour):
    try:
        run_query(supabase.table("bookings").insert({
            "villa": villa,
            "sub_community": sub_community,
            "court": court,
            "date": date_str,
            "start_hour": start_hour
        }))
        log_detail = f"{sub_community} Villa {villa} booked {court} for {date_str} at {start_hour:02d}:00"
        add_log("Booking Created", log_detail)
        return True
    except APIError as e:
        if e.code == "23505":
            return False
        raise e
    except Exception:
        return False

def get_user_bookings(villa, sub_community):
    today_str = get_today().strftime('%Y-%m-%d')
    now_hour = get_utc_plus_4().hour
    response = run_query(
        supabase.table("bookings").select("id, court, date, start_hour")\
        .eq("villa", villa)\
        .eq("sub_community", sub_community)\
        .or_(f"date.gt.{today_str},and(date.eq.{today_str},start_hour.gte.{now_hour})")\
        .order("date")\
        .order("start_hour")
    )
    return response.data if response else []

def delete_booking(booking_id, villa, sub_community):
    record = run_query(supabase.table("bookings").select("court, date, start_hour").eq("id", booking_id).single())
    if record and record.data:
        b = record.data
        log_detail = f"{sub_community} Villa {villa} cancelled {b['court']} for {b['date']} at {b['start_hour']:02d}:00"
        add_log("Booking Deleted", log_detail)
    run_query(supabase.table("bookings").delete().eq("id", booking_id).eq("villa", villa).eq("sub_community", sub_community))

def get_logs_last_14_days():
    cutoff = (get_utc_plus_4() - timedelta(days=14)).isoformat()
    response = run_query(
        supabase.table("logs").select("timestamp, event_type, details")\
        .gte("timestamp", cutoff)\
        .order("timestamp", desc=True)
    )
    return response.data if response else []

def get_villas_with_active_bookings():
    today_str = get_today().strftime('%Y-%m-%d')
    now_hour = get_utc_plus_4().hour
    try:
        res_future = run_query(supabase.table("bookings").select("villa, sub_community").gt("date", today_str))
        res_today = run_query(supabase.table("bookings").select("villa, sub_community").eq("date", today_str).gte("start_hour", now_hour))
        
        all_rows = (res_future.data if res_future else []) + (res_today.data if res_today else [])
        unique_villas = sorted(list(set([f"{row['sub_community']} - {row['villa']}" for row in all_rows])))
        return unique_villas
    except Exception:
        return []

def get_all_villas_with_any_bookings():
    response = run_query(supabase.table("bookings").select("villa, sub_community"))
    if not response or not response.data: return []
    unique_villas = sorted(list(set([f"{row['sub_community']} - {row['villa']}" for row in response.data])))
    return unique_villas

def get_bookings_for_villa(villa, sub_community):
    response = run_query(
        supabase.table("bookings").select("id, court, date, start_hour")\
        .eq("villa", villa)\
        .eq("sub_community", sub_community)\
        .order("date", desc=True)\
        .order("start_hour", desc=True)
    )
    return response.data if response else []

def _process_background_tasks():
    try:
        from database_cleanup import run_db_cleanup
        run_db_cleanup(supabase, courts)
    except Exception:
        pass

def get_active_bookings_for_villa_display(villa_identifier):
    try:
        sub_comm, villa_num = villa_identifier.split(" - ")
        today_str = get_today().strftime('%Y-%m-%d')
        now_hour = get_utc_plus_4().hour
        response = run_query(
            supabase.table("bookings").select("court, date, start_hour")\
            .eq("villa", villa_num)\
            .eq("sub_community", sub_comm)\
            .or_(f"date.gt.{today_str},and(date.eq.{today_str},start_hour.gte.{now_hour})")\
            .order("date")\
            .order("start_hour")
        )
        return [f"{b['date']} | {b['start_hour']:02d}:00 | {b['court']}" for b in response.data]
    except Exception:
        return []

def get_peak_time_data():
    response = run_query(supabase.table("bookings").select("date, start_hour"))
    if not response or not response.data: return pd.DataFrame()
    df = pd.DataFrame(response.data)
    if df.empty: return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.day_name()
    return df

def get_available_hours(court, date_str):
    response = run_query(supabase.table("bookings").select("start_hour").eq("court", court).eq("date", date_str))
    if not response or not response.data:
        booked_hours = []
    else:
        booked_hours = [row['start_hour'] for row in response.data]
    available = []
    for h in get_start_hours_for_date(date_str):
        if h not in booked_hours and not is_slot_in_past(date_str, h):
            available.append(h)
    return available

def logout_action():
    """Centrally handles logout, clears session and localStorage."""
    # Use JS to clear localStorage and force a clean reload with a logout flag
    st_javascript("localStorage.removeItem('court_villa_lock'); setTimeout(() => { window.location.href = window.location.origin + window.location.pathname + '?logout=1'; }, 300);")
    for key in ["authenticated", "sub_community", "villa"]:
        if key in st.session_state:
            del st.session_state[key]
    st.info("Logging out... Please wait.")
    time.sleep(1.2)
    st.rerun()

# --- UI STYLING ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Audiowide&display=swap" rel="stylesheet">
<style>
.stApp { background: linear-gradient(to bottom, #010f1a, #052134); background-attachment: scroll; }
[data-testid="stHeader"] { background: linear-gradient(to bottom, #052134 , #010f1a) !important; }
h1, h2, h3, .stTitle { font-family: 'Audiowide', cursive !important; color: #2c3e50; }
.stButton>button { background-color: #4CAF50; color: white; font-family: 'Audiowide', cursive; }
.stDataFrame th { font-family: 'Audiowide', cursive; font-size: 12px; background-color: #2c3e50 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# --- LOGIC FOR FULL FRAME PAGE ---
if st.query_params.get("view") == "full":
    st.title("📅 Full 14-Day Schedule")
    if st.button("⬅️ Back to Booking App"):
        st.query_params.clear()
        st.rerun()
    for d in get_next_14_days():
        d_str = d.strftime('%Y-%m-%d')
        st.subheader(f"{d_str} ({d.strftime('%A')})")
        bookings_with_details = get_bookings_for_day_with_details(d_str)
        data = {}
        for h in get_start_hours_for_date(d_str):
            label = f"{h:02d}:00 - {h+1:02d}:00"
            row = []
            for court in courts:
                key = (court, h)
                if is_slot_in_past(d_str, h): row.append("—")
                elif key in bookings_with_details:
                    full_comm, villa_num = bookings_with_details[key].rsplit(" - ", 1)
                    abbr = abbreviate_community(full_comm)
                    row.append(f"{abbr}-{villa_num}")
                else: row.append("Available")
            data[label] = row
        st.dataframe(pd.DataFrame(data, index=courts).style.map(color_cell), width="stretch")
        st.divider()
    st.stop()

# --- MAIN APP ---
st.subheader("🎾 Book that Court ...")    
st.caption("An Un-Official & Community Driven Booking Solution.")

today_str_check = get_today().strftime('%Y-%m-%d')
if today_str_check <= "2026-03-22":
    st.info("Ramadan Timings 7AM to 12AM slots.")
else:
    st.info("Standard Timings 7AM to 10PM slots.")

try:
    _process_background_tasks()
    villas_active = get_villas_with_active_bookings()
    today_str = get_today().strftime('%Y-%m-%d')
    now_hour = get_utc_plus_4().hour
    
    # Count total active bookings (future and today-active)
    res_f = run_query(supabase.table("bookings").select("id", count="exact").gt("date", today_str))
    res_t = run_query(supabase.table("bookings").select("id", count="exact").eq("date", today_str).gte("start_hour", now_hour))
    
    total_residences = len(villas_active)
    count_f = res_f.count if res_f and res_f.count is not None else 0
    count_t = res_t.count if res_t and res_t.count is not None else 0
    total_bookings = count_f + count_t
    
    st.write(f"**{total_residences}** Residences have **{total_bookings}** active bookings.")
except Exception:
    st.write("Unable to load live stats (Network refreshing...)")
    villas_active = []

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# --- AUTHENTICATION LOGIC ---
if not st.session_state.authenticated:
    # Use query params to detect manual logout
    logout_mode = st.query_params.get("logout") == "1"
    
    # Get fingerprint and existing lock
    # Note: st_javascript returns 0 or None while loading
    stored_lock = st_javascript("localStorage.getItem('court_villa_lock') || 'no_lock';")
    last_reset_seen = st_javascript("localStorage.getItem('court_last_reset_seen') || '1970-01-01T00:00:00';")
    # Improved Fingerprint (Persistent Unique ID + Hardware)
    # Fixes collisions for iPhone 14/Safari users who share identical hardware specs
    client_fp = st_javascript("(function(){var d=localStorage.getItem('court_device_uuid');if(!d){d='D-'+Math.random().toString(36).substring(2,11)+'-'+Date.now();localStorage.setItem('court_device_uuid',d);}var h=btoa([Intl.DateTimeFormat().resolvedOptions().timeZone,screen.width+'x'+screen.height,navigator.hardwareConcurrency||'',navigator.deviceMemory||'',navigator.maxTouchPoints||'',navigator.platform,navigator.language,navigator.userAgent].join('|'));return d+':'+h;})()")
    
    # Wait for signals to stabilize (prevent loop while st_javascript is 0)
    if stored_lock == 0 or last_reset_seen == 0 or client_fp == 0:
        st.markdown("### 🔒 Security Check...")
        st.info("Verifying device identity. Please wait...")
        st.stop()

    # Fetch latest global reset timestamp from DB
    global_reset_ts = get_global_reset_ts()

    # Store in session state
    if client_fp and client_fp != 0: st.session_state.client_fp = client_fp

    # 1. Global Reset Enforcement (with session guard to prevent loops)
    if 'last_reset_triggered' not in st.session_state:
        st.session_state.last_reset_triggered = None

    if last_reset_seen and last_reset_seen != 'no_lock':
        if str(global_reset_ts) > str(last_reset_seen) and st.session_state.last_reset_triggered != global_reset_ts:
            st.session_state.last_reset_triggered = global_reset_ts
            st_javascript(f"localStorage.removeItem('court_villa_lock'); localStorage.setItem('court_last_reset_seen', '{global_reset_ts}');")
            st.rerun()

    # 2. Primary Auto-Login (localStorage)
    if not logout_mode and stored_lock and stored_lock != "no_lock":
        try:
            locked_sub, locked_villa = stored_lock.split("-")
            st.session_state.sub_community, st.session_state.villa = locked_sub, locked_villa
            st.session_state.authenticated = True
            st.rerun()
        except:
            st_javascript("localStorage.removeItem('court_villa_lock');")
            st.rerun()

    # 3. Registration Form
    st.subheader("Villa Login")
    st.info("First-time login will lock this device to your villa.")
    
    col1, col2 = st.columns(2)
    with col1:
        sub_community_input = st.selectbox("Select Your Sub-Community", options=sub_community_list, index=None)
    with col2:
        # Filter input to only allow digits for villa number
        villa_input_raw = st.text_input("Enter Villa Number").strip()
        villa_input = "".join(filter(str.isdigit, villa_input_raw))

    if st.button("Login", type="primary", use_container_width=True):
        if not sub_community_input or not villa_input:
            if villa_input_raw and not villa_input:
                st.error("Please enter a numeric villa number (e.g. 255).")
            else:
                st.error("Please select a sub-community and enter your villa number.")
        else:
            # Check if FP is loaded
            fp = st.session_state.get('client_fp')
            if not fp or fp == 0 or fp == 'unknown':
                st.warning("⚠️ Verifying device security. Please wait a moment and try again.")
            else:
                # Check for existing lock in logs
                owner = check_device_lock(villa_input, sub_community_input)
                if owner:
                    st.error(f"🚫 Access Denied: This device is already associated with **{owner}**. Switching villas is not permitted.")
                    st.info(f"🆔 **Your Device ID:** `{fp}`\n\nIf you think this is an error, copy this Device ID and send it to dev for a device reset.")
                    add_log("Access Denied", f"Login blocked for Villa ({sub_community_input} - {villa_input}): Already locked to {owner}")
                else:
                    current_choice = f"{sub_community_input}-{villa_input}"
                    st_javascript(f"localStorage.setItem('court_villa_lock', '{current_choice}'); localStorage.setItem('court_last_reset_seen', '{global_reset_ts}');")
                    st.session_state.sub_community, st.session_state.villa = sub_community_input, villa_input
                    st.session_state.authenticated = True
                    # Clear query params to remove ?logout=1 if it exists
                    st.query_params.clear()
                    add_log("Device Registered", f"New login for {sub_community_input} - Villa {villa_input}")
                    st.rerun()
    
    st.write("")
    if st.button("🚪 Reset / Change Villa", use_container_width=True, key="reg_logout"):
        logout_action()
    
    st.stop()

sub_community, villa = st.session_state.sub_community, st.session_state.villa
st.success(f"✅ Logged in as: **{sub_community} - Villa {villa}**")

tab1, tab2, tab3, tab4 = st.tabs(["📅 Availability", "➕ Book", "📋 My Bookings", "📜 Activity Log"])

with tab1:
    st.subheader("Court Availability")
    date_options = [f"{d.strftime('%Y-%m-%d')} ({d.strftime('%A')})" for d in get_next_14_days()]
    selected_date_full = st.selectbox("Select Date:", date_options)
    selected_date = selected_date_full.split(" (")[0]
    bookings_with_details = get_bookings_for_day_with_details(selected_date)
    data = {}
    for h in get_start_hours_for_date(selected_date):
        label = f"{h:02d}:00 - {h+1:02d}:00"
        row = []
        for court in courts:
            key = (court, h)
            if is_slot_in_past(selected_date, h): row.append("—")
            elif key in bookings_with_details:
                full_comm, villa_num = bookings_with_details[key].rsplit(" - ", 1)
                row.append(f"{abbreviate_community(full_comm)}-{villa_num}")
            else: row.append("Available")
        data[label] = row
    st.dataframe(pd.DataFrame(data, index=courts).style.map(color_cell), width="stretch")
    st.link_button("🌐 View Full 14-Day Schedule (Full Page)", url="/?view=full")
    
    st.divider()
    st.markdown("### ⚡ Quick Book")
    q_col1, q_col2, q_col3, q_col4 = st.columns([2, 2, 2, 2])
    with q_col1: q_court = st.selectbox("Select Court", options=courts, key="q_court_select")
    with q_col2:
        q_free_hours = get_available_hours(q_court, selected_date)
        if not q_free_hours:
            st.warning("No slots available"); q_time = None
        else:
            q_time_options = [f"{h:02d}:00" for h in q_free_hours]
            q_time = st.selectbox("Select Time", options=q_time_options, key="q_time_select")
    with q_col3:
        st.write(""); st.write("") 
        q_2_hours = st.checkbox("Book for 2 hours", key="q_2_hours_check")
        q_slots = 2 if q_2_hours else 1
    with q_col4:
        st.write(""); st.write("") 
        if st.button("🚀 Book Now", key="q_book_btn", use_container_width=True):
            if q_time:
                active_count = get_active_bookings_count(villa, sub_community)
                daily_count = get_daily_bookings_count(villa, sub_community, selected_date)
                
                start_h = int(q_time.split(":")[0])
                slots_to_book = list(range(start_h, start_h + q_slots))
                valid_hours = get_start_hours_for_date(selected_date)
                
                # Check availability and limits for all requested slots
                unavailable = []
                for h in slots_to_book:
                    if h not in valid_hours or is_slot_booked(q_court, selected_date, h) or is_slot_in_past(selected_date, h):
                        unavailable.append(f"{h:02d}:00")
                
                if unavailable:
                    st.error(f"Slot(s) {', '.join(unavailable)} are unavailable.")
                elif active_count + q_slots > 6:
                    st.error(f"Limit Reached (Max 6 active). You can book {max(0, 6-active_count)} more.")
                    add_log("Access Denied", f"{sub_community} Villa {villa} reached active booking limit (6)")
                elif daily_count + q_slots > 2:
                    st.error(f"Daily Limit Reached (Max 2 per day). You can book {max(0, 2-daily_count)} more today.")
                    add_log("Access Denied", f"{sub_community} Villa {villa} reached daily limit (2) for {selected_date}")
                else:
                    success = True
                    booked_slots = []
                    for h in slots_to_book:
                        if book_slot(villa, sub_community, q_court, selected_date, h):
                            booked_slots.append(h)
                        else:
                            success = False
                            break
                    
                    if success:
                        st.balloons()
                        st.success(f"Booked {q_slots} slot(s) for {q_court} starting at {q_time}")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("❌ One or more slots were taken! Please refresh.")

    st.divider()
    st.subheader("📊 Community Usage Insights")
    usage_data = get_peak_time_data()
    if not usage_data.empty:
        col_charts1, col_charts2 = st.columns([1, 1])
        with col_charts1:
            st.write("**🔥 Busiest Hours**")
            hour_counts = usage_data['start_hour'].value_counts().sort_index()
            chart_df = pd.DataFrame({"Bookings": hour_counts.values}, index=[f"{h:02d}:00" for h in hour_counts.index])
            st.bar_chart(chart_df, color="#4CAF50")
        with col_charts2:
            st.write("**📅 Busiest Days**")
            day_counts = usage_data['day_of_week'].value_counts()
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_counts = day_counts.reindex(days_order).fillna(0)
            st.area_chart(day_counts, color="#0d5384")
        st.write("**Weekly Intensity Heatmap**")
        heatmap_data = usage_data.groupby(['day_of_week', 'start_hour']).size().unstack(fill_value=0)
        heatmap_data = heatmap_data.reindex(days_order).fillna(0)
        try:
            st.dataframe(heatmap_data.style.background_gradient(cmap="YlGnBu"), width="stretch")
        except Exception:
            # Fallback if matplotlib/gradient fails
            st.dataframe(heatmap_data, width="stretch")
    else: st.info("Charts will appear here once more bookings are made!")

    st.divider()
    st.subheader("🔍 Booking Lookup")
    if villas_active:
        look_villa = st.selectbox("Select Villa to see details:", options=["-- Select --"] + villas_active)
        if look_villa != "-- Select --":
            active_list = get_active_bookings_for_villa_display(look_villa)
            if active_list: st.selectbox("Active bookings for this villa:", options=active_list)
            else: st.write("No active bookings found for this villa.")

    st.divider()
    if st.button("🚪 Logout / Change Villa", use_container_width=True, key="tab1_logout"):
        logout_action()

with tab2:
    st.subheader("Book a New Slot")
    
    date_options = [f"{d.strftime('%Y-%m-%d')} ({d.strftime('%A')})" for d in get_next_14_days()]
    selected_date_full = st.selectbox("Date:", date_options)
    date_choice = selected_date_full.split(" (")[0]
    
    # Dynamic timing info for the selected date
    if date_choice <= "2026-03-22":
        timing_msg = "7AM to 12AM slots."
    else:
        timing_msg = "7AM to 10PM slots."
    
    st.info(f"App allows 6 Active bookings spanning 14 days, A maximum of 2 active bookings per day. Current date choice timing: **{timing_msg}**")
    
    court_choice = st.selectbox("Court:", courts)
    free_hours = get_available_hours(court_choice, date_choice)
    if not free_hours:
        st.warning(f"😔 Sorry, no slots available for {court_choice} on {date_choice}."); time_choice = None
    else:
        time_options = [f"{h:02d}:00 - {h+1:02d}:00" for h in free_hours]
        time_choice = st.selectbox("Time Slot:", time_options)
    
    slots_2_hours = st.checkbox("Book for 2 hours", key="tab2_slots_2_hours")
    slots_choice = 2 if slots_2_hours else 1

    active_count = get_active_bookings_count(villa, sub_community)
    daily_count = get_daily_bookings_count(villa, sub_community, date_choice)
    col_status1, col_status2 = st.columns(2)
    with col_status1: st.info(f"Total active bookings: **{active_count} / 6**")
    with col_status2: st.info(f"Bookings for {date_choice}: **{daily_count} / 2**")
    
    if st.button("Book This Slot", type="primary"):
        # RE-CALCULATE latest counts to prevent stale limit issues
        active_count_latest = get_active_bookings_count(villa, sub_community)
        daily_count_latest = get_daily_bookings_count(villa, sub_community, date_choice)
        
        if not time_choice:
            st.error("Please select an available time slot.")
        else:
            start_h = int(time_choice.split(":")[0])
            slots_to_book = list(range(start_h, start_h + slots_choice))
            valid_hours = get_start_hours_for_date(date_choice)

            # Check availability and limits
            unavailable = []
            for h in slots_to_book:
                if h not in valid_hours or is_slot_booked(court_choice, date_choice, h) or is_slot_in_past(date_choice, h):
                    unavailable.append(f"{h:02d}:00")

            if unavailable:
                st.error(f"Slot(s) {', '.join(unavailable)} are unavailable.")
            elif active_count_latest + slots_choice > 6: 
                st.error(f"🚫 Overall limit reached. You can book {max(0, 6-active_count_latest)} more slots.")
                add_log("Access Denied", f"{sub_community} Villa {villa} reached active booking limit (6)")
            elif daily_count_latest + slots_choice > 2:
                st.error(f"🚫 Daily limit reached. You can book {max(0, 2-daily_count_latest)} more on {date_choice}.")
                add_log("Access Denied", f"{sub_community} Villa {villa} reached daily limit (2) for {date_choice}")
            else:
                success = True
                booked_slots = []
                for h in slots_to_book:
                    if book_slot(villa, sub_community, court_choice, date_choice, h):
                        booked_slots.append(h)
                    else:
                        success = False
                        break
                
                if success:
                    st.balloons()
                    st.success(f"✅ SUCCESS! {court_choice} booked for {date_choice} starting at {start_h:02d}:00 ({slots_choice} slot(s))")
                    time.sleep(2.5) 
                    st.rerun()
                else:
                    st.error("❌ One or more slots were taken! Please refresh.")

with tab3:
    st.subheader("📋 My Bookings")
    court_locations = {
        "Mira 2": "https://maps.google.com/?q=25.003702,55.306740",
        "Mira 4": "https://maps.google.com/?q=25.010338,55.305798",
        "Mira 5A": "https://maps.google.com/?q=25.007513,55.303432",
        "Mira 5B": "https://maps.google.com/?q=25.007513,55.303432",
        "Mira Oasis 1": "https://maps.google.com/?q=25.010536,55.296654",
        "Mira Oasis 2": "https://maps.google.com/?q=25.016439,55.298626",
        "Mira Oasis 3A": "https://maps.google.com/?q=25.012520,55.298313",
        "Mira Oasis 3B": "https://maps.google.com/?q=25.012520,55.298313",
        "Mira Oasis 3C": "https://maps.google.com/?q=25.015327,55.301998"
    }
    if sub_community == "Mira 1" and villa in ["229", "231", "233"]:
        my_b = []
        for v_num in ["229", "231", "233"]:
            vb = get_user_bookings(v_num, "Mira 1")
            for b in vb: b['orig_v'] = v_num; b['orig_sc'] = "Mira 1"
            my_b.extend(vb)
        limit_val = 6 # Individual limit for Mira 1 villas
    else:
        my_b = get_user_bookings(villa, sub_community)
        for b in my_b: b['orig_v'] = villa; b['orig_sc'] = sub_community
        limit_val = 6

    # --- Summary Section ---
    today_str = get_today().strftime('%Y-%m-%d')
    total_active = len(my_b)
    today_bookings = len([b for b in my_b if b['date'] == today_str])

    col_sum1, col_sum2 = st.columns(2)
    with col_sum1:
        st.metric("Total Active Bookings", f"{total_active} / {limit_val}")
    with col_sum2:
        st.metric("Today's Bookings", f"{today_bookings} / 2")
    st.divider()

    if not my_b: st.info("You have no active bookings.")
    else:
        df_my_b = pd.DataFrame(my_b).sort_values(['date', 'court', 'start_hour'])
        merged_bookings = []
        if not df_my_b.empty:
            current_booking = None
            for _, row in df_my_b.iterrows():
                if current_booking is None:
                    current_booking = {'court': row['court'], 'date': row['date'], 'start_hours': [row['start_hour']], 'ids': [row['id']], 'v': row['orig_v'], 'sc': row['orig_sc']}
                else:
                    if (row['date'] == current_booking['date'] and row['court'] == current_booking['court'] and row['orig_v'] == current_booking['v'] and row['orig_sc'] == current_booking['sc'] and row['start_hour'] == max(current_booking['start_hours']) + 1):
                        current_booking['start_hours'].append(row['start_hour']); current_booking['ids'].append(row['id'])
                    else:
                        merged_bookings.append(current_booking)
                        current_booking = {'court': row['court'], 'date': row['date'], 'start_hours': [row['start_hour']], 'ids': [row['id']], 'v': row['orig_v'], 'sc': row['orig_sc']}
            merged_bookings.append(current_booking)

        for i, b in enumerate(merged_bookings):
            b_date = datetime.strptime(b['date'], '%Y-%m-%d')
            day_name = b_date.strftime('%A')
            formatted_date = b_date.strftime('%b %d, %Y')
            
            # Calculate time range
            start_time = min(b['start_hours'])
            end_time = max(b['start_hours']) + 1
            time_display = f"{start_time:02d}:00 - {end_time:02d}:00"
            
            # ID Display logic
            id_list = sorted(b['ids'])
            id_display = f"#{id_list[0]}" if len(id_list) == 1 else f"#{id_list[0]}-{id_list[-1]}"
            
            # Get location URL
            map_url = court_locations.get(b['court'], "#")
            
            # Use a container to group the card and the button
            with st.container():
                # CSS Card Styling (Location link moved under court name)
                st.markdown(f"""
                    <div style="
                        background-color: #0d5384; 
                        padding: 18px; 
                        border-radius: 12px 12px 0px 0px; 
                        border-left: 6px solid #4CAF50; 
                        color: white;
                        box-shadow: 0px 4px 10px rgba(0,0,0,0.4);
                        margin-top: 15px;
                    ">
                        <div style="font-family: 'Audiowide'; color: rgba(255,255,255,0.6); font-size: 0.8rem; margin-bottom: 5px;">
                            BOOKING CONF.: {id_display}
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 2px;">
                            <span style="font-family: 'Audiowide'; font-size: 1.3rem; color: #ccff00;">🎾 {b['court']}</span>
                            <span style="font-size: 1.1rem; font-weight: bold; color: white;">{b['sc']} - {b['v']}</span>
                        </div>
                        <div style="margin-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px;">
                            <a href="{map_url}" target="_blank" style="color: #ccff00; text-decoration: none; font-size: 0.9rem; font-weight: bold;">
                                📍 View Location Pin
                            </a>
                        </div>
                        <div>
                            <span style="font-size: 1.0rem; opacity: 0.9;">{day_name}, {formatted_date}</span>
                        </div>
                        <div style="font-size: 1.5rem; font-weight: bold; margin-top: 5px; font-family: 'Audiowide'; color: white;">
                            ⏰ {time_display}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Integrated Action Button
                if st.button(f"❌ Cancel Booking {id_display}", key=f"cancel_{i}", use_container_width=True):
                    for bid in b['ids']: delete_booking(bid, b['v'], b['sc'])
                    st.success(f"Successfully cancelled booking {id_display}")
                    time.sleep(1.5); st.rerun()
                st.markdown('<div style="margin-bottom: 25px;"></div>', unsafe_allow_html=True)
        
        st.divider()
        if st.button("🚪 Logout / Change Villa", use_container_width=True):
            logout_action()

with tab4:
    st.subheader("Community Activity Log (Last 14 Days)")
    st.caption("Timezone: UTC+4")
    
    # Check for admin status via session state (from the password field at the bottom)
    admin_pass_val = st.session_state.get("log_admin_pass", "")
    is_admin = admin_pass_val == st.secrets.get("ADMIN_PASSWORD", "admin123")

    logs = get_logs_last_14_days()
    if logs:
        log_df = pd.DataFrame(logs, columns=["timestamp", "event_type", "details"])
        
        # Standard filters
        filters = (
            (log_df['event_type'] != "Debug") &
            (log_df['event_type'] != "System Maintenance") &
            (~log_df['details'].str.contains("System-Synced", case=False, na=False))
        )
        
        # If NOT admin, also filter out "Limit Enforcement"
        if not is_admin:
            filters &= (log_df['event_type'] != "Limit Enforcement")
            
        display_df = log_df[filters].copy()        
        if is_admin:
            # In admin mode, let's extract Fingerprints into their own column for easy copying
            display_df['Fingerprint'] = display_df['details'].str.extract(r'⟦FP:(.*?)⟧')
            display_df['details'] = display_df['details'].str.replace(r'⟦FP:.*?⟧ ', '', regex=True)
            cols = ['timestamp', 'event_type', 'Fingerprint', 'details']
        else:
            # In user mode, hide the fingerprint codes for cleaner UI
            display_df['details'] = display_df['details'].str.replace(r'⟦FP:.*?⟧ ', '', regex=True)
            cols = ['timestamp', 'event_type', 'details']

        display_df['timestamp'] = pd.to_datetime(display_df['timestamp'], format='ISO8601').dt.strftime('%b %d, %H:%M')
        
        def style_rows(row):
            styles = [''] * len(row)
            if row.event_type == "Booking Created": styles[1] = 'background-color: #d4edda; color: #155724; font-weight: bold;'
            elif row.event_type in ["Booking Deleted", "Booking Cancelled"]: styles[1] = 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
            elif row.event_type == "Access Denied": styles[1] = 'background-color: #ffcc00; color: black; font-weight: bold;'
            return styles
            
        st.dataframe(display_df[cols].style.apply(style_rows, axis=1), hide_index=True, width="stretch")
    else: st.info("No activity.")

    # Show Device ID to ALL users
    st.divider()
    curr_fp = st.session_state.get('client_fp')
    if curr_fp:
        st.info(f"🆔 **Your Device ID:** `{curr_fp}`")
        st.caption("If you're unable to switch villas, copy this ID and send it to the admin for a manual reset.")

    st.divider()
    st.subheader("🛠️ Admin Tools")
    admin_pass = st.text_input("Admin Password", type="password", key="log_admin_pass")
    
    if is_admin:
        st.success("Admin Access Granted")
        st.markdown("### 🏘️ Villa Booking Management")
        
        all_villas = get_all_villas_with_any_bookings()
        selected_villa = st.selectbox("Select Villa to Manage", options=["-- Select --"] + all_villas, key="admin_manage_villa")
        
        if selected_villa != "-- Select --":
            try:
                sub_comm, villa_num = selected_villa.split(" - ")
                bookings = get_bookings_for_villa(villa_num, sub_comm)
                if bookings:
                    # Create a copy with a "Delete" column
                    df_bookings = pd.DataFrame(bookings)
                    # For UI display: Format the date and time
                    df_bookings['Time'] = df_bookings['start_hour'].apply(lambda x: f"{x:02d}:00")
                    # Prepare for selection
                    df_bookings.insert(0, "Select", False)
                    
                    st.write(f"Showing bookings for **{selected_villa}**:")
                    
                    # Use data_editor for selection
                    edited_df = st.data_editor(
                        df_bookings[["Select", "id", "date", "Time", "court"]],
                        column_config={
                            "Select": st.column_config.CheckboxColumn(
                                "Delete?",
                                help="Select to delete",
                                default=False,
                            ),
                            "id": "ID",
                            "date": "Date",
                            "Time": "Time",
                            "court": "Court"
                        },
                        disabled=["id", "date", "Time", "court"],
                        hide_index=True,
                        key="admin_booking_editor"
                    )
                    
                    if st.button("Delete Selected Bookings", type="primary"):
                        to_delete = edited_df[edited_df["Select"] == True]
                        if not to_delete.empty:
                            with st.spinner(f"Deleting {len(to_delete)} bookings..."):
                                for _, row in to_delete.iterrows():
                                    delete_booking(row['id'], villa_num, sub_comm)
                            st.success(f"Successfully deleted {len(to_delete)} bookings for {selected_villa}.")
                            time.sleep(1.5)
                            st.rerun()
                        else:
                            st.warning("Please select at least one booking to delete.")
                else:
                    st.info(f"No bookings found for {selected_villa}.")
            except Exception as e:
                st.error(f"Error loading bookings: {str(e)}")

        st.divider()
        st.markdown("### Device Lock Management")
        
        # Option 1: Global Reset
        if st.button("⚠️ Global Reset (Unlocks ALL devices)", type="secondary"):
            add_log("Global Reset", "Admin triggered a system-wide device unlock.")
            st.success("Global reset logged! Users will be unlocked upon their next refresh.")
            time.sleep(2)
            st.rerun()
            
        st.divider()
        # Option 2: Reset specific device
        col_res1, col_res2 = st.columns([3, 1])
        with col_res1:
            target_fp = st.text_input("Target Fingerprint to Reset", placeholder="Paste fingerprint from log here...")
        with col_res2:
            st.write(""); st.write("")
            if st.button("🔓 Reset FP"):
                if target_fp:
                    add_log("Lock Reset", f"Admin reset lock for ⟦FP:{target_fp}⟧")
                    st.success(f"Lock reset for {target_fp[:10]}...")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("Enter a fingerprint")

        # Option 3: Reset THIS device
        curr_fp = st.session_state.get('client_fp')
        if curr_fp:
            st.write(f"Your Device Fingerprint: `{curr_fp}`")
            if st.button("🔓 Reset MY Device Lock"):
                add_log("Lock Reset", f"Admin reset lock for ⟦FP:{curr_fp}⟧")
                st_javascript("localStorage.removeItem('court_villa_lock');")
                st.success("Your device has been unlocked.")
                time.sleep(2)
                st.rerun()
    elif admin_pass:
        st.error("Incorrect Password")

st.divider()
st.subheader("💾 Data Backup")
def get_zip_data():
    try:
        res_b = run_query(supabase.table("bookings").select("*"))
        res_l = run_query(supabase.table("logs").select("*"))
        b_data = res_b.data if res_b else []
        l_data = res_l.data if res_l else []
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as vz:
            vz.writestr(f"bookings_{get_today()}.csv", pd.DataFrame(b_data).to_csv(index=False))
            vz.writestr(f"logs_{get_today()}.csv", pd.DataFrame(l_data).to_csv(index=False))
        return buf.getvalue()
    except: return None

if st.button("Generate Backup Link"):
    data = get_zip_data()
    if data: st.download_button(label="Click here to Download ZIP", data=data, file_name=f"court_booking_backup_{get_today()}.zip", mime="application/zip")
    else: st.error("Failed to fetch data for backup.")

col1, col2 = st.columns([1, 5])
with col1: st.markdown(f'<img src="https://raw.githubusercontent.com/mahadevbk/courtbooking/main/qr-code.miracourtbooking.streamlit.app.png" height="100">', unsafe_allow_html=True)
with col2: st.markdown("""
    <div style='background-color: #0d5384; padding: 1rem; border-left: 5px solid #fff500; border-radius: 0.5rem; color: white;'>
    Built with ❤️ using <a href='https://streamlit.io/' style='color: #ccff00;'>Streamlit</a> — free and open source.
    <a href='https://devs-scripts.streamlit.app/' style='color: #ccff00;'>Other Scripts by dev</a> on Streamlit.
    </div>
    """, unsafe_allow_html=True)
