import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import base64
import time
from datetime import datetime, timedelta
from pathlib import Path
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# JavaScript-injected audio alert
def play_alert_sound():
    ALERT_AUDIO_PATH = "flood.wav"
    if not os.path.exists(ALERT_AUDIO_PATH):
        st.error("Alert sound file not found!")
        return
    audio_bytes = Path(ALERT_AUDIO_PATH).read_bytes()
    b64_audio = base64.b64encode(audio_bytes).decode()
    audio_id = f"audio_{int(time.time() * 1000)}"
    components.html(f"""
        <audio id="{audio_id}" autoplay>
            <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
        </audio>
        <script>
            var audio = document.getElementById("{audio_id}");
            if (audio) {{ audio.play(); }}
        </script>
    """, height=0)

# Enhanced logging functions
def initialize_logs():
    """Initialize log files with proper structure"""
    log_filepath = "flood_events_log.csv"
    daily_summary_filepath = "daily_summary_log.csv"
    
    # Initialize flood events log
    if not os.path.exists(log_filepath):
        events_df = pd.DataFrame(columns=[
            "Event_ID", "Event_Type", "Start_Time", "End_Time", 
            "Duration_Seconds", "Date", "Severity"
        ])
        events_df.to_csv(log_filepath, index=False)
    
    # Initialize daily summary log
    if not os.path.exists(daily_summary_filepath):
        summary_df = pd.DataFrame(columns=[
            "Date", "Total_Events", "Total_Duration_Minutes", 
            "Longest_Event_Minutes", "First_Event_Time", "Last_Event_Time"
        ])
        summary_df.to_csv(daily_summary_filepath, index=False)
    
    return log_filepath, daily_summary_filepath

def log_flood_event(event_type, start_time, end_time=None, event_id=None):
    """Log flood events with detailed information"""
    log_filepath, _ = initialize_logs()
    events_df = pd.read_csv(log_filepath)
    
    if event_type == "START":
        # Create new event entry
        event_id = f"FE_{int(time.time())}"
        new_event = {
            "Event_ID": event_id,
            "Event_Type": "ONGOING",
            "Start_Time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
            "End_Time": None,
            "Duration_Seconds": None,
            "Date": start_time.strftime('%Y-%m-%d'),
            "Severity": "Medium"  # Can be enhanced based on motion intensity
        }
        events_df = pd.concat([events_df, pd.DataFrame([new_event])], ignore_index=True)
        events_df.to_csv(log_filepath, index=False)
        return event_id
    
    elif event_type == "END" and event_id:
        # Update existing event with end time
        mask = events_df['Event_ID'] == event_id
        if mask.any():
            start_str = events_df.loc[mask, 'Start_Time'].iloc[0]
            start_dt = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
            duration = (end_time - start_dt).total_seconds()
            
            events_df.loc[mask, 'Event_Type'] = 'COMPLETED'
            events_df.loc[mask, 'End_Time'] = end_time.strftime('%Y-%m-%d %H:%M:%S')
            events_df.loc[mask, 'Duration_Seconds'] = duration
            
            # Determine severity based on duration
            if duration > 300:  # 5 minutes
                severity = "High"
            elif duration > 60:  # 1 minute
                severity = "Medium"
            else:
                severity = "Low"
            events_df.loc[mask, 'Severity'] = severity
            
            events_df.to_csv(log_filepath, index=False)
            update_daily_summary(start_dt.date())

def update_daily_summary(date):
    """Update daily summary statistics"""
    log_filepath, daily_summary_filepath = initialize_logs()
    events_df = pd.read_csv(log_filepath)
    daily_df = pd.read_csv(daily_summary_filepath)
    
    # Filter events for the specific date
    date_str = date.strftime('%Y-%m-%d')
    day_events = events_df[
        (events_df['Date'] == date_str) & 
        (events_df['Event_Type'] == 'COMPLETED')
    ]
    
    if len(day_events) > 0:
        total_events = len(day_events)
        total_duration_seconds = day_events['Duration_Seconds'].sum()
        total_duration_minutes = total_duration_seconds / 60
        longest_event_minutes = day_events['Duration_Seconds'].max() / 60
        
        # Get first and last event times
        start_times = pd.to_datetime(day_events['Start_Time'])
        first_event = start_times.min().strftime('%H:%M:%S')
        last_event = start_times.max().strftime('%H:%M:%S')
        
        # Update or create daily summary entry
        date_mask = daily_df['Date'] == date_str
        summary_data = {
            "Date": date_str,
            "Total_Events": total_events,
            "Total_Duration_Minutes": round(total_duration_minutes, 2),
            "Longest_Event_Minutes": round(longest_event_minutes, 2),
            "First_Event_Time": first_event,
            "Last_Event_Time": last_event
        }
        
        if date_mask.any():
            for key, value in summary_data.items():
                daily_df.loc[date_mask, key] = value
        else:
            daily_df = pd.concat([daily_df, pd.DataFrame([summary_data])], ignore_index=True)
        
        daily_df.to_csv(daily_summary_filepath, index=False)

# Enhanced detection with flood duration tracking
def run_detection(stream_url, slanted_roi):
    def initialize_video_capture(url):
        cap = cv2.VideoCapture(url)
        return cap if cap.isOpened() else None

    cap = initialize_video_capture(stream_url)
    retry_attempts = 0
    max_retries = 10
    retry_interval = 5
    prev_frame = None
    
    # Flood tracking variables
    flood_active = False
    current_flood_start = None
    current_event_id = None
    frame_index = 0
    fps = None
    
    # UI setup
    video_col, flow_col = st.columns(2)
    video_placeholder = video_col.empty()
    flow_placeholder = flow_col.empty()
    
    # Status display
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()

    while True:
        if cap is None or not cap.isOpened():
            if retry_attempts < max_retries:
                st.warning(f"Stream connection lost. Retrying ({retry_attempts+1}/{max_retries})...")
                time.sleep(retry_interval)
                cap = initialize_video_capture(stream_url)
                retry_attempts += 1
                if cap and cap.isOpened():
                    retry_attempts = 0
            else:
                st.error("Failed to reconnect to stream after multiple attempts.")
                break

        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            continue

        current_time = datetime.now()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if np.mean(gray_frame) < 10 or cv2.countNonZero(gray_frame) < 1000:
            continue

        if prev_frame is None:
            prev_frame = gray_frame
            continue

        # Optical flow processing
        mask = np.zeros_like(gray_frame)
        cv2.fillPoly(mask, [slanted_roi], 255)
        roi_curr = cv2.bitwise_and(gray_frame, mask)
        roi_prev = cv2.bitwise_and(prev_frame, mask)

        flow = cv2.calcOpticalFlowFarneback(
            roi_prev, roi_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag_thresh = np.where(magnitude > 20, 255, 0).astype(np.uint8)
        mag_thresh_masked = cv2.bitwise_and(mag_thresh, mask)

        # Frame preparation
        output_frame = frame.copy()
        cv2.polylines(output_frame, [slanted_roi], True, (0, 255, 0), 2)

        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
        timestamp_seconds = frame_index / fps
        timestamp = f"Time: {int(timestamp_seconds//60):02}:{int(timestamp_seconds%60):02}"
        cv2.putText(output_frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Flood detection logic
        motion_area = np.count_nonzero(mag_thresh_masked)
        flood_detected = motion_area > 500
        
        if flood_detected and not flood_active:
            # Flood event started
            flood_active = True
            current_flood_start = current_time
            current_event_id = log_flood_event("START", current_flood_start)
            play_alert_sound()
            
        elif not flood_detected and flood_active:
            # Flood event ended
            flood_active = False
            if current_event_id and current_flood_start:
                log_flood_event("END", current_flood_start, current_time, current_event_id)
            current_flood_start = None
            current_event_id = None

        # Display flood status
        if flood_active:
            cv2.putText(output_frame, "FLOOD DETECTED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            duration = (current_time - current_flood_start).total_seconds()
            cv2.putText(output_frame, f"Duration: {int(duration)}s", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Update displays
        video_placeholder.image(output_frame, channels="BGR", caption="Original Frame with ROI")
        flow_placeholder.image(mag_thresh_masked, caption="Motion Threshold Mask", clamp=True)
        
        # Status update
        status_text = "FLOOD ACTIVE" if flood_active else "MONITORING"
        if flood_active:
            duration = int((current_time - current_flood_start).total_seconds())
            status_placeholder.error(f"{status_text} - Duration: {duration}s")
        else:
            status_placeholder.success(status_text)

        prev_frame = gray_frame
        frame_index += 1

        if st.button("Stop Monitoring"):
            # End any active flood event
            if flood_active and current_event_id:
                log_flood_event("END", current_flood_start, current_time, current_event_id)
            break

    cap.release()
    st.success("Monitoring stopped.")

# Analytics and reporting functions
def create_daily_events_chart(events_df, date_range_days=30):
    """Create daily events bar chart"""
    if events_df.empty:
        return None
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=date_range_days)
    
    # Filter data for date range
    events_df['Date'] = pd.to_datetime(events_df['Date']).dt.date
    filtered_df = events_df[
        (events_df['Date'] >= start_date) & 
        (events_df['Date'] <= end_date) &
        (events_df['Event_Type'] == 'COMPLETED')
    ]
    
    if filtered_df.empty:
        return None
    
    # Group by date and count events
    daily_counts = filtered_df.groupby('Date').size().reset_index(name='Event_Count')
    
    fig = px.bar(
        daily_counts, 
        x='Date', 
        y='Event_Count',
        title=f'Daily Flood Events (Last {date_range_days} Days)',
        labels={'Event_Count': 'Number of Events', 'Date': 'Date'}
    )
    fig.update_layout(showlegend=False)
    return fig

def create_duration_analysis_chart(events_df):
    """Create duration analysis charts"""
    completed_events = events_df[events_df['Event_Type'] == 'COMPLETED'].copy()
    if completed_events.empty:
        return None
    
    completed_events['Duration_Minutes'] = completed_events['Duration_Seconds'] / 60
    
    # Create subplot with multiple charts
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Duration Distribution', 'Severity Distribution', 
                       'Duration Over Time', 'Daily Total Duration'),
        specs=[[{"type": "histogram"}, {"type": "pie"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Duration histogram
    fig.add_trace(
        go.Histogram(x=completed_events['Duration_Minutes'], name="Duration"),
        row=1, col=1
    )
    
    # Severity pie chart
    severity_counts = completed_events['Severity'].value_counts()
    fig.add_trace(
        go.Pie(labels=severity_counts.index, values=severity_counts.values, name="Severity"),
        row=1, col=2
    )
    
    # Duration over time
    completed_events['Start_Time'] = pd.to_datetime(completed_events['Start_Time'])
    fig.add_trace(
        go.Scatter(
            x=completed_events['Start_Time'], 
            y=completed_events['Duration_Minutes'],
            mode='markers',
            name="Duration Over Time"
        ),
        row=2, col=1
    )
    
    # Daily total duration
    daily_duration = completed_events.groupby('Date')['Duration_Minutes'].sum().reset_index()
    fig.add_trace(
        go.Bar(x=daily_duration['Date'], y=daily_duration['Duration_Minutes'], name="Daily Duration"),
        row=2, col=2
    )
    
    fig.update_layout(title_text="Flood Event Analysis", showlegend=False)
    return fig

def display_analytics_dashboard():
    """Display comprehensive analytics dashboard"""
    st.header(" Flood Detection Analytics")
    
    # Load data
    log_filepath, daily_summary_filepath = initialize_logs()
    
    try:
        events_df = pd.read_csv(log_filepath)
        daily_df = pd.read_csv(daily_summary_filepath)
    except:
        st.warning("No data available yet. Start monitoring to generate analytics.")
        return
    
    if events_df.empty:
        st.info("No flood events recorded yet.")
        return
    
    # Summary metrics
    completed_events = events_df[events_df['Event_Type'] == 'COMPLETED']
    total_events = len(completed_events)
    
    if total_events > 0:
        avg_duration = completed_events['Duration_Seconds'].mean() / 60
        max_duration = completed_events['Duration_Seconds'].max() / 60
        total_duration = completed_events['Duration_Seconds'].sum() / 60
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Events", total_events)
        col2.metric("Avg Duration", f"{avg_duration:.1f} min")
        col3.metric("Max Duration", f"{max_duration:.1f} min")
        col4.metric("Total Duration", f"{total_duration:.1f} min")
    
    # Date range selector
    st.subheader("Select Analysis Period")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Last 7 Days"):
            days = 7
    with col2:
        if st.button("Last 30 Days"):
            days = 30
    with col3:
        if st.button("Last 6 Months"):
            days = 180
    
    # Custom date range
    st.subheader("Custom Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now().date())
    
    if st.button("Apply Custom Range"):
        days = (end_date - start_date).days
    
    # Default to 30 days if not set
    if 'days' not in locals():
        days = 30
    
    # Charts
    st.subheader("Daily Events Chart")
    daily_chart = create_daily_events_chart(events_df, days)
    if daily_chart:
        st.plotly_chart(daily_chart, use_container_width=True)
    else:
        st.info("No data available for the selected period.")
    
    st.subheader("Duration Analysis")
    duration_chart = create_duration_analysis_chart(events_df)
    if duration_chart:
        st.plotly_chart(duration_chart, use_container_width=True)
    
    # Detailed tables
    st.subheader("Recent Events")
    if not completed_events.empty:
        display_events = completed_events.copy()
        display_events['Duration_Minutes'] = (display_events['Duration_Seconds'] / 60).round(2)
        display_events = display_events[['Event_ID', 'Start_Time', 'End_Time', 'Duration_Minutes', 'Severity']]
        st.dataframe(display_events.head(20), use_container_width=True)
    
    st.subheader("Daily Summary")
    if not daily_df.empty:
        st.dataframe(daily_df.head(20), use_container_width=True)
    
    # Download options
    st.subheader("Download Data")
    col1, col2 = st.columns(2)
    
    with col1:
        if not events_df.empty:
            csv_events = events_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Events Log",
                data=csv_events,
                file_name=f"flood_events_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if not daily_df.empty:
            csv_daily = daily_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Daily Summary",
                data=csv_daily,
                file_name=f"daily_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# Main Streamlit App
def main():
    st.set_page_config(page_title="TDU Flood Detection System", layout="wide")
    
    # Sidebar navigation
    st.sidebar.title("TDU Flood Detection")
    page = st.sidebar.selectbox("Choose a page:", ["Live Monitoring", "Analytics Dashboard"])
    
    if page == "Live Monitoring":
        st.title("Live Flood Detection Monitoring")
        
        # Initialize logs
        initialize_logs()
        
        # ROI Setup
        slanted_roi = np.array([[509, 197], [487, 195], [155, 300], [167, 339], [185, 347], [539, 229]]).reshape((-1, 1, 2))
        
       
        default_url = "rtsp://username:password@192.168.0.100:554/Streaming/Channels/1/"
        stream_url = st.text_input("Enter RTSP Stream URL", value=default_url)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Monitoring", type="primary"):
                run_detection(stream_url, slanted_roi)
        
        with col2:
            if st.button("View Analytics"):
                st.switch_page("Analytics Dashboard")
    
    elif page == "Analytics Dashboard":
        display_analytics_dashboard()

if __name__ == "__main__":
    main()