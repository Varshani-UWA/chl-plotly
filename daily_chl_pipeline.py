#!/usr/bin/env python3
"""
daily_chl_pipeline.py

Daily Sentinel-3 OLCI (S3A/S3B) chlorophyll monitoring pipeline:
- FIRST RUN: downloads all October 2025 data to populate initial CSV
- SUBSEQUENT RUNS: search & download yesterday's NRT files (earthaccess)
- extract 3x3 mean chlor_a at target coordinate
- append to continuous CSV timeseries
- generate 5-day and monthly plots (plotnine)
- optional upload to Google Drive (pydrive2)
- cleanup old files locally

Configure the paths and options in the CONFIG section below.
"""

import os
import glob
import time
import shutil
import subprocess
from datetime import datetime, timedelta
import logging

import earthaccess
import xarray as xr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ---------- CONFIG ----------
BBOX = (115.4, -32.65, 115.8, -31.70)  # lon_min, lat_min, lon_max, lat_max
TARGET_LAT = -32.20085
TARGET_LON = 115.77047

DOWNLOAD_DIR = r"C:/Users/23755118/OneDrive - UWA/Documents/PhD_Varshani/CODING/chl_time"
STORE_CSV = os.path.join(DOWNLOAD_DIR, "chl_timeseries.csv")
PLOT_DIR = os.path.join(DOWNLOAD_DIR, "plots")
KEEP_DAYS = 30  # local cleanup: remove downloaded files older than this

# Initial bulk download settings
INITIAL_BULK_DOWNLOAD = True  # Set to False after first successful run
BULK_START_DATE = "2025-10-01"  # October 1, 2025
BULK_END_DATE = "2025-10-16"    # Today (October 16, 2025)

# Google Drive upload config (optional)
ENABLE_DRIVE_UPLOAD = False
DRIVE_FOLDER_ID = "1VVtBvjgFK5d8R1ri0uMC_EUB_jy_oWUJ"  # change if using upload

# Git automation config
ENABLE_GIT_PUSH = True  # Set to True to enable automatic git commits
GIT_REPO_PATH = DOWNLOAD_DIR  # Path to your git repository
GIT_COMMIT_MESSAGE_TEMPLATE = "Auto-update chlorophyll data: {date}"
# ----------------------------

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s: %(message)s",
)
logger = logging.getLogger("chl_pipeline")
# ----------------------------

# ---------- Utility / core functions ----------
def get_yesterday_str(utc=True):
    """Return yesterday in YYYY-MM-DD (UTC) or local if utc=False (not used here)."""
    if utc:
        ref = datetime.utcnow()
    else:
        ref = datetime.now()
    yesterday = ref - timedelta(days=1)
    return yesterday.strftime("%Y-%m-%d")


def earthdata_login_check():
    """Attempt a login with earthaccess. Raises on failure."""
    try:
        logger.info("Logging in to Earthdata via earthaccess...")
        session = earthaccess.login() 
        
        logger.info("Earthdata login OK.")
        return session
    except Exception as e:
        logger.exception("Earthdata login failed. Check credentials (.netrc) and network.")
        raise


def fetch_date_range_files(start_date, end_date, bbox, download_dir, short_names=None):
    """
    Search and download files for a date range.
    start_date, end_date: strings in YYYY-MM-DD format
    Returns list of downloaded file paths.
    """
    if short_names is None:
        short_names = ["OLCIS3A_L2_EFR_OC_NRT", "OLCIS3B_L2_EFR_OC_NRT"]

    os.makedirs(download_dir, exist_ok=True)
    downloaded_files = []

    # Perform login (this will raise if credentials missing)
    earthdata_login_check()

    for sat in short_names:
        try:
            logger.info(f"Searching {sat} for date range {start_date} to {end_date} in bbox {bbox} ...")
            results = earthaccess.search_data(
                short_name=sat,
                temporal=(start_date, end_date),
                bounding_box=bbox
            )
            if not results:
                logger.info(f"No results for {sat} in date range {start_date} to {end_date}")
                continue

            logger.info(f"Found {len(results)} items for {sat}. Downloading...")
            # earthaccess.download returns list of downloaded file paths (or similar)
            try:
                earthaccess.download(results, download_dir)
            except TypeError:
                # older/newer earthaccess versions may have different signatures
                earthaccess.download(results, path=download_dir)

            # wait briefly to allow files to appear if necessary
            time.sleep(2)
            # collect any new NC files from this run
            new_files = glob.glob(os.path.join(download_dir, "*.nc"))
            logger.info(f"Download directory now has {len(new_files)} .nc files (may include older files).")
            downloaded_files.extend(new_files)
        except Exception as e:
            logger.exception(f"Error searching/downloading for {sat}: {e}")

    # deduplicate and return absolute unique list
    downloaded_files = sorted(set([os.path.abspath(p) for p in downloaded_files]))
    logger.info(f"Total unique files downloaded: {len(downloaded_files)}")
    return downloaded_files


def fetch_daily_files(date_str, bbox, download_dir, short_names=None):
    """
    Search and download files for given date_str (YYYY-MM-DD).
    Returns list of downloaded file paths.
    """
    if short_names is None:
        short_names = ["OLCIS3A_L2_EFR_OC_NRT", "OLCIS3B_L2_EFR_OC_NRT"]

    os.makedirs(download_dir, exist_ok=True)
    downloaded_files = []

    # Perform login (this will raise if credentials missing)
    earthdata_login_check()

    for sat in short_names:
        try:
            logger.info(f"Searching {sat} for {date_str} in bbox {bbox} ...")
            results = earthaccess.search_data(
                short_name=sat,
                temporal=(date_str, date_str),
                bounding_box=bbox
            )
            if not results:
                logger.info(f"No results for {sat} on {date_str}")
                continue

            logger.info(f"Found {len(results)} items for {sat} on {date_str}. Downloading...")
            # earthaccess.download returns list of downloaded file paths (or similar)
            try:
                earthaccess.download(results, download_dir)
            except TypeError:
                # older/newer earthaccess versions may have different signatures
                earthaccess.download(results, path=download_dir)

            # wait briefly to allow files to appear if necessary
            time.sleep(1)
            # collect any new NC files from this run
            new_files = glob.glob(os.path.join(download_dir, "*.nc"))
            logger.info(f"Download directory now has {len(new_files)} .nc files (may include older files).")
            downloaded_files.extend(new_files)
        except Exception as e:
            logger.exception(f"Error searching/downloading for {sat}: {e}")

    # deduplicate and return absolute unique list
    downloaded_files = sorted(set([os.path.abspath(p) for p in downloaded_files]))
    return downloaded_files


def extract_nearest_3x3_satellite_2d(dataset, target_lat, target_lon, var_name='chlor_a'):
    """
    Extract 3x3 values around target lat/lon for satellite data with 2D coords.
    dataset: xarray.Dataset with dataset.latitude and dataset.longitude 2D arrays and variable var_name
    returns float (mean) or np.nan on failure.
    """
    try:
        lat_coords = dataset.latitude.values
        lon_coords = dataset.longitude.values
        data_values = dataset[var_name].values

        # If shapes match, find nearest directly
        if lat_coords.shape == data_values.shape:
            distances = np.sqrt((lat_coords - target_lat) ** 2 + (lon_coords - target_lon) ** 2)
            min_idx = np.unravel_index(np.nanargmin(distances), distances.shape)
            line_idx, pixel_idx = min_idx
            line_start = max(0, line_idx - 1)
            line_end = min(data_values.shape[0], line_idx + 2)
            pixel_start = max(0, pixel_idx - 1)
            pixel_end = min(data_values.shape[1], pixel_idx + 2)
            region_3x3 = data_values[line_start:line_end, pixel_start:pixel_end]
        else:
            # coords subsampled => create coordinate pairs and find nearest point
            coord_points = np.column_stack([lat_coords.ravel(), lon_coords.ravel()])
            target_point = np.array([[target_lat, target_lon]])
            distances = np.sqrt(np.sum((coord_points - target_point) ** 2, axis=1))
            nearest_coord_idx = np.nanargmin(distances)
            coord_line_idx, coord_pixel_idx = np.unravel_index(nearest_coord_idx, lat_coords.shape)
            scale_line = int(round(data_values.shape[0] / lat_coords.shape[0]))
            scale_pixel = int(round(data_values.shape[1] / lat_coords.shape[1]))
            data_line_idx = min(data_values.shape[0] - 1, coord_line_idx * max(1, scale_line))
            data_pixel_idx = min(data_values.shape[1] - 1, coord_pixel_idx * max(1, scale_pixel))
            line_start = max(0, data_line_idx - 1)
            line_end = min(data_values.shape[0], data_line_idx + 2)
            pixel_start = max(0, data_pixel_idx - 1)
            pixel_end = min(data_values.shape[1], data_pixel_idx + 2)
            region_3x3 = data_values[line_start:line_end, pixel_start:pixel_end]

        chl_mean = np.nanmean(region_3x3)
        return float(chl_mean) if np.isfinite(chl_mean) else np.nan
    except Exception as e:
        logger.exception(f"Failed to extract 3x3 region: {e}")
        return np.nan


def process_downloaded_files(files, target_lat, target_lon, store_csv):
    """
    Process list of .nc files, extract chl mean for each file, and append to CSV store.
    Files may include many days; function will extract date from filename if possible.
    """
    rows = []
    for fpath in files:
        try:
            # attempt to infer date from filename (common OLCI naming has .YYYYMMDD)
            fname = os.path.basename(fpath)
            # attempt common patterns: .YYYYMMDD or _YYYYMMDDT...
            date_str = None
            # examples to try:
            for token in fname.replace(".", "_").split("_"):
                if len(token) >= 8 and token[:8].isdigit():
                    # take first 8 digits as date
                    candidate = token[:8]
                    try:
                        dt = datetime.strptime(candidate, "%Y%m%d")
                        date_str = dt.strftime("%Y-%m-%d")
                        break
                    except Exception:
                        continue

            # fallback: file modified time
            if date_str is None:
                mtime = datetime.utcfromtimestamp(os.path.getmtime(fpath))
                date_str = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

            # open datatree/dataset and extract
            try:
                datatree = xr.open_datatree(fpath)
                dataset = xr.merge(datatree.to_dict().values())
            except Exception:
                # fallback to open_dataset (some files are simple netcdf)
                dataset = xr.open_dataset(fpath)

            chl_mean = extract_nearest_3x3_satellite_2d(dataset, target_lat, target_lon, var_name='chlor_a')
            # close objects (safe if they don't exist)
            try:
                dataset.close()
            except Exception:
                pass
            try:
                datatree.close()
            except Exception:
                pass

            rows.append((date_str, chl_mean, fname))
            logger.info(f"Processed {fname} -> date {date_str}, chl={chl_mean}")
        except Exception as e:
            logger.exception(f"Error processing file {fpath}: {e}")
    
    # Build DataFrame and append to CSV (deduplicate by date)
    if rows:
        df_new = pd.DataFrame(rows, columns=["date", "chlor_a", "source_file"])
        df_new["date"] = pd.to_datetime(df_new["date"], format="%Y-%m-%d", errors="coerce")
        df_new = df_new.dropna(subset=["date"])
        df_new = df_new.sort_values("date")
        df_new = df_new.drop_duplicates(subset=["date"], keep="last")

        # load existing
        if os.path.exists(store_csv):
            df_old = pd.read_csv(store_csv, parse_dates=["date"])
            df_merged = pd.concat([df_old, df_new[["date", "chlor_a"]]])
            df_merged = df_merged.drop_duplicates(subset=["date"], keep="last")
            df_merged = df_merged.sort_values("date")
        else:
            df_merged = df_new[["date", "chlor_a"]].copy()

        df_merged.to_csv(store_csv, index=False, date_format="%Y-%m-%d")
        logger.info(f"Timeseries store updated: {store_csv} (rows: {len(df_merged)})")
        return df_merged
    else:
        logger.info("No new rows extracted from files.")
        # return existing dataframe or empty df
        if os.path.exists(store_csv):
            return pd.read_csv(store_csv, parse_dates=["date"])
        else:
            return pd.DataFrame(columns=["date", "chlor_a"])


def generate_plots(csv_path, out_dir):
    """Generate interactive Plotly plots instead of static plotnine plots"""
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values("date").dropna(subset=["chlor_a"])
    
    if df.empty:
        logger.warning("No data available for plotting")
        return {}

    latest_date = df['date'].max()
    out_paths = {}

    # ---------- 5-day plot ----------
    start_5d = latest_date - timedelta(days=4)
    df_5d = df[df["date"].between(start_5d, latest_date)]

    if not df_5d.empty:
        fig_5d = go.Figure()
        
        # Add scatter points
        fig_5d.add_trace(go.Scatter(
            x=df_5d["date"], 
            y=df_5d["chlor_a"],
            mode="markers+lines",
            name="Chlorophyll-a",
            marker=dict(size=10, color="#50C701"),
            line=dict(color="#50C701", width=2),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Chl-a: %{y:.4f} mg/m³<extra></extra>"
        ))

        # Add moving average if enough points
        if len(df_5d) >= 3:
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                x_numeric = (df_5d["date"] - df_5d["date"].min()).dt.total_seconds()
                lowess_res = lowess(df_5d["chlor_a"], x_numeric, frac=0.5)
                xs = df_5d["date"].min() + pd.to_timedelta(lowess_res[:,0], unit='s')
                ys = lowess_res[:,1]
                fig_5d.add_trace(go.Scatter(
                    x=xs, y=ys, 
                    mode="lines", 
                    name="Trend",
                    line=dict(color="black", width=2, dash='dash')
                ))
            except Exception as e:
                logger.warning(f"Could not add trend line: {e}")

        fig_5d.update_layout(
            title=f"5-Day Chlorophyll Trend (as of {latest_date.date()})",
            xaxis_title="Date",
            yaxis_title="Chlorophyll-a (mg/m³)",
            hovermode="x unified",
            template="plotly_white",
            height=500,
            width=900,
            showlegend=True
        )

        p5_path_html = os.path.join(out_dir, f"chl_5day_{latest_date.date()}.html")
        p5_path_png = os.path.join(out_dir, f"chl_5day_{latest_date.date()}.png")
        pio.write_html(fig_5d, file=p5_path_html, include_plotlyjs='cdn')
        
        # Also save static PNG
        try:
            fig_5d.write_image(p5_path_png, width=900, height=500)
        except Exception as e:
            logger.warning(f"Could not save PNG (kaleido may not be installed): {e}")
        
        out_paths["5day_html"] = p5_path_html
        out_paths["5day_png"] = p5_path_png
        logger.info(f"5-day plot saved: {p5_path_html}")

    # ---------- Monthly plot ----------
    month_start = latest_date.replace(day=1)
    df_month = df[df["date"].between(month_start, latest_date)]

    if not df_month.empty:
        fig_month = go.Figure()
        
        # Add scatter points
        fig_month.add_trace(go.Scatter(
            x=df_month["date"], 
            y=df_month["chlor_a"],
            mode="markers+lines",
            name="Chlorophyll-a",
            marker=dict(size=8, color="#50C701"),
            line=dict(color="#50C701", width=2),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Chl-a: %{y:.4f} mg/m³<extra></extra>"
        ))

        # Add moving average if enough points
        if len(df_month) >= 3:
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                x_numeric = (df_month["date"] - df_month["date"].min()).dt.total_seconds()
                lowess_res = lowess(df_month["chlor_a"], x_numeric, frac=0.3)
                xs = df_month["date"].min() + pd.to_timedelta(lowess_res[:,0], unit='s')
                ys = lowess_res[:,1]
                fig_month.add_trace(go.Scatter(
                    x=xs, y=ys, 
                    mode="lines", 
                    name="Trend",
                    line=dict(color="black", width=2, dash='dash')
                ))
            except Exception as e:
                logger.warning(f"Could not add trend line: {e}")

        fig_month.update_layout(
            title=f"{latest_date.strftime('%B %Y')} Chlorophyll Trend",
            xaxis_title="Date",
            yaxis_title="Chlorophyll-a (mg/m³)",
            hovermode="x unified",
            template="plotly_white",
            height=500,
            width=1000,
            showlegend=True
        )

        pmonth_path_html = os.path.join(out_dir, f"chl_month_{latest_date.strftime('%Y-%m')}.html")
        pmonth_path_png = os.path.join(out_dir, f"chl_month_{latest_date.strftime('%Y-%m')}.png")
        pio.write_html(fig_month, file=pmonth_path_html, include_plotlyjs='cdn')
        
        # Also save static PNG
        try:
            fig_month.write_image(pmonth_path_png, width=1000, height=500)
        except Exception as e:
            logger.warning(f"Could not save PNG (kaleido may not be installed): {e}")
        
        out_paths["month_html"] = pmonth_path_html
        out_paths["month_png"] = pmonth_path_png
        logger.info(f"Monthly plot saved: {pmonth_path_html}")

    # ---------- Full interactive timeseries ----------
    if len(df) > 0:
        fig_full = go.Figure()

        # Add daily points
        fig_full.add_trace(go.Scatter(
            x=df["date"], 
            y=df["chlor_a"],
            mode="markers",
            name="Daily chlor_a",
            marker=dict(size=6, color="#50C701"),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Chl-a: %{y:.4f} mg/m³<extra></extra>"
        ))

        # Add 7-day moving average if enough points
        if len(df) >= 7:
            df_temp = df.copy()
            df_temp["ma7"] = df_temp["chlor_a"].rolling(window=7, min_periods=1, center=True).mean()
            fig_full.add_trace(go.Scatter(
                x=df_temp["date"], 
                y=df_temp["ma7"],
                mode="lines",
                name="7-day moving avg",
                line=dict(color="blue", width=2)
            ))

        # Add LOESS if enough points
        if len(df) >= 4 and df["chlor_a"].nunique() > 1:
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                x_numeric = (df["date"] - df["date"].min()).dt.total_seconds()
                lowess_res = lowess(df["chlor_a"], x_numeric, frac=0.25)
                xs = df["date"].min() + pd.to_timedelta(lowess_res[:,0], unit='s')
                ys = lowess_res[:,1]
                fig_full.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="lines",
                    name="LOESS trend",
                    line=dict(color="black", width=2, dash='dash')
                ))
            except Exception as e:
                logger.warning(f"Could not add LOESS trend: {e}")

        fig_full.update_layout(
            title="Interactive Chlorophyll Timeseries - Full Dataset",
            xaxis_title="Date",
            yaxis_title="Chlorophyll-a (mg/m³)",
            hovermode="x unified",
            template="plotly_white",
            height=600,
            width=1200,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Add range selector
        fig_full.update_xaxes(
            rangeselector=dict(
                buttons=[
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ]
            ),
            rangeslider=dict(visible=True),
            type="date"
        )

        pfull_path = os.path.join(out_dir, "chlorophyll_timeseries_interactive.html")
        pio.write_html(fig_full, file=pfull_path, include_plotlyjs='cdn')
        out_paths["full_interactive"] = pfull_path
        logger.info(f"Full interactive plot saved: {pfull_path}")

    return out_paths


# ---------- Optional uploader (Google Drive via pydrive2) ----------
def upload_to_google_drive(local_path, folder_id):
    """
    Upload file to Google Drive. Requires pydrive2 and a configured OAuth2 flow.
    This function is optional; if you enable, configure pydrive2 auth on the machine.
    """
    try:
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
    except ImportError:
        logger.error("pydrive2 not installed. Install with 'pip install pydrive2' to enable upload.")
        return None

    try:
        gauth = GoogleAuth()
        # This uses local webserver auth flow on first run; make sure machine supports it.
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)

        file_drive = drive.CreateFile({'title': os.path.basename(local_path),
                                       'parents': [{'id': folder_id}]})
        file_drive.SetContentFile(local_path)
        file_drive.Upload()
        link = file_drive.get('alternateLink')
        logger.info(f"Uploaded to Drive: {link}")
        return link
    except Exception as e:
        logger.exception(f"Failed to upload {local_path} to Google Drive: {e}")
        return None


def cleanup_old_files(directory, keep_days=30):
    """Delete files in `directory` older than keep_days (creation/modification time)."""
    cutoff = datetime.utcnow() - timedelta(days=keep_days)
    removed = 0
    for f in glob.glob(os.path.join(directory, "*")):
        try:
            mtime = datetime.utcfromtimestamp(os.path.getmtime(f))
            if mtime < cutoff:
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)
                removed += 1
        except Exception:
            logger.exception(f"Failed cleaning {f}")
    logger.info(f"Cleanup done. Removed {removed} files older than {keep_days} days.")


def git_commit_and_push(repo_path, files_to_add=None, commit_message=None):
    """
    Commit and push changes to Git repository.
    
    Args:
        repo_path: Path to git repository
        files_to_add: List of files to add (relative to repo_path), or None to add all
        commit_message: Commit message, or None for default
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Change to repo directory
        original_dir = os.getcwd()
        os.chdir(repo_path)
        
        # Check if it's a git repo
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            logger.error(f"Not a git repository: {repo_path}")
            logger.info("Initialize with: git init")
            os.chdir(original_dir)
            return False
        
        # Configure git if needed (for automated commits)
        subprocess.run(
            ["git", "config", "user.email", "automated@pipeline.local"],
            capture_output=True,
            timeout=10
        )
        subprocess.run(
            ["git", "config", "user.name", "Automated Pipeline"],
            capture_output=True,
            timeout=10
        )
        
        # Add files
        if files_to_add is None:
            # Add all changes
            subprocess.run(["git", "add", "-A"], capture_output=True, timeout=30)
        else:
            for file in files_to_add:
                subprocess.run(["git", "add", file], capture_output=True, timeout=10)
        
        # Check if there are changes to commit
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if not result.stdout.strip():
            logger.info("No changes to commit")
            os.chdir(original_dir)
            return True
        
        # Commit
        if commit_message is None:
            commit_message = f"Auto-update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        result = subprocess.run(
            ["git", "commit", "-m", commit_message],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            logger.error(f"Git commit failed: {result.stderr}")
            os.chdir(original_dir)
            return False
        
        logger.info(f"Git commit successful: {commit_message}")
        
        # Get current branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            logger.error("Could not determine current branch")
            os.chdir(original_dir)
            return False
        
        current_branch = result.stdout.strip()
        logger.info(f"Current branch: {current_branch}")
        
        # Push to current branch
        result = subprocess.run(
            ["git", "push", "origin", current_branch],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            # If push fails, might need to set upstream
            logger.warning(f"Initial push failed, trying with -u flag: {result.stderr}")
            result = subprocess.run(
                ["git", "push", "-u", "origin", current_branch],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                logger.error(f"Git push failed: {result.stderr}")
                logger.warning("Changes committed locally but push failed. Will retry on next run.")
                os.chdir(original_dir)
                return True
        
        logger.info(f"Git push successful to {current_branch}")
        os.chdir(original_dir)
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("Git operation timed out")
        try:
            os.chdir(original_dir)
        except Exception:
            pass
        return False
    except Exception as e:
        logger.exception(f"Git operation failed: {e}")
        try:
            os.chdir(original_dir)
        except Exception:
            pass
        return False


def check_first_run():
    """
    Check if this is the first run by looking for the CSV file.
    Returns True if first run (CSV doesn't exist or is empty).
    """
    if not os.path.exists(STORE_CSV):
        return True
    
    try:
        df = pd.read_csv(STORE_CSV)
        return len(df) == 0
    except Exception:
        return True


def main():
    logger.info("=== Starting daily chlorophyll pipeline ===")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Check if this is the first run or bulk download is explicitly enabled
    is_first_run = check_first_run()
    
    if INITIAL_BULK_DOWNLOAD and is_first_run:
        logger.info("=" * 60)
        logger.info("FIRST RUN DETECTED: Performing initial bulk download")
        logger.info(f"Downloading data from {BULK_START_DATE} to {BULK_END_DATE}")
        logger.info("=" * 60)
        
        files = fetch_date_range_files(BULK_START_DATE, BULK_END_DATE, BBOX, DOWNLOAD_DIR)
        
        if not files:
            logger.warning("No files downloaded during bulk download. Check date range and bbox.")
        else:
            logger.info(f"Bulk download complete. Processing {len(files)} files...")
            df = process_downloaded_files(files, TARGET_LAT, TARGET_LON, STORE_CSV)
            logger.info(f"Initial CSV populated with {len(df)} records")
            
        logger.info("=" * 60)
        logger.info("IMPORTANT: Set INITIAL_BULK_DOWNLOAD = False in CONFIG")
        logger.info("for subsequent runs to enable daily mode")
        logger.info("=" * 60)
    else:
        # Regular daily run
        date_str = get_yesterday_str(utc=True)
        logger.info(f"Daily mode: Searching for data for {date_str}")

        files = fetch_daily_files(date_str, BBOX, DOWNLOAD_DIR)
        if not files:
            logger.info("No files downloaded for yesterday. Exiting (no update).")
        else:
            df = process_downloaded_files(files, TARGET_LAT, TARGET_LON, STORE_CSV)

    # Generate plots if we have data
    if os.path.exists(STORE_CSV):
        df = pd.read_csv(STORE_CSV, parse_dates=["date"])
        if not df.empty:
            out_paths = generate_plots(STORE_CSV, PLOT_DIR)

            # Optional Google Drive upload
            if ENABLE_DRIVE_UPLOAD and out_paths and "month_html" in out_paths:
                link = upload_to_google_drive(out_paths["month_html"], DRIVE_FOLDER_ID)
                if link:
                    logger.info(f"Monthly plot uploaded: {link}")

    # Git commit and push
    if ENABLE_GIT_PUSH:
        logger.info("Attempting to commit and push changes to Git...")
        date_str = get_yesterday_str(utc=True) if not (INITIAL_BULK_DOWNLOAD and is_first_run) else "bulk-download"
        commit_msg = GIT_COMMIT_MESSAGE_TEMPLATE.format(date=date_str)
        
        # Specify important files to commit
        files_to_commit = [
            "chl_timeseries.csv",
            "plots/*.html",
            "plots/*.png",
            "automation_log.txt"
        ]
        
        success = git_commit_and_push(GIT_REPO_PATH, files_to_add=None, commit_message=commit_msg)
        if success:
            logger.info("Git commit and push completed successfully")
        else:
            logger.warning("Git commit/push had issues - check logs")

    # Cleanup old files
    cleanup_old_files(DOWNLOAD_DIR, KEEP_DAYS)

    logger.info("=== Pipeline finished ===")


if __name__ == "__main__":
    main()