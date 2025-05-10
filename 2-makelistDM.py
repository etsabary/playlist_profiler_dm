#!/usr/bin/env python3
"""
find_scored_subgenre_tracks_multisource.py
--------------------------------------------------
Scans selected data source(s) (Discogs, MusicBrainz, or Combined) JSON data
to find and score tracks based on the combined positional score AND user
profile rank weight of interactively selected top subgenres from a
user-selected weighted profile file.

Includes Year and Country in the output, selecting the earliest valid year
among duplicates with the same highest score.

Outputs a single ranked CSV file named based on the input profile and data source,
saved into a 'files' subfolder.
"""

import json
import os
import glob
import re
import unicodedata
import multiprocessing as mp
import csv
import sys
import time
from tqdm import tqdm
from functools import partial
import math # For score calculation if needed
import tkinter as tk
from tkinter import filedialog

# --- Configuration ---
# Relative paths to the data dump folders
DISCOGS_RELATIVE_DIR = "../essential_discogs_data"
MUSICBRAINZ_RELATIVE_DIR = "../essential_musicbrainz_data" # Added for MusicBrainz

OUTPUT_SUBFOLDER_NAME = "files" # Added for output directory

NUM_PROCESSES = 0 # 0 for auto (half cores)
DEFAULT_NUM_TOP_SUBGENRES = 10
MIN_USER_RANK_WEIGHT = 0.2
MIN_VALID_YEAR = 1900
MAX_VALID_YEAR = 2040 # Adjusted for a bit more future-proofing

# --- Paths ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError: # Fallback for interactive environments
    script_dir = os.getcwd()

# Specific data paths will be determined by user choice later
# output_csv_path will be set dynamically

# --- Normalization Helpers ---
_PUNCT_SPACE_RE = re.compile(r"[^\w]", re.UNICODE)

def normalize_subgenre_simple(text: str) -> str:
    """Simple normalization for subgenre matching: lower, remove non-alphanumeric."""
    if not text or not isinstance(text, str): return ""
    try:
        return _PUNCT_SPACE_RE.sub("", text.lower())
    except Exception: # Broad exception for safety, though less likely here
        return ""

_PUNCT_RE_ARTIST = re.compile(r"[^\w\s]", re.UNICODE) # Keeps whitespace for initial step
_SPACE_RE_ARTIST = re.compile(r"\s+", re.UNICODE) # For consolidating spaces
def normalize_artist_title(text: str) -> str:
    """Normalizes artist/title strings: NFD, remove diacritics, lower, remove punct (except space), compact space."""
    if not text or not isinstance(text, str): return ""
    try:
        text_nfd = unicodedata.normalize("NFD", text)
        text_nodiacritics = "".join(c for c in text_nfd if unicodedata.category(c) != 'Mn')
        text_lower = text_nodiacritics.lower()
        # Remove content within parentheses and the parentheses themselves, replacing with a space
        text_no_paren = re.sub(r'\s*\(.*?\)\s*', ' ', text_lower)
        # Remove content within square brackets and the brackets themselves, replacing with a space
        # --- Conditional bracket handling ---
        stripped_for_brackets_check = text_no_paren.strip()
        match_entire_bracketed = re.fullmatch(r'\[(.*)\]', stripped_for_brackets_check) # (.*) is greedy but ok with fullmatch

        if match_entire_bracketed:
            text_no_brackets = match_entire_bracketed.group(1) # Keep content if entire title was [...]
        else:
            # Otherwise, remove all bracketed parts as if they are suffixes/tags
            text_no_brackets = re.sub(r'\s*\[.*?\]\s*', ' ', text_no_paren)
        # Replace punctuation with space first, then compact multiple spaces
        text_nopunct_spaced = _PUNCT_RE_ARTIST.sub(" ", text_no_brackets) # Operate on text_no_brackets
        text_compact = _SPACE_RE_ARTIST.sub(" ", text_nopunct_spaced).strip()
        return text_compact
    except Exception as e: # Fallback if complex normalization fails
        try: # Simpler fallback
            return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', str(text).lower())).strip()
        except Exception:
            return "" # Final fallback

_GENERIC_ARTIST_TOKENS = {"various", "unknown", "va", "various artists", "unknown artist"}
def is_unknown_artist(name: str) -> bool:
    """Checks if an artist name is a generic placeholder like 'Various Artists'."""
    if not name or not isinstance(name, str): return True
    return normalize_artist_title(name) in _GENERIC_ARTIST_TOKENS

def get_specific_artist(track_artists, release_artists) -> str:
    """
    Determines the specific artist(s) for a track, preferring track-level artists.
    Filters out generic "unknown" or "various" artists.
    Returns a sorted, unique, ' / '-separated string of artist names.
    """
    valid_artists = []
    # Prefer track-specific artists if available and not generic
    if track_artists:
        valid_track_artists = [a for a in track_artists if a and not is_unknown_artist(a)]
        valid_artists.extend(valid_track_artists)

    # If no valid track artists, use release-level artists (if not generic)
    if not valid_artists and release_artists:
        valid_release_artists = [a for a in release_artists if a and not is_unknown_artist(a)]
        valid_artists.extend(valid_release_artists)

    if not valid_artists: return "" # No valid specific artist found
    # Return sorted, unique list of artists joined by " / "
    return " / ".join(sorted(list(set(valid_artists))))

def parse_and_validate_year(year_str):
    """Parses year string and validates it within MIN_VALID_YEAR and MAX_VALID_YEAR. Returns int or None."""
    if not year_str or not isinstance(year_str, str) or not year_str.isdigit():
        return None
    try:
        year = int(year_str)
        if MIN_VALID_YEAR <= year <= MAX_VALID_YEAR:
            return year
        else:
            return None # Year out of valid range
    except ValueError: # Not a valid integer string
        return None

# --- THIS IS THE FUNCTION THAT WAS MISSING OR MISPLACED IN YOUR LOCAL FILE ---
def sanitize_filename(filename):
    """Removes or replaces characters invalid for filenames."""
    filename = filename.replace('/', '_').replace('\\', '_')
    filename = re.sub(r'[<>:"|?*]', '', filename)
    filename = re.sub(r'\s+', '_', filename)
    # Remove common profile suffixes if present, as per your original script for this step
    filename = re.sub(r'_profile_weighted$', '', filename, flags=re.IGNORECASE)
    # To also remove suffixes from the two-stage profiler, you might use:
    # filename = re.sub(r'_profile_two_stage_pos\d+p\d+(_(discogs|musicbrainz|combined))?$', '', filename, flags=re.IGNORECASE)
    # The line below is more specific to the output of the *previous* script if its name was like "myplaylist_profile_two_stage_pos0p7_discogs"
    # It will remove that specific suffix.
    filename = re.sub(r'_profile_two_stage_pos\d+p\d+_(discogs|musicbrainz|combined)$', '', filename, flags=re.IGNORECASE)
    # If the profile name itself contains "_profile_two_stage_pos0p7" without the source, this will catch it:
    filename = re.sub(r'_profile_two_stage_pos\d+p\d+$', '', filename, flags=re.IGNORECASE)
    return filename
# --- END OF sanitize_filename DEFINITION ---

# --- Worker Function ---
def worker_score_tracks(files, user_rank_weights):
    """
    Processes a list of JSON data files, calculates combined score for tracks
    based on user-selected subgenres and their weights, extracts year/country.
    Returns a list of found tracks: (score, artist, title, subgenres_list, year_str, country_str).
    Field names like 'subgenres', 'release_artists', 'tracks' are assumed to be harmonized
    between Discogs and MusicBrainz JSON structures.
    """
    found_tracks = [] # List to store (combined_score, artist, title, subgenres_list, year_str, country_str)
    # pid = os.getpid() # For logging, not strictly used in output here

    target_subgenres_set = set(user_rank_weights.keys()) # Normalized selected subgenres

    for fpath in files: # Iterate through each file in the chunk assigned to this worker
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                releases = json.load(f) # Load the list of releases from the JSON file
        except Exception as e:
            # Optionally log error: print(f"Worker {pid}: Error reading/parsing {fpath}: {e}", file=sys.stderr)
            continue # Skip to next file if current one is problematic

        for release in releases: # Iterate through each release in the file
            original_release_subgenres = release.get("subgenres", [])
            if not original_release_subgenres or not isinstance(original_release_subgenres, list):
                continue # Skip release if no subgenres or invalid format

            # Normalize subgenres from the current release for matching
            release_subgenres_norm = [normalize_subgenre_simple(sg) for sg in original_release_subgenres if sg]

            # Quick check: if no intersection between release's subgenres and target subgenres, skip release
            if not target_subgenres_set.intersection(release_subgenres_norm):
                continue

            # Extract release year and country (these apply to all tracks in this release)
            release_year_str = release.get("year", "")
            release_country = release.get("country", "")
            release_artists_list = release.get("release_artists", []) # Artists for the overall release

            for track in release.get("tracks", []): # Iterate through tracks in this release
                track_title = track.get("title", "")
                if not track_title: # Skip track if no title
                    continue

                # Determine the specific artist for this track
                specific_artist = get_specific_artist(track.get("artists", []), release_artists_list)
                if not specific_artist: # Skip if no valid artist can be determined
                    continue

                # Calculate combined score for the track based on release subgenres
                combined_score = 0.0
                for i, sg_norm in enumerate(release_subgenres_norm): # i is 0-based index
                    user_weight = user_rank_weights.get(sg_norm, 0.0) # Get weight if it's a target subgenre
                    if user_weight > 0: # If this subgenre from release is one of user's targets
                        positional_weight = 1.0 / (i + 1) # Positional weight within this release's subgenre list
                        combined_score += positional_weight * user_weight

                if combined_score > 0: # Only add track if it has a positive score
                    # Append tuple: (score, original_artist, original_title, original_subgenres_list, year_str, country_str)
                    found_tracks.append((combined_score, specific_artist, track_title,
                                         original_release_subgenres, release_year_str, release_country))
    return found_tracks

# --- File Chunking for Multiprocessing ---
def chunkify(items, num_chunks):
    """Splits a list into a specified number of chunks, as evenly as possible."""
    num_chunks = max(1, num_chunks) # Ensure at least one chunk
    chunk_size = len(items) // num_chunks
    remainder = len(items) % num_chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(items[start:end])
        start = end
    return [c for c in chunks if c] # Return only non-empty chunks

# --- Profile Loading and Subgenre Processing ---
def get_sorted_subgenres_with_prob(profile_path):
    """Loads the profile, extracts subgenre distribution, returns list of (name, probability) sorted by probability."""
    print(f"Loading user profile from: {profile_path}")
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "subgenre_distribution" not in data:
            print(f"Error: 'subgenre_distribution' key not found in {profile_path}", file=sys.stderr)
            return None

        # Items are (name, probability). Sort by probability (value, index 1) in descending order.
        sorted_subgenres = sorted(data["subgenre_distribution"].items(), key=lambda item: item[1], reverse=True)

        if not sorted_subgenres: # Check if the distribution was empty
            print("Error: No subgenres found in profile's subgenre_distribution.", file=sys.stderr)
            return None
        return sorted_subgenres # Returns list of (name, probability) tuples
    except FileNotFoundError:
        print(f"Error: Profile file not found at {profile_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error loading or parsing profile {profile_path}: {e}", file=sys.stderr)
        return None

def calculate_user_weights(selected_subgenre_names, min_weight=0.2):
    """
    Calculates rank-based weights for the user's *selected* subgenre names.
    The rank is based on their order in the `selected_subgenre_names` list.
    """
    weights = {} # {normalized_subgenre_name: weight}
    n_actual = len(selected_subgenre_names)
    if n_actual == 0: return weights
    if n_actual == 1: # If only one subgenre selected, it gets full weight (1.0)
        weights[normalize_subgenre_simple(selected_subgenre_names[0])] = 1.0
        return weights

    # Linear interpolation for weights from 1.0 down to min_weight
    max_weight = 1.0
    denominator = n_actual - 1 # Avoid division by zero if n_actual is 1 (handled above)
    weight_range = max_weight - min_weight

    # Rank is 1-based relative to the selected list (order of selection matters here if manual)
    for rank_1_based, sg_name in enumerate(selected_subgenre_names, 1):
        # Weight decreases linearly from max_weight for rank 1 to min_weight for rank n_actual
        weight = ((n_actual - rank_1_based) * weight_range / denominator) + min_weight
        weights[normalize_subgenre_simple(sg_name)] = weight
    return weights

def select_target_subgenres(all_sorted_subgenres_with_prob):
    """
    Handles interactive selection of target subgenres from the loaded profile.
    Takes a list of (name, probability) tuples, returns a list of selected subgenre names.
    The returned list of names preserves the order of selection or rank from profile.
    """
    if not all_sorted_subgenres_with_prob: return None

    max_available = len(all_sorted_subgenres_with_prob)
    print(f"\n--- Subgenre Selection ---"); print(f"Profile contains {max_available} subgenres with probabilities.")

    while True:
        print("\nChoose subgenre selection method:")
        print(f"  1. Top N subgenres (starting from Rank 1, default N={DEFAULT_NUM_TOP_SUBGENRES})")
        print(f"  2. Top N subgenres (starting from a specific Rank)")
        print(f"  3. Manual Selection by Number (from displayed list)")
        choice = input("Enter choice (1, 2, or 3): ").strip()

        if choice == '1':
            num_to_select = min(DEFAULT_NUM_TOP_SUBGENRES, max_available)
            selected_subgenres_names = [sg_tuple[0] for sg_tuple in all_sorted_subgenres_with_prob[:num_to_select]]
            print(f"Selected Top {len(selected_subgenres_names)} subgenres (by profile rank).")
            return selected_subgenres_names
        elif choice == '2':
            try:
                num_str = input(f"How many subgenres to select? (1-{max_available}, default {min(DEFAULT_NUM_TOP_SUBGENRES, max_available)}): ")
                num_to_select = min(DEFAULT_NUM_TOP_SUBGENRES, max_available) if not num_str else int(num_str)
                if not 1 <= num_to_select <= max_available:
                    print(f"Invalid number. Please enter between 1 and {max_available}."); continue

                start_rank_str = input(f"Start from which rank? (1-{max_available - num_to_select + 1}, default 1): ")
                start_rank = 1 if not start_rank_str else int(start_rank_str)

                max_possible_start_rank = max_available - num_to_select + 1
                if not 1 <= start_rank <= max_possible_start_rank:
                    print(f"Invalid start rank. Please enter between 1 and {max_possible_start_rank}."); continue

                start_index = start_rank - 1 # Convert 1-based rank to 0-based index
                end_index = start_index + num_to_select
                selected_subgenres_names = [sg_tuple[0] for sg_tuple in all_sorted_subgenres_with_prob[start_index:end_index]]
                print(f"Selected {len(selected_subgenres_names)} subgenres starting from rank {start_rank}.")
                return selected_subgenres_names
            except ValueError: print("Invalid input. Please enter numbers."); continue
        elif choice == '3':
            print("\nAvailable Subgenres (Rank. Name - Probability):")
            for i, (name, prob) in enumerate(all_sorted_subgenres_with_prob):
                print(f"  {i+1}. {name} ({prob:.4f})") # Display 1-based rank

            while True: # Loop for manual selection input
                selection_str = input(f"\nEnter numbers of subgenres to select, separated by commas (e.g., 1,3,5): ").strip()
                if not selection_str: print("No selection made."); break # Allow exiting manual selection
                try:
                    selected_indices_1based = [int(x.strip()) for x in selection_str.split(',') if x.strip()]
                    valid_selection = True; temp_selected_names = []; seen_indices_0based = set()

                    # Convert to 0-based and validate, store original names
                    for idx_1based in selected_indices_1based:
                        idx_0based = idx_1based - 1
                        if not 0 <= idx_0based < max_available:
                            print(f"Error: Index {idx_1based} is out of range (1-{max_available})."); valid_selection = False; break
                        if idx_0based in seen_indices_0based: # Avoid duplicates from user input
                            # print(f"Warning: Index {idx_1based} selected multiple times. Will be included once."); # Optional warning
                            continue
                        temp_selected_names.append(all_sorted_subgenres_with_prob[idx_0based][0])
                        seen_indices_0based.add(idx_0based)

                    if valid_selection and temp_selected_names:
                        print(f"Selected {len(temp_selected_names)} subgenres manually.")
                        # To maintain an order for rank weighting, we sort the *selected names*
                        # by their original profile rank.
                        name_to_original_rank = {name: i for i, (name, _) in enumerate(all_sorted_subgenres_with_prob)}
                        final_manual_selection_sorted_by_profile_rank = sorted(
                            temp_selected_names,
                            key=lambda name: name_to_original_rank[name]
                        )
                        return final_manual_selection_sorted_by_profile_rank
                    elif not temp_selected_names and valid_selection: print("No valid indices entered.")
                except ValueError: print("Invalid input. Please enter comma-separated numbers only.")
        else: print("Invalid choice. Please enter 1, 2, or 3.")

# --- ADDED: Function to get data source choice ---
def get_data_source_choice():
    """Prompts the user to select a data source for finding tracks."""
    while True:
        print("\nSelect data source for finding tracks:")
        print("  1: Discogs only")
        print("  2: MusicBrainz only")
        print("  3: Combined (Discogs + MusicBrainz)")
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

# --- ADDED: Function to get data files from selected sources ---
def get_data_files_from_sources(data_folders_list, source_tag_for_log):
    """
    Finds all releases_*.json files in the specified list of data folders.
    Logs which source is being scanned.
    """
    all_json_files = []
    print(f"\n--- Scanning for data files from source(s): {source_tag_for_log.upper()} ---")
    for data_folder_relative_path in data_folders_list:
        abs_data_folder = os.path.abspath(os.path.join(script_dir, data_folder_relative_path))

        source_name_display = "Unknown Source" # Determine display name for logging
        if DISCOGS_RELATIVE_DIR in data_folder_relative_path: source_name_display = "Discogs"
        elif MUSICBRAINZ_RELATIVE_DIR in data_folder_relative_path: source_name_display = "MusicBrainz"

        print(f"Looking for {source_name_display} data in: {abs_data_folder}")
        if not os.path.isdir(abs_data_folder):
            print(f"Warning: {source_name_display} data folder not found: {abs_data_folder}")
            continue

        json_files_in_folder = sorted(glob.glob(os.path.join(abs_data_folder, 'releases_*.json')))
        all_json_files.extend(json_files_in_folder)
        print(f"Found {len(json_files_in_folder)} {source_name_display} JSON files in this folder.")

    if not all_json_files:
        print(f"Warning: No 'releases_*.json' files found in any of the specified data folders for {source_tag_for_log.upper()}.")
    else:
        print(f"Total {len(all_json_files)} JSON files found from {source_tag_for_log.upper()} to process.")
    return all_json_files

# --- Main Execution ---
def main():
    start_time = time.time()
    print(f"--- Find Scored Subgenre Tracks (Multi-Source) ---")
    print(f"Script execution started: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # --- Select Profile File ---
    root_tk = tk.Tk(); root_tk.withdraw() # Initialize and hide Tkinter root window
    selected_profile_path = filedialog.askopenfilename(
        title="Select the Weighted Profile JSON file",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    root_tk.destroy() # Clean up Tkinter window
    if not selected_profile_path: print("No profile file selected. Exiting."); sys.exit(1)
    print(f"Using profile file: {selected_profile_path}")

    # --- Get Data Source Choice from User ---
    data_source_choice = get_data_source_choice()
    source_tag = ""
    data_folders_to_scan = []

    if data_source_choice == '1':
        source_tag = "discogs"
        data_folders_to_scan = [DISCOGS_RELATIVE_DIR]
        print(f"User selected: Discogs data only.")
    elif data_source_choice == '2':
        source_tag = "musicbrainz"
        data_folders_to_scan = [MUSICBRAINZ_RELATIVE_DIR]
        print(f"User selected: MusicBrainz data only.")
    elif data_source_choice == '3':
        source_tag = "combined"
        data_folders_to_scan = [DISCOGS_RELATIVE_DIR, MUSICBRAINZ_RELATIVE_DIR]
        print(f"User selected: Combined Discogs and MusicBrainz data.")

    # 1. Load profile and get sorted list of (subgenre_name, probability) tuples
    all_sorted_subgenres_with_prob = get_sorted_subgenres_with_prob(selected_profile_path)
    if not all_sorted_subgenres_with_prob: sys.exit(1) # Exit if profile loading failed

    # 2. Interactive Selection of target subgenres (returns list of original names)
    # The order of names in this list is important for rank weighting.
    selected_subgenre_names = select_target_subgenres(all_sorted_subgenres_with_prob)
    if not selected_subgenre_names: print("No target subgenres selected. Exiting."); sys.exit(0)

    # 3. Calculate user rank weights based on the *selected* list of subgenre names
    # user_rank_weights maps: {normalized_subgenre_name: rank_weight}
    user_rank_weights = calculate_user_weights(selected_subgenre_names, MIN_USER_RANK_WEIGHT)
    if not user_rank_weights: print("Error: Could not calculate user rank weights.", file=sys.stderr); sys.exit(1)

    print(f"\n--- Using {len(selected_subgenre_names)} Selected Subgenres with Calculated Rank Weights ---")
    # Display the selected subgenres and their calculated weights for user confirmation
    # Sort by weight for display, purely for readability
    temp_sorted_weights_for_display = sorted(user_rank_weights.items(), key=lambda item: item[1], reverse=True)
    for i, (sg_norm, weight) in enumerate(temp_sorted_weights_for_display):
        # Find original name for display (user_rank_weights uses normalized names as keys)
        original_name_for_display = "Unknown (check normalization)" # Fallback
        for orig_name in selected_subgenre_names:
            if normalize_subgenre_simple(orig_name) == sg_norm:
                original_name_for_display = orig_name
                break
        print(f"  {i+1}. '{original_name_for_display}' (Norm: '{sg_norm}') -> Rank Weight: {weight:.4f}")
    print("------------------------------------------------------------------------------------")

    # --- Generate dynamic output CSV filename and path ---
    profile_basename = os.path.basename(selected_profile_path)
    profile_name_no_ext = os.path.splitext(profile_basename)[0]
    # Sanitize the profile name part
    safe_profile_name_base = sanitize_filename(profile_name_no_ext) # Call to sanitize_filename

    # Create output subfolder if it doesn't exist
    output_dir_path = os.path.join(script_dir, OUTPUT_SUBFOLDER_NAME)
    try:
        os.makedirs(output_dir_path, exist_ok=True)
        print(f"Output CSV will be saved in: {os.path.abspath(output_dir_path)}")
    except OSError as e:
        print(f"Error creating output directory '{output_dir_path}': {e}. Saving in script directory instead.")
        output_dir_path = script_dir # Fallback

    # Construct filename including source tag and number of subgenres
    output_csv_filename_dynamic = f"{safe_profile_name_base}_ranked_tracks_{len(selected_subgenre_names)}subgenres_{source_tag}.csv"
    output_csv_path = os.path.join(output_dir_path, output_csv_filename_dynamic)
    print(f"Output CSV file will be named: {output_csv_filename_dynamic}")

    # 4. Get list of data files to process from selected source(s)
    files_to_process = get_data_files_from_sources(data_folders_to_scan, source_tag)
    if not files_to_process:
        print(f"Error: No data files found for source(s) '{source_tag.upper()}'. Cannot proceed.", file=sys.stderr)
        sys.exit(1)

    # 5. Setup Multiprocessing
    if NUM_PROCESSES == 0: # Auto-detect: use half of available cores
        n_workers = max(1, mp.cpu_count() // 2)
    else: # Use specified number of processes
        n_workers = NUM_PROCESSES
    # Ensure number of workers doesn't exceed number of files (or chunks)
    if len(files_to_process) < n_workers: n_workers = max(1, len(files_to_process))

    all_found_tracks_lists = [] # To store lists of tracks from each worker
    # Create partial function for worker, fixing the user_rank_weights argument
    worker_func_partial = partial(worker_score_tracks, user_rank_weights=user_rank_weights)

    if n_workers < 2 or len(files_to_process) < 2: # Run in single-process mode if few workers or files
        print("Running in single-process mode (less than 2 workers or files)...")
        with tqdm(total=1, desc=f"Processing {source_tag.capitalize()} Files (Single Core)") as pbar:
            all_found_tracks_lists = [worker_func_partial(files_to_process)]
            pbar.update(1)
    else: # Multiprocessing mode
        file_chunks = chunkify(files_to_process, n_workers)
        print(f"Scanning {len(files_to_process)} files with {n_workers} worker processes...")
        try:
            with mp.Pool(n_workers) as pool:
                results_iterator = pool.imap_unordered(worker_func_partial, file_chunks)
                all_found_tracks_lists = list(tqdm(results_iterator, total=len(file_chunks), desc=f"Worker Progress ({source_tag.capitalize()})"))
        except Exception as e:
            print(f"\nError during multiprocessing: {e}", file=sys.stderr)
            if 'pool' in locals() and pool is not None: pool.terminate(); pool.join()
            sys.exit(1)

    # Note: start_time is from the very beginning of main()
    processing_files_duration = time.time() - start_time
    print(f"\nFinished processing data files. Wall time for this stage might be less due to earlier setup. Total elapsed since script start: {processing_files_duration:.2f} seconds.")


    # 6. Aggregate results from all workers
    print("Aggregating results from workers...")
    all_tracks_flat = [track for track_list in all_found_tracks_lists for track in track_list]

    if not all_tracks_flat:
        print(f"Warning: No tracks found matching any of your selected target subgenres from source(s) '{source_tag.upper()}'.", file=sys.stderr)
        sys.exit(0)
    print(f"Found {len(all_tracks_flat):,} raw track occurrences matching targets from source(s) '{source_tag.upper()}'.")

    # 7. Deduplicate tracks
    print(f"\nDeduplicating tracks, keeping highest score & then earliest valid year...")
    unique_tracks_combined = {}
    duplicates_processed_count = 0

    for combined_score, artist, title, subgenres, year_str, country in all_tracks_flat:
        key = (normalize_artist_title(artist), normalize_artist_title(title))
        new_year_int = parse_and_validate_year(year_str)

        if key not in unique_tracks_combined:
            unique_tracks_combined[key] = (combined_score, artist, title, subgenres, new_year_int, country)
        else:
            duplicates_processed_count += 1
            current_best_score, _, _, _, curr_year_int, _ = unique_tracks_combined[key] # Unpack only needed parts

            if combined_score > current_best_score:
                unique_tracks_combined[key] = (combined_score, artist, title, subgenres, new_year_int, country)
            elif combined_score == current_best_score:
                if new_year_int is not None:
                    if curr_year_int is None or new_year_int < curr_year_int:
                        unique_tracks_combined[key] = (combined_score, artist, title, subgenres, new_year_int, country)

    ranked_list_final = []
    for score, art, tit, sgs, yr_int, ctry in unique_tracks_combined.values():
        subgenres_json_str = json.dumps(sgs)
        year_output_str = str(yr_int) if yr_int is not None else ""
        ranked_list_final.append([score, art, tit, year_output_str, ctry, subgenres_json_str])

    ranked_list_final.sort(key=lambda x: x[0], reverse=True)
    unique_track_count = len(ranked_list_final)
    print(f"Processed {duplicates_processed_count:,} duplicate occurrences.")
    print(f"Found {unique_track_count:,} unique tracks ranked by combined score.")

    # 8. Write Single Combined CSV
    print(f"Writing results to '{output_csv_path}'...")
    write_start_time = time.time()
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Artist', 'Title', 'Combined Score', 'Year', 'Country', 'Subgenres (JSON)'])
            for score_val, artist_val, title_val, year_out_val, country_out_val, subgenres_str_val in ranked_list_final:
                writer.writerow([artist_val, title_val, f"{score_val:.6f}", year_out_val, country_out_val, subgenres_str_val])
        write_duration = time.time() - write_start_time
        print(f"Successfully wrote CSV file in {write_duration:.2f} seconds.")
    except Exception as e:
        print(f"\nError writing CSV file {output_csv_path}: {e}", file=sys.stderr)

    total_script_duration = time.time() - start_time
    print(f"\n--- Overall Summary ---")
    print(f"Data source(s) processed: {source_tag.upper()}")
    print(f"Targeted Subgenres ({len(selected_subgenre_names)}): {', '.join(selected_subgenre_names)}")
    print(f"Found {unique_track_count:,} unique ranked tracks.")
    print(f"Output File: '{os.path.abspath(output_csv_path)}'")
    print(f"Total script execution time: {total_script_duration:.2f} seconds ({total_script_duration/3600:.2f} hours).")

if __name__ == "__main__":
    mp.freeze_support()
    main()
