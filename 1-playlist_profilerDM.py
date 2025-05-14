import os
import json
import csv
import glob
import re
import unicodedata
import time
import tkinter as tk
from tkinter import filedialog
from multiprocessing import Pool, Manager, cpu_count, freeze_support
from functools import partial
from tqdm import tqdm
import sys
import hashlib
import math
from collections import Counter

# --- Combined Configuration ---

# Data Source Folder Paths
# IMPORTANT: Adjust these paths if your data is located elsewhere relative to the script.
DISCOGS_DATA_FOLDER = '../essential_discogs_data'
MUSICBRAINZ_DATA_FOLDER = '../essential_musicbrainz_data' # Path for MusicBrainz JSONs

# Profiler Configuration
MIN_VALID_YEAR = 1900
MAX_VALID_YEAR = 2035
SUBGENRE_POS_WEIGHT_BASE = 0.7
LABEL_POS_WEIGHT_BASE = 0.7 # Added: Positional weight base for labels, can be tuned separately

# Weighting Configuration for Profiler (Release Context)
WEIGHT_PRIMARY_ARTIST_RELEASE = 3.0
WEIGHT_FEATURE_APPEARANCE = 1.0
WEIGHT_COMPILATION = 0.2
WEIGHT_UNKNOWN_CONTEXT = 0.5

# Output subfolder name
OUTPUT_SUBFOLDER_NAME = "files"

# --- Calculate number of processes for Matching part ---
TOTAL_CORES = cpu_count()
NUM_PROCESSES_MATCHER = max(1, TOTAL_CORES // 2)

# --- Helper Functions ---

def normalize_string(text, is_artist=False):
    """
    Normalizes strings for better matching (lowercase, remove punctuation, diacritics).
    Handles common variations like 'The' prefix for artists and version indicators for titles.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFD', text)
    text = "".join(c for c in text if unicodedata.category(c) != 'Mn')
    text = text.lower()

    if is_artist:
        if text.startswith('the '):
            text = text[4:]
        text = re.sub(r'\s*\(\d+\)$', '', text).strip()
    else:
        original_text = text
        text = re.sub(r'\s*\(.*?\)\s*', ' ', text).strip()
        text = re.sub(r'\s*\[.*?\]\s*', ' ', text).strip()
        patterns_to_remove_targeted = [
            r'\s+-\s*\S+\s*edit$', r'\s+feat\s*\..*$', r'\s+ft\s*\..*$',
            r'\s+remaster.*$', r'\s+live.*$', r'\s+mix$', r'\s+version$',
            r'\s+edit$', r'\s+remix$', r'\s+acoustic$', r'\s+pt\s*\.?\s*\d+$',
            r'\s+vol\s*\.?\s*\d+$', r'\s+part\s*\.?\s*\d+$', r'\s+demo$',
            r'\s+original$', r'\s+radio$', r'\s+album$', r'\s+single$',
            r'\s+instrumental$', r'\s+a?cap?p?ella$'
        ]
        current_text_for_targeted = text
        for pattern in patterns_to_remove_targeted:
            current_text_for_targeted = re.sub(pattern, '', current_text_for_targeted, flags=re.IGNORECASE).strip()
            if not current_text_for_targeted and text:
                 current_text_for_targeted = text
                 break
        text = current_text_for_targeted
        if not text.strip() and original_text.strip():
             temp_orig_no_brackets = re.sub(r'\s*\(.*?\)\s*', ' ', original_text).strip()
             temp_orig_no_brackets = re.sub(r'\s*\[.*?\]\s*', ' ', temp_orig_no_brackets).strip()
             if not temp_orig_no_brackets:
                 text = original_text
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def read_playlist_lines(file_path):
    """Attempts to read the playlist file using common encodings."""
    encodings_to_try = ['utf-16', 'utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                print(f"Successfully opened playlist with encoding: {encoding}")
                return f.readlines()
        except (UnicodeDecodeError, UnicodeError): continue
        except FileNotFoundError:
            print(f"Error: Playlist file not found at {file_path}"); return None
        except Exception as e:
            print(f"An unexpected error occurred opening file with {encoding}: {e}"); return None
    print(f"Error: Could not decode playlist file with any attempted encoding: {encodings_to_try}")
    return None

def parse_playlist(file_path):
    """Parses the user's playlist file, returning a set of normalized (title, artist) tuples."""
    print(f"Attempting to parse playlist file: {file_path}")
    playlist_tracks = set()
    lines = read_playlist_lines(file_path)
    if lines is None: return None
    try:
        reader = csv.reader(lines, delimiter='\t', quotechar='"', skipinitialspace=True)
        header, header_found, header_line_index = [], False, -1
        for i, line_content in enumerate(reader):
            if not isinstance(line_content, list) or not line_content: continue
            if line_content[0].startswith('\ufeff'): line_content[0] = line_content[0].lstrip('\ufeff')
            if "Name" in line_content and "Artist" in line_content:
                header, header_found, header_line_index = line_content, True, i
                print(f"Found header at line {i+1}: {header}"); break
        if not header_found:
            print("Error: Could not find a valid header row containing 'Name' and 'Artist' columns."); return None
        try: name_idx, artist_idx = header.index("Name"), header.index("Artist")
        except ValueError:
            print("Error: Playlist file header must contain 'Name' and 'Artist' columns."); return None
        current_line_num = header_line_index + 1
        try:
            data_reader = csv.reader(lines[header_line_index + 1:], delimiter='\t', quotechar='"', skipinitialspace=True)
            line_num_offset = header_line_index + 1
            for i, row in enumerate(data_reader):
                current_line_num = line_num_offset + i + 1
                if not isinstance(row, list) or len(row) <= max(name_idx, artist_idx): continue
                track_title, artist_name = row[name_idx].strip(), row[artist_idx].strip()
                if track_title and artist_name:
                    norm_title = normalize_string(track_title, is_artist=False)
                    norm_artist = normalize_string(artist_name, is_artist=True)
                    if norm_title and norm_artist: playlist_tracks.add((norm_title, norm_artist))
        except csv.Error as e: print(f"CSV Error reading data row around line {current_line_num}: {e}")
        except Exception as e: print(f"Error processing data row around line {current_line_num}: {e}")
        print(f"Successfully parsed and normalized {len(playlist_tracks)} unique tracks from playlist.")
        if not playlist_tracks: print("Warning: No valid tracks found or processed in the playlist.")
        return playlist_tracks
    except Exception as e:
        print(f"An unexpected error occurred during playlist parsing: {e}"); return None

def get_data_files(data_folders_list):
    """
    Finds all releases_*.json files in the specified list of data folders.
    Logs which source (Discogs/MusicBrainz) is being scanned based on folder path.
    """
    all_json_files = []
    script_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()

    for data_folder_relative_path in data_folders_list:
        abs_data_folder = os.path.abspath(os.path.join(script_dir, data_folder_relative_path))

        # Determine source name for logging
        source_name_for_log = "Unknown Source"
        if data_folder_relative_path == DISCOGS_DATA_FOLDER:
            source_name_for_log = "Discogs"
        elif data_folder_relative_path == MUSICBRAINZ_DATA_FOLDER:
            source_name_for_log = "MusicBrainz"

        print(f"Looking for {source_name_for_log} data in: {abs_data_folder}")
        if not os.path.isdir(abs_data_folder):
            print(f"Warning: {source_name_for_log} data folder not found: {abs_data_folder}")
            continue # Skip this folder if not found

        json_files_in_folder = glob.glob(os.path.join(abs_data_folder, 'releases_*.json'))
        json_files_in_folder.sort() # Ensure consistent order
        all_json_files.extend(json_files_in_folder)
        print(f"Found {len(json_files_in_folder)} {source_name_for_log} JSON files in {abs_data_folder}.")

    if not all_json_files:
        print(f"Warning: No 'releases_*.json' files found in any of the specified data folders.")
    else:
        print(f"Total {len(all_json_files)} JSON files found from the selected source(s) for processing.")
    return all_json_files


def get_release_fingerprint(release_data):
    """Creates a SHA1 hash of key release fields for deduplicating releases."""
    key_fields = (
        release_data.get('discogs_release_title', ''), # Assuming 'discogs_release_title' is harmonized
        release_data.get('year', ''),
        tuple(sorted(release_data.get('labels', []) if release_data.get('labels') else [])),
        tuple(sorted(release_data.get('discogs_artists', []) if release_data.get('discogs_artists') else [])) # Assuming 'discogs_artists' is harmonized
    )
    return hashlib.sha1(str(key_fields).encode('utf-8')).hexdigest()

def sanitize_filename(filename):
    """Removes or replaces characters invalid for filenames."""
    filename = filename.replace('/', '_').replace('\\', '_')
    filename = re.sub(r'[<>:"|?*]', '', filename)
    filename = re.sub(r'\s+', '_', filename)
    return filename

def process_discogs_chunk(file_paths, playlist_tracks_set, progress_data):
    """Worker function to process a chunk of JSON files (Discogs or MusicBrainz)."""
    # This function's name "process_discogs_chunk" is kept for now, but it handles generic JSON files.
    matches = {}
    worker_pid = os.getpid()
    PROGRESS_UPDATE_INTERVAL = 10000
    for file_path in file_paths:
        sys.stdout.write(f"[Matcher Worker {worker_pid}] Starting file: {os.path.basename(file_path)}\n"); sys.stdout.flush()
        file_start_time = time.time()
        releases_in_file, matches_in_file, releases_processed_interval = 0, 0, 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                try: data = json.load(f)
                except json.JSONDecodeError:
                    sys.stderr.write(f"[Matcher Worker {worker_pid}] Skipping {os.path.basename(file_path)}: JSON decode error.\n"); continue
                if not isinstance(data, list): continue
                total_releases_in_file = len(data)
                for release in data:
                    releases_in_file += 1; releases_processed_interval += 1
                    if releases_processed_interval >= PROGRESS_UPDATE_INTERVAL:
                        percent = (releases_in_file / total_releases_in_file) * 100 if total_releases_in_file else 0
                        sys.stdout.write(f"[Matcher Worker {worker_pid}] ...processed {releases_in_file:,}/{total_releases_in_file:,} ({percent:.1f}%) in {os.path.basename(file_path)}\n"); sys.stdout.flush()
                        releases_processed_interval = 0
                    release_tracks = release.get('tracks', [])
                    if not release_tracks or not isinstance(release_tracks, list): continue
                    for track in release_tracks:
                        if not isinstance(track, dict): continue
                        track_title = track.get('title', '')
                        track_artists_raw = track.get('artists', [])
                        release_artists_raw = release.get('release_artists', []) # Assumed harmonized name
                        if not isinstance(track_artists_raw, list): track_artists_raw = []
                        if not isinstance(release_artists_raw, list): release_artists_raw = []
                        all_artists_raw = list(set(a for a in track_artists_raw + release_artists_raw if a and isinstance(a, str)))
                        if not track_title or not all_artists_raw: continue
                        norm_discogs_title = normalize_string(track_title, is_artist=False)
                        norm_discogs_artists_set = {normalize_string(a, is_artist=True) for a in all_artists_raw if normalize_string(a, is_artist=True)}
                        if not norm_discogs_title or not norm_discogs_artists_set: continue
                        for p_title, p_artist in playlist_tracks_set:
                            if p_title == norm_discogs_title and p_artist in norm_discogs_artists_set:
                                playlist_key = (p_title, p_artist)
                                if playlist_key not in matches:
                                    matches[playlist_key] = {'match_count': 0, 'releases': []}
                                matches[playlist_key]['match_count'] += 1; matches_in_file += 1
                                match_data = { # Field names assumed to be harmonized
                                    'discogs_release_title': release.get('release'),
                                    'discogs_track_title': track_title,
                                    'discogs_artists': all_artists_raw,
                                    'year': release.get('year'), 'country': release.get('country'),
                                    'labels': release.get('labels'), 'genres': release.get('genres'),
                                    'subgenres': release.get('subgenres'), 'duration': track.get('duration'),
                                    'source_file': os.path.basename(file_path)
                                }
                                matches[playlist_key]['releases'].append(match_data)
        except Exception as e: sys.stderr.write(f"[Matcher Worker {worker_pid}] Error processing {os.path.basename(file_path)}: {e}\n")
        finally:
            with progress_data['lock']:
                 progress_data['files_processed'] += 1
                 progress_data['releases_scanned'] += releases_in_file
                 progress_data['matches_found'] += matches_in_file
            file_end_time = time.time()
            sys.stdout.write(f"[Matcher Worker {worker_pid}] Finished {os.path.basename(file_path)} in {file_end_time - file_start_time:.2f}s. Found {matches_in_file} raw matches in {releases_in_file} releases.\n"); sys.stdout.flush()
    return matches

def get_decade(year_str):
    """Converts a year string to a decade string (e.g., '1980s')."""
    if not year_str or not isinstance(year_str, str) or not year_str.isdigit(): return None
    try:
        year = int(year_str)
        if MIN_VALID_YEAR <= year <= MAX_VALID_YEAR: return f"{(year // 10) * 10}s"
        return None
    except ValueError: return None

def normalize_distribution(weighted_counter):
    """Normalizes a weighted Counter into a probability distribution (sums to 1.0), sorted."""
    total_weight = sum(weighted_counter.values())
    if total_weight == 0: return {}
    distribution = {item: weight / total_weight for item, weight in weighted_counter.items()}
    return dict(sorted(distribution.items(), key=lambda item: item[1], reverse=True))

def determine_release_weight(release_data, original_playlist_artist_normalized):
    """Determines the context weight for a release relative to the playlist artist."""
    release_artists_raw = release_data.get('discogs_artists', []) # Assumed harmonized name
    if not isinstance(release_artists_raw, list): release_artists_raw = []
    normalized_release_artists = {normalize_string(a, is_artist=True) for a in release_artists_raw if a and isinstance(a, str)}
    if 'various artists' in normalized_release_artists or 'various' in normalized_release_artists:
        return WEIGHT_COMPILATION
    if original_playlist_artist_normalized in normalized_release_artists:
        return WEIGHT_PRIMARY_ARTIST_RELEASE
    # This condition might be less frequently hit if the above is true.
    if original_playlist_artist_normalized in normalized_release_artists and len(normalized_release_artists) > 1:
         return WEIGHT_FEATURE_APPEARANCE
    return WEIGHT_UNKNOWN_CONTEXT

# --- Main Operational Functions ---

def run_playlist_matching(playlist_file_path, data_folders_to_scan, source_tag_for_logging):
    """Orchestrates playlist matching against specified data folders."""
    print(f"\n--- Starting Phase 1: Playlist Matching (Source: {source_tag_for_logging.upper()}) ---")
    parse_start_time = time.time()
    user_playlist_tracks = parse_playlist(playlist_file_path)
    if not user_playlist_tracks: print("Failed to parse playlist or playlist is empty. Cannot proceed."); return None
    print(f"Playlist parsing took: {time.time() - parse_start_time:.2f} seconds.")

    all_json_files = get_data_files(data_folders_to_scan) # Use the generalized function
    if not all_json_files: print(f"No data files found from source(s): {source_tag_for_logging}. Cannot proceed."); return None

    total_files_to_process = len(all_json_files)
    print(f"\nPreparing to scan data with {NUM_PROCESSES_MATCHER} worker(s)...")
    chunk_size = max(1, (total_files_to_process + NUM_PROCESSES_MATCHER - 1) // NUM_PROCESSES_MATCHER)
    file_chunks = [all_json_files[i:i + chunk_size] for i in range(0, total_files_to_process, chunk_size)]
    print(f"Divided {total_files_to_process} data files into {len(file_chunks)} chunks.")

    manager = Manager()
    progress_shared_data = manager.dict({'files_processed': 0, 'releases_scanned': 0, 'matches_found': 0, 'lock': manager.Lock()})
    worker_func = partial(process_discogs_chunk, playlist_tracks_set=user_playlist_tracks, progress_data=progress_shared_data)

    print(f"\nStarting data scan from source(s): {source_tag_for_logging.upper()} (this may take a while)...")
    final_aggregated_matches = {}; scan_start_time = time.time(); pool = None
    try:
        pool = Pool(processes=NUM_PROCESSES_MATCHER)
        async_result = pool.imap_unordered(worker_func, file_chunks); pool.close()
        with tqdm(total=total_files_to_process, desc=f"Scanning {source_tag_for_logging.capitalize()} Files", unit="file") as pbar:
            while True:
                with progress_shared_data['lock']:
                    current_processed = progress_shared_data['files_processed']
                    releases_scanned = progress_shared_data['releases_scanned']
                    matches_found_raw = progress_shared_data['matches_found']
                update_amount = current_processed - pbar.n
                if update_amount > 0: pbar.update(update_amount)
                pbar.set_postfix_str(f"Releases Scanned: {releases_scanned:,}, Raw Matches: {matches_found_raw:,}")
                if current_processed >= total_files_to_process: break
                time.sleep(0.5)
            if pbar.n < total_files_to_process: pbar.update(total_files_to_process - pbar.n)
            with progress_shared_data['lock']:
                 releases_scanned = progress_shared_data['releases_scanned']
                 matches_found_raw = progress_shared_data['matches_found']
            pbar.set_postfix_str(f"Releases Scanned: {releases_scanned:,}, Raw Matches: {matches_found_raw:,}")
        print("\nAll data files processed by matcher workers. Collecting results...")
        results_list = list(async_result)
        print("Merging results from matcher workers..."); merge_start_time = time.time()
        for worker_matches in results_list:
            for key, data in worker_matches.items():
                if key not in final_aggregated_matches:
                     final_aggregated_matches[key] = {'match_count': 0, 'releases': []}
                final_aggregated_matches[key]['match_count'] += data['match_count']
                final_aggregated_matches[key]['releases'].extend(data['releases'])
        print(f"Matcher results merging complete in {time.time() - merge_start_time:.2f} seconds.")
    except KeyboardInterrupt: print("\nKeyboardInterrupt detected! Terminating matcher workers...")
    except Exception as e: print(f"\nAn error occurred during parallel matching: {e}"); import traceback; traceback.print_exc()
    finally:
         if pool: print("Terminating and joining worker pool..."); pool.terminate(); pool.join()
         if 'async_result' in locals() and async_result and hasattr(async_result, '_taskqueue'):
             while True:
                 try: async_result._taskqueue.get_nowait()
                 except Exception: break
    if 'scan_start_time' not in locals(): print("Matching phase did not complete due to an early error."); return None

    scan_duration = time.time() - scan_start_time
    final_releases_scanned = progress_shared_data['releases_scanned']
    print(f"\nData scan from source(s) {source_tag_for_logging.upper()} finished in {scan_duration:.2f} seconds ({scan_duration/3600:.2f} hours).")
    print(f"Total releases scanned from source(s) {source_tag_for_logging.upper()}: {final_releases_scanned:,}")

    print("\nDeduplicating found releases per playlist track..."); dedup_start_time = time.time()
    final_unique_matches = {}; total_unique_discogs_releases_count = 0
    for key, data in final_aggregated_matches.items():
        seen_fingerprints = set(); unique_releases_for_track = []
        for release_data in data['releases']:
            fingerprint = get_release_fingerprint(release_data)
            if fingerprint not in seen_fingerprints:
                seen_fingerprints.add(fingerprint); unique_releases_for_track.append(release_data)
        if unique_releases_for_track:
            final_unique_matches[key] = {
                'discogs_match_count': data['match_count'], # Retaining 'discogs_match_count' for now, implies general match count
                'matched_releases': unique_releases_for_track
            }
            total_unique_discogs_releases_count += len(unique_releases_for_track)
    print(f"Deduplication complete in {time.time() - dedup_start_time:.2f} seconds.")
    serializable_matches_for_profiler = {f"{title} ||| {artist}": data for (title, artist), data in final_unique_matches.items()}

    print(f"\n--- Matcher Summary (Source: {source_tag_for_logging.upper()}) ---")
    print(f"Found entries for {len(final_unique_matches)} out of {len(user_playlist_tracks)} unique playlist tracks.")
    print(f"Total raw track matches found (before deduplication): {progress_shared_data['matches_found']:,}")
    print(f"Total unique releases matched across all playlist tracks (after deduplication): {total_unique_discogs_releases_count:,}")
    print(f"--- Finished Phase 1: Playlist Matching (Source: {source_tag_for_logging.upper()}) ---\n")
    return serializable_matches_for_profiler

def build_and_save_user_profile(matches_data, output_profile_filepath, source_playlist_filename, source_tag, data_sources_metadata):
    """Builds the weighted user profile with two-stage aggregation and saves it."""
    print(f"--- Starting Phase 2: Building User Profile (Source: {source_tag.upper()}) ---")
    if not matches_data: print("No match data provided to build profile. Exiting profile phase."); return

    global_playlist_subgenre_contributions = Counter()
    global_playlist_genre_contributions = Counter()
    global_playlist_label_contributions = Counter()
    global_playlist_decade_contributions = Counter()
    global_playlist_country_contributions = Counter()

    print("\nProcessing matched releases with two-stage weighting for profile...")
    for track_key, track_data in tqdm(matches_data.items(), desc=f"Profiling Playlist Songs ({source_tag.capitalize()})"):
        try:
            _, original_playlist_artist_norm = track_key.split(' ||| ', 1)
        except ValueError:
            sys.stderr.write(f"Warning: Skipping malformed track key for profiling: {track_key}\n"); continue

        song_specific_subgenre_counts = Counter()
        song_specific_genre_counts = Counter()
        song_specific_label_counts = Counter()
        song_specific_decade_counts = Counter()
        song_specific_country_counts = Counter()
        matched_releases = track_data.get('matched_releases', [])
        if not matched_releases or not isinstance(matched_releases, list): continue

        for release in matched_releases:
            if not isinstance(release, dict): continue
            base_weight = determine_release_weight(release, original_playlist_artist_norm)
            subgenres = release.get('subgenres')
            if subgenres and isinstance(subgenres, list):
                for idx, subgenre_name in enumerate(subgenres):
                    if subgenre_name and isinstance(subgenre_name, str):
                        positional_weight = math.pow(SUBGENRE_POS_WEIGHT_BASE, idx)
                        final_weight_for_subgenre = base_weight * positional_weight
                        song_specific_subgenre_counts[subgenre_name.strip()] += final_weight_for_subgenre # Ensure strip for subgenres too
            for genre_name in (g for g in release.get('genres', []) if g and isinstance(g,str)):
                song_specific_genre_counts[genre_name.strip()] += base_weight

            labels_list = release.get('labels')
            if labels_list and isinstance(labels_list, list):
                for idx, label_name_raw in enumerate(labels_list):
                    if not label_name_raw or not isinstance(label_name_raw, str):
                        continue
                    # Process label, ensuring it's valid and not in the ignore list
                    current_label_value = label_name_raw.strip() # Ensure label_name_raw is stripped
                    current_label_lower = current_label_value.lower()

                    skip_label = False
                    if not current_label_value: # Skip if empty after stripping
                        skip_label = True
                    # Check conditions for skipping the label (case-insensitive)
                    elif current_label_lower.startswith("no label"): # Condition 1
                        skip_label = True
                    elif current_label_lower.startswith('["no label"]'): # Condition 2
                        skip_label = True
                    elif current_label_lower.startswith("not on label"): # Condition 3
                        skip_label = True
                    elif "self-released" in current_label_lower: # Condition 4
                        skip_label = True

                    if skip_label:
                        continue

                    # If label is valid, calculate its weight and add to song-specific counts
                    label_positional_weight = math.pow(LABEL_POS_WEIGHT_BASE, idx)
                    final_weight_for_label = base_weight * label_positional_weight
                    song_specific_label_counts[current_label_value] += final_weight_for_label

            year_str = release.get('year'); decade = get_decade(year_str)
            if decade: song_specific_decade_counts[decade] += base_weight
            country = release.get('country')
            if country and isinstance(country, str): song_specific_country_counts[country.strip()] += base_weight

        song_normalized_subgenre_profile = normalize_distribution(song_specific_subgenre_counts)
        song_normalized_genre_profile = normalize_distribution(song_specific_genre_counts)
        song_normalized_label_profile = normalize_distribution(song_specific_label_counts)
        song_normalized_decade_profile = normalize_distribution(song_specific_decade_counts)
        song_normalized_country_profile = normalize_distribution(song_specific_country_counts)

        for subgenre, score in song_normalized_subgenre_profile.items(): global_playlist_subgenre_contributions[subgenre] += score
        for genre, score in song_normalized_genre_profile.items(): global_playlist_genre_contributions[genre] += score
        for label, score in song_normalized_label_profile.items(): global_playlist_label_contributions[label] += score
        for decade_val, score in song_normalized_decade_profile.items(): global_playlist_decade_contributions[decade_val] += score
        for country_val, score in song_normalized_country_profile.items(): global_playlist_country_contributions[country_val] += score

    print("\nAggregating contributions from all songs complete.")
    print(f"Found {len(global_playlist_subgenre_contributions)} unique contributing subgenres before final normalization.")

    print("\nNormalizing global contributions into final profile distributions...")
    final_user_profile_output = {
        "subgenre_distribution": normalize_distribution(global_playlist_subgenre_contributions),
        "genre_distribution": normalize_distribution(global_playlist_genre_contributions),
        "label_distribution": normalize_distribution(global_playlist_label_contributions),
        "decade_distribution": normalize_distribution(global_playlist_decade_contributions),
        "country_distribution": normalize_distribution(global_playlist_country_contributions),
        "_metadata": {
            "source_playlist_file": source_playlist_filename,
            "data_sources_used": data_sources_metadata, # List of sources like ["discogs", "musicbrainz"]
            "profile_type": f"two_stage_aggregation_subpos{str(SUBGENRE_POS_WEIGHT_BASE).replace('.', 'p')}_labpos{str(LABEL_POS_WEIGHT_BASE).replace('.', 'p')}_{source_tag}",
            "playlist_tracks_in_profile": len(matches_data),
            "total_song_contributions_to_subgenres": round(sum(global_playlist_subgenre_contributions.values()), 5),
            "total_song_contributions_to_genres": round(sum(global_playlist_genre_contributions.values()), 5),
            "total_song_contributions_to_labels": round(sum(global_playlist_label_contributions.values()), 5),
            "total_song_contributions_to_decades": round(sum(global_playlist_decade_contributions.values()), 5),
            "total_song_contributions_to_countries": round(sum(global_playlist_country_contributions.values()), 5),
            "profile_build_time_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "subgenre_positional_weight_base": SUBGENRE_POS_WEIGHT_BASE,
            "label_positional_weight_base": LABEL_POS_WEIGHT_BASE # Added for transparency
        }
    }
    print("Final profile normalization complete.")
    print(f"\nSaving user profile to: {output_profile_filepath}")
    try:
        with open(output_profile_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_user_profile_output, f, indent=4, ensure_ascii=False)
        print("User profile saved successfully.")
    except Exception as e: print(f"Error saving profile to JSON: {e}")
    print(f"--- Finished Phase 2: Building User Profile (Source: {source_tag.upper()}) ---")

def get_data_source_choice():
    """Prompts the user to select a data source and returns the choice."""
    while True:
        print("\nSelect data source for profiling:")
        print("  1: Discogs only")
        print("  2: MusicBrainz only")
        print("  3: Combined (Discogs + MusicBrainz)")
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

# --- Main Execution Script ---
if __name__ == "__main__":
    freeze_support()
    overall_start_time = time.time()

    print(f"--- Combined Playlist Matcher & User Profile Builder (Multi-Source, Two-Stage Profiling) ---")
    print(f"Script execution started: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Matcher will use {NUM_PROCESSES_MATCHER} worker process(es) (System CPU Cores: {TOTAL_CORES}).")
    print(f"Profiler Subgenre Positional Weight Base: {SUBGENRE_POS_WEIGHT_BASE}")
    print(f"Profiler Label Positional Weight Base: {LABEL_POS_WEIGHT_BASE}") # Added print for new constant
    print(f"Discogs Data Folder (relative to script): {DISCOGS_DATA_FOLDER}")
    print(f"MusicBrainz Data Folder (relative to script): {MUSICBRAINZ_DATA_FOLDER}")
    print(f"Output files will be saved to subfolder: '{OUTPUT_SUBFOLDER_NAME}'")

    # Get user's choice for data source
    data_source_choice = get_data_source_choice()
    source_tag = ""
    data_folders_to_scan = []
    data_sources_metadata = []

    if data_source_choice == '1':
        source_tag = "discogs"
        data_folders_to_scan = [DISCOGS_DATA_FOLDER]
        data_sources_metadata = ["discogs"]
        print(f"User selected: Discogs data only.")
    elif data_source_choice == '2':
        source_tag = "musicbrainz"
        data_folders_to_scan = [MUSICBRAINZ_DATA_FOLDER]
        data_sources_metadata = ["musicbrainz"]
        print(f"User selected: MusicBrainz data only.")
    elif data_source_choice == '3':
        source_tag = "combined"
        # Important: The order here might matter if there are identical files in both,
        # though glob should handle it. For processing, it just creates one large list of files.
        data_folders_to_scan = [DISCOGS_DATA_FOLDER, MUSICBRAINZ_DATA_FOLDER]
        data_sources_metadata = ["discogs", "musicbrainz"]
        print(f"User selected: Combined Discogs and MusicBrainz data.")

    root = tk.Tk(); root.withdraw()
    playlist_filepath = filedialog.askopenfilename(
        title="Select Your Playlist File (e.g., .txt or .csv)",
        filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    if not playlist_filepath: print("No playlist file selected. Exiting."); sys.exit(1)
    print(f"\nSelected playlist file: {playlist_filepath}")

    script_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    output_dir_path = os.path.join(script_dir, OUTPUT_SUBFOLDER_NAME)
    try:
        os.makedirs(output_dir_path, exist_ok=True)
        print(f"Output files will be saved in: {os.path.abspath(output_dir_path)}")
    except OSError as e:
        print(f"Error creating output directory '{output_dir_path}': {e}. Saving in script directory instead.")
        output_dir_path = script_dir

    matched_data_for_profiler = run_playlist_matching(playlist_filepath, data_folders_to_scan, source_tag)

    if matched_data_for_profiler is not None and matched_data_for_profiler:
        playlist_basename = os.path.basename(playlist_filepath)
        playlist_name_no_ext = os.path.splitext(playlist_basename)[0]
        safe_playlist_name_base = sanitize_filename(playlist_name_no_ext)
        profile_base_str = str(SUBGENRE_POS_WEIGHT_BASE).replace('.', 'p')
        # Include the source_tag in the output filename
        output_profile_filename = f"{safe_playlist_name_base}_profile_two_stage_pos{profile_base_str}_{source_tag}.json"
        absolute_output_profile_path = os.path.abspath(os.path.join(output_dir_path, output_profile_filename))
        build_and_save_user_profile(matched_data_for_profiler, absolute_output_profile_path, playlist_basename, source_tag, data_sources_metadata)
    elif matched_data_for_profiler is None: print("\nProfile building skipped due to errors or interruption in the matching phase.")
    else: print("\nNo matches found in the selected data source(s) for the provided playlist. Profile building skipped.")

    overall_duration = time.time() - overall_start_time
    print(f"\n--- Total script execution time: {overall_duration:.2f} seconds ({overall_duration/3600:.2f} hours). ---")
