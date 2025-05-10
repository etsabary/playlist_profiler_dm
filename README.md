# Playlist Profiler and Generator DM

This project is a three-step pipeline designed to:
1. Analyze your existing music playlists against extensive music databases (Discogs, MusicBrainz) to understand your musical taste.
2. Recommend new tracks based on this taste profile, scored and ranked according to your preferences.
3. Allow you to filter these recommendations and generate a final playlist file.

## Prerequisites

*   **Python 3**: Ensure Python 3 is installed on your system.
*   **Python Libraries**: The required libraries are listed in `requirements.txt`.
*   **Music Data**:
    *   You need access to collections of music metadata in JSON format, specifically `releases_*.json` files.

### Essential Music Data Details

**Important Clarification:** The scripts in this project (`1-playlist_profilerDM.py` and `2-makelistDM.py`) **do not parse the raw, full data dumps from Discogs or MusicBrainz.**

These scripts expect that you have already processed the original dumps (which often come in XML or other large-scale formats) into a collection of smaller JSON files, specifically `releases_*.json` files.

**What these `releases_*.json` files should contain:**

While the exact parsing method is up to you, each `releases_*.json` file is expected to represent a musical release (album, EP, single, etc.) and should contain structured data that includes, at a minimum:

*   Track titles
*   Artist names (primary artists, featured artists if possible)
*   Release title
*   Genres
*   Subgenres
*   Record labels
*   Release year/date
*   Country of release

**How to obtain the required `releases_*.json` files:**

1.  **Download Raw Dumps**: Obtain the data dumps from [Discogs](https://data.discogs.com/) and [MusicBrainz](https://musicbrainz.org/doc/MusicBrainz_Database/Download).
2.  **Parse the Dumps**:
    *   You will need to use or create separate tools/scripts to parse these raw dumps and convert them into the `releases_*.json` format.
    *   The owner of this repository also provides example parsers that might serve as a starting point or be used directly:
        *   For Discogs: [etsabary/discogs_data_parser](https://github.com/etsabary/discogs_data_parser)
        *   For MusicBrainz: [etsabary/musicbrainz_data_parser](https://github.com/etsabary/musicbrainz_data_parser)
        (Note: These are typically located in a folder one level up from this `playlist_profiler_dm` repository, alongside the `essential_discogs_data` and `essential_musicbrainz_data` folders they help create.)
    *   Look for existing third-party parsers for Discogs/MusicBrainz data that can output JSON per release. For example, some tools might convert Discogs XML dumps into line-delimited JSON (`.jsonl`) which you might then need to split or process further into the `releases_*.json` structure expected here.
    *   If writing your own parser, aim to extract the fields mentioned above for each release into its own JSON file. The naming convention `releases_*.json` suggests multiple files, possibly one per release or batched releases.

**Data Placement:**

*   Once you have your `releases_*.json` files, place them into two folders:
    *   `essential_discogs_data` (for Discogs-derived JSONs)
    *   `essential_musicbrainz_data` (for MusicBrainz-derived JSONs)
*   As stated in the main setup, these two folders must be located **one level above** the directory where these scripts are cloned (e.g., if scripts are in `~/playlist_tools/`, data should be in `~/essential_discogs_data/` and `~/essential_musicbrainz_data/`).

This pre-processing step is crucial as the provided scripts are designed to work with this specific pre-processed JSON structure for efficient analysis and recommendation.
    *   These files should be placed in folders named `essential_discogs_data` and `essential_musicbrainz_data`.
    *   These data folders must be located **one level above** the directory containing these scripts (e.g., if scripts are in `~/playlist_tools/`, data should be in `~/essential_discogs_data/`).
    *   The scripts `1-playlist_profilerDM.py` and `2-makelistDM.py` expect these relative paths (`../essential_discogs_data`, `../essential_musicbrainz_data`).

## Setup

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Install Dependencies**:
    It's recommended to use a virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate
    pip install -r requirements.txt
    ```
3.  **Prepare Data**:
    *   Create two folders, `essential_discogs_data` and `essential_musicbrainz_data`, in the directory **above** where you cloned this repository.
    *   Populate these folders with your `releases_*.json` files from Discogs and MusicBrainz respectively.

## Workflow / How to Use

The process involves running three scripts in sequence. You can run them using the `.command` files (e.g., by double-clicking on macOS/Linux if they are executable, or via `./1-playlist_profilerDM.command` in the terminal) or directly using `python3 <script_name>.py`.

### Step 1: Profile Your Music Taste

*   **Script**: `1-playlist_profilerDM.command` or `python3 1-playlist_profilerDM.py`
*   **Purpose**: This script analyzes a playlist file you provide. It matches the tracks against the Discogs and/or MusicBrainz data to create a detailed "taste profile" based on attributes like genres, subgenres, labels, release decades, and countries.
*   **Input**:
    1.  You'll be prompted to choose a data source:
        *   Discogs only
        *   MusicBrainz only
        *   Combined (Discogs + MusicBrainz)
    2.  A file dialog will open, asking you to select your playlist file. This file should be a text-based format (e.g., `.txt`, `.csv`) exported from your music player, typically with tab-separated values, and **must** contain columns named "Name" (for track title) and "Artist".
*   **Output**:
    *   A JSON file (e.g., `YourPlaylistName_profile_two_stage_pos0p7_discogs.json`) saved in the `files/` subfolder. This file contains your weighted musical taste profile.

### Step 2: Find and Rank Recommended Tracks

*   **Script**: `2-makelistDM.command` or `python3 2-makelistDM.py`
*   **Purpose**: This script uses the taste profile generated in Step 1 to scan the Discogs/MusicBrainz data again. It finds and scores tracks that align with your profiled preferences, particularly focusing on subgenres.
*   **Input**:
    1.  A file dialog will ask you to select the `_profile_*.json` file created in Step 1.
    2.  You'll be prompted to choose a data source for finding new tracks (Discogs, MusicBrainz, or Combined).
    3.  You will then interactively select your target subgenres from a list derived from your profile (e.g., top N, top N from a rank, or manual selection).
*   **Output**:
    *   A CSV file (e.g., `YourPlaylistName_profile_ranked_tracks_10subgenres_discogs.csv`) saved in the `files/` subfolder. This file contains a list of recommended tracks, their artists, titles, calculated scores, release years, countries, and associated subgenres (in JSON format). Tracks are ranked by score.

### Step 3: Generate Your Playlist

*   **Script**: `3-playlistDM.command` or `python3 3-playlistDM.py`
*   **Purpose**: This script launches a Graphical User Interface (GUI) that allows you to load the ranked tracks from Step 2, filter them based on score and release year, and then generate a final playlist.
*   **Input**:
    1.  A file dialog will ask you to select the `_ranked_tracks_*.csv` file created in Step 2.
    2.  Using the GUI:
        *   Adjust sliders to define a desired **score range** and **year range** for tracks.
        *   Set parameters like **Playlist Size** (total number of tracks) and **Max Tracks Per Artist**.
*   **Output**:
    *   A playlist file (you'll be prompted to save it, typically as a `.txt` file). The format is `Artist - Title` per line.
    *   The playlist content is also copied to your system clipboard if the `pyperclip` Python library is installed.

## Running the Scripts

As mentioned, you can execute each step by:
*   Running the `.command` files (e.g., `./1-playlist_profilerDM.command` in a terminal, or double-clicking in a GUI if your system supports it for executable shell scripts).
*   Running the Python scripts directly: `python3 1-playlist_profilerDM.py`, then `python3 2-makelistDM.py`, and finally `python3 3-playlistDM.py`.

Make sure you are in the script's directory when running them, or provide the correct path to the script.

## Output Files

All generated files (profiles from Step 1, ranked track lists from Step 2) are saved in the `files/` subfolder, which will be created in the same directory as the scripts if it doesn't already exist. Playlists from Step 3 are saved to a location of your choice.

## Notes

*   The paths to the Discogs and MusicBrainz data folders (`../essential_discogs_data`, `../essential_musicbrainz_data`) are hardcoded at the top of `1-playlist_profilerDM.py` and `2-makelistDM.py`. If your data is elsewhere, you'll need to modify these paths in the scripts.
*   The scripts use multiprocessing for faster data processing in Steps 1 and 2. The number of processes can sometimes be configured within the scripts if needed.
*   Step 3 (GUI) requires a desktop environment to run.
