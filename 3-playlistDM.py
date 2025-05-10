import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import os
import random
try:
    import pyperclip # For clipboard functionality
    PYPERCLIP_AVAILABLE = True
except ImportError:
    print("Warning: pyperclip module not found. 'Copy to Clipboard' functionality will be disabled.")
    print("Install it using: pip install pyperclip")
    PYPERCLIP_AVAILABLE = False

# --- Constants ---
MIN_VALID_YEAR = 1900
MAX_VALID_YEAR = 2040

class PlaylistGeneratorApp:
    """
    A GUI application to select tracks based on score and year ranges,
    set parameters, generate a playlist file, and copy it to the clipboard.
    """
    def __init__(self, master):
        """Initialize the application."""
        self.master = master
        self.master.title("Playlist Generator from Ranked CSV")
        self.master.geometry("600x550")

        # --- Data Variables ---
        self.file_path = tk.StringVar()
        self.df = None # Holds the loaded DataFrame
        self.df_filtered_score = None # Holds the DataFrame filtered by score
        self.df_filtered_final = None # Holds the DataFrame filtered by score AND year

        # Score Range Data
        self.min_score = tk.DoubleVar(value=0.0)
        self.max_score = tk.DoubleVar(value=1.0)
        self.current_min_score = tk.DoubleVar(value=0.0)
        self.current_max_score = tk.DoubleVar(value=1.0)

        # Year Range Data
        self.min_year = tk.IntVar(value=MIN_VALID_YEAR)
        self.max_year = tk.IntVar(value=MAX_VALID_YEAR)
        self.current_min_year = tk.IntVar(value=MIN_VALID_YEAR)
        self.current_max_year = tk.IntVar(value=MAX_VALID_YEAR)

        # Count Data (reflects combined filtering)
        self.track_count = tk.IntVar(value=0)
        self.artist_count = tk.IntVar(value=0)
        self.total_tracks = tk.IntVar(value=0) # Total tracks in file

        # Playlist Parameter Data
        self.playlist_size = tk.IntVar(value=1000)
        self.max_per_artist = tk.IntVar(value=5)
        self.no_artist_limit = tk.BooleanVar(value=False)

        # --- Slider Canvas Variables ---
        self.canvas_width = 400
        self.canvas_height = 50
        self.padding = 20
        self.handle_width = 10
        self.handle_height = 20
        # Score Slider Elements
        self.score_canvas = None
        self.score_min_handle_id = None
        self.score_max_handle_id = None
        # Year Slider Elements
        self.year_canvas = None
        self.year_min_handle_id = None
        self.year_max_handle_id = None
        # General Drag State
        self.selected_handle_id = None
        self.dragging_canvas = None # Keep track of which canvas is active

        # --- Create Widgets ---
        self.create_widgets()

    def parse_and_validate_year(self, year_val):
        """Parses year value and validates it. Returns int or None."""
        if pd.isna(year_val): return None
        try:
            year = int(year_val)
            if MIN_VALID_YEAR <= year <= MAX_VALID_YEAR: return year
            else: return None
        except (ValueError, TypeError):
             try:
                 year = int(float(year_val))
                 if MIN_VALID_YEAR <= year <= MAX_VALID_YEAR: return year
                 else: return None
             except (ValueError, TypeError): return None

    def create_widgets(self):
        """Create and layout the GUI widgets."""
        # --- File Selection ---
        file_frame = ttk.Frame(self.master, padding="10")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(file_frame, text="Select Ranked CSV File", command=self.select_file).pack(side=tk.LEFT, padx=5)
        ttk.Label(file_frame, textvariable=self.file_path, relief=tk.SUNKEN, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # --- Overall Info Frame ---
        overall_info_frame = ttk.Frame(self.master, padding="5")
        overall_info_frame.pack(fill=tk.X, padx=5)
        ttk.Label(overall_info_frame, text="Min Score:").grid(row=0, column=0, padx=5, sticky=tk.W)
        ttk.Label(overall_info_frame, textvariable=self.min_score, width=10).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(overall_info_frame, text="Max Score:").grid(row=0, column=2, padx=(10, 5), sticky=tk.W)
        ttk.Label(overall_info_frame, textvariable=self.max_score, width=10).grid(row=0, column=3, sticky=tk.W)
        ttk.Label(overall_info_frame, text="Min Year:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(overall_info_frame, textvariable=self.min_year, width=10).grid(row=1, column=1, pady=5, sticky=tk.W)
        ttk.Label(overall_info_frame, text="Max Year:").grid(row=1, column=2, padx=(10, 5), pady=5, sticky=tk.W)
        ttk.Label(overall_info_frame, textvariable=self.max_year, width=10).grid(row=1, column=3, pady=5, sticky=tk.W)
        ttk.Label(overall_info_frame, text="Total Tracks:").grid(row=0, column=4, padx=(10, 5), sticky=tk.W)
        ttk.Label(overall_info_frame, textvariable=self.total_tracks, width=10).grid(row=0, column=5, sticky=tk.W)

        # --- Score Range Slider ---
        ttk.Label(self.master, text="Score Range:", padding=(10, 5, 0, 0)).pack(anchor=tk.W)
        self.score_canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg="lightgrey", bd=0, highlightthickness=0)
        self.score_canvas.pack(pady=(0, 10))
        self.draw_slider('score') # Draw score slider

        # --- Year Range Slider ---
        ttk.Label(self.master, text="Year Range:", padding=(10, 5, 0, 0)).pack(anchor=tk.W)
        self.year_canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg="lightgrey", bd=0, highlightthickness=0)
        self.year_canvas.pack(pady=(0, 10))
        self.draw_slider('year') # Draw year slider

        # --- Selected Range Info ---
        range_frame = ttk.Frame(self.master, padding="5")
        range_frame.pack(fill=tk.X, padx=5)
        # Score
        ttk.Label(range_frame, text="Sel. Min Score:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.min_score_val_label = ttk.Label(range_frame, textvariable=self.current_min_score, width=10)
        self.min_score_val_label.grid(row=0, column=1, sticky=tk.W)
        ttk.Label(range_frame, text="Sel. Max Score:").grid(row=0, column=2, padx=(10, 5), sticky=tk.W)
        self.max_score_val_label = ttk.Label(range_frame, textvariable=self.current_max_score, width=10)
        self.max_score_val_label.grid(row=0, column=3, sticky=tk.W)
        # Year
        ttk.Label(range_frame, text="Sel. Min Year:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.min_year_val_label = ttk.Label(range_frame, textvariable=self.current_min_year, width=10)
        self.min_year_val_label.grid(row=1, column=1, pady=5, sticky=tk.W)
        ttk.Label(range_frame, text="Sel. Max Year:").grid(row=1, column=2, padx=(10, 5), pady=5, sticky=tk.W)
        self.max_year_val_label = ttk.Label(range_frame, textvariable=self.current_max_year, width=10)
        self.max_year_val_label.grid(row=1, column=3, pady=5, sticky=tk.W)
        # Counts (Combined Filter)
        ttk.Label(range_frame, text="Tracks in Range:").grid(row=0, column=4, padx=(10, 5), sticky=tk.W)
        self.count_label = ttk.Label(range_frame, textvariable=self.track_count, width=10)
        self.count_label.grid(row=0, column=5, sticky=tk.W)
        ttk.Label(range_frame, text="Unique Artists:").grid(row=1, column=4, padx=(10, 5), pady=5, sticky=tk.W)
        self.artist_count_label = ttk.Label(range_frame, textvariable=self.artist_count, width=10)
        self.artist_count_label.grid(row=1, column=5, pady=5, sticky=tk.W)


        # --- Playlist Parameters ---
        params_frame = ttk.LabelFrame(self.master, text="Playlist Parameters", padding="10")
        params_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(params_frame, text="Playlist Size:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.size_scale = ttk.Scale(params_frame, from_=50, to=5000, orient=tk.HORIZONTAL, length=200, variable=self.playlist_size, command=lambda s: self.playlist_size.set(int(float(s))))
        self.size_scale.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Label(params_frame, textvariable=self.playlist_size, width=5).grid(row=0, column=2, padx=5, pady=5)
        ttk.Label(params_frame, text="Max Tracks/Artist:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.artist_scale = ttk.Scale(params_frame, from_=1, to=10, orient=tk.HORIZONTAL, length=200, variable=self.max_per_artist, command=lambda s: self.max_per_artist.set(int(float(s))))
        self.artist_scale.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Label(params_frame, textvariable=self.max_per_artist, width=5).grid(row=1, column=2, padx=5, pady=5)
        self.no_limit_check = ttk.Checkbutton(params_frame, text="No Limit", variable=self.no_artist_limit, command=self.toggle_artist_limit)
        self.no_limit_check.grid(row=1, column=3, padx=10, pady=5)

        # --- Generate Button ---
        self.generate_button = ttk.Button(self.master, text="Generate Playlist", command=self.generate_playlist)
        self.generate_button.pack(pady=10)

        # --- Bind Events to Canvases ---
        for canvas in [self.score_canvas, self.year_canvas]:
            canvas.bind("<ButtonPress-1>", self.on_press)
            canvas.bind("<B1-Motion>", self.on_drag)
            canvas.bind("<ButtonRelease-1>", self.on_release)

        # Initially disable controls
        self.disable_controls()

    def draw_slider(self, slider_type):
        """Draws slider elements for 'score' or 'year'."""
        if slider_type == 'score':
            canvas = self.score_canvas; min_val = self.min_score.get(); max_val = self.max_score.get()
            min_tag = "score_min_handle"; max_tag = "score_max_handle"; min_color = "blue"; max_color = "red"
        elif slider_type == 'year':
            canvas = self.year_canvas; min_val = self.min_year.get(); max_val = self.max_year.get()
            min_tag = "year_min_handle"; max_tag = "year_max_handle"; min_color = "darkgreen"; max_color = "orange"
        else: return

        min_x = self.value_to_x(min_val, slider_type); max_x = self.value_to_x(max_val, slider_type)
        canvas.delete(f"{slider_type}_line", f"{slider_type}_handle") # Clear specific slider elements

        line_y = self.canvas_height // 2
        canvas.create_line(self.padding, line_y, self.canvas_width - self.padding, line_y, width=3, fill="grey", tags=f"{slider_type}_line")
        min_handle_id = canvas.create_rectangle(min_x - self.handle_width / 2, line_y - self.handle_height / 2, min_x + self.handle_width / 2, line_y + self.handle_height / 2, fill=min_color, outline="black", tags=(min_tag, f"{slider_type}_handle", "handle"))
        max_handle_id = canvas.create_rectangle(max_x - self.handle_width / 2, line_y - self.handle_height / 2, max_x + self.handle_width / 2, line_y + self.handle_height / 2, fill=max_color, outline="black", tags=(max_tag, f"{slider_type}_handle", "handle"))

        if slider_type == 'score': self.score_min_handle_id = min_handle_id; self.score_max_handle_id = max_handle_id
        else: self.year_min_handle_id = min_handle_id; self.year_max_handle_id = max_handle_id

    def select_file(self):
        """Open file dialog to select CSV and load data."""
        fpath = filedialog.askopenfilename(title="Select Ranked CSV File", filetypes=[("CSV files", "*.csv")])
        if fpath: self.file_path.set(os.path.basename(fpath)); self.load_data(fpath)
        else: self.file_path.set(""); self.df = None; self.disable_controls()

    def load_data(self, fpath):
        """Load data from CSV, find score/year ranges, and update GUI."""
        try:
            cols_to_read = ['Artist', 'Title', 'Combined Score', 'Year']; self.df = pd.read_csv(fpath, usecols=lambda c: c in cols_to_read)
            required_cols = ['Artist', 'Title', 'Combined Score', 'Year']
            if not all(col in self.df.columns for col in required_cols): missing = [col for col in required_cols if col not in self.df.columns]; messagebox.showerror("Error", f"CSV file missing required columns: {', '.join(missing)}"); raise ValueError("Missing required columns")
            self.df['Combined Score'] = pd.to_numeric(self.df['Combined Score'], errors='coerce'); self.df.dropna(subset=['Combined Score'], inplace=True)
            self.df['Year_parsed'] = self.df['Year'].apply(self.parse_and_validate_year); valid_years_df = self.df.dropna(subset=['Year_parsed'])
            self.df.dropna(subset=['Artist', 'Title'], inplace=True); self.df['Artist'] = self.df['Artist'].astype(str); self.df['Title'] = self.df['Title'].astype(str)
            if self.df.empty: messagebox.showerror("Error", "No valid data rows found after cleaning."); raise ValueError("Empty DataFrame after cleaning")

            min_s = self.df['Combined Score'].min(); max_s = self.df['Combined Score'].max()
            min_y = int(valid_years_df['Year_parsed'].min()) if not valid_years_df.empty else MIN_VALID_YEAR
            max_y = int(valid_years_df['Year_parsed'].max()) if not valid_years_df.empty else MAX_VALID_YEAR
            total_n = len(self.df)

            self.min_score.set(round(min_s, 6)); self.max_score.set(round(max_s, 6)); self.current_min_score.set(round(min_s, 6)); self.current_max_score.set(round(max_s, 6))
            self.min_year.set(min_y); self.max_year.set(max_y); self.current_min_year.set(min_y); self.current_max_year.set(max_y)
            self.total_tracks.set(total_n)

            self.enable_controls(); self.reset_slider_positions(); self.update_range_info()
            print(f"Loaded data: Score Range=[{min_s:.4f}, {max_s:.4f}], Year Range=[{min_y}, {max_y}], Total Tracks={total_n}")
        except Exception as e: messagebox.showerror("Error", f"Failed to load or process CSV file:\n{fpath}\n\nError: {e}"); self.df = None; self.file_path.set(""); self.disable_controls()

    # --- Coordinate/Value Conversion ---
    def value_to_x(self, value, slider_type):
        """Convert a value (score or year) to an x-coordinate."""
        if slider_type == 'score': min_val = self.min_score.get(); max_val = self.max_score.get()
        elif slider_type == 'year': min_val = self.min_year.get(); max_val = self.max_year.get()
        else: return self.padding
        val_range = max_val - min_val; canvas_range = self.canvas_width - 2 * self.padding
        if val_range == 0: return self.padding # Avoid division by zero
        # Ensure proportion calculation handles potential floating point issues
        proportion = (float(value) - float(min_val)) / float(val_range) if val_range != 0 else 0
        x = self.padding + proportion * canvas_range
        return max(self.padding, min(x, self.canvas_width - self.padding))

    def x_to_value(self, x, slider_type):
        """Convert an x-coordinate to a value (score or year)."""
        if slider_type == 'score': min_val = self.min_score.get(); max_val = self.max_score.get()
        elif slider_type == 'year': min_val = self.min_year.get(); max_val = self.max_year.get()
        else: return 0
        val_range = max_val - min_val; canvas_range = self.canvas_width - 2 * self.padding
        if canvas_range <= 0: return min_val # Avoid division by zero or negative range
        proportion = (float(x) - self.padding) / float(canvas_range)
        # Clamp proportion
        proportion = max(0.0, min(1.0, proportion))
        value = float(min_val) + proportion * float(val_range)
        # Clamp value to theoretical min/max before rounding/casting
        value = max(float(min_val), min(value, float(max_val)))
        return int(round(value)) if slider_type == 'year' else round(value, 6)


    def reset_slider_positions(self):
        """Move slider handles to the current min/max values."""
        if self.score_min_handle_id and self.score_max_handle_id:
            min_x = self.value_to_x(self.min_score.get(), 'score'); max_x = self.value_to_x(self.max_score.get(), 'score'); line_y = self.canvas_height // 2
            self.score_canvas.coords(self.score_min_handle_id, min_x - self.handle_width / 2, line_y - self.handle_height / 2, min_x + self.handle_width / 2, line_y + self.handle_height / 2)
            self.score_canvas.coords(self.score_max_handle_id, max_x - self.handle_width / 2, line_y - self.handle_height / 2, max_x + self.handle_width / 2, line_y + self.handle_height / 2)
        if self.year_min_handle_id and self.year_max_handle_id:
            min_x = self.value_to_x(self.min_year.get(), 'year'); max_x = self.value_to_x(self.max_year.get(), 'year'); line_y = self.canvas_height // 2
            self.year_canvas.coords(self.year_min_handle_id, min_x - self.handle_width / 2, line_y - self.handle_height / 2, min_x + self.handle_width / 2, line_y + self.handle_height / 2)
            self.year_canvas.coords(self.year_max_handle_id, max_x - self.handle_width / 2, line_y - self.handle_height / 2, max_x + self.handle_width / 2, line_y + self.handle_height / 2)

    # --- Event Handlers ---
    def on_press(self, event):
        """Handle mouse button press on either canvas. Prioritize handle closest to click when overlapping."""
        canvas = event.widget; click_x = event.x
        self.selected_handle_id = None; self.dragging_canvas = None
        slider_type = 'score' if canvas == self.score_canvas else 'year'
        min_handle_id = self.score_min_handle_id if slider_type == 'score' else self.year_min_handle_id
        max_handle_id = self.score_max_handle_id if slider_type == 'score' else self.year_max_handle_id
        if not min_handle_id or not max_handle_id: return

        min_coords = canvas.coords(min_handle_id); max_coords = canvas.coords(max_handle_id)
        min_center_x = (min_coords[0] + min_coords[2]) / 2; max_center_x = (max_coords[0] + max_coords[2]) / 2
        line_y = self.canvas_height // 2
        if not (line_y - self.handle_height / 2 <= event.y <= line_y + self.handle_height / 2): return

        overlap_threshold = self.handle_width * 0.75 # Allow slight overlap before special handling
        handles_overlap = abs(min_center_x - max_center_x) < overlap_threshold
        clicked_on_min = min_coords[0] <= click_x <= min_coords[2]
        clicked_on_max = max_coords[0] <= click_x <= max_coords[2]

        if handles_overlap and (clicked_on_min or clicked_on_max):
            dist_min_center = abs(click_x - min_center_x) # Use distance to center for overlap
            dist_max_center = abs(click_x - max_center_x)
            if dist_min_center <= dist_max_center: self.selected_handle_id = min_handle_id
            else: self.selected_handle_id = max_handle_id
        elif clicked_on_min: self.selected_handle_id = min_handle_id
        elif clicked_on_max: self.selected_handle_id = max_handle_id

        if self.selected_handle_id: self.dragging_canvas = canvas; canvas.itemconfig(self.selected_handle_id, fill="yellow")


    def on_drag(self, event):
        """Handle mouse drag event on either canvas."""
        if self.selected_handle_id and self.dragging_canvas and self.df is not None:
            canvas = self.dragging_canvas
            slider_type = 'score' if canvas == self.score_canvas else 'year'
            new_x = max(self.padding, min(event.x, self.canvas_width - self.padding))

            min_handle_id = self.score_min_handle_id if slider_type == 'score' else self.year_min_handle_id
            max_handle_id = self.score_max_handle_id if slider_type == 'score' else self.year_max_handle_id

            min_coords = canvas.coords(min_handle_id); max_coords = canvas.coords(max_handle_id)
            min_x_center = (min_coords[0] + min_coords[2]) / 2
            max_x_center = (max_coords[0] + max_coords[2]) / 2

            # --- MODIFIED Collision Check: Prevent centers crossing by 1 pixel ---
            min_separation = 1 # Minimum pixel separation between centers
            if self.selected_handle_id == min_handle_id:
                new_x = min(new_x, max_x_center - min_separation)
            elif self.selected_handle_id == max_handle_id:
                new_x = max(new_x, min_x_center + min_separation)
            # --- End Modification ---

            # Move handle visually
            line_y = self.canvas_height // 2
            canvas.coords(self.selected_handle_id, new_x - self.handle_width / 2, line_y - self.handle_height / 2, new_x + self.handle_width / 2, line_y + self.handle_height / 2)

            # Update corresponding current value variable based on final handle positions
            min_handle_x_final = (canvas.coords(min_handle_id)[0] + canvas.coords(min_handle_id)[2]) / 2
            max_handle_x_final = (canvas.coords(max_handle_id)[0] + canvas.coords(max_handle_id)[2]) / 2

            current_min_val = self.x_to_value(min_handle_x_final, slider_type)
            current_max_val = self.x_to_value(max_handle_x_final, slider_type)

            # Ensure min <= max after conversion, especially for integers
            if current_min_val > current_max_val:
                 # If they crossed due to rounding, set them equal
                 if slider_type == 'year':
                     # Set the one being dragged to the value of the other one
                     if self.selected_handle_id == min_handle_id: current_min_val = current_max_val
                     else: current_max_val = current_min_val
                 else: # For score, just swap
                     current_min_val, current_max_val = current_max_val, current_min_val

            # Update Tkinter variables
            if slider_type == 'score': self.current_min_score.set(current_min_val); self.current_max_score.set(current_max_val)
            else: self.current_min_year.set(current_min_val); self.current_max_year.set(current_max_val)

            self.update_range_info() # Update counts based on both sliders

    def on_release(self, event):
        """Handle mouse button release."""
        if self.selected_handle_id and self.dragging_canvas:
            canvas = self.dragging_canvas
            if canvas == self.score_canvas: original_color = "blue" if self.selected_handle_id == self.score_min_handle_id else "red"
            elif canvas == self.year_canvas: original_color = "darkgreen" if self.selected_handle_id == self.year_min_handle_id else "orange"
            else: original_color = "grey"
            canvas.itemconfig(self.selected_handle_id, fill=original_color)
            self.selected_handle_id = None; self.dragging_canvas = None

    def update_range_info(self):
        """Filter DataFrame based on BOTH score and year ranges."""
        if self.df is not None:
            min_s = self.current_min_score.get(); max_s = self.current_max_score.get()
            min_y = self.current_min_year.get(); max_y = self.current_max_year.get()
            if min_s > max_s: min_s, max_s = max_s, min_s
            if min_y > max_y: min_y, max_y = max_y, min_y

            try:
                self.df_filtered_final = self.df[
                    (self.df['Combined Score'] >= min_s) & (self.df['Combined Score'] <= max_s) &
                    (self.df['Year_parsed'] >= min_y) & (self.df['Year_parsed'] <= max_y)
                ].copy()
                count = len(self.df_filtered_final); artist_c = self.df_filtered_final['Artist'].nunique() if count > 0 else 0
                self.track_count.set(count); self.artist_count.set(artist_c)
            except Exception as e: print(f"Error during combined filtering/counting: {e}"); self.track_count.set(0); self.artist_count.set(0); self.df_filtered_final = None
        else: self.track_count.set(0); self.artist_count.set(0); self.df_filtered_final = None

    def toggle_artist_limit(self):
        """Enable/disable the max tracks per artist scale."""
        if self.no_artist_limit.get(): self.artist_scale.config(state=tk.DISABLED)
        else: self.artist_scale.config(state=tk.NORMAL)

    def generate_playlist(self):
        """Generate playlist based on selected ranges and parameters."""
        if self.df_filtered_final is None or self.df_filtered_final.empty: messagebox.showwarning("No Tracks", "No tracks selected in the current score AND year range."); return
        target_size = self.playlist_size.get(); max_artist = self.max_per_artist.get() if not self.no_artist_limit.get() else float('inf')
        print(f"\nGenerating playlist..."); print(f"  Target Size: {target_size}"); print(f"  Max per Artist: {'No Limit' if max_artist == float('inf') else max_artist}"); print(f"  Tracks available in selected range: {len(self.df_filtered_final)}")
        shuffled_df = self.df_filtered_final.sample(frac=1).reset_index(drop=True)
        playlist = []; artist_counts = {}
        for index, row in shuffled_df.iterrows():
            if len(playlist) >= target_size: break
            artist = row['Artist']; title = row['Title']; current_artist_count = artist_counts.get(artist, 0)
            if current_artist_count < max_artist: playlist.append(f"{artist} - {title}"); artist_counts[artist] = current_artist_count + 1
        print(f"Selected {len(playlist)} tracks for the playlist.")
        if not playlist: messagebox.showinfo("Empty Playlist", "Could not select any tracks based on the criteria."); return
        playlist_text = "\n".join(playlist); save_path = filedialog.asksaveasfilename(title="Save Playlist As", defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f: f.write(playlist_text); print(f"Playlist saved to: {save_path}")
                if PYPERCLIP_AVAILABLE:
                    try: pyperclip.copy(playlist_text); print("Playlist copied to clipboard."); messagebox.showinfo("Playlist Generated", f"Playlist saved to:\n{save_path}\n\nAlso copied to clipboard.")
                    except Exception as clip_err: print(f"Error copying to clipboard: {clip_err}"); messagebox.showinfo("Playlist Saved", f"Playlist saved to:\n{save_path}\n\n(Could not copy to clipboard: {clip_err})")
                else: messagebox.showinfo("Playlist Saved", f"Playlist saved to:\n{save_path}\n\n(Clipboard functionality disabled - install pyperclip).")
            except Exception as e: messagebox.showerror("Save Error", f"Failed to save playlist file:\n{save_path}\n\nError: {e}")
        else: print("Playlist generation cancelled by user.")

    def disable_controls(self):
        """Disable interactive elements when no file is loaded."""
        for canvas in [self.score_canvas, self.year_canvas]:
            if canvas: canvas.unbind("<ButtonPress-1>"); canvas.unbind("<B1-Motion>"); canvas.unbind("<ButtonRelease-1>")
        for label in [self.min_score_val_label, self.max_score_val_label, self.min_year_val_label, self.max_year_val_label, self.count_label, self.artist_count_label]:
             label.config(foreground="grey")
        self.size_scale.config(state=tk.DISABLED); self.artist_scale.config(state=tk.DISABLED); self.no_limit_check.config(state=tk.DISABLED); self.generate_button.config(state=tk.DISABLED)
        self.min_score.set(0.0); self.max_score.set(1.0); self.current_min_score.set(0.0); self.current_max_score.set(0.0)
        self.min_year.set(MIN_VALID_YEAR); self.max_year.set(MAX_VALID_YEAR); self.current_min_year.set(MIN_VALID_YEAR); self.current_max_year.set(MAX_VALID_YEAR)
        self.track_count.set(0); self.artist_count.set(0); self.total_tracks.set(0)
        self.playlist_size.set(1000); self.max_per_artist.set(5); self.no_artist_limit.set(False)
        self.draw_slider('score'); self.draw_slider('year')

    def enable_controls(self):
        """Enable interactive elements after a file is loaded."""
        for canvas in [self.score_canvas, self.year_canvas]:
             if canvas: canvas.bind("<ButtonPress-1>", self.on_press); canvas.bind("<B1-Motion>", self.on_drag); canvas.bind("<ButtonRelease-1>", self.on_release)
        for label in [self.min_score_val_label, self.max_score_val_label, self.min_year_val_label, self.max_year_val_label, self.count_label, self.artist_count_label]:
             label.config(foreground="black")
        self.size_scale.config(state=tk.NORMAL); self.no_limit_check.config(state=tk.NORMAL); self.toggle_artist_limit(); self.generate_button.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = PlaylistGeneratorApp(root)
    root.mainloop()
