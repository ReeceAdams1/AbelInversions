import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os
import json
import abel_methods as am

class AbelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Abel Inversion Analysis Tool")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Menu
        menubar = tk.Menu(root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load Settings", command=self.load_settings)
        filemenu.add_command(label="Save Settings", command=self.save_settings)
        menubar.add_cascade(label="File", menu=filemenu)
        root.config(menu=menubar)

        # Status Bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Data storage
        self.raw_data = None
        self.filename = ""
        self.center_test_data = None

        # --- Layout ---
        # Left frame for controls
        control_frame = ttk.Frame(root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Right frame for plots
        plot_frame = ttk.Frame(root, padding="10")
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Controls ---
        
        # File Selection
        ttk.Label(control_frame, text="Data File:").pack(anchor=tk.W)
        self.btn_load = ttk.Button(control_frame, text="Load CSV/Txt", command=self.load_file)
        self.btn_load.pack(fill=tk.X, pady=5)
        self.lbl_filename = ttk.Label(control_frame, text="No file loaded", wraplength=200)
        self.lbl_filename.pack(anchor=tk.W, pady=5)

        # Separator
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        # Data Extraction Settings
        ttk.Label(control_frame, text="Data Extraction", font='bold').pack(anchor=tk.W)
        
        ttk.Label(control_frame, text="Row Index:").pack(anchor=tk.W)
        self.var_row_index = tk.IntVar(value=0)
        self.ent_row_index = ttk.Entry(control_frame, textvariable=self.var_row_index)
        self.ent_row_index.pack(fill=tk.X)
        self.ent_row_index.bind('<Return>', self.on_param_change)

        ttk.Label(control_frame, text="Average +/- Rows:").pack(anchor=tk.W)
        self.var_avg_rows = tk.IntVar(value=0)
        self.ent_avg_rows = ttk.Entry(control_frame, textvariable=self.var_avg_rows)
        self.ent_avg_rows.pack(fill=tk.X)
        self.ent_avg_rows.bind('<Return>', self.on_param_change)

        # Separator
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        # Analysis Settings
        ttk.Label(control_frame, text="Analysis Parameters", font='bold').pack(anchor=tk.W)

        ttk.Label(control_frame, text="Pixel Size (cm):").pack(anchor=tk.W)
        self.var_pixel_size = tk.DoubleVar(value=300e-4 / 21.26)
        ttk.Entry(control_frame, textvariable=self.var_pixel_size).pack(fill=tk.X)

        ttk.Label(control_frame, text="Center Pixel:").pack(anchor=tk.W)
        self.var_center_px = tk.IntVar(value=458)
        center_frame = ttk.Frame(control_frame)
        center_frame.pack(fill=tk.X)
        ttk.Entry(center_frame, textvariable=self.var_center_px).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(center_frame, text="Auto", width=5, command=self.auto_find_center).pack(side=tk.RIGHT, padx=2)

        # Cutoff Settings
        self.var_use_cutoff = tk.BooleanVar(value=False)
        cutoff_frame = ttk.Frame(control_frame)
        cutoff_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Checkbutton(cutoff_frame, text="Enable Width Cutoff", variable=self.var_use_cutoff, command=self.run_analysis).pack(side=tk.LEFT)
        ttk.Button(cutoff_frame, text="Auto Detect", width=10, command=self.auto_detect_widths).pack(side=tk.RIGHT, padx=2)

        ttk.Label(control_frame, text="Left Cutoff (px):").pack(anchor=tk.W)
        self.var_left_cutoff = tk.IntVar(value=318)
        ttk.Entry(control_frame, textvariable=self.var_left_cutoff).pack(fill=tk.X)

        ttk.Label(control_frame, text="Right Cutoff (px):").pack(anchor=tk.W)
        self.var_right_cutoff = tk.IntVar(value=628)
        ttk.Entry(control_frame, textvariable=self.var_right_cutoff).pack(fill=tk.X)

        # Smoothing
        smooth_header_frame = ttk.Frame(control_frame)
        smooth_header_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(smooth_header_frame, text="Smoothing", font='bold').pack(side=tk.LEFT)
        
        self.var_smoothing_method = tk.StringVar(value='savgol')
        self.cmb_smoothing = ttk.Combobox(smooth_header_frame, textvariable=self.var_smoothing_method, 
                                          values=['savgol', 'gaussian', 'none'], width=10, state='readonly')
        self.cmb_smoothing.pack(side=tk.RIGHT)
        self.cmb_smoothing.bind('<<ComboboxSelected>>', lambda e: self.run_analysis())

        smooth_frame = ttk.Frame(control_frame)
        smooth_frame.pack(fill=tk.X)
        
        # Left Smoothing
        ttk.Label(smooth_frame, text="L Win:").grid(row=0, column=0)
        self.var_smooth_win_l = tk.IntVar(value=11)
        tk.Scale(smooth_frame, variable=self.var_smooth_win_l, from_=5, to=51, resolution=2, orient=tk.HORIZONTAL, showvalue=0, command=lambda x: self.run_analysis()).grid(row=0, column=1, sticky='ew')
        ttk.Label(smooth_frame, textvariable=self.var_smooth_win_l).grid(row=0, column=2)
        
        ttk.Label(smooth_frame, text="Poly:").grid(row=0, column=3)
        self.var_smooth_poly_l = tk.IntVar(value=3)
        ttk.Entry(smooth_frame, textvariable=self.var_smooth_poly_l, width=3).grid(row=0, column=4)

        # Right Smoothing
        ttk.Label(smooth_frame, text="R Win:").grid(row=1, column=0)
        self.var_smooth_win_r = tk.IntVar(value=11)
        tk.Scale(smooth_frame, variable=self.var_smooth_win_r, from_=5, to=51, resolution=2, orient=tk.HORIZONTAL, showvalue=0, command=lambda x: self.run_analysis()).grid(row=1, column=1, sticky='ew')
        ttk.Label(smooth_frame, textvariable=self.var_smooth_win_r).grid(row=1, column=2)

        ttk.Label(smooth_frame, text="Poly:").grid(row=1, column=3)
        self.var_smooth_poly_r = tk.IntVar(value=3)
        ttk.Entry(smooth_frame, textvariable=self.var_smooth_poly_r, width=3).grid(row=1, column=4)

        # Tapering and Background
        self.var_taper_edges = tk.BooleanVar(value=False)
        
        opts_frame = ttk.Frame(control_frame)
        opts_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Checkbutton(opts_frame, text="Taper Edges", variable=self.var_taper_edges, command=self.run_analysis).pack(side=tk.LEFT)

        # Container for Methods and Plot Options side-by-side
        methods_plot_container = ttk.Frame(control_frame)
        methods_plot_container.pack(fill=tk.X, pady=(10, 0))

        # Left Column: Inversion Methods
        methods_frame = ttk.Frame(methods_plot_container)
        methods_frame.pack(side=tk.LEFT, fill=tk.Y, expand=True, anchor=tk.N)
        
        ttk.Label(methods_frame, text="Inversion Methods:", font='bold').pack(anchor=tk.W)
        self.methods = ['basex', 'onion_peeling', 'three_point', 'hansenlaw']
        self.method_vars = {}
        
        for method in self.methods:
            var = tk.BooleanVar(value=(method == 'basex'))
            self.method_vars[method] = var
            ttk.Checkbutton(methods_frame, text=method, variable=var, command=self.run_analysis).pack(anchor=tk.W)

        # Right Column: Plot Options
        plot_opt_frame = ttk.Frame(methods_plot_container)
        plot_opt_frame.pack(side=tk.LEFT, fill=tk.Y, expand=True, anchor=tk.N, padx=(10, 0))
        
        ttk.Label(plot_opt_frame, text="Plot Options:", font='bold').pack(anchor=tk.W)
        
        self.var_show_left = tk.BooleanVar(value=True)
        self.var_show_right = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(plot_opt_frame, text="Show Left", variable=self.var_show_left, command=self.on_view_toggle).pack(anchor=tk.W)
        ttk.Checkbutton(plot_opt_frame, text="Show Right", variable=self.var_show_right, command=self.on_view_toggle).pack(anchor=tk.W)

        # Run Button
        self.btn_run = ttk.Button(control_frame, text="Update / Run Analysis", command=self.run_analysis)
        self.btn_run.pack(fill=tk.X, pady=20)
        
        # Bind Enter key to run analysis
        self.root.bind('<Return>', lambda event: self.run_analysis())

        # --- Plots ---
        self.notebook = ttk.Notebook(plot_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_2d = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_2d, text="2D Preview")
        
        self.tab_1d = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_1d, text="1D Profile")

        self.tab_result = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_result, text="Inversion Result")

        # Stats Frame inside Result Tab
        self.stats_frame = ttk.LabelFrame(self.tab_result, text="Plasma Statistics")
        self.stats_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.lbl_stats = ttk.Label(self.stats_frame, text="Run analysis to see statistics.")
        self.lbl_stats.pack(padx=5, pady=5)

        self.tab_center_test = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_center_test, text="Center Accuracy")
        
        # Controls for Center Test
        ct_control_frame = ttk.Frame(self.tab_center_test)
        ct_control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Label(ct_control_frame, text="Range (+/- px):").pack(side=tk.LEFT, padx=2)
        self.var_center_range = tk.IntVar(value=10)
        ttk.Entry(ct_control_frame, textvariable=self.var_center_range, width=5).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(ct_control_frame, text="Step:").pack(side=tk.LEFT, padx=2)
        self.var_center_step = tk.IntVar(value=1)
        ttk.Entry(ct_control_frame, textvariable=self.var_center_step, width=5).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(ct_control_frame, text="Run Center Test", command=self.run_center_test).pack(side=tk.LEFT, padx=10)

        # Plots for Center Test
        self.fig_ct, (self.ax_ct_prof, self.ax_ct_peak) = plt.subplots(1, 2, figsize=(10, 4))
        self.canvas_ct = FigureCanvasTkAgg(self.fig_ct, master=self.tab_center_test)
        self.canvas_ct.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_ct = NavigationToolbar2Tk(self.canvas_ct, self.tab_center_test)

        self.tab_cutoff_test = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_cutoff_test, text="Cutoff Accuracy")
        
        # Controls for Cutoff Test
        cut_control_frame = ttk.Frame(self.tab_cutoff_test)
        cut_control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Label(cut_control_frame, text="Range Below:").pack(side=tk.LEFT, padx=2)
        self.var_cutoff_range_below = tk.IntVar(value=10)
        ttk.Entry(cut_control_frame, textvariable=self.var_cutoff_range_below, width=5).pack(side=tk.LEFT, padx=2)

        ttk.Label(cut_control_frame, text="Range Above:").pack(side=tk.LEFT, padx=2)
        self.var_cutoff_range_above = tk.IntVar(value=10)
        ttk.Entry(cut_control_frame, textvariable=self.var_cutoff_range_above, width=5).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(cut_control_frame, text="Step:").pack(side=tk.LEFT, padx=2)
        self.var_cutoff_step = tk.IntVar(value=1)
        ttk.Entry(cut_control_frame, textvariable=self.var_cutoff_step, width=5).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(cut_control_frame, text="Run Cutoff Test", command=self.run_cutoff_test).pack(side=tk.LEFT, padx=10)

        # Plots for Cutoff Test
        self.fig_cut, (self.ax_cut_prof, self.ax_cut_peak) = plt.subplots(1, 2, figsize=(10, 4))
        self.canvas_cut = FigureCanvasTkAgg(self.fig_cut, master=self.tab_cutoff_test)
        self.canvas_cut.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_cut = NavigationToolbar2Tk(self.canvas_cut, self.tab_cutoff_test)

        # Initialize figures
        self.fig_2d, self.ax_2d = plt.subplots(figsize=(5, 4))
        self.cbar_2d = None
        self.canvas_2d = FigureCanvasTkAgg(self.fig_2d, master=self.tab_2d)
        self.canvas_2d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_2d = NavigationToolbar2Tk(self.canvas_2d, self.tab_2d)

        self.fig_1d, self.ax_1d = plt.subplots(figsize=(5, 4))
        self.canvas_1d = FigureCanvasTkAgg(self.fig_1d, master=self.tab_1d)
        self.canvas_1d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_1d = NavigationToolbar2Tk(self.canvas_1d, self.tab_1d)

        self.fig_res, self.ax_res = plt.subplots(figsize=(5, 4))
        self.canvas_res = FigureCanvasTkAgg(self.fig_res, master=self.tab_result)
        self.canvas_res.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_res = NavigationToolbar2Tk(self.canvas_res, self.tab_result)

    def on_closing(self):
        plt.close('all')
        self.root.quit()
        self.root.destroy()

    def load_settings(self):
        init_dir = r"C:\Users\Reece\Documents\GitHub\AbelInversions"
        if not os.path.exists(init_dir):
            init_dir = os.getcwd()
            
        file_path = filedialog.askopenfilename(initialdir=init_dir, filetypes=[("JSON files", "*.json")])
        if not file_path: return
        try:
            with open(file_path, 'r') as f:
                settings = json.load(f)
            
            self.var_row_index.set(settings.get('row_index', 0))
            self.var_avg_rows.set(settings.get('avg_rows', 0))
            self.var_pixel_size.set(settings.get('pixel_size', 1.0))
            self.var_center_px.set(settings.get('center_px', 0))
            self.var_left_cutoff.set(settings.get('left_cutoff', 100))
            self.var_right_cutoff.set(settings.get('right_cutoff', 100))
            self.var_smooth_win_l.set(settings.get('smooth_win_l', 11))
            self.var_smooth_poly_l.set(settings.get('smooth_poly_l', 3))
            self.var_smooth_win_r.set(settings.get('smooth_win_r', 11))
            self.var_smooth_poly_r.set(settings.get('smooth_poly_r', 3))
            self.var_use_cutoff.set(settings.get('use_cutoff', False))
            
            for m, val in settings.get('methods', {}).items():
                if m in self.method_vars:
                    self.method_vars[m].set(val)
            
            self.run_analysis()
            messagebox.showinfo("Settings", "Settings loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings: {e}")

    def save_settings(self):
        init_dir = r"C:\Users\Reece\Documents\GitHub\AbelInversions"
        if not os.path.exists(init_dir):
            init_dir = os.getcwd()
            
        file_path = filedialog.asksaveasfilename(initialdir=init_dir, defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if not file_path: return
        try:
            settings = {
                'row_index': self.var_row_index.get(),
                'avg_rows': self.var_avg_rows.get(),
                'pixel_size': self.var_pixel_size.get(),
                'center_px': self.var_center_px.get(),
                'left_cutoff': self.var_left_cutoff.get(),
                'right_cutoff': self.var_right_cutoff.get(),
                'smooth_win_l': self.var_smooth_win_l.get(),
                'smooth_poly_l': self.var_smooth_poly_l.get(),
                'smooth_win_r': self.var_smooth_win_r.get(),
                'smooth_poly_r': self.var_smooth_poly_r.get(),
                'use_cutoff': self.var_use_cutoff.get(),
                'methods': {m: var.get() for m, var in self.method_vars.items()}
            }
            with open(file_path, 'w') as f:
                json.dump(settings, f, indent=4)
            messagebox.showinfo("Settings", "Settings saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def load_file(self):
        init_dir = r"C:\Users\Reece\Documents\GitHub\AbelInversions"
        if not os.path.exists(init_dir):
            init_dir = os.getcwd()
            
        file_path = filedialog.askopenfilename(initialdir=init_dir, filetypes=[("Data files", "*.csv;*.txt"), ("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")])
        if not file_path:
            return
        
        try:
            # Try loading with automatic delimiter detection (or fallback)
            try:
                self.raw_data = np.loadtxt(file_path, delimiter=',')
            except:
                self.raw_data = np.loadtxt(file_path)
            
            if self.raw_data.size == 0:
                raise ValueError("Loaded data is empty.")

            # Handle single column 2D arrays -> 1D
            if self.raw_data.ndim == 2 and (self.raw_data.shape[0] == 1 or self.raw_data.shape[1] == 1):
                self.raw_data = self.raw_data.flatten()

            self.filename = os.path.basename(file_path)
            self.lbl_filename.config(text=self.filename)
            
            # Reset row index if out of bounds
            if self.raw_data.ndim == 2:
                max_rows = self.raw_data.shape[0]
                if self.var_row_index.get() >= max_rows:
                    self.var_row_index.set(max_rows // 2)
                # Enable row controls
                self.ent_row_index.config(state='normal')
                self.ent_avg_rows.config(state='normal')
            else:
                # Disable row controls for 1D
                self.ent_row_index.config(state='disabled')
                self.ent_avg_rows.config(state='disabled')
            
            self.update_2d_plot()
            self.run_analysis()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def update_2d_plot(self):
        if self.raw_data is None:
            return
        
        # Remove colorbar before clearing axes to avoid layout issues
        if self.cbar_2d:
            try:
                self.cbar_2d.remove()
            except Exception:
                pass
            self.cbar_2d = None

        self.ax_2d.clear()
        
        if self.raw_data.ndim == 2:
            im = self.ax_2d.imshow(self.raw_data, aspect='auto', cmap='viridis')
            # Draw line for selected row
            row = self.var_row_index.get()
            self.ax_2d.axhline(row, color='r', linestyle='--')
            self.ax_2d.set_title(f"2D Data (Selected Row: {row})")
            
            # Add colorbar
            self.cbar_2d = self.fig_2d.colorbar(im, ax=self.ax_2d)
        else:
            # Plot the 1D data in the 2D preview window just to show it exists
            self.ax_2d.plot(self.raw_data)
            self.ax_2d.set_title("1D Data Loaded")
            
        self.canvas_2d.draw()

    def on_param_change(self, event=None):
        # Triggered when row index changes via entry
        self.update_2d_plot()

    def extract_profile(self):
        if self.raw_data is None:
            return None
        
        if self.raw_data.ndim == 1:
            return self.raw_data
        
        row = self.var_row_index.get()
        max_rows = self.raw_data.shape[0]
        
        if row < 0 or row >= max_rows:
            return None
            
        n_avg = self.var_avg_rows.get()
        if n_avg > 0:
            start = max(0, row - n_avg)
            end = min(max_rows, row + n_avg + 1)
            return np.nanmean(self.raw_data[start:end, :], axis=0)
        else:
            return self.raw_data[row, :]

    def auto_find_center(self):
        data = self.extract_profile()
        if data is None or data.size == 0:
            return

        center_com = am.find_center_of_mass(data)
        center_gauss = am.find_center_gaussian(data, center_guess=center_com)

        # Use CoM as primary if available, or Gaussian
        if center_gauss is not None:
            self.var_center_px.set(center_gauss)
            msg = f"Found center at {center_gauss} (Gaussian fit)"
            if center_com is not None:
                msg += f"\n(CoM was {center_com})"
            messagebox.showinfo("Auto Center", msg)
            self.run_analysis()
        elif center_com is not None:
            self.var_center_px.set(center_com)
            messagebox.showinfo("Auto Center", f"Found center at {center_com} (CoM)")
            self.run_analysis()
        else:
            messagebox.showwarning("Auto Center", "Could not find center.")

    def auto_detect_widths(self):
        data = self.extract_profile()
        if data is None or data.size == 0:
            return
        
        try:
            center = self.var_center_px.get()
            left_cutoff, right_cutoff, threshold, msg = am.detect_widths(data, center)
            
            # Update vars
            self.var_left_cutoff.set(left_cutoff)
            self.var_right_cutoff.set(right_cutoff)
            self.var_use_cutoff.set(True) 
            
            messagebox.showinfo("Auto Widths", msg)
            self.run_analysis()
            
        except Exception as e:
            messagebox.showerror("Auto Width Error", str(e))

    def on_view_toggle(self):
        self.run_analysis()
        # If Center Accuracy tab is active, update it too
        try:
            current_tab = self.notebook.index("current")
            if current_tab == 3:
                 if hasattr(self, 'center_test_data') and self.center_test_data:
                     self.plot_center_test_results()
            elif current_tab == 4:
                 if hasattr(self, 'cutoff_test_data') and self.cutoff_test_data:
                     self.plot_cutoff_test_results()
        except:
            pass

    def run_analysis(self):
        self.status_var.set("Processing...")
        self.root.update_idletasks()
        
        data = self.extract_profile()
        if data is None or data.size == 0:
            self.status_var.set("Ready")
            return

        # Keep NaNs for plotting (gaps)
        data_plot = data.copy()

        # Update 2D plot line
        self.update_2d_plot()

        # Get params
        try:
            center = self.var_center_px.get()
            left_cutoff = self.var_left_cutoff.get()
            right_cutoff = self.var_right_cutoff.get()
            pixel_size = self.var_pixel_size.get()
            
            sw_l = self.var_smooth_win_l.get()
            sp_l = self.var_smooth_poly_l.get()
            sw_r = self.var_smooth_win_r.get()
            sp_r = self.var_smooth_poly_r.get()
        except:
            self.status_var.set("Ready")
            return

        # 1D Plot
        self.ax_1d.clear()
        self.ax_1d.plot(data_plot, label='Raw Profile')
        self.ax_1d.axvline(center, color='r', linestyle='--', label='Center')
        
        if self.var_use_cutoff.get():
            self.ax_1d.axvline(left_cutoff, color='k', linestyle=':', label='Left Cutoff')
            self.ax_1d.axvline(right_cutoff, color='k', linestyle=':', label='Right Cutoff')
            
        self.ax_1d.legend()
        self.ax_1d.set_title("Extracted 1D Profile")
        self.canvas_1d.draw()

        # Processing
        try:
            # Calculate widths from absolute cutoffs
            l_width = center - left_cutoff
            r_width = right_cutoff - center
            
            # Ensure positive widths
            if l_width < 0: l_width = 0
            if r_width < 0: r_width = 0

            left_smooth, right_smooth = am.prepare_profiles(
                data, center, l_width, r_width,
                (sw_l, sp_l), (sw_r, sp_r),
                smoothing_method=self.var_smoothing_method.get(),
                use_cutoff=self.var_use_cutoff.get(),
                taper_edges=self.var_taper_edges.get()
            )

            # Update 1D plot with taper if enabled
            if self.var_taper_edges.get():
                # Calculate x coordinates for plotting
                x_l_plot = center - np.arange(len(left_smooth))
                x_r_plot = center + np.arange(len(right_smooth))
                
                self.ax_1d.plot(x_l_plot, left_smooth, 'g--', alpha=0.7, label='Tapered L')
                self.ax_1d.plot(x_r_plot, right_smooth, 'm--', alpha=0.7, label='Tapered R')
                self.ax_1d.legend()
                self.canvas_1d.draw()

            # Result Plot
            self.ax_res.clear()

            methods_to_run = [m for m in self.methods if self.method_vars[m].get()]
            
            if not methods_to_run:
                 self.ax_res.text(0.5, 0.5, "No method selected", ha='center')
                 self.canvas_res.draw()
                 self.status_var.set("Ready")
                 return

            for m in methods_to_run:
                # Inversion
                recon_l = am.perform_inversion(left_smooth, m, pixel_size)
                recon_r = am.perform_inversion(right_smooth, m, pixel_size)

                x_l = np.arange(len(recon_l)) * pixel_size
                x_r = np.arange(len(recon_r)) * pixel_size
                
                show_l = self.var_show_left.get()
                show_r = self.var_show_right.get()

                if len(methods_to_run) > 1:
                    if show_l: self.ax_res.plot(x_l, recon_l, '--', label=f'{m} (L)')
                    if show_r: self.ax_res.plot(x_r, recon_r, '-', label=f'{m} (R)')
                else:
                    if show_l: self.ax_res.plot(x_l, recon_l, 'b--', label='Left Inverted')
                    if show_r: self.ax_res.plot(x_r, recon_r, 'r-', label='Right Inverted')

            self.ax_res.set_xlabel("Radius (cm)")
            self.ax_res.set_ylabel("Plasma Density ($cm^{-3}$)")
            self.ax_res.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            if self.ax_res.get_legend_handles_labels()[0]: # Only show legend if something is plotted
                self.ax_res.legend()
            title = "Inversion Comparison" if len(methods_to_run) > 1 else f"Inversion Result ({methods_to_run[0]})"
            self.ax_res.set_title(title)
            self.canvas_res.draw()

            # Calculate Stats for the first method
            if methods_to_run:
                peak_l = np.max(recon_l) if recon_l.size > 0 else 0.0
                peak_r = np.max(recon_r) if recon_r.size > 0 else 0.0
                
                fwhm_l = am.calculate_fwhm(recon_l, x_l) if recon_l.size > 0 else 0.0
                fwhm_r = am.calculate_fwhm(recon_r, x_r) if recon_r.size > 0 else 0.0
                
                stats_text = f"Method: {methods_to_run[-1]}\n"
                if show_l: stats_text += f"Peak Density (L): {peak_l:.2e} | FWHM (L): {fwhm_l:.4f} cm\n"
                if show_r: stats_text += f"Peak Density (R): {peak_r:.2e} | FWHM (R): {fwhm_r:.4f} cm"
                
                self.lbl_stats.config(text=stats_text)

        except Exception as e:
            print(f"Analysis Error: {e}")
            messagebox.showerror("Analysis Error", str(e))
        finally:
            self.status_var.set("Ready")

    def run_center_test(self):
        self.calculate_center_test()
        self.plot_center_test_results()

    def calculate_center_test(self):
        if self.raw_data is None:
            return

        data = self.extract_profile()
        if data is None:
            return
            
        data_analysis = np.nan_to_num(data, nan=0.0)
        
        try:
            base_center = self.var_center_px.get()
            r_val = self.var_center_range.get()
            step = self.var_center_step.get()
            
            pixel_size = self.var_pixel_size.get()
            left_cutoff = self.var_left_cutoff.get()
            right_cutoff = self.var_right_cutoff.get()
            
            sw_l = self.var_smooth_win_l.get()
            sp_l = self.var_smooth_poly_l.get()
            sw_r = self.var_smooth_win_r.get()
            sp_r = self.var_smooth_poly_r.get()
            
            centers = range(base_center - r_val, base_center + r_val + 1, step)
            
            # Use the first selected method
            methods_to_run = [m for m in self.methods if self.method_vars[m].get()]
            if not methods_to_run:
                messagebox.showwarning("Warning", "No inversion method selected.")
                return
            method = methods_to_run[0] # Use first one for test
            
            results = []
            valid_centers = []

            for idx, center in enumerate(centers):
                # Calculate widths from absolute cutoffs
                l_width = center - left_cutoff
                r_width = right_cutoff - center
                
                # Bounds check
                if self.var_use_cutoff.get():
                    if l_width < 0 or r_width < 0:
                        continue
                else:
                    if center < 0 or center >= len(data_analysis):
                        continue
                    
                left_smooth, right_smooth = am.prepare_profiles(
                    data, center, l_width, r_width,
                    (sw_l, sp_l), (sw_r, sp_r),
                    smoothing_method=self.var_smoothing_method.get(),
                    use_cutoff=self.var_use_cutoff.get(),
                    taper_edges=self.var_taper_edges.get()
                )
                
                # Inversion
                recon_l = am.perform_inversion(left_smooth, method, pixel_size)
                recon_r = am.perform_inversion(right_smooth, method, pixel_size)
                    
                x_l = np.arange(len(recon_l)) * pixel_size
                x_r = np.arange(len(recon_r)) * pixel_size
                
                results.append({
                    'recon_l': recon_l,
                    'recon_r': recon_r,
                    'x_l': x_l,
                    'x_r': x_r,
                    'peak_l': np.max(recon_l) if recon_l.size > 0 else 0.0,
                    'peak_r': np.max(recon_r) if recon_r.size > 0 else 0.0,
                    'avg_l': np.mean(recon_l) if recon_l.size > 0 else 0.0,
                    'avg_r': np.mean(recon_r) if recon_r.size > 0 else 0.0
                })
                valid_centers.append(center)

            self.center_test_data = {
                'centers': valid_centers,
                'results': results,
                'method': method
            }
            
        except Exception as e:
            messagebox.showerror("Error", f"Center Test Failed: {e}")
            self.center_test_data = None

    def plot_center_test_results(self):
        if not hasattr(self, 'center_test_data') or self.center_test_data is None:
            return

        try:
            data = self.center_test_data
            valid_centers = data['centers']
            results = data['results']
            
            self.ax_ct_prof.clear()
            self.ax_ct_peak.clear()
            
            # Remove previous colorbar axes if it exists
            if hasattr(self, 'ct_cax') and self.ct_cax:
                try:
                    self.ct_cax.remove()
                except:
                    pass
                self.ct_cax = None
            
            import matplotlib.cm as cm
            from matplotlib.colors import Normalize
            from matplotlib.cm import ScalarMappable
            
            # Create colormap based on center values
            norm = Normalize(vmin=min(valid_centers), vmax=max(valid_centers))
            cmap = cm.viridis
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            
            peaks_l = []
            peaks_r = []
            
            for idx, res in enumerate(results):
                center_val = valid_centers[idx]
                color = cmap(norm(center_val))
                
                if self.var_show_left.get():
                    self.ax_ct_prof.plot(res['x_l'], res['recon_l'], color=color, alpha=0.5)
                    peaks_l.append(res['avg_l'])
                
                if self.var_show_right.get():
                    self.ax_ct_prof.plot(res['x_r'], res['recon_r'], color=color, linestyle='--', alpha=0.5)
                    peaks_r.append(res['avg_r'])

            self.ax_ct_prof.set_title(f"Profiles (Color: Center {valid_centers[0]}->{valid_centers[-1]})")
            self.ax_ct_prof.set_xlabel("Radius (cm)")
            self.ax_ct_prof.set_ylabel("Density ($cm^{-3}$)")
            self.ax_ct_prof.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
            # Add colorbar inside the plot
            # Position: x, y, width, height (in axes coordinates)
            self.ct_cax = self.ax_ct_prof.inset_axes([0.85, 0.6, 0.03, 0.35]) 
            cbar = self.fig_ct.colorbar(sm, cax=self.ct_cax, orientation='vertical')
            cbar.set_label('Center Pixel')
            
            if self.var_show_left.get() and peaks_l:
                self.ax_ct_peak.plot(valid_centers, np.array(peaks_l), 'o-', label='Left Avg')
            if self.var_show_right.get() and peaks_r:
                self.ax_ct_peak.plot(valid_centers, np.array(peaks_r), 's--', label='Right Avg')
                
            self.ax_ct_peak.set_title("Average Density vs Center Pixel")
            self.ax_ct_peak.set_xlabel("Center Pixel")
            self.ax_ct_peak.set_ylabel("Average Density ($cm^{-3}$)")
            self.ax_ct_peak.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            if self.var_show_left.get() or self.var_show_right.get():
                self.ax_ct_peak.legend()
            self.ax_ct_peak.grid(True)
            
            self.canvas_ct.draw()
        except Exception as e:
            print(f"Plotting error: {e}")

    def run_cutoff_test(self):
        self.calculate_cutoff_test()
        self.plot_cutoff_test_results()

    def calculate_cutoff_test(self):
        if self.raw_data is None:
            return

        data = self.extract_profile()
        if data is None:
            return
            
        data_analysis = np.nan_to_num(data, nan=0.0)
        
        try:
            center = self.var_center_px.get()
            base_left = self.var_left_cutoff.get()
            base_right = self.var_right_cutoff.get()
            
            r_below = self.var_cutoff_range_below.get()
            r_above = self.var_cutoff_range_above.get()
            step = self.var_cutoff_step.get()
            
            pixel_size = self.var_pixel_size.get()
            
            sw_l = self.var_smooth_win_l.get()
            sp_l = self.var_smooth_poly_l.get()
            sw_r = self.var_smooth_win_r.get()
            sp_r = self.var_smooth_poly_r.get()
            
            # Deltas to apply to the cutoffs. 
            # Positive delta = wider window (left moves left, right moves right)
            if step < 1: step = 1
            deltas = range(-abs(r_below), abs(r_above) + 1, step)
            
            # Use the first selected method
            methods_to_run = [m for m in self.methods if self.method_vars[m].get()]
            if not methods_to_run:
                messagebox.showwarning("Warning", "No inversion method selected.")
                return
            method = methods_to_run[0] 
            
            results = []
            valid_deltas = []

            for delta in deltas:
                test_left = base_left - delta
                test_right = base_right + delta
                
                # Calculate widths
                l_width = center - test_left
                r_width = test_right - center
                
                # Bounds check
                if l_width < 0 or r_width < 0:
                    continue
                # We allow test_left/right to go out of bounds (clamped by prepare_profiles)

                # We must force use_cutoff=True for this test to make sense, 
                # or at least pass the widths correctly.
                # prepare_profiles uses l_width and r_width if use_cutoff is True.
                
                left_smooth, right_smooth = am.prepare_profiles(
                    data, center, l_width, r_width,
                    (sw_l, sp_l), (sw_r, sp_r),
                    smoothing_method=self.var_smoothing_method.get(),
                    use_cutoff=True, # Force True for this test
                    taper_edges=self.var_taper_edges.get()
                )
                
                # Inversion
                recon_l = am.perform_inversion(left_smooth, method, pixel_size)
                recon_r = am.perform_inversion(right_smooth, method, pixel_size)
                    
                x_l = np.arange(len(recon_l)) * pixel_size
                x_r = np.arange(len(recon_r)) * pixel_size
                
                results.append({
                    'recon_l': recon_l,
                    'recon_r': recon_r,
                    'x_l': x_l,
                    'x_r': x_r,
                    'peak_l': np.max(recon_l) if recon_l.size > 0 else 0.0,
                    'peak_r': np.max(recon_r) if recon_r.size > 0 else 0.0,
                    'avg_l': np.mean(recon_l) if recon_l.size > 0 else 0.0,
                    'avg_r': np.mean(recon_r) if recon_r.size > 0 else 0.0
                })
                valid_deltas.append(delta)

            self.cutoff_test_data = {
                'deltas': valid_deltas,
                'results': results,
                'method': method,
                'requested_range': (-r_below, r_above)
            }
            
        except Exception as e:
            messagebox.showerror("Error", f"Cutoff Test Failed: {e}")
            self.cutoff_test_data = None

    def plot_cutoff_test_results(self):
        if not hasattr(self, 'cutoff_test_data') or self.cutoff_test_data is None:
            return

        try:
            data = self.cutoff_test_data
            valid_deltas = data['deltas']
            results = data['results']
            
            self.ax_cut_prof.clear()
            self.ax_cut_peak.clear()
            
            # Remove previous colorbar axes if it exists
            if hasattr(self, 'cut_cax') and self.cut_cax:
                try:
                    self.cut_cax.remove()
                except:
                    pass
                self.cut_cax = None
            
            import matplotlib.cm as cm
            from matplotlib.colors import Normalize
            from matplotlib.cm import ScalarMappable
            
            # Create colormap based on delta values
            req_min, req_max = data.get('requested_range', (min(valid_deltas) if valid_deltas else 0, max(valid_deltas) if valid_deltas else 0))
            norm = Normalize(vmin=req_min, vmax=req_max)
            cmap = cm.viridis
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            
            peaks_l = []
            peaks_r = []
            
            for idx, res in enumerate(results):
                delta = valid_deltas[idx]
                color = cmap(norm(delta))
                
                if self.var_show_left.get():
                    self.ax_cut_prof.plot(res['x_l'], res['recon_l'], color=color, alpha=0.5)
                    peaks_l.append(res['avg_l'])
                
                if self.var_show_right.get():
                    self.ax_cut_prof.plot(res['x_r'], res['recon_r'], color=color, linestyle='--', alpha=0.5)
                    peaks_r.append(res['avg_r'])

            self.ax_cut_prof.set_title(f"Profiles (Color: Delta {req_min}->{req_max})")
            self.ax_cut_prof.set_xlabel("Radius (cm)")
            self.ax_cut_prof.set_ylabel("Density ($cm^{-3}$)")
            self.ax_cut_prof.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
            # Add colorbar inside the plot
            self.cut_cax = self.ax_cut_prof.inset_axes([0.85, 0.6, 0.03, 0.35]) 
            cbar = self.fig_cut.colorbar(sm, cax=self.cut_cax, orientation='vertical')
            cbar.set_label('Cutoff Delta (px)')
            
            if self.var_show_left.get() and peaks_l:
                self.ax_cut_peak.plot(valid_deltas, np.array(peaks_l), 'o-', label='Left Avg')
            if self.var_show_right.get() and peaks_r:
                self.ax_cut_peak.plot(valid_deltas, np.array(peaks_r), 's--', label='Right Avg')
                
            self.ax_cut_peak.set_title("Average Density vs Cutoff Delta")
            self.ax_cut_peak.set_xlabel("Cutoff Delta (px)")
            self.ax_cut_peak.set_ylabel("Average Density ($cm^{-3}$)")
            self.ax_cut_peak.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            if self.var_show_left.get() or self.var_show_right.get():
                self.ax_cut_peak.legend()
            self.ax_cut_peak.grid(True)
            
            self.canvas_cut.draw()
        except Exception as e:
            print(f"Plotting error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AbelApp(root)
    root.mainloop()