# Alchemy — Abel Inversion Analysis Tool

Alchemy is a Python desktop application for extracting radial profiles from two-dimensional plasma-diagnostic data and performing inverse Abel transforms. It was developed to support interferometric plasma-density analysis by combining data extraction, preprocessing, multiple inversion methods, sensitivity studies, visualization, and report generation in one interface.

## Why this project exists

Axisymmetric plasma diagnostics often produce line-integrated measurements rather than the local radial quantity of interest. Recovering a radial density profile requires an inverse Abel transform, but the result can be highly sensitive to preprocessing choices such as:

- Center selection
- Signal cutoffs
- Smoothing method and strength
- Edge behavior
- Pixel calibration
- Noise and asymmetry between the left and right sides

Alchemy makes these choices visible and repeatable so the analyst can compare methods, inspect assumptions, and quantify sensitivity instead of treating the inversion as a single black-box calculation.

## Core capabilities

### Data loading and extraction

- Load two-dimensional image data or one-dimensional profiles from CSV and text files.
- Select an individual image row for analysis.
- Average neighboring rows to improve signal-to-noise ratio.
- Preview the two-dimensional dataset and selected extraction region.

### Centering and signal boundaries

- Estimate the center using a center-of-mass method.
- Estimate the center using a Gaussian fit.
- Fine-tune the center manually.
- Detect left and right cutoffs from an estimated noise floor.
- Override cutoffs manually when the automatic result is not physically appropriate.

### Preprocessing

- Savitzky–Golay smoothing
- Gaussian smoothing
- Independent left/right smoothing parameters
- Optional edge tapering to reduce truncation artifacts
- Separate treatment of the left and right radial profiles

### Abel inversion methods

Alchemy uses PyAbel and supports:

- BASEX
- Hansen–Law
- Onion peeling
- Three-point inversion

Multiple methods can be selected at the same time for comparison.

### Analysis and visualization

- Two-dimensional data preview
- Extracted one-dimensional profile
- Smoothed left/right profiles
- Overlaid inversion results from multiple methods
- Peak-value calculation
- Full width at half maximum calculation
- Center-selection sensitivity study
- Cutoff-selection sensitivity study
- Interactive Matplotlib navigation
- Saved settings for repeatable analysis
- In-application report output

## Repository structure

```text
.
├── Alchemy.py
├── backend/
│   ├── abel_methods.py
│   └── debug_abel_install.py
├── README.md
└── supporting data or configuration files
```

### Main components

- `Alchemy.py` contains the Tkinter graphical interface, file handling, analysis controls, plots, sensitivity-test workflows, settings management, and report interface.
- `backend/abel_methods.py` contains reusable numerical functions for center estimation, cutoff detection, smoothing, profile preparation, Abel transforms, and FWHM calculation.
- `backend/debug_abel_install.py` helps verify the local PyAbel installation.

## Requirements

- Python 3.8 or newer recommended
- NumPy
- Matplotlib
- SciPy
- PyAbel
- Tkinter, which is included with many Python installations

Install the Python packages with:

```bash
pip install numpy matplotlib scipy PyAbel
```

On some Linux distributions, Tkinter must be installed separately. For example:

```bash
sudo apt-get install python3-tk
```

## Installation

Clone this repository:

```bash
git clone https://github.com/ReeceAdams1/AbelInversions.git
cd AbelInversions
```

Install the dependencies, then launch the application:

```bash
python Alchemy.py
```

## Basic workflow

### 1. Load data

Select **Load CSV/Txt** and open either a two-dimensional dataset or a one-dimensional profile.

### 2. Choose an extraction region

For two-dimensional data:

- Set the desired row index.
- Increase **Average +/- Rows** to average adjacent rows when additional noise reduction is needed.

### 3. Set the physical calibration

Enter the pixel size in centimeters. This calibration affects the radial coordinate and the scale of the inversion result.

### 4. Determine the center

Use the automatic center option or enter the center pixel manually. Center selection is one of the most consequential choices in an Abel inversion, so inspect the profile and use the center-sensitivity tab when necessary.

### 5. Set cutoffs and smoothing

Optionally enable width cutoffs and use **Auto Detect** as an initial estimate. Choose Savitzky–Golay, Gaussian, or no smoothing, then set independent left/right parameters.

### 6. Select inversion methods

Choose one or more of BASEX, Hansen–Law, onion peeling, and three-point inversion.

### 7. Run and compare

Select **Update / Run Analysis** or press Enter. Compare the left/right solutions, inversion methods, peak values, and FWHM.

### 8. Test sensitivity

Use the **Center Accuracy** and **Cutoff Accuracy** tabs to sweep the selected parameter and observe how the recovered profile and peak value change.

## Numerical implementation

The backend prepares independent left and right profiles about the selected center. Depending on the chosen settings, it can:

1. Replace non-finite data.
2. Extract each radial half-profile.
3. Apply Savitzky–Golay or Gaussian smoothing.
4. Extend a truncated edge using an exponential taper.
5. Pass the profile and pixel spacing to the selected PyAbel inversion routine.
6. Calculate peak value and FWHM from the transformed profile.

Automatic cutoff detection estimates a noise floor from the quieter edge of the signal and uses a threshold based on both statistical variation and a fraction of the peak amplitude. The detected result is intended as an analyst aid, not a substitute for reviewing the physical signal.

## Research application

Alchemy was used in my UC Berkeley plasma-imaging research to analyze lineouts extracted from interferometric electron-density maps. The broader workflow included image alignment, fringe detection and labeling, phase-shift reconstruction, planar density mapping, and reverse Abel inversion to recover radial plasma-density profiles.

For Shot 36751, the analysis produced a reported peak radial density of approximately:

```text
4.38 × 10^19 e/cm³ ± 2.26 × 10^18 e/cm³
```

The uncertainty and interpretation depend on the image calibration, center selection, preprocessing, inversion method, and assumptions of approximate cylindrical symmetry.

## My contribution

I developed the Alchemy analysis application and integrated the graphical workflow, preprocessing controls, multiple PyAbel methods, center and cutoff sensitivity studies, statistics, visualization, settings management, and report-oriented analysis needed for the plasma-imaging project.

## Validation and good analysis practice

Before relying on a recovered profile:

- Compare more than one inversion method.
- Compare left and right profiles for asymmetry.
- Sweep the center and cutoff selections.
- Inspect sensitivity to smoothing parameters.
- Confirm the pixel calibration and physical units.
- Test the workflow on synthetic profiles with known forward and inverse transforms.
- Treat large edge oscillations or center spikes as possible numerical artifacts until investigated.

## Limitations

- The inverse Abel transform assumes approximate axial symmetry.
- Noise, asymmetry, limited field of view, and incorrect centering can strongly affect the result.
- Automatic center and cutoff estimates may require manual correction.
- Different inversion methods have different noise and regularization behavior.
- The current repository does not yet include a formal automated test suite or a packaged executable.

## Suggested future improvements

- Add synthetic validation datasets with known analytical inverses.
- Add automated tests for centering, tapering, FWHM, and inversion wrappers.
- Add exported figures and a sample end-to-end analysis.
- Add a `requirements.txt` or `pyproject.toml`.
- Package the application for installation or standalone distribution.
- Add uncertainty propagation beyond parameter-sensitivity sweeps.

## Acknowledgments

This project uses the open-source PyAbel library for Abel-transform algorithms and was developed in the context of plasma-imaging research at UC Berkeley.
