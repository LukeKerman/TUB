# AstroTUBIN Project

## Overview

AstroTUBIN is a data processing and analysis pipeline designed for astronomical observations. The pipeline includes a Jupyter notebook (`AstroTUBIN_data_pipeline.ipynb`), a set of Python functions (`AstroTUBIN_functions.py`), and additional resources like pointing information (`pointing_ra_dec`) and mask weights (`mask_weights.npy`) to facilitate the analysis of celestial objects.

## Setup

1. **Clone Repository:** Clone this repository to your local machine to get started.
2. **Environment Setup:** It's recommended to create a virtual environment using Python 3.x and install the required packages listed in `requirements.txt`.
3. **Directory Structure:** Ensure `AstroTUBIN_data_pipeline.ipynb` and `AstroTUBIN_functions.py` are placed in the same directory. The `Testset10` directory should contain sample images for testing purposes.

## Components

- **AstroTUBIN_data_pipeline.ipynb:** A Jupyter notebook that guides users through the data processing and analysis steps.
- **AstroTUBIN_functions.py:** Contains all the functions used by the data pipeline notebook.
- **pointing_ra_dec:** A file with pointing right ascension and declination information necessary for the analysis.
- **mask_weights.npy:** Numpy array file with mask weights used in data processing to handle image artifacts or regions of no interest.

## Usage

1. **Launch Jupyter Notebook:** Open `AstroTUBIN_data_pipeline.ipynb` in Jupyter Notebook or JupyterLab.
2. **Follow Instructions:** The notebook contains step-by-step instructions for processing and analyzing your data. Adjust parameters as needed based on your dataset.
3. **Testing:** Use the images in `Testset10` for testing the pipeline with sample data.

## Contributing

Contributions to AstroTUBIN are welcome! Please read the contributing guidelines before submitting pull requests.

