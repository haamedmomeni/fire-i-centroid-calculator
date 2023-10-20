# Fire I Tilt Calcualtor 

## Description

This script performs an analysis on a series of images by identifying and processing bright spots, and then generates a plot based on the processed data.

This code was originally developed to analyze the motion of hotspots as captured by Fire-I cameras.

## Installation

### Prerequisites

- Python 3.x
- Pip package manager

### Setting Up

1. Clone the repository to your local machine:

` git clone https://github.com/haamedmomeni/fire-i-centroid-calculator`

2. Navigate to the directory:

`cd fire-i-centroid-calculator`

3. Install the required Python libraries and dependencies:

`pip install -r requirements.txt`


## Usage

1. Make sure your images are stored in the appropriate directory. By default, the script looks for images in a directory named `06-27-23-fast-rep`.

2. Run the script:

`python main.py`

3. The script processes the images and generates a CSV file named `results.csv` with the analysis results.

4. After processing the images, the script generates a plot displaying the analysis results.

## Customization

- `OUTPUT_FILE`: The name of the output CSV file with the results. Default is `results.csv`.
- `INPUT_DIR`: The directory where the images are located. Default is `06-27-23-fast-rep`.
- `WILDCARD_JPGS_STR`: The wildcard string to match image files. Default is `*.JPG`.
- `REFERENCE_IMG_STR`: The reference string to identify a specific image in the directory. Default is `Frame_2_`.

You can change these variables in the script to match your directory structure or naming conventions.

## Contributing

If you want to contribute to this project, please create a pull request.

## License

[MIT](https://choosealicense.com/licenses/mit/)
