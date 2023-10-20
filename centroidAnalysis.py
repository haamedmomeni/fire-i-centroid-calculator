import numpy as np
from datetime import datetime
from PIL import Image, ImageFilter, ImageChops
import logging


def centroid(frame, threshold=0.25):
    """
    Calculate the centroid of a frame after thresholding the pixel values.

    Parameters:
    frame (numpy.ndarray): 2D array representing the frame.
    threshold (float): Threshold factor to determine which pixels to consider in the centroid calculation.
                       Pixels with value less than threshold * frame.max() are discarded.

    Returns:
    tuple: The (x, y) coordinates of the centroid. Returns (0, 0) if all pixels are below the threshold.
    """

    if not isinstance(frame, np.ndarray) or frame.ndim != 2:
        raise ValueError("Input frame must be a 2D numpy array.")

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    # Thresholding the frame
    thresholded = np.where(frame < threshold * frame.max(), 0, frame)

    # Calculate the sum of the thresholded values
    sumframe = thresholded.sum()

    # Check if sumframe is zero to avoid division by zero
    if sumframe == 0:
        return (0, 0)

    # Creating arrays of indices
    ix, iy = np.indices(frame.shape)

    # Calculate the centroid
    totx = np.sum(ix * thresholded)
    toty = np.sum(iy * thresholded)

    return totx / sumframe, toty / sumframe


def calculate_centroid(image: np.ndarray, x_guess: int, y_guess: int, WINDOW_SIZE=50) -> tuple:
    """
    Calculate the centroid within a window around the provided coordinates.

    Parameters:
    image (np.ndarray): The image in which to calculate the centroid.
    x_guess (int): Initial x-coordinate guess for the centroid's position.
    y_guess (int): Initial y-coordinate guess for the centroid's position.

    Returns:
    tuple: The (x, y) coordinates of the centroid.
    """
    left_edge = max(x_guess - WINDOW_SIZE, 0)
    right_edge = min(x_guess + WINDOW_SIZE, image.shape[0])
    bottom_edge = max(y_guess - WINDOW_SIZE, 0)
    top_edge = min(y_guess + WINDOW_SIZE, image.shape[1])
    x, y = centroid(image[left_edge:right_edge, bottom_edge:top_edge])  # Assuming 'centroid' is a predefined function
    return x + left_edge, y + bottom_edge


def get_maxima(seen_image: np.ndarray, two_spots: bool = False, WINDOW_SIZE=50, MIN_VALID_SPOT_BRIGHTNESS=50) -> dict:
    """
    Identify and calculate the positions of one or two bright spots in an image.

    Parameters:
    seen_image (np.ndarray): 2D array representing the image.
    two_spots (bool): Whether or not to calculate the positions for two spots.

    Returns:
    dict: The positions of the spots with keys 'x1', 'y1', 'x2', and 'y2'.
    """
    try:
        # Calculate centroid for the first bright spot
        xmax_guess, ymax_guess = np.unravel_index(np.argmax(seen_image), seen_image.shape)
        x1, y1 = calculate_centroid(seen_image, xmax_guess, ymax_guess)

        x2, y2 = (0, 0)  # Default coordinates if there's no second spot or if it's invalid

        if two_spots:
            # Nullify the area around the first spot and find the second one
            tmp_image = np.copy(seen_image)
            tmp_image[
                max(xmax_guess - WINDOW_SIZE, 0):min(xmax_guess + WINDOW_SIZE, seen_image.shape[0]),
                max(ymax_guess - WINDOW_SIZE, 0):min(ymax_guess + WINDOW_SIZE, seen_image.shape[1])
            ] = 0

            xmax_guess2, ymax_guess2 = np.unravel_index(np.argmax(tmp_image), tmp_image.shape)

            # Validate the brightness of the second spot
            if seen_image[xmax_guess2, ymax_guess2] > MIN_VALID_SPOT_BRIGHTNESS:
                x2, y2 = calculate_centroid(seen_image, xmax_guess2, ymax_guess2)

        return x1, y1, x2, y2

    except IndexError as e:  # Specific exception
        logging.error(f"An error occurred: {e}")  # Log the error
        return 0, 0, 0, 0  # Return zeros in case of an error

# Ensure to configure logging appropriately in your main application
# For a simple print-out, you could use: logging.basicConfig(level=logging.ERROR)


def calculate_image_median(im_rgb: Image.Image, filter_size: int = 3) -> float:
    """
    Calculate the median value of an image after converting it to grayscale and applying a median filter.

    Parameters:
    im_rgb (PIL.Image.Image): The input image in RGB format.
    filter_size (int): The size of the median filter to be applied. Defaults to 3.

    Returns:
    float: The median value of the processed image.
    """
    im_gray_scale = im_rgb.convert('L')
    im_removed_hot_spot = im_gray_scale.filter(ImageFilter.MedianFilter(size=filter_size))
    im_array = np.asarray(im_removed_hot_spot)
    median_value = np.median(im_array)
    return median_value


def calculate_hot_pixels(im_rgb: Image.Image, filter_size: int = 3) -> Image.Image:
    """
    Identify hot pixels in an image by subtracting the original image from a median-filtered version.

    Parameters:
    im_rgb (PIL.Image.Image): The input image in RGB format.
    filter_size (int): The size of the median filter to be applied. Defaults to 3.

    Returns:
    PIL.Image.Image: An image highlighting the hot pixels.
    """
    im_gray_scale = im_rgb.convert('L')
    im_removed_hot_spot = im_gray_scale.filter(ImageFilter.MedianFilter(size=filter_size))
    im_hot_spot = ImageChops.subtract(im_gray_scale, im_removed_hot_spot)
    return im_hot_spot


def preprocess_image(im_rgb: Image.Image, hot_pixels_img: Image.Image, median_value: int) -> tuple:
    """
    Preprocess an image by converting to grayscale, removing hot pixels, normalizing based on median,
    and identifying coordinates of two primary maxima.

    Parameters:
    im_rgb (PIL.Image.Image): The input image in RGB format.
    hot_pixels_img (PIL.Image.Image): An image highlighting the hot pixels.

    Returns:
    tuple: Coordinates x1, y1, x2, y2 of two maxima in the image.

    Raises:
    ValueError: If maxima extraction fails.
    """
    im_gray_scale = im_rgb.convert('L')
    im_removed_hot_spot = ImageChops.subtract(im_gray_scale, hot_pixels_img)
    im_array = np.asarray(im_removed_hot_spot)

    # Normalization
    if median_value is None:
        median_value = np.median(im_array)

    im_array = im_array - median_value # Subtracting median value of the image array

    return im_array


def get_file_date(filename: str) -> datetime:
    """
    Extracts the datetime from a filename with a specific format.

    Parameters:
    filename (str): The filename to parse, expected format: PREFIX_day_month_year_hour_minute_second_millisecond.JPG

    Returns:
    datetime: The datetime object extracted from the filename.

    Raises:
    ValueError: If the filename format is incorrect.
    """
    # Ensure the correct delimiter is used for different filesystems or adjust accordingly
    parts = filename.split('_')

    # Validate filename format
    if len(parts) < 9:
        raise ValueError(
            "Filename format is incorrect, expected format: PREFIX_day_month_year_hour_minute_second_millisecond.JPG")

    day, month, year, hour, minute, second = parts[2:8]
    millisec = parts[8].split(".")[0]  # Assuming the file extension is present

    # create date string
    date_str = '{year}-{month}-{day} {hour}:{minute}:{second}.{millisec}'.format(year=year, month=month, day=day,
                                                                                 hour=hour, minute=minute,
                                                                                 second=second, millisec=millisec)

    # date string to datetime object
    return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')
