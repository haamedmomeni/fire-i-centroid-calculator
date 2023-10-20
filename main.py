import os
import glob
import pandas as pd
import progressbar
from PIL import Image
from matplotlib import pyplot as plt

from centroidAnalysis import (
    calculate_image_median,
    calculate_hot_pixels,
    get_file_date,
    preprocess_image,
    get_maxima
)

OUTPUT_FILE = 'results.csv'
INPUT_DIR = "06-27-23-fast-rep"
WILDCARD_JPGS_STR = "*.JPG"
REFERENCE_IMG_STR = "Frame_2_"

search_str = os.path.join(INPUT_DIR, WILDCARD_JPGS_STR)
image_paths = glob.glob(search_str)

secondFramePath = [f for f in image_paths if REFERENCE_IMG_STR in f][0]
secondFrame = Image.open(secondFramePath)
medianValue = calculate_image_median(secondFrame)
hotPixelsFrame = calculate_hot_pixels(secondFrame)

widgets = [' [', progressbar.Timer(format='elapsed time: %(elapsed)s'), '] ',
           progressbar.Bar('*'),
           ' (', progressbar.ETA(), ') ',
           ]

bar = progressbar.ProgressBar(max_value=len(image_paths),
                              widgets=widgets).start()

data_points = []
for idx, filePath in enumerate(image_paths):
    bar.update(idx)

    imageTime = get_file_date(filePath)

    with Image.open(filePath) as imRGB:  # Using 'with' ensures file is closed after operations are done.
        im_array = preprocess_image(imRGB, hotPixelsFrame, medianValue)
        coords = get_maxima(im_array, True)
        # Values are sorted here, with the assumption of that the motion of spots are small.
        x1, y1, x2, y2 = sorted(coords)
        data_points.append({'Time': imageTime, 'X1': x1, 'Y1': y1, 'X2': x2, 'Y2': y2})
bar.finish()

df = pd.DataFrame(data_points)
df_sorted = df.sort_values(by=['Time'], ascending=False)
df_sorted.to_csv(OUTPUT_FILE, index=False, header=True)


plt.plot(df_sorted['Time'], df_sorted['X1'], ".", label='X1')
plt.plot(df_sorted['Time'], df_sorted['Y1'], ".", label='Y1')
plt.plot(df_sorted['Time'], df_sorted['X2'], ".", label='X2')
plt.plot(df_sorted['Time'], df_sorted['Y2'], ".", label='Y2')

plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Your Plot Title')
plt.legend()

plt.show()
