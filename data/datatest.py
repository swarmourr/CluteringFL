import pandas as pd
import numpy as np
import ast
df = pd.read_csv("data/mnist/10_partitions/0.csv")
#

import numpy as np
import pandas as pd

def convert_string_to_array(s):
    # Remove extra brackets and split by spaces
    numbers = s.strip('[]').replace('[', '').replace(']', '').split()
    # Convert to integers
    return np.array([int(num) for num in numbers if num])

# Apply the conversion to the 'image' column
df['image'] = df['image'].apply(convert_string_to_array)

# Convert the series of arrays into a 2D numpy array
image_arrays = np.stack(df['image'].values)

# Create a new DataFrame with each pixel as a feature
pixel_df = pd.DataFrame(image_arrays, columns=[f'pixel_{i}' for i in range(image_arrays.shape[1])])

# Add the label column
pixel_df['label'] = df['label']


print(pixel_df)