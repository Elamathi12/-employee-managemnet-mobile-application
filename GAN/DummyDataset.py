import os
import pandas as pd

# Directory to save the CSV files
output_directory = "D:/Mathi/PROJECTS/GAN/Datasets"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate and create 1000 CSV files
for i in range(1, 1001):
    filename = os.path.join(output_directory, f"Dataset_{i}.csv")
    
    # Create a DataFrame with numbers from 1 to 1000 and a column named 'Numbers'
    data = {'Numbers': range(1, 1001), 'Condition': range(1, 1001)}  # Both columns have the same entries
    df = pd.DataFrame(data)
    
    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)
