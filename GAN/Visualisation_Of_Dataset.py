# Import the Required Libraries
import pandas as pd
import matplotlib.pyplot as plt

num_dataset = pd.read_excel("D:/Mathi/PROJECTS/GAN/Datasets/Dataset_5.xlsx")

plt.plot(num_dataset['Numbers'])
plt.xlabel('Index')
plt.ylabel('Value')
plt.title("Visualisation")
plt.show()