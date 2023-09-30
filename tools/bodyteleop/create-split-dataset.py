import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("ds_info.csv")

# Splitting the data into train, validation, and test
train, temp = train_test_split(data, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save them into separate CSV files
train.to_csv("train_dataset.csv", index=False)
val.to_csv("val_dataset.csv", index=False)
test.to_csv("test_dataset.csv", index=False)

print(f"Total data: {len(data)}")
print(f"Training data: {len(train)}")
print(f"Validation data: {len(val)}")
print(f"Test data: {len(test)}")
