from sklearn.model_selection import train_test_split

# Read the full dataset
with open("oscar.eo.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Split into train (90%) and validation (10%)
train_lines, valid_lines = train_test_split(lines, test_size=0.1, random_state=42)

# Save train and validation sets
with open("oscar.eo_train.txt", "w", encoding="utf-8") as f:
    f.writelines(train_lines)

with open("oscar.eo_valid.txt", "w", encoding="utf-8") as f:
    f.writelines(valid_lines)
