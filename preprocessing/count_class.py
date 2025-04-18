import os
import json
from collections import defaultdict

# Directory containing labelme JSON files
json_dir = "./original_annotation"

# Load all JSON files from the directory
json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

# Function to count unique classes and their instances
def count_classes(json_dir, json_files):
    class_counts = defaultdict(int)
    instance_counts = defaultdict(int)
    
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for shape in data['shapes']:
            class_name = shape['label']
            class_counts[class_name] += 1
            instance_counts[class_name] += len(shape['points'])
    
    return class_counts, instance_counts

# Count unique classes and instances
class_counts, instance_counts = count_classes(json_dir, json_files)

# Print the counts
print("Class Counts:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count} files")

