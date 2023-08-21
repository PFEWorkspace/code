import csv
import random

# Function to generate random data for each field
def generate_random_instance(instance_id):
    availability = random.choices(["true", "false"], weights=[0.8, 0.2])[0]
    honesty = 0
    dataset_size = random.randint(10, 100) * 10
    frequency = random.randint(5, 30) * 10
    transmission_rate = random.randint(15, 100) * 10
    task = random.choices([0, 1], weights=[0.7, 0.3])[0]
    dropout = random.choices(["true", "false"], weights=[0.1, 0.9])[0]
    malicious = random.choices(["true", "false"],weights=[0.05, 0.95])[0]
    return [instance_id, availability, honesty, dataset_size, frequency, transmission_rate, task, dropout, malicious]

# Number of instances to generate
num_instances = 1000

# CSV file name
csv_file = "generated_nodes.csv"

# Generating data and writing to CSV
with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)

    # Writing header row
    header = ["nodeId", "availability", "honesty", "datasetSize", "freq", "transRate", "task", "dropout", "malicious"]
    writer.writerow(header)

    # Generating and writing data for each instance
    for i in range(0, num_instances):
        instance_data = generate_random_instance(i)
        writer.writerow(instance_data)

print(f"CSV file '{csv_file}' generated successfully with {num_instances} instances.")
