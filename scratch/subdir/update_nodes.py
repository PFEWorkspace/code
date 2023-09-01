
from utils.CSVManager import CSVFileManager
from FL_tasks import FLNodeStruct
import csv
import random

file_path = "./generated_nodes.csv"
manager = CSVFileManager(file_path, FLNodeStruct._fields_)
instances = manager.retrieve_instances()
dropout_percent = 0.7
malicious_percent = 0.3
with open(file_path,"w",newline="") as file:
        writer = csv.writer(file)

        # Writing header row
        header = ["nodeId", "availability", "honesty", "datasetSize", "freq", "transRate", "task", "dropout", "malicious"]
        writer.writerow(header)

        for node in instances:
            availability = 1 if node.availability == True else 0
            dropout = random.choices([1, 0], weights=[dropout_percent, 1-dropout_percent])[0]
            malicious = random.choices([1, 0],weights=[malicious_percent, 1-malicious_percent])[0]
            row = [node.nodeId, availability,0, node.datasetSize, node.freq, node.transRate, node.task, dropout, malicious]
            writer.writerow(row)
