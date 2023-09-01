from utils.CSVManager import CSVFileManager
from FL_tasks import FLNodeStruct

file_path = "./generated_nodes.csv"
manager = CSVFileManager(file_path, FLNodeStruct._fields_)
instances = manager.retrieve_instances()
for index in range(0,50):
    manager.modify_instance_field(index,"honesty", 0.0)