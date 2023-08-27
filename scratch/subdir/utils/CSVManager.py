import csv
import os
import ctypes

class CSVFileManager:
    def __init__(self, file_path, structure_fields):
        self.file_path = file_path
        self.structure_fields = structure_fields
        self.field_names = [field[0] for field in self.structure_fields]

        # Create the file if it doesn't exist
        self._create_csv_file_if_not_exists()

    def _create_csv_file_if_not_exists(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.field_names)
                writer.writeheader()

    def write_instance(self, instance):
        if not isinstance(instance, ctypes.Structure):
            raise ValueError("Invalid instance. Should be of type ctypes.Structure.")
        
        converted_instance = {}
        for field_name, field_type in self.structure_fields:
            value = getattr(instance, field_name)
            if field_type is ctypes.c_bool:
                
                if field_name in ["dropout","malicious","availability"]:
                    converted_instance[field_name] = 1 if value==True else 0
                else : 
                    converted_instance[field_name] = str(value).lower()
            else:
                converted_instance[field_name] = str(value)

        with open(self.file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.field_names)
            writer.writerow(converted_instance)

    def retrieve_instances(self):
        instances = []
        if not os.path.exists(self.file_path):
            return instances

        with open(self.file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=self.field_names)
            next(reader)  # Skip header row
            for row in reader:
                instance = self._create_instance_from_row(row)
                instances.append(instance)

        return instances

    def modify_instance_field(self, instance_id, field_name, new_value):
        instances = self.retrieve_instances()
        for instance in instances:
            if getattr(instance, self.field_names[0]) == instance_id:
                setattr(instance, field_name, new_value)
                self._rewrite_csv_file(instances)
                return True
        return False

    def delete_instance(self, instance_id):
        instances = self.retrieve_instances()
        updated_instances = [instance for instance in instances if getattr(instance, self.field_names[0]) != instance_id]
        if len(updated_instances) < len(instances):
            self._rewrite_csv_file(updated_instances)
            return True
        return False

    def _create_instance_from_row(self, row):
        instance = self._get_empty_instance()

        for field_name, field_type in self.structure_fields:
            if field_type is ctypes.c_bool:
                if field_name in ["dropout","malicious","availability"]:
                    setattr(instance, field_name, int(row[field_name])==1)
                else:    
                    setattr(instance, field_name, row[field_name].lower() == 'true')
            elif field_type in (ctypes.c_int, ctypes.c_long, ctypes.c_longlong):
                setattr(instance, field_name, int(row[field_name]))
            elif field_type in (ctypes.c_float, ctypes.c_double):
                setattr(instance, field_name, float(row[field_name]))
            else:
                setattr(instance, field_name, field_type(row[field_name]))
        return instance

    def _get_empty_instance(self):
        class EmptyStructure(ctypes.Structure):
            _pack_ = 1
            _fields_ = self.structure_fields
        return EmptyStructure()

    def _rewrite_csv_file(self, instances):
        os.remove(self.file_path)
        self._create_csv_file_if_not_exists()
        for instance in instances:
            self.write_instance(instance)

    def get_instance_id(self, field_name):
        if not any(name == field_name for name, _ in self.structure_fields):
            raise ValueError(f"Field name '{field_name}' does not exist in the structure fields.")

        instances = self.retrieve_instances()
        max_id = max(getattr(instance, field_name) for instance in instances) if instances else 0
        return max_id      



# Example usage
if __name__ == "__main__":
    import ctypes

    class MLModel(ctypes.Structure):
        _pack_ = 1  # Pack the structure to match the C++ layout
        _fields_ = [
            ("modelId", ctypes.c_int),
            ("nodeId", ctypes.c_int),
            ("taskId", ctypes.c_int),
            ("round", ctypes.c_int),
            ("type", ctypes.c_int),
            ("positiveVote", ctypes.c_int),
            ("negativeVote", ctypes.c_int),
            ("evaluator1", ctypes.c_int),
            ("evaluator2", ctypes.c_int),
            ("evaluator3", ctypes.c_int),
            ("aggregated", ctypes.c_bool),
            ("aggModelId", ctypes.c_int),
            ("accuracy", ctypes.c_double)
        ]
    

    file_path = "./models.csv"
    manager = CSVFileManager(file_path, MLModel._fields_)

    # Example: Writing instances to CSV
    model1 = MLModel(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, True, 11, 0.95)
    model2 = MLModel(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, False, 12, 0.85)

    manager.write_instance(model1)
    manager.write_instance(model2)

    # Example: Retrieving instances from CSV
    instances = manager.retrieve_instances()
    for instance in instances:
        print(instance.modelId, instance.accuracy)

    # Example: Modifying an instance's field
    instance_id_to_modify = 1
    new_accuracy = 0.99
    manager.modify_instance_field(instance_id_to_modify, "accuracy", new_accuracy)

    # Example: Deleting an instance
    instance_id_to_delete = 2
    manager.delete_instance(instance_id_to_delete)
