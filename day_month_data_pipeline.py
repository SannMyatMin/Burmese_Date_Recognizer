import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

class DayMonth_data_pipeline:
    def __init__(self, data_dir, batch_size=32, img_size=(64,64)):
        self.data_dir          = Path(data_dir)
        self.batch_size        = batch_size
        self.image_size        = img_size
        self.image_dataframe   = None
        self.class_names       = ["၀", "၁", "၂", "၃", "၄", "၅", "၆", "၇", "၈", "၉", "၁၀", 
                                  "၁၁", "၁၂", "၁၃", "၁၄", "၁၅","၁၆", "၁၇", "၁၈", "၁၉", "၂၀", 
                                  "၂၁", "၂၂", "၂၃", "၂၄", "၂၅", "၂၆", "၂၇", "၂၈", "၂၉", "၃၀", "၃၁"]
        
    def create_image_dataframe(self):
        data = []
        for num_folder in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, num_folder)
            if os.path.isdir(folder_path):
                for image_name in os.listdir(folder_path):
                    if image_name.endswith(('.jpg', ".png", ".jpeg")):
                        data.append({
                            "image_path": str(os.path.join(folder_path, image_name)),
                            "label": int(num_folder)
                        })
        self.image_dataframe = pd.DataFrame(data)
        return self.image_dataframe

    def get_train_val_data(self):
        df = self.create_image_dataframe()
        train, val = train_test_split(
            df,
            test_size = 0.2,
            stratify = df["label"],
            random_state = 42
        )
        return train, val
    