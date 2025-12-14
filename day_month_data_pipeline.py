import pandas
from pathlib import Path

class DayMonth_data_pipeline:
    def __init__(self, dataset_dir, batch_size=32, img_size=(64,64)):
        self.dataset_dir = Path(dataset_dir)
        self.batch_size  = batch_size
        self.image_size  = img_size
        self.dataframe   = None
        self.class_names = ["၀", "၁", "၂", "၃", "၄", "၅", "၆", "၇", "၈", "၉", "၁၀", 
                            "၁၁", "၁၂", "၁၃", "၁၄", "၁၅","၁၆", "၁၇", "၁၈", "၁၉", "၂၀", 
                            "၂၁", "၂၂", "၂၃", "၂၄", "၂၅", "၂၆", "၂၇", "၂၈", "၂၉", "၃၀", "၃၁"]
        
    