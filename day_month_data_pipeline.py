import os
import pandas as pd
import tensorflow as tf
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
    
    def _load_and_process_df(self, path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    def create_tf_dataset(self, df):
        ds = tf.data.Dataset.from_tensor_slices((df["image_path"].values, df["label"].values))
        ds = ds.map(self._load_and_process_df, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def get_train_val_data(self):
        df = self.create_image_dataframe()
        train, val = train_test_split(
            df,
            test_size = 0.2,
            stratify = df["label"],
            random_state = 42
        )
        train = self.create_tf_dataset(train)
        val = self.create_tf_dataset(val)
        return train, val
    
