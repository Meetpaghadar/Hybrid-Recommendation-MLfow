import pandas as pd
from data_cleaning import data_for_content_filtering
from content_based import transform_data, save_transformed_data

filtered_data_path = "Data/collab_filtered_data.csv"


save_path = "Data/transformed_hybrid_data.npz"


def main(data_path, save_path):
  
    filtered_data = pd.read_csv(data_path)
    filtered_data_cleaned = data_for_content_filtering(filtered_data)
    filtered_data_cleaned = filtered_data_cleaned.astype(str)
    transformed_data = transform_data(filtered_data_cleaned)
    save_transformed_data(transformed_data, save_path)
    

if __name__ == "__main__":
    main(filtered_data_path, save_path)