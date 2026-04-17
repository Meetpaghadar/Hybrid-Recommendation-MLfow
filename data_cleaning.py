# import pandas as pd




# def clean_data(data):
#     """
#     Cleans the input DataFrame by performing the following operations:
#     1. Removes duplicate rows based on the 'spotify_id' column.
#     2. Drops the 'genre' and 'spotify_id' columns.
#     3. Fills missing values in the 'tags' column with the string 'no_tags'.
#     4. Converts the 'name', 'artist', and 'tags' columns to lowercase.

#     Parameters:
#     data (pd.DataFrame): The input DataFrame containing the data to be cleaned.

#     Returns:
#     pd.DataFrame: The cleaned DataFrame.
#     """
#     return (
#         data
#         .drop_duplicates(subset="track_id")
#         .drop(columns=["genre","spotify_id"])
#         .fillna({"tags":"no_tags"})
#         .assign(
#             name=lambda x: x["name"].str.lower(),
#             artist=lambda x: x["artist"].str.lower(),
#             tags=lambda x: x["tags"].str.lower()
#         )
#         .reset_index(drop=True)
#     )
    
    
# def data_for_content_filtering(data):
#     """
#     Cleans the input DataFrame by dropping specific columns.

#     This function takes a DataFrame and removes the columns "track_id", "name",
#     and "spotify_preview_url". It is intended to prepare the data for content based
#     filtering by removing unnecessary features.

#     Parameters:
#     data (pandas.DataFrame): The input DataFrame containing songs information.

#     Returns:
#     pandas.DataFrame: A DataFrame with the specified columns removed.
#     """
#     return (
#         data
#         .drop(columns=["track_id","name","spotify_preview_url"])
#     )
    
    
# def main(data_path):
#     """
#     Main function to load, clean, and save data.
#     Parameters:
#     data_path (str): The file path to the raw data CSV file.
#     Returns:
#     None
#     """
#     # load the data
#     data = pd.read_csv('Data\Music Info.csv')
    
#     # perform data cleaning
#     cleaned_data = clean_data(data)
    
#     # saved cleaned data
#     cleaned_data.to_csv("data/cleaned_data.csv",index=False)
    

# if __name__ == "__main__":
#     main(DATA_PATH)

import pandas as pd


def clean_data(data):
    """
    Cleans the input DataFrame:
    1. Removes duplicates based on track_id
    2. Drops unnecessary columns
    3. Fills missing tags
    4. Converts text columns to lowercase
    """

    return (
        data
        .drop_duplicates(subset="track_id")
        .drop(columns=["genre", "spotify_id"])
        .fillna({"tags": "no_tags"})
        .assign(
            name=lambda x: x["name"].str.lower(),
            artist=lambda x: x["artist"].str.lower(),
            tags=lambda x: x["tags"].str.lower()
        )
        .reset_index(drop=True)
    )


def data_for_content_filtering(data):
    """
    Prepares dataset for content-based filtering
    """

    return data.drop(columns=["track_id", "name", "spotify_preview_url"])


def main(data_path):
    """
    Loads, cleans, and saves dataset
    """

    # Load data (safe path handling)
    data = pd.read_csv(data_path)

    # Clean data
    cleaned_data = clean_data(data)

    # Save cleaned data
    cleaned_data.to_csv("data/cleaned_data.csv", index=False)

    print(" Data cleaning completed successfully!")


# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    main("Data/Music Info.csv")