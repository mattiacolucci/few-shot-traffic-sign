import os
import shutil
import pandas as pd
import random

def create_folders_by_artists(df,artist,image_folder,output_folder,style):
    # take all images from the artist (which will be between 30 and 50
    # 70% for the train, 10% for the val and 20% for the test
    images = df[(df['artist'] == artist) & (df['style']==style)]['file_name'].tolist()
    images_train = images[:len(images)*70//100]
    images_val = images[len(images)*70//100:len(images)*80//100]
    images_test = images[len(images)*80//100:]

    # create train folder
    artist_folder = os.path.join(output_folder, "train", artist)
    os.makedirs(artist_folder, exist_ok=True)

    for image_name in images_train:
        # Move the image to the artist's folder
        source_path = os.path.join(image_folder, image_name)
        destination_path = os.path.join(artist_folder, image_name)

        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)
        else:
            print(f"Warning: Image {image_name} not found in {image_folder}.")


    # create test folder
    artist_folder = os.path.join(output_folder, "test", artist)
    os.makedirs(artist_folder, exist_ok=True)

    for image_name in images_test:
        # Move the image to the artist's folder
        source_path = os.path.join(image_folder, image_name)
        destination_path = os.path.join(artist_folder, image_name)

        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)
        else:
            print(f"Warning: Image {image_name} not found in {image_folder}.")

    
    # create val folder
    artist_folder = os.path.join(output_folder, "val", artist)
    os.makedirs(artist_folder, exist_ok=True)

    for image_name in images_val:
        # Move the image to the artist's folder
        source_path = os.path.join(image_folder, image_name)
        destination_path = os.path.join(artist_folder, image_name)

        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)
        else:
            print(f"Warning: Image {image_name} not found in {image_folder}.")



def create_dataset(image_folder, csv_file, output_folder):
    min_artworks=30
    max_artworks=50
    
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # count the occurances of each style
    style_counts = df['style'].value_counts()

    # sorted styles
    sorted_styles = df['style'].unique().tolist()
    sorted_styles.sort(key=lambda x: style_counts[x], reverse=True)

    # take top 10 styles
    sorted_styles = sorted_styles[:10]

    artists=set()

    # for each style
    for st in sorted_styles:
        # Count the number of artworks of each artist of style st
        artist_counts = df[df['style'] == st]['artist'].value_counts()

        # get all artists of the style
        style_artists = df[df['style'] == st]['artist'].unique().tolist()

        # Filter out artists that are already in the `artists` set, that have not made any artworks of the style
        # getting only artists with more than `min_artworks` and less than `max_artworks` artworks of that style
        artist_counts = artist_counts[(~artist_counts.index.isin(artists)) & (artist_counts.index.isin(style_artists)) & (artist_counts<=max_artworks)]
        
        # sort artists by num of artworks
        artist_counts.sort_values(ascending=False, inplace=True)

        # take 10 artists with the most number of artworks of the style
        top_artists = artist_counts.head(10)

        # Add the selected artists to the `artists` set
        artists.update(top_artists.index.tolist())

        print(f"- Style '{st}' with {style_counts[st]} artworks")
        print(f"Top artists for style '{st}':")
        for a in top_artists.index.tolist():
            print(f"    - {a}: {artist_counts[a]} artworks")
            
            # For each artist, pick random 40 images, so to create the few-shots training dataset
            # also, take other different 20 samples to create the validation and other 20 to create the test dataset
            create_folders_by_artists(df,a,image_folder,output_folder,st)

def analyze(csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Display basic information about the dataset
    print("\nBasic information about the dataset:")
    print(df.info())

    # Display summary statistics for numerical columns
    print("\nSummary statistics for numerical columns:")
    print(df.describe())  

    # Display the distribution of artworks per artist (if applicable)
    if 'artist' in df.columns:
        print("\nDistribution of artworks per artist:")
        print(df['artist'].value_counts().describe())
  
    

# Example usage
if __name__ == "__main__":
    image_folder = "./datasets/artworks/artgraph/images"  # Replace with the path to your image folder
    csv_file = "./datasets/artworks/artgraph.csv"     # Replace with the path to your CSV file
    output_folder = "./datasets/artworks_style"  # Replace with the path to the output folder

    #organize_images_by_artist(image_folder, csv_file, output_folder)
    create_dataset(image_folder, csv_file, output_folder)
    #analyze(csv_file)