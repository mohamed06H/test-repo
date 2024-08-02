import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyarrow
import fastparquet

import boto3
from botocore.exceptions import NoCredentialsError




# ------------ Functions 
def save_to_s3(df, bucket, path):
    """Save DataFrame to S3 as a Parquet file."""
    try:
        # Save DataFrame to Parquet file locally
        df.to_parquet('/tmp/temp.parquet')

        # Initialize S3 client
        s3 = boto3.client('s3')

        # Upload file to S3
        s3.upload_file('/tmp/temp.parquet', bucket, path)
        print(f"File uploaded to s3://{bucket}/{path}")
    except NoCredentialsError:
        print("Credentials not available.")

def generate_image(image_shape: tuple[int, int, int]):
    """ Generates an RGB Image represented by 3 matrices of dimensions (m, n)
    """
    if image_shape[2] == 3:
        image = np.random.randint(0, 256, size=image_shape)
        return image
    else:
        raise ValueError("Image shape is not correct, a tuple (m, n, 3) is required")

def generate_random_batches(n_batches: int, n_images: int, image_shape: tuple[int, int, int]):
    """ Generates a list of random image batches
    """
    return [[generate_image(image_shape) for _ in range(n_images)] for _ in range(n_batches)]


def plot_batches(batches):
    """ Plot images in row"""
    # Flatten images to plot them on the same line
    flattened_images = [image for batch in batches for image in batch]

    num_images = len(flattened_images)

    # Create subplots 
    if num_images > 1:
        fig, axes = plt.subplots(1, num_images)
        for ax, image in zip(axes, flattened_images):
            ax.imshow(image)
            ax.axis()
    else:
        plt.imshow(flattened_images[0])

def verify_overlapping_positions(x1, y1, x2, y2, square_size) -> bool:
    """ Check if two squares overlap.
    """
    if abs(x1 - x2) < square_size or abs(y1 - y2) < square_size:
        return True
    return False


def apply_square(image, x_pos, y_pos, square_size, color):
    """
    Apply Square of a given color to an image.
    x_pos and y_pos represent the top left of the square to add.
    """
    image[y_pos:y_pos+square_size, x_pos:x_pos+square_size, :] = color

    return image


def alter_image(image, square_size):
    """ Adds one black and one white square to an image, added squares do not overlap
    """

    # Find two random non-overlapping positions for each black and white squares, top-left position of the square
    # Generate two random coordinates (x1, y1) (x2, y2) within the image, that verfiy both these conditions:
    # |x1 - X2| > square_size and |y1 - y2| > square_size
    m = image.shape[0] # height
    n = image.shape[1] # width
    # Square size must fit in the image
    if square_size >= min(m,n)/2 or square_size <= 0:
        raise ValueError(f"Square size is bigger than the image size/2 or negative: {square_size}")
    
    overlapping = True
    while overlapping:
        # Random positions constrained only by the limits of the image
        x1 = np.random.randint(0, n - square_size)
        y1 = np.random.randint(0, m - square_size)

        x2 = np.random.randint(0, n - square_size) 
        y2 = np.random.randint(0, m - square_size)
       
        overlapping = verify_overlapping_positions(x1, y1, x2, y2, square_size)
    
    # print(x1, y1)
    # print(x2, y2)

    # Apply squares to the image on the positions found
    altered_image = apply_square(image, x1, y1, square_size, color=0) # Black square
    altered_image = apply_square(image, x2, y2, square_size, color=255) # White square

    return altered_image

def add_randomly_placed_squares(image_batches, square_size):
    """ Inserts two squares white and black to each image in the batches"""
    return [[alter_image(image, square_size) for image in batch] for batch in image_batches]

def random_crop(image, new_size):
    """ Returns a random subset of the image of size new_size"""
    # Pick random top-left postion with respect to the new size and image size
    m = image.shape[0] # number of rows
    n = image.shape[1] # number of columns

    if new_size > min(m,n) or new_size <= 0:
        raise ValueError(f"New size is bigger than the image or negative: {new_size}")
    x_pos =  np.random.randint(0, n - new_size+1)
    y_pos =  np.random.randint(0, m - new_size+1)

    croped_image = image[y_pos:y_pos+new_size, x_pos:x_pos+new_size, :]
    return croped_image

def random_crop_batches(batches, new_size):
    """ Randomly crops all images in batches of images to a given size"""
    return [[random_crop(image, new_size) for image in batch] for batch in batches]

def count_pixels(image, pixel:tuple[int, int, int]):
    """ Counts the occurences of an RGB value in an image
    """
    pixel_count = np.sum(np.all(image==pixel, axis=-1)) # Condition must match all the three RGB layers -> axis = 2 or -1
    return pixel_count


def get_batch_stats(images, batch_id:str):
    """ Returns pd.dataframe stats about black and white pixels in a list of images"""

    white_counts = []
    black_counts = []

    for image in images:
        white_counts.append(count_pixels(image, (255, 255, 255)))
        black_counts.append(count_pixels(image, (0, 0, 0)))
    
    # Average
    white_avg = np.average(white_counts)
    black_avg = np.average(black_counts)

    # Min 
    white_min = np.min(white_counts)
    black_min = np.min(black_counts)

    # Max
    white_max = np.max(white_counts)
    black_max = np.max(black_counts)

    # Std
    white_std = np.std(white_counts)
    black_std = np.std(black_counts)

    data = {
        'batch_id': batch_id,
        'white_avg': white_avg,
        'white_min': white_min,
        'white_max': white_max,
        'white_std': white_std,

        'black_avg': black_avg,
        'black_min': black_min,
        'black_max': black_max,
        'black_std': black_std,
    }

    return pd.Series(data)

def get_stats(batches):
    rows = [get_batch_stats(batch, f"batch_{idx}") for batch, idx in zip(batches, range(len(batches)))]
    df = pd.DataFrame(rows)
    return df


# ------------ main
def main(n_batches: int, n_images:int):
    # Retrieve the WORKDIR environment variable
    
    S3_BUCKET = os.environ.get('S3_BUCKET')

    # Instruction 1
    generated_batches = generate_random_batches(n_batches=n_batches, n_images=n_images, image_shape=(256, 512, 3))
    # plot_batches(generated_batches)
    
    # Instruction 2
    altered_batches = add_randomly_placed_squares(generated_batches, square_size=50)
    # plot_batches(altered_batches)
    
    # Instruction 3
    croped_batches = random_crop_batches(batches=altered_batches, new_size=200)
    # plot_batches(croped_batches)

    # Instruction 4
    df = get_stats(croped_batches)
    print(df.head())

    # Instruction 5 - Save to a parquet file
    if n_batches == 5 and n_images == 20:
        print("Saving Images to S3 ...")
        save_to_s3(df, S3_BUCKET, 'test/test.parquet')
        


if __name__ == "__main__":
     # Setup argument parser
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument('--n_batches', type=int, required=True, help='Number of batches to generate')
    parser.add_argument('--n_images', type=int, required=True, help='Number of images per batch')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call main function with parsed arguments
    main(args.n_batches, args.n_images)