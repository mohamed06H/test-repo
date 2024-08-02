# test-repo

See the notebook for step by step implementation with visual tests

## Build and use app

docker build -t image_processing .

docker run image_processing --n_batches 5 --n_images 20  

When n_batches == 5 and n_images == 20 the resulting stats dataframe is uploaded to s3 in parquet format.

