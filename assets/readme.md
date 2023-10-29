# Assets

### How to format and preprocess?

1. Install synthlung by running `pip install .`
2. Download the MSD Lung Tumor dataset from [here](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2).
3. Extract the zip file into `/assets/`.
4. Run `synthlung format --dataset msd` to adjust dataset format
5. Run `synthlung seed --dataset` to extract tumor seeds from the dataset
6. Run `synthlung host --dataset msd` to extract lung masks from the images
