import argparse
from synthlung.utils.tumor_isolation_pipeline import TumorCropPipeline
import json

def tumor_crop():
    json_file_path = "./../assets/source_images/msd/dataset.json"

    with open(json_file_path, 'r') as json_file:
        image_dict = json.load(json_file)
    crop_pipeline = TumorCropPipeline()
    crop_pipeline(image_dict)

def main():
    parser = argparse.ArgumentParser(description="Program description")

    parser.add_argument("action", choices=["crop", "filter"], help="Action to perform")
    args = parser.parse_args()

    if args.action == "crop":
        tumor_crop()
    else:
        print("Action not recognized")

tumor_crop()