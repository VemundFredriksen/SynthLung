import argparse
from synthlung.utils.tumor_isolation_pipeline import TumorCropPipeline
from synthlung.utils.dataset_formatter import MSDImageSourceFormatter
import json

def seed_msd():
    json_file_path = "./assets/source_images/msd/dataset.json"

    with open(json_file_path, 'r') as json_file:
        image_dict = json.load(json_file)
    crop_pipeline = TumorCropPipeline()
    crop_pipeline(image_dict)

def format_msd():
    formatter = MSDImageSourceFormatter()
    formatter.format()

def main():
    parser = argparse.ArgumentParser(description="Create your synthetic lung tumors!")

    parser.add_argument("action", choices=["format", "seed"], help="Action to perform")
    parser.add_argument("--dataset", help="Dataset to format", choices=["msd"])
    args = parser.parse_args()

    if args.action == "format":
        if(args.dataset == "msd"):
            format_msd()
    elif args.action == "seed":
        if(args.dataset == "msd"):
            seed_msd()
    else:
        print("Action not recognized")

seed_msd()