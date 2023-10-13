import argparse
from synthlung.utils.tumor_isolation_pipeline import TumorCropPipeline
from synthlung.utils.dataset_formatter import MSDImageSourceFormatter, MSDGenerateJSONFormatter
from synthlung.utils.tumor_insertion_pipeline import InsertTumorPipeline
import json

def seed_msd():
    json_file_path = "./assets/source/msd/dataset.json"

    with open(json_file_path, 'r') as json_file:
        image_dict = json.load(json_file)
    crop_pipeline = TumorCropPipeline()
    crop_pipeline(image_dict)
    formatter = MSDGenerateJSONFormatter("./assets/seeds/msd/")
    formatter.generate_json()

def format_msd():
    formatter = MSDImageSourceFormatter()
    formatter.format()
    formatter.generate_json()

def generate_randomized_tumors():
    tumor_inserter = InsertTumorPipeline()
    json_file_path = "./assets/source/msd/dataset.json"
    with open(json_file_path, 'r') as json_file:
        image_dict = json.load(json_file)

    json_seed_path = "./assets/seeds/msd/dataset.json"
    with open(json_seed_path, 'r') as json_file:
        seeds_dict = json.load(json_file)

    tumor_inserter(image_dict, seeds_dict)

def main():
    parser = argparse.ArgumentParser(description="Create your synthetic lung tumors!")

    parser.add_argument("action", choices=["format", "seed", "generate"], help="Action to perform")
    parser.add_argument("--dataset", help="Dataset to format", choices=["msd"])
    args = parser.parse_args()

    if args.action == "format":
        if(args.dataset == "msd"):
            format_msd()
    elif args.action == "seed":
        if(args.dataset == "msd"):
            seed_msd()
    elif args.action == "generate":
        if(args.dataset == "msd"):
            generate_randomized_tumors()
    else:
        print("Action not recognized")
