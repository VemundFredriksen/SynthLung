import argparse
import json
import tqdm

from synthlung.utils.tumor_isolation_pipeline import TumorCropPipeline
from synthlung.utils.dataset_formatter import MSDImageSourceFormatter, JsonSeedGenerator, JsonTrainingGenerator
from synthlung.utils.tumor_insertion_pipeline import InsertTumorPipeline
from synthlung.utils.lung_segmentation_pipeline import LungMaskPipeline, HostJsonGenerator

from synthlung.train_pipeline.train import TrainPipeline

from lungmask import LMInferer

def seed():
    json_file_path = "./assets/source/dataset.json"

    with open(json_file_path, 'r') as json_file:
        image_dict = json.load(json_file)
    crop_pipeline = TumorCropPipeline()
    crop_pipeline(image_dict)
    formatter = JsonSeedGenerator("./assets/seeds/")
    formatter.generate_json_seeds()

def format_msd():
    formatter = MSDImageSourceFormatter()
    formatter.format()
    formatter.generate_json()

def generate_randomized_tumors():
    tumor_inserter = InsertTumorPipeline()
    json_file_path = "./assets/source/dataset.json"
    with open(json_file_path, 'r') as json_file:
        image_dict = json.load(json_file)

    json_seed_path = "./assets/seeds/dataset.json"
    with open(json_seed_path, 'r') as json_file:
        seeds_dict = json.load(json_file)

    tumor_inserter(image_dict, seeds_dict)
    round_dict = tumor_inserter.getDict()

    path = round_dict[0]["randomized_image"].split("0_image")[0]
    formatter = JsonTrainingGenerator(path)
    formatter.generate_json()
    return path

def mask_hosts():
    lung_masker = LMInferer()
    host_masker = LungMaskPipeline(lung_masker)
    json_file_path = "./assets/source/dataset.json"
    with open(json_file_path, 'r') as json_file:
        image_dict = json.load(json_file)
    
    host_masker(image_dict)
    json_generator = HostJsonGenerator('./assets/hosts/')
    json_generator.generate_json()

def train():
    path = "./synthlung/config.json"
    with open(path, "r") as f:
        data = json.load(f)

    trainPipeline = TrainPipeline(data)
    trainPipeline.verify_config()
    trainPipeline()
    
    exit(0)

    for n in range(10):
        path = generate_randomized_tumors()
        with open(f"{path}dataset.json", 'r') as json_file:
            image_dict = json.load(json_file)
        trainPipeline(image_dict)

def main():
    parser = argparse.ArgumentParser(description="Create your synthetic lung tumors!")

    parser.add_argument("action", choices=["format", "seed", "host", "generate", "train"], help="Action to perform")
    parser.add_argument("--dataset", help="Dataset to format", choices=["msd"])
    args = parser.parse_args()

    if args.action == "format":
        if(args.dataset == "msd"):
            format_msd()
    elif args.action == "seed":
        seed()
    elif args.action == "generate":
        if(args.dataset == "msd"):
            generate_randomized_tumors()
    elif args.action == "host":
        if(args.dataset == "msd"):
            mask_hosts()
    elif args.action == "train":
        if(args.dataset == "msd"):
            train()
    else:
        print("Action not recognized")
