import argparse
from utils.tumor_isolation_pipeline import TumorCropPipeline

image_dict = [
    {
        'image' : "D:\\Images\\LungTumor\\imagesTr\\lung_001.nii.gz",
        'label' : "D:\\Images\\LungTumor\\labelsTr\\lung_001.nii.gz"
    },
    {
        'image' : "D:\\Images\\LungTumor\\imagesTr\\lung_003.nii.gz",
        'label' : "D:\\Images\\LungTumor\\labelsTr\\lung_003.nii.gz"
    }
]

def tumor_crop():
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

main()