import pytest
from synthlung.utils.dataset_formatter import MSDImageSourceFormatter
import shutil
from pathlib import Path
import json

def test_msd_formatter_format():
    # Arrange
    source_dir = './tests/source'
    target_dir = './tests/target'

    shutil.os.makedirs(f'{source_dir}/imagesTr/')
    shutil.os.makedirs(f'{source_dir}/labelsTr/')
    dummy_images = ["lung_001.nii.gz", "lung_002.nii.gz", "lung_003.nii.gz"]
    for image in dummy_images:
        image_path = Path(f"{source_dir}/imagesTr/{image}")
        label_path = Path(f"{source_dir}/labelsTr/{image}")
        Path.touch(image_path)
        Path.touch(label_path)
    

    msd_image_source_formatter = MSDImageSourceFormatter(source_dir, target_dir)

    # Act
    msd_image_source_formatter.format()

    # Assert
    expected_files = ["source_msd_lung_001_image.nii.gz", "source_msd_lung_001_label.nii.gz", "source_msd_lung_002_image.nii.gz", "source_msd_lung_002_label.nii.gz", "source_msd_lung_003_image.nii.gz", "source_msd_lung_003_label.nii.gz"]
    actual_files = sorted(shutil.os.listdir(target_dir))
    
    assert expected_files == actual_files

    # Cleanup
    shutil.rmtree(source_dir)
    shutil.rmtree(target_dir)

def test_msd_formatter_generate_json():
    # Arrange
    target_dir = './tests/target/'
    shutil.os.makedirs(target_dir)

    dummy_images = ["source_msd_lung_001_image.nii.gz", "source_msd_lung_001_label.nii.gz", "source_msd_lung_002_image.nii.gz", "source_msd_lung_002_label.nii.gz"]
    for image in dummy_images:
        image_path = Path(f"{target_dir}/{image}")
        Path.touch(image_path)
    

    msd_image_source_formatter = MSDImageSourceFormatter(target_dir, target_dir)

    # Act
    msd_image_source_formatter.generate_json()

    # Assert
    expected_json = [
        {
            "image": f"{target_dir}source_msd_lung_001_image.nii.gz",
            "label": f"{target_dir}source_msd_lung_001_label.nii.gz"
        },
        {
            "image": f"{target_dir}source_msd_lung_002_image.nii.gz",
            "label": f"{target_dir}source_msd_lung_002_label.nii.gz"
        }
    ]

    actual_json = None
    with open(f"{target_dir}/dataset.json", 'r') as json_file:
        actual_json = json.load(json_file)

    assert expected_json == actual_json

    # Cleanup
    shutil.rmtree(target_dir)