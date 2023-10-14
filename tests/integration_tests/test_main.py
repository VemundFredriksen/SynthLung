import pytest
from synthlung.utils.dataset_formatter import MSDImageSourceFormatter
import shutil
from pathlib import Path

def test_main_format_msd():
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
    actual_files = shutil.os.listdir(target_dir)
    
    assert expected_files == actual_files

    # Cleanup
    shutil.rmtree(source_dir)
    shutil.rmtree(target_dir)
    