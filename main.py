"""
This script extracts features from images, trains a classifier, and stores the classifier in a file.
It uses AWS S3 to store the images and features.
The script loads the annotations for the images from a JSON file and the labelset from a CSV file.
It then extracts features from the images using an EfficientNetExtractor and stores them in S3.
Finally, it trains a classifier using the extracted features and stores it in a file.
"""
import csv
import json
from operator import itemgetter
import os
import boto3
import botocore.exceptions
from botocore.exceptions import NoCredentialsError, ClientError
from pathlib import Path
from spacer import config
from spacer.data_classes import ImageLabels
from spacer.extract_features import EfficientNetExtractor
from spacer.messages import (
    ClassifyFeaturesMsg,
    DataLocation,
    ExtractFeaturesMsg,
    TrainClassifierMsg,
)
from spacer.tasks import classify_features, extract_features, train_classifier

# Load the secret.json file
with open('secrets.json', 'r') as f:
    secrets = json.load(f)
# Create an S3 resource using credentials from the secret.json file
s3 = boto3.resource(
    's3',
    region_name=secrets['AWS_REGION'],
    aws_access_key_id=secrets['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=secrets['AWS_SECRET_ACCESS_KEY']
)

# The bucket you have read permissions for
bucket_name = 'pyspacer-test'  
# Create a bucket object
bucket = s3.Bucket(bucket_name)
try:
    # List objects within the bucket
    for obj in bucket.objects.all():
        print(f'Object: {obj.key}')
except botocore.exceptions.ClientError as e:
    print(f'An error occurred: {e}')
except botocore.exceptions.NoCredentialsError as e:
    print("Credentials not available")

# Module directory is local directory
module_dir = Path.cwd()
output_dir = module_dir / 'output'
TOP_SCORES_PER_POINT = 5

# Create client

s3_client = boto3.client(
    's3',
    region_name=secrets['AWS_REGION'],
    aws_access_key_id=secrets['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=secrets['AWS_SECRET_ACCESS_KEY']
)

# Download everything following the same folder structure in the bucket
# Should add a test here for file size, user prompt, etc.

def image_to_feature_filepath(image_filepath: Path) -> Path:
    return output_dir / 'features' / (image_filepath.stem + '.featurevector')

def image_filepath_to_id(image_filepath: Path) -> str:
    # Image IDs aren't a pyspacer construct; this is just how this
    # example organizes the images and annotations.
    return image_filepath.stem.split('_')[-1]

if __name__ == '__main__':

    annotated_image_dir = module_dir / 'images-annotated'
    unannotated_image_dir = module_dir / 'images-unannotated'
    extractor_weights_filepath = module_dir / 'efficientnet_b0_ver1.pt'

    # The local directory to which you want to download the files is the current working directory
    local_directory = Path.cwd()

    # List and download all objects from the bucket
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name):
        for obj in page.get('Contents', []):
            # Define the local path where the file should be downloaded
            local_file_path = local_directory / obj['Key']
            local_file_dir = local_file_path.parent

            # Create directories if they don't exist
            os.makedirs(local_file_dir, exist_ok=True)

            # Download the file
            s3_client.download_file(bucket_name, obj['Key'], str(local_file_path))
            print(f"Downloaded {obj['Key']} to {local_file_path}")

    print("Download completed")
    with open(module_dir / 'annotations.json') as f:
        all_annotations = json.load(f)

    label_ids_to_codes = dict()
    with open(module_dir / 'labelset.csv') as f:
        reader = csv.reader(f)
        for label_id, label_code in reader:
            label_ids_to_codes[label_id] = label_code

    annotated_image_filepaths = []
    # Sort for predictable ordering
    for filename in sorted(os.listdir(annotated_image_dir)):
        annotated_image_filepaths.append(annotated_image_dir / filename)
    unannotated_image_filepaths = []
    for filename in sorted(os.listdir(unannotated_image_dir)):
        unannotated_image_filepaths.append(unannotated_image_dir / filename)
    all_image_filepaths = (
        annotated_image_filepaths + unannotated_image_filepaths)

    # Extract features

    print("------------------------------")

    for image_filepath in all_image_filepaths:
        feature_filepath = image_to_feature_filepath(image_filepath)
        image_id = image_filepath_to_id(image_filepath)
        annotations = all_annotations[image_id]

        message = ExtractFeaturesMsg(
            job_token=image_filepath.name,
            extractor=EfficientNetExtractor(
                data_locations=dict(
                    weights=DataLocation(
                        'filesystem', str(extractor_weights_filepath)),
                ),
            ),
            rowcols=[(row, col) for row, col, _ in annotations],
            image_loc=DataLocation('filesystem', str(image_filepath)),
            feature_loc=DataLocation('filesystem', str(feature_filepath)),
        )
        return_message = extract_features(message)
        print(
            f"Feature vector stored at: {feature_filepath}"
            f" (extract time: {return_message.runtime:.1f} s)")

    # Train classifier

    train_labels_data = dict()
    val_labels_data = dict()

    for i, image_filepath in enumerate(annotated_image_filepaths):
        # First 16 annotated images go in train labels, last 4 go in val
        if i < 16:
            data = train_labels_data
        else:
            data = val_labels_data

        feature_filepath = image_to_feature_filepath(image_filepath)
        image_id = image_filepath_to_id(image_filepath)
        annotations = all_annotations[image_id]
        data[feature_filepath] = annotations

    classifier_filepath = output_dir / 'classifier1.pkl'
    valresult_filepath = output_dir / 'valresult.json'

    message = TrainClassifierMsg(
        job_token='classifier1',
        trainer_name='minibatch',
        nbr_epochs=10,
        clf_type='MLP',
        train_labels=ImageLabels(data=train_labels_data),
        val_labels=ImageLabels(data=val_labels_data),
        features_loc=DataLocation('filesystem', ''),
        previous_model_locs=[],
        model_loc=DataLocation('filesystem', str(classifier_filepath)),
        valresult_loc=DataLocation('filesystem', str(valresult_filepath)),
    )
    return_message = train_classifier(message)

    ref_accs_str = ", ".join(
        [f"{100*acc:.1f}" for acc in return_message.ref_accs])

    print("------------------------------")
    print(f"Classifier stored at: {classifier_filepath}")
    print(f"New model's accuracy: {100*return_message.acc:.1f}%")
    print(
        "New model's accuracy progression (calculated on part of train_labels)"
        f" after each epoch of training: {ref_accs_str}")

    print(f"Evaluation results:")
    with open(valresult_filepath) as f:
        valresult = json.load(f)
    label_list = [
        label_ids_to_codes[str(label_id)] for label_id in valresult['classes']]
    for ground_truth_i, prediction_i, score in zip(
        valresult['gt'], valresult['est'], valresult['scores']
    ):
        print(f"Actual = {label_list[ground_truth_i]}, Predicted = {label_list[prediction_i]}, Confidence = {100*score:.1f}%")

    print(f"Train time: {return_message.runtime:.1f} s")

    # Classify

    for image_filepath in unannotated_image_filepaths:
        feature_filepath = image_to_feature_filepath(image_filepath)

        message = ClassifyFeaturesMsg(
            job_token=image_filepath.name,
            feature_loc=DataLocation('filesystem', feature_filepath),
            classifier_loc=DataLocation('filesystem', classifier_filepath),
        )
        return_message = classify_features(message)

        print("------------------------------")
        print(f"Classification result for {image_filepath.name}:")

        label_ids = return_message.classes
        for i, (row, col, scores) in enumerate(return_message.scores):
            top_scores = sorted(
                zip(label_ids, scores), key=itemgetter(1), reverse=True)
            top_scores_str = ", ".join([
                f"{label_ids_to_codes[str(label_id)]} = {100*score:.1f}%"
                for label_id, score in top_scores[:TOP_SCORES_PER_POINT]
            ])
            print(f"- Row {row}, column {col}: {top_scores_str}")

        print(f"Classification time: {return_message.runtime:.1f} s")

    print("------------------------------")
    print("Clear the output dir before rerunning this script.")