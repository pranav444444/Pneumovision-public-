import os
from dataclasses import dataclass

from torch import device

from xray.constant.training_pipeline import *


@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.s3_data_folder: str = S3_DATA_FOLDER

        self.bucket_name: str = BUCKET_NAME

        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)

        self.data_path: str = os.path.join(
            self.artifact_dir, "data_ingestion", self.s3_data_folder
        )

        self.train_data_path: str = os.path.join(self.data_path, "train")

        self.test_data_path: str = os.path.join(self.data_path, "test")


@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.RESIZE: int = RESIZE

        self.CENTERCROP: int = CENTERCROP

        self.RANDOMROTATION: int = RANDOMROTATION

        self.normalize_transforms: dict = {
            "mean": NORMALIZE_LIST_1,
            "std": NORMALIZE_LIST_2,
        }

        self.train_data_loader_params: dict = {
            "batch_size": BATCH_SIZE,
            "shuffle": TRAIN_SHUFFLE,
            "pin_memory": PIN_MEMORY,
            "num_workers": NUM_WORKERS,
            "persistent_workers": PERSISTENT_WORKERS,
        }

        self.test_data_loader_params: dict = {
            "batch_size": BATCH_SIZE,
            "shuffle": TEST_SHUFFLE,
            "pin_memory": PIN_MEMORY,
            "num_workers": NUM_WORKERS,
            "persistent_workers": PERSISTENT_WORKERS,
        }

        self.artifact_dir: str = os.path.join(
            ARTIFACT_DIR, TIMESTAMP, "data_transformation"
        )

        self.train_transforms_file: str = os.path.join(
            self.artifact_dir, TRAIN_TRANSFORMS_FILE
        )

        self.test_transforms_file: str = os.path.join(
            self.artifact_dir, TEST_TRANSFORMS_FILE
        )


@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.artifact_dir: str = os.path.join(
            ARTIFACT_DIR,
            TIMESTAMP,
            "model_training"
        )

        self.trained_bentoml_model_name: str = BENTOML_MODEL_NAME

        self.trained_model_path: str = os.path.join(
            self.artifact_dir,
            TRAINED_MODEL_NAME
        )

        self.inference_transforms_key: str = INFERENCE_TRANSFORMS_KEY

        self.epochs: int = EPOCH

        self.optimizer_params: dict = {
            "lr": 0.001
        }

        self.scheduler_params: dict = {
            "step_size": STEP_SIZE,
            "gamma": GAMMA
        }

        self.device: device = DEVICE


@dataclass
class ModelEvaluationConfig:
    def __init__(self):
        # Evaluation only needs the computation device.
        # Loss and metric values are calculated locally
        # inside model_evaluation.py.
        self.device: device = DEVICE


# Model Pusher Configurations
@dataclass
class ModelPusherConfig:
    def __init__(self):
        self.bentoml_model_name: str = BENTOML_MODEL_NAME

        self.bentoml_service_name: str = BENTOML_SERVICE_NAME

        self.inference_transforms_key: str = INFERENCE_TRANSFORMS_KEY

        self.bentoml_ecr_image: str = BENTOML_ECR_IMAGE

        self.aws_region: str = AWS_REGION

        self.ecr_registry: str = ECR_REGISTRY