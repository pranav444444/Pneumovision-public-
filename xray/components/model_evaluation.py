import sys
from typing import Tuple

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader

from xray.entity.artifacts_entity import (
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from xray.entity.config_entity import ModelEvaluationConfig
from xray.exception import XRayException
from xray.logger import logging


class ModelEvaluation:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        # Contains the test DataLoader produced by Data Transformation
        self.data_transformation_artifact = data_transformation_artifact

        # Contains device and other evaluation-related configuration
        self.model_evaluation_config = model_evaluation_config

        # Contains the path of the trained model
        self.model_trainer_artifact = model_trainer_artifact

    def configuration(self) -> Tuple[DataLoader, Module, Module]:
        """
        Prepare the objects required for model evaluation.

        Returns:
            test_dataloader
            trained model
            CrossEntropyLoss criterion
        """
        logging.info(
            "Entered the configuration method of Model evaluation class"
        )

        try:
            # Test DataLoader received from Data Transformation stage
            test_dataloader: DataLoader = (
                self.data_transformation_artifact.transformed_test_object
            )

            # Load the complete model saved by ModelTrainer.
            # This is intentionally kept compatible with:
            # torch.save(model, trained_model_path)
            model: Module = torch.load(
                self.model_trainer_artifact.trained_model_path
            )

            # Move model to GPU if available, otherwise CPU
            model = model.to(
                self.model_evaluation_config.device
            )

            # Final notebook implementation:
            # raw logits + CrossEntropyLoss
            criterion: Module = CrossEntropyLoss(
                reduction="sum"
            )

            # Switch model to evaluation mode
            model.eval()

            logging.info(
                "Exited the configuration method of Model evaluation class"
            )

            return test_dataloader, model, criterion

        except Exception as e:
            raise XRayException(e, sys)

    def test_net(self) -> float:
        """
        Evaluate the trained CNN on the test dataset.

        Calculates:
        - Average test loss
        - Accuracy
        - Precision
        - Recall
        - F1-score
        - Confusion Matrix
        - Classification Report
        """

        logging.info(
            "Entered the test_net method of Model evaluation class"
        )

        try:
            test_dataloader, model, criterion = (
                self.configuration()
            )

            test_loss: float = 0.0

            # Store true labels and model predictions
            all_labels = []
            all_predictions = []

            # Disable gradient calculation during evaluation
            with torch.no_grad():

                for images, labels in test_dataloader:

                    images = images.to(
                        self.model_evaluation_config.device
                    )

                    labels = labels.to(
                        self.model_evaluation_config.device
                    )

                    # Forward pass
                    # Model returns RAW LOGITS
                    outputs = model(images)

                    # Calculate test loss
                    test_loss += criterion(
                        outputs,
                        labels
                    ).item()

                    # Select class having the highest logit
                    predictions = torch.argmax(
                        outputs,
                        dim=1
                    )

                    # Move predictions/labels to CPU and
                    # store them for metric calculation
                    all_labels.extend(
                        labels.cpu().numpy()
                    )

                    all_predictions.extend(
                        predictions.cpu().numpy()
                    )

            total_samples = len(
                test_dataloader.dataset
            )

            # Average loss per test image
            average_test_loss = (
                test_loss / total_samples
            )

            # Overall classification accuracy
            accuracy = accuracy_score(
                all_labels,
                all_predictions
            )

            # Class 1 = PNEUMONIA
            precision = precision_score(
                all_labels,
                all_predictions,
                zero_division=0,
            )

            recall = recall_score(
                all_labels,
                all_predictions,
                zero_division=0,
            )

            f1 = f1_score(
                all_labels,
                all_predictions,
                zero_division=0,
            )

            # Confusion Matrix
            conf_matrix = confusion_matrix(
                all_labels,
                all_predictions
            )

            # Detailed report for NORMAL and PNEUMONIA
            class_report = classification_report(
                all_labels,
                all_predictions,
                target_names=[
                    "NORMAL",
                    "PNEUMONIA",
                ],
                zero_division=0,
            )

            # Convert accuracy to percentage for consistency
            accuracy_percentage = accuracy * 100

            print("\n" + "=" * 70)
            print("FINAL MODEL EVALUATION")
            print("=" * 70)

            print(
                f"Average Test Loss : "
                f"{average_test_loss:.4f}"
            )

            print(
                f"Accuracy          : "
                f"{accuracy_percentage:.2f}%"
            )

            print(
                f"Precision         : "
                f"{precision:.4f}"
            )

            print(
                f"Recall            : "
                f"{recall:.4f}"
            )

            print(
                f"F1 Score          : "
                f"{f1:.4f}"
            )

            print("\nConfusion Matrix:")
            print(conf_matrix)

            print("\nClassification Report:")
            print(class_report)

            # Save important information in logs
            logging.info(
                f"Average Test Loss: "
                f"{average_test_loss:.4f}"
            )

            logging.info(
                f"Accuracy: "
                f"{accuracy_percentage:.2f}%"
            )

            logging.info(
                f"Precision: {precision:.4f}"
            )

            logging.info(
                f"Recall: {recall:.4f}"
            )

            logging.info(
                f"F1 Score: {f1:.4f}"
            )

            logging.info(
                f"Confusion Matrix:\n{conf_matrix}"
            )

            logging.info(
                f"Classification Report:\n"
                f"{class_report}"
            )

            logging.info(
                "Exited the test_net method of "
                "Model evaluation class"
            )

            return accuracy_percentage

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_model_evaluation(
        self,
    ) -> ModelEvaluationArtifact:
        """
        Main method for the Model Evaluation stage.
        """

        logging.info(
            "Entered the initiate_model_evaluation method "
            "of Model evaluation class"
        )

        try:
            accuracy = self.test_net()

            # Keep the existing artifact structure so that
            # downstream components of the mentor pipeline
            # do not break.
            model_evaluation_artifact: ModelEvaluationArtifact = (
                ModelEvaluationArtifact(
                    model_accuracy=accuracy
                )
            )

            logging.info(
                "Exited the initiate_model_evaluation method "
                "of Model evaluation class"
            )

            return model_evaluation_artifact

        except Exception as e:
            raise XRayException(e, sys)