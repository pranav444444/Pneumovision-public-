import os
import sys

import bentoml
import joblib
import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from tqdm import tqdm

from xray.entity.artifacts_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from xray.entity.config_entity import ModelTrainerConfig
from xray.exception import XRayException
from xray.logger import logging
from xray.ml.model.arch import Net


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        # Configuration required for model training
        self.model_trainer_config: ModelTrainerConfig = model_trainer_config

        # Output received from the Data Transformation stage
        self.data_transformation_artifact: DataTransformationArtifact = (
            data_transformation_artifact
        )

        # Custom CNN architecture
        self.model: Module = Net()

        # Final notebook implementation:
        # Raw logits from the model + CrossEntropyLoss
        self.criterion = CrossEntropyLoss()

    def train(self, optimizer: Optimizer) -> None:
        """
        Train the CNN for one epoch.

        Flow:
        DataLoader batch
        -> Forward pass
        -> CrossEntropyLoss
        -> Backpropagation
        -> Optimizer update
        """
        logging.info("Entered the train method of Model trainer class")

        try:
            # Enables training behaviour for layers such as BatchNorm
            self.model.train()

            train_loader = (
                self.data_transformation_artifact.transformed_train_object
            )

            pbar = tqdm(train_loader)

            correct: int = 0
            processed: int = 0

            for batch_idx, (data, target) in enumerate(pbar):

                # Move images and labels to GPU if available,
                # otherwise CPU
                data = data.to(self.model_trainer_config.device)
                target = target.to(self.model_trainer_config.device)

                # Clear gradients calculated in the previous iteration
                optimizer.zero_grad()

                # Forward propagation
                # Net() now returns RAW LOGITS
                y_pred = self.model(data)

                # Final notebook implementation:
                # CrossEntropyLoss directly accepts raw logits
                loss = self.criterion(y_pred, target)

                # Backpropagation
                loss.backward()

                # Update CNN parameters
                optimizer.step()

                # Predicted class = index of highest logit
                pred = y_pred.argmax(dim=1, keepdim=True)

                correct += pred.eq(
                    target.view_as(pred)
                ).sum().item()

                processed += len(data)

                # Display current training progress
                pbar.set_description(
                    desc=(
                        f"Loss={loss.item():.4f} "
                        f"Batch_id={batch_idx} "
                        f"Accuracy={100 * correct / processed:0.2f}"
                    )
                )

            logging.info(
                f"Training Accuracy: {100 * correct / processed:.2f}%"
            )

            logging.info(
                "Exited the train method of Model trainer class"
            )

        except Exception as e:
            raise XRayException(e, sys)

    def test(self) -> None:
        """
        Evaluate the model after an epoch.

        This method currently keeps basic:
        - Test loss
        - Test accuracy

        Detailed Precision, Recall, F1-score and Confusion Matrix
        will belong to the separate Model Evaluation component.
        """
        logging.info("Entered the test method of Model trainer class")

        try:
            # Enables evaluation behaviour for BatchNorm etc.
            self.model.eval()

            test_loader = (
                self.data_transformation_artifact.transformed_test_object
            )

            test_loss: float = 0.0
            correct: int = 0

            # Sum losses across all samples so we can calculate
            # average loss over the complete test dataset
            test_criterion = CrossEntropyLoss(reduction="sum")

            # No gradients required during evaluation
            with torch.no_grad():

                for data, target in test_loader:

                    data = data.to(
                        self.model_trainer_config.device
                    )

                    target = target.to(
                        self.model_trainer_config.device
                    )

                    # Forward propagation
                    output = self.model(data)

                    # CrossEntropyLoss on RAW LOGITS
                    test_loss += test_criterion(
                        output, target
                    ).item()

                    # Select class with maximum logit
                    pred = output.argmax(
                        dim=1,
                        keepdim=True
                    )

                    correct += pred.eq(
                        target.view_as(pred)
                    ).sum().item()

            total_test_samples = len(test_loader.dataset)

            # Average loss per sample
            test_loss /= total_test_samples

            test_accuracy = (
                100.0 * correct / total_test_samples
            )

            print(
                "\nTest set: Average loss: {:.4f}, "
                "Accuracy: {}/{} ({:.2f}%)\n".format(
                    test_loss,
                    correct,
                    total_test_samples,
                    test_accuracy,
                )
            )

            logging.info(
                "Test set: Average loss: {:.4f}, "
                "Accuracy: {}/{} ({:.2f}%)".format(
                    test_loss,
                    correct,
                    total_test_samples,
                    test_accuracy,
                )
            )

            logging.info(
                "Exited the test method of Model trainer class"
            )

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Main method responsible for the complete model-training stage.
        """
        logging.info(
            "Entered the initiate_model_trainer method "
            "of Model trainer class"
        )

        try:
            # Move model to CUDA GPU when available,
            # otherwise it will run on CPU
            model: Module = self.model.to(
                self.model_trainer_config.device
            )

            # Final notebook optimizer:
            # Adam with learning rate = 0.001
            optimizer: Optimizer = Adam(
                model.parameters(),
                **self.model_trainer_config.optimizer_params
            )

            # Final notebook learning-rate scheduler:
            # StepLR(step_size=6, gamma=0.5)
            scheduler: _LRScheduler = StepLR(
                optimizer=optimizer,
                **self.model_trainer_config.scheduler_params
            )

            # Final notebook uses 10 epochs
            for epoch in range(
                1,
                self.model_trainer_config.epochs + 1
            ):
                print(
                    "\n"
                    + "=" * 80
                    + f"\nEPOCH {epoch}/{self.model_trainer_config.epochs}"
                    + "\n"
                    + "=" * 80
                )

                # Current learning rate
                current_lr = optimizer.param_groups[0]["lr"]

                print(
                    f"Learning Rate: {current_lr:.6f}"
                )

                logging.info(
                    f"Epoch {epoch} - "
                    f"Learning Rate: {current_lr:.6f}"
                )

                # Train for one epoch
                self.train(
                    optimizer=optimizer
                )

                # Evaluate after the epoch
                self.test()

                # Update learning rate according to StepLR
                scheduler.step()

            # Create model-training artifact folder
            os.makedirs(
                self.model_trainer_config.artifact_dir,
                exist_ok=True,
            )

            # IMPORTANT:
            # Keep saving the complete model for now because
            # the existing ModelEvaluation component expects
            # torch.load(trained_model_path) to return the model.
            torch.save(
                model,
                self.model_trainer_config.trained_model_path
            )

            logging.info(
                "Trained model saved at: "
                f"{self.model_trainer_config.trained_model_path}"
            )

            # Load deterministic preprocessing for inference/deployment.
            # We use the TEST transform because it does not contain
            # random augmentation such as rotation or horizontal flipping.
            inference_transform_obj = joblib.load(
                self.data_transformation_artifact.test_transform_file_path
            )

            # Save the trained PyTorch model into BentoML's model store
            # together with the preprocessing required during inference.
            bentoml.pytorch.save_model(
                name=self.model_trainer_config.trained_bentoml_model_name,
                                model=model,
                    custom_objects={
                        self.model_trainer_config.inference_transforms_key:
                            inference_transform_obj
                    },
                )

            # Output of the Model Training component
            model_trainer_artifact: ModelTrainerArtifact = (
                ModelTrainerArtifact(
                    trained_model_path=(
                        self.model_trainer_config.trained_model_path
                    )
                )
            )

            logging.info(
                "Exited the initiate_model_trainer method "
                "of Model trainer class"
            )

            return model_trainer_artifact

        except Exception as e:
            raise XRayException(e, sys)
