import os
import shutil
import subprocess
import sys

from xray.entity.artifacts_entity import ModelPusherArtifact
from xray.entity.config_entity import ModelPusherConfig
from xray.exception import XRayException
from xray.logger import logging


class ModelPusher:
    def __init__(
        self,
        model_pusher_config: ModelPusherConfig
    ):
        self.model_pusher_config = model_pusher_config

    def build_and_push_bento_image(self):
        logging.info(
            "Entered build_and_push_bento_image method "
            "of ModelPusher class"
        )

        try:
            # -------------------------------------------------
            # Complete AWS ECR image URI
            # -------------------------------------------------
            ecr_image_uri = (
                f"{self.model_pusher_config.ecr_registry}/"
                f"{self.model_pusher_config.bentoml_ecr_image}:latest"
            )

            # -------------------------------------------------
            # STEP 1: Build Bento
            # -------------------------------------------------
            logging.info(
                "Building Bento from bentofile.yaml"
            )

            build_result = subprocess.run(
                [
                    "bentoml",
                    "build",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            logging.info(
                "Built Bento from bentofile.yaml"
            )

            # -------------------------------------------------
            # STEP 2: Get latest Bento path
            # -------------------------------------------------
            bentoml_home = os.path.expanduser(
                "~/bentoml"
            )

            bento_service_dir = os.path.join(
                bentoml_home,
                "bentos",
                self.model_pusher_config.bentoml_service_name,
            )

            latest_file = os.path.join(
                bento_service_dir,
                "latest"
            )

            with open(
                latest_file,
                "r",
                encoding="utf-8"
            ) as file:
                latest_bento_tag = file.read().strip()

            bento_dir = os.path.join(
                bento_service_dir,
                latest_bento_tag,
            )

            logging.info(
                f"Latest Bento directory: {bento_dir}"
            )

            # -------------------------------------------------
            # STEP 3: Copy BentoML model into Bento build context
            # -------------------------------------------------
            local_model_dir = os.path.join(
                bentoml_home,
                "models",
                self.model_pusher_config.bentoml_model_name,
            )

            bento_models_dir = os.path.join(
                bento_dir,
                "models",
            )

            os.makedirs(
                bento_models_dir,
                exist_ok=True,
            )

            destination_model_dir = os.path.join(
                bento_models_dir,
                self.model_pusher_config.bentoml_model_name,
            )

            if os.path.exists(destination_model_dir):
                shutil.rmtree(
                    destination_model_dir
                )

            shutil.copytree(
                local_model_dir,
                destination_model_dir,
            )

            logging.info(
                "Copied BentoML model into Bento build context"
            )

            # -------------------------------------------------
            # STEP 4: Create custom Dockerfile
            # -------------------------------------------------
            generated_dockerfile = os.path.join(
                bento_dir,
                "env",
                "docker",
                "Dockerfile",
            )

            custom_dockerfile = os.path.join(
                bento_dir,
                "Dockerfile.custom",
            )

            with open(
                generated_dockerfile,
                "r",
                encoding="utf-8",
            ) as file:
                dockerfile_content = file.read()

            original_copy_line = (
                "COPY --chown=bentoml:bentoml . ./"
            )

            model_copy_block = """
# Create BentoML model store
RUN mkdir -p /home/bentoml/models

# Copy registered model into BentoML model store
COPY --chown=bentoml:bentoml ./models /home/bentoml/models
"""

            dockerfile_content = (
                dockerfile_content.replace(
                    original_copy_line,
                    original_copy_line
                    + "\n"
                    + model_copy_block,
                    1,
                )
            )

            with open(
                custom_dockerfile,
                "w",
                encoding="utf-8",
            ) as file:
                file.write(
                    dockerfile_content
                )

            logging.info(
                "Created custom Dockerfile"
            )

            # -------------------------------------------------
            # STEP 5: Build Docker image directly
            # -------------------------------------------------
            logging.info(
                "Building Docker image from Bento"
            )

            subprocess.run(
                [
                    "docker",
                    "build",
                    "-t",
                    ecr_image_uri,
                    "-f",
                    custom_dockerfile,
                    bento_dir,
                ],
                check=True,
            )

            logging.info(
                f"Created Docker image: {ecr_image_uri}"
            )

            # -------------------------------------------------
            # STEP 6: Get AWS ECR login password
            # -------------------------------------------------
            logging.info(
                "Getting AWS ECR login password"
            )

            login_password = subprocess.run(
                [
                    "aws",
                    "ecr",
                    "get-login-password",
                    "--region",
                    self.model_pusher_config.aws_region,
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout

            # -------------------------------------------------
            # STEP 7: Login Docker into AWS ECR
            # -------------------------------------------------
            logging.info(
                "Logging Docker into AWS ECR"
            )

            subprocess.run(
                [
                    "docker",
                    "login",
                    "--username",
                    "AWS",
                    "--password-stdin",
                    self.model_pusher_config.ecr_registry,
                ],
                input=login_password,
                check=True,
                text=True,
            )

            logging.info(
                "Logged Docker into AWS ECR"
            )

            # -------------------------------------------------
            # STEP 8: Push Docker image to AWS ECR
            # -------------------------------------------------
            logging.info(
                "Pushing Docker image to AWS ECR"
            )

            subprocess.run(
                [
                    "docker",
                    "push",
                    ecr_image_uri,
                ],
                check=True,
            )

            logging.info(
                f"Pushed Docker image to AWS ECR: "
                f"{ecr_image_uri}"
            )

            logging.info(
                "Exited build_and_push_bento_image method "
                "of ModelPusher class"
            )

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_model_pusher(
        self,
    ) -> ModelPusherArtifact:
        """
        Initiates the Model Pusher stage.

        Steps:
        1. Build Bento
        2. Prepare Bento model store
        3. Create custom Dockerfile
        4. Build Docker image
        5. Authenticate Docker with AWS ECR
        6. Push Docker image to AWS ECR
        7. Return ModelPusherArtifact
        """

        logging.info(
            "Entered initiate_model_pusher method "
            "of ModelPusher class"
        )

        try:
            self.build_and_push_bento_image()

            model_pusher_artifact = ModelPusherArtifact(
                bentoml_model_name=(
                    self.model_pusher_config.bentoml_model_name
                ),
                bentoml_service_name=(
                    self.model_pusher_config.bentoml_service_name
                ),
            )

            logging.info(
                "Exited initiate_model_pusher method "
                "of ModelPusher class"
            )

            return model_pusher_artifact

        except Exception as e:
            raise XRayException(e, sys)