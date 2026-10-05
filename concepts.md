# Deep Learning Concepts

---

## CNN

Convolutional Neural Network

Used for image classification because it learns features automatically.

---

## Epoch

One complete pass through the training dataset.

---

## Batch

Number of images processed together.








## Production ML Pipeline Architecture

### Core Flow

Constants â†’ Config â†’ Component â†’ Artifact â†’ Next Component

### Constants
- Store fixed/common project values in one central location.
- Examples: bucket name, batch size, epochs, image size, learning rate.
- Avoid hard-coding the same values across multiple files.

### Config Entity
- Contains the settings/paths required by a particular pipeline component.
- Config tells a component what it needs before execution.
- Example: `DataIngestionConfig` stores the S3 bucket name and local train/test paths.

### Component
- Performs the actual work of a pipeline stage.
- Examples: DataIngestion, DataTransformation, ModelTrainer.
- DataIngestion uses its config and S3 helper to download the dataset.

### Artifact Entity
- Represents the output produced by a pipeline component.
- The artifact is passed to the next pipeline stage.
- Example: `DataIngestionArtifact` contains `train_file_path` and `test_file_path`.

### Config vs Artifact
- Config = input/settings required by a component.
- Artifact = output/result produced by a component.

### S3Operation
- Reusable helper class for AWS S3 operations.
- `sync_folder_from_s3()` â†’ S3 to local.
- `sync_folder_to_s3()` â†’ local to S3.
- Current implementation uses AWS CLI commands through `os.system()`.

### Data Ingestion Flow
AWS S3 â†’ S3Operation â†’ DataIngestion â†’ Local Artifact Directory â†’ DataIngestionArtifact â†’ DataTransformation

### Timestamped Artifacts
- Each pipeline run uses a timestamped artifact directory.
- Helps keep outputs from different pipeline runs separate.


### AWS CLI in the Project

AWS CLI allows AWS services to be accessed from the terminal.

Important commands:

- `aws --version` â†’ checks whether AWS CLI is installed.
- `aws s3 ls` â†’ lists S3 buckets.
- `aws s3 ls s3://bucket/path/` â†’ lists objects/folders at an S3 location.
- `aws s3 sync SOURCE DESTINATION` â†’ synchronizes files between two locations.

Examples:

Local â†’ S3:
`aws s3 sync Data s3://bucket/data/`

S3 â†’ Local:
`aws s3 sync s3://bucket/data/ artifacts/.../data/`

In this project, `s3_operations.py` constructs these CLI commands and runs them using `os.system()`.

### Artifact Directory

`ARTIFACT_DIR = "artifacts"` only defines the folder name; it does not create the directory.

The actual `artifacts/` directory appears when a pipeline component runs and writes/downloads its output there.

For Data Ingestion:

AWS S3 â†’ `aws s3 sync` â†’ `artifacts/<timestamp>/data_ingestion/data/`






## Data Transformation Pipeline

Data Transformation converts ingested image folders into model-ready PyTorch DataLoaders.

### Flow

DataIngestionArtifact
â†’ ImageFolder
â†’ Image Transformations
â†’ DataLoader
â†’ DataTransformationArtifact
â†’ Model Trainer

### Training Transform

Resize(224)
â†’ CenterCrop(224)
â†’ RandomHorizontalFlip
â†’ RandomRotation(10)
â†’ ToTensor
â†’ Normalize

Random augmentation is applied only to training data.

### Test Transform

Resize(224)
â†’ CenterCrop(224)
â†’ ToTensor
â†’ Normalize

Testing uses deterministic preprocessing to keep evaluation consistent.

### ImageFolder

`ImageFolder` loads images from class-based folders and assigns labels automatically.

For this project:

NORMAL â†’ 0
PNEUMONIA â†’ 1

### Dataset vs DataLoader

Dataset:
- Stores samples and labels.
- Applies transformations to individual samples.

DataLoader:
- Creates batches.
- Controls shuffling.
- Supports parallel data loading.

Final DataLoader settings:

Train:
- batch_size = 16
- shuffle = True
- pin_memory = True
- num_workers = 2
- persistent_workers = True

Test:
- batch_size = 16
- shuffle = False
- pin_memory = True
- num_workers = 2
- persistent_workers = True

### num_workers

`num_workers=2` allows multiple CPU worker processes to load and transform images in parallel.

It does not speed up GPU computation directly; it reduces GPU waiting time for the next batch.

### persistent_workers

Keeps DataLoader worker processes alive across epochs, reducing repeated worker startup overhead.

### Transform Persistence

The train and test preprocessing pipelines are saved using Joblib as `.pkl` files so the preprocessing configuration can be reused later.





## Model Training Pipeline

ModelTrainer receives the train/test DataLoaders from DataTransformationArtifact and trains the custom CNN.

### Final Training Configuration

- Model: Custom CNN (`Net`)
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Learning rate: 0.001
- Epochs: 10
- Scheduler: StepLR(step_size=6, gamma=0.5)

### One Training Iteration

optimizer.zero_grad()
â†’ model(images)
â†’ CrossEntropyLoss(logits, labels)
â†’ loss.backward()
â†’ optimizer.step()

### Model Output

The CNN returns two raw logits:

[logit_NORMAL, logit_PNEUMONIA]

No Sigmoid or Softmax is applied before CrossEntropyLoss.

### Training vs Evaluation

`model.train()`:
- Enables training behavior.
- Used during parameter updates.

`model.eval()`:
- Enables evaluation behavior.
- Used during testing/inference.

`torch.no_grad()` disables unnecessary gradient calculations during evaluation.

### Optimizer vs Scheduler

Optimizer:
- Updates model parameters.
- `optimizer.step()` occurs after backpropagation for each training batch.

Scheduler:
- Changes the optimizer learning rate.
- `scheduler.step()` occurs once per epoch in this pipeline.

### Model Saving

The trained model is saved to the timestamped model-training artifact directory.

The deterministic test transformation is stored with the BentoML model for inference because random training augmentation should not be applied to user-uploaded X-rays.






## Model Evaluation

Model Evaluation measures the final trained CNN performance on the test dataset without updating model parameters.

### Evaluation Flow

Test DataLoader
+ Trained Model
â†’ model.eval()
â†’ torch.no_grad()
â†’ Forward Pass
â†’ Raw Logits
â†’ CrossEntropyLoss
â†’ Argmax Predictions
â†’ Collect Actual & Predicted Labels
â†’ Calculate Metrics

### Metrics

Accuracy:
Overall percentage of correct predictions.

Precision:
Of all images predicted as PNEUMONIA, how many were actually PNEUMONIA.

Recall:
Of all actual PNEUMONIA images, how many were correctly detected.

F1-score:
Harmonic balance between precision and recall.

Confusion Matrix:
Shows TN, FP, FN, and TP counts.

Classification Report:
Provides precision, recall, and F1-score separately for NORMAL and PNEUMONIA.

### Evaluation Mode

`model.eval()` switches layers such as BatchNorm to inference behavior.

`torch.no_grad()` disables gradient calculations because no model parameters are updated.

### Positive Class

ImageFolder mapping:

NORMAL â†’ 0
PNEUMONIA â†’ 1

Therefore binary precision, recall, and F1 treat PNEUMONIA as the positive class.

### Test Loss

CrossEntropyLoss uses `reduction="sum"` during evaluation.

Losses for all test images are accumulated and divided by the total number of samples to obtain average test loss.

- must remember
Evaluation does NOT train.

No optimizer.
No backward().
No optimizer.step().

model.eval()
+
torch.no_grad()

CNN gives raw logits.
argmax gives predicted class.

0 = NORMAL
1 = PNEUMONIA

Accuracy = overall correctness.
Precision = reliability of pneumonia predictions.
Recall = how many real pneumonia cases were found.
F1 = balance of precision and recall.

Confusion Matrix shows the actual error types.

ModelEvaluationArtifact currently returns accuracy,
while other metrics are printed and logged.






## Model Pusher, BentoML, Docker, and AWS ECR

### What is the Model Pusher?

The Model Pusher is the deployment-oriented stage of the ML pipeline.

Its job is to take the trained model service, package it into a Docker image, and push that image to AWS ECR.

Overall flow:

Trained PyTorch Model
â†’ BentoML Model Store
â†’ BentoML Service
â†’ Bento Build
â†’ Docker Image
â†’ AWS ECR Login
â†’ Docker Push
â†’ Image stored in ECR

---

### Why is Model Pusher needed?

A `.pth` or `.pt` model file alone is not a complete deployable application.

Deployment also needs:

- model architecture
- preprocessing
- inference logic
- API/service
- Python dependencies
- runtime environment

The Model Pusher packages all these together.

---

### BentoML Model vs BentoML Service

BentoML Model:
The trained PyTorch model stored in BentoML.

For this project:

`xray_model`

BentoML Service:
The API/service that uses the model for inference.

For this project:

`xray_service`

The service accepts a chest X-ray and returns:

`NORMAL`

or

`PNEUMONIA`

---

### BentoML Model Store

The trained CNN is registered in BentoML using:

`bentoml.pytorch.save_model()`

The deterministic inference transform is stored along with the model.

This allows the deployed service to use the same preprocessing required by the trained CNN.

---

### Inference Transform

Training transforms contain random augmentation such as:

- RandomHorizontalFlip
- RandomRotation

These should not be used during prediction.

Therefore inference uses:

Resize(224)
â†’ CenterCrop(224)
â†’ ToTensor()
â†’ Normalize()

This makes prediction preprocessing deterministic.

---

### `model_service.py`

`model_service.py` defines the inference API.

Flow:

Uploaded X-ray
â†’ convert to RGB
â†’ inference transform
â†’ tensor `[3, 224, 224]`
â†’ `unsqueeze(0)`
â†’ tensor `[1, 3, 224, 224]`
â†’ BentoML Runner
â†’ CNN
â†’ raw logits
â†’ `argmax`
â†’ class index
â†’ NORMAL / PNEUMONIA

---

### Why `unsqueeze(0)`?

A single transformed image has shape:

`[3, 224, 224]`

The CNN expects batched input:

`[Batch, Channels, Height, Width]`

Therefore:

`unsqueeze(0)`

changes:

`[3, 224, 224]`

to:

`[1, 3, 224, 224]`

---

### Prediction Label Mapping

The class mapping is:

`0 â†’ NORMAL`

`1 â†’ PNEUMONIA`

Therefore:

`PREDICTION_LABEL = {0: CLASS_LABEL_1, 1: CLASS_LABEL_2}`

Integer keys are required because `argmax()` returns an integer class index.

---

### What is `bentofile.yaml`?

`bentofile.yaml` tells BentoML how to package the service.

It defines:

- service entry point
- included source files
- labels
- Python dependencies
- PyTorch package source

Service entry point:

`xray.ml.model.model_service:svc`

This means BentoML imports the `svc` object from:

`xray/ml/model/model_service.py`

---

### Bento

A Bento is the packaged ML service artifact created by:

`bentoml build`

It contains:

- service code
- dependencies
- API definition
- model references
- project files

A Bento is then used to build the Docker image.

---

### Why direct Docker build is used

The original implementation used:

`bentoml containerize`

In the current Windows + BentoML 1.0.x environment, that command failed internally with a `NotImplementedError`.

Therefore the working approach uses:

Bento
â†’ generated Dockerfile
â†’ custom Dockerfile
â†’ `docker build`
â†’ Docker image

---

### Why the Model Store must be copied into Docker

The local BentoML model exists in the local BentoML model store.

Inside the container, BentoML expects models under:

`/home/bentoml/models`

Because direct Docker building bypasses some automatic BentoML packaging behavior, the registered model must be explicitly copied into this location.

The custom Dockerfile therefore:

1. Copies the Bento application.
2. Creates `/home/bentoml/models`.
3. Copies `xray_model` into the container model store.

---

### Docker

Docker packages the complete inference environment.

The Docker image contains:

- Python runtime
- BentoML
- PyTorch
- torchvision
- Pillow
- trained CNN
- model architecture
- inference transform
- prediction API
- required dependencies

This makes the service portable and reproducible.

---

### Local Docker Testing

Before pushing the image to AWS, the container is tested locally.

Command:

`docker run --rm -p 3000:3000 <image>`

This maps:

Host port 3000
â†’ Container port 3000

The service becomes available at:

`http://localhost:3000`

The `/predict` endpoint was tested using a real chest X-ray and returned the expected prediction.

---

### Amazon ECR

ECR stands for:

Amazon Elastic Container Registry

It stores Docker/container images.

For this project:

Repository:

`xray_bento_image`

Region:

`ap-south-1`

---

### S3 vs ECR

S3:
Used for objects such as datasets and files.

ECR:
Used for Docker container images.

In this project:

S3
â†’ chest X-ray dataset

ECR
â†’ deployable Docker image

---

### ECR Authentication

Docker must authenticate before pushing to a private ECR repository.

Flow:

AWS CLI
â†’ `aws ecr get-login-password`
â†’ temporary authentication password
â†’ `docker login`
â†’ Docker authenticated with ECR

---

### Docker Push

The Docker image is uploaded using:

`docker push`

After successful upload, ECR stores:

- image tag
- digest
- image size
- architecture
- operating system
- push timestamp

---

### Tag vs Digest

Tag:

`latest`

A human-readable, mutable reference.

Digest:

`sha256:...`

An immutable cryptographic identifier of the exact image.

Therefore:

Tag
â†’ convenient name

Digest
â†’ exact version

---

### Image Index and Manifests

Modern Docker BuildKit may push an OCI Image Index.

The index can reference:

- actual Linux/AMD64 runtime image
- additional provenance/metadata manifests

This does not mean multiple completely separate application images were accidentally uploaded.

---

### Local Docker Size vs ECR Size

The local Docker image can appear much larger than the ECR image.

Reason:

Local Docker:
expanded/uncompressed layers

ECR:
compressed stored layers

Therefore a multi-GB local image can appear significantly smaller in ECR.

---

### Final Working Deployment Dependencies

The working deployment environment uses:

- BentoML 1.0.25
- PyTorch 1.13.1
- torchvision 0.14.1
- NumPy 1.26.4
- setuptools 80.9.0

These versions were pinned to avoid compatibility issues discovered during deployment.

---

### Important Deployment Issues Encountered

1. BentoML 1.0.10 had a Starlette compatibility issue.

2. `bentoml containerize` failed with an internal `NotImplementedError`.

3. NumPy 2.x caused compatibility warnings with older PyTorch components.

4. Newer setuptools removed/changed `pkg_resources` behavior required by the old stack.

5. Newer PyTorch changed `torch.load()` behavior to safer `weights_only=True`, which prevented BentoML from loading the full serialized custom `Net` object.

6. The BentoML model store had to be explicitly copied into the Docker image when using direct Docker build.

---

### Final Model Pusher Flow

ModelTrainer
â†’ save `xray_model` in BentoML
â†’ `model_service.py`
â†’ `bentoml build`
â†’ Bento created
â†’ model copied into Docker build context
â†’ custom Dockerfile prepared
â†’ `docker build`
â†’ local container test
â†’ `/predict` test
â†’ AWS ECR login
â†’ `docker push`
â†’ image stored in ECR

---

### Must Remember

PyTorch
â†’ runs the CNN

BentoML
â†’ exposes the CNN as an inference service

Docker
â†’ packages the service into a portable container

AWS ECR
â†’ stores the Docker image

Inference must use deterministic preprocessing.

A Docker image should be tested locally before pushing to ECR.

A successful image build does not guarantee a successful runtime.

Always test:
build
â†’ run
â†’ predict
â†’ then push.
