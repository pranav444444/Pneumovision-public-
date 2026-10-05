# PneumoVision: Chest X-Ray Pneumonia Detection

PneumoVision is an end-to-end deep learning project for **chest X-ray
image classification**, designed to distinguish between **Pneumonia**
and **Normal** chest X-ray images.

The project started as a deep learning experimentation workflow in
Google Colab and evolved into a modular, production-oriented ML
application with model evaluation, artifact management, BentoML model
packaging, FastAPI inference, Docker containerization, automated CI
validation with GitHub Actions, and deployment on Render.

> **Important:** PneumoVision is an educational AI/ML research and
> decision-support prototype. It is **not a medical diagnostic system**
> and must not be used as a substitute for evaluation by qualified
> healthcare professionals.

## Project Overview

Pneumonia is a respiratory infection that can affect the lungs and may
be visible in chest radiographs. Chest X-rays are commonly used as part
of the clinical assessment of suspected pneumonia.

This project explores how a convolutional neural network can learn
visual patterns from chest X-ray images and classify them into two
categories:

-   **NORMAL**
-   **PNEUMONIA**

The project focuses not only on model performance, but also on the
complete ML engineering lifecycle:

``` text
Dataset
   ↓
Data Preparation
   ↓
CNN Experimentation
   ↓
13 Experiments
   ↓
Final Model Selection
   ↓
Production-Oriented Python Pipeline
   ↓
Model Evaluation
   ↓
BentoML Model Packaging
   ↓
FastAPI Inference API
   ↓
Docker Containerization
   ↓
GitHub Actions CI
   ↓
Render Deployment
```

## Key Results

The model was improved through **13 experiments**, progressively
changing training configuration, evaluation, optimization, data loading,
and reproducibility settings.

  Metric            Initial Baseline   Final Model
  --------------- ------------------ -------------
  Accuracy                    79.27%    **95.82%**
  Precision                      ---    **96.97%**
  Recall                         ---    **97.31%**
  F1-Score                       ---    **97.14%**
  Training Time             27 min\*    **16 min**

The final model improved accuracy from **79.27% to 95.82%**,
representing a **16.55 percentage-point improvement** or approximately
**20.9% relative improvement**.

The final model achieved a **97.14% F1-score** and **97.31% recall**.

\*The training-time comparison refers to the 10-epoch configuration
before the later data-loading optimization.

## Experimentation Journey

The model was developed iteratively in **Google Colab using a T4 GPU**.
Rather than treating the first successful training run as the final
model, multiple experiments were performed to understand the effect of
training, evaluation, optimization, augmentation, and data-loading
decisions.

### Experiment Summary

  ------------------------------------------------------------------------------------------------
  Experiment   Change                   Accuracy    Precision       Recall           F1   Training
                                                                                              Time
  ------------ -------------------- ------------ ------------ ------------ ------------ ----------
  Exp 1        Baseline                   79.27%          ---          ---          ---     40 min

  Exp 2        Better evaluation          74.06%       75.58%       95.20%       84.27%     12 min

  Exp 3        Batch size 2 → 16          89.85%       90.17%       96.61%       93.28%     11 min

  Exp 4        CrossEntropyLoss +         93.77%       96.55%       94.85%       95.69%     11 min
               raw logits                                                               

  Exp 5        Epochs 4 → 10              95.73%       96.21%       98.01%       97.10%     27 min

  Exp 6        Adam optimizer             95.82%       96.64%       97.66%       97.15%     22 min

  Exp 7        Removed                    95.82%       96.11%       98.25%       97.17%   \~23--24
               ColorJitter + local                                                             min
               SSD data                                                                 

  Exp 8        Class weights              95.39%       97.06%       96.61%       96.83%        ---

  Exp 9        Dropout                    94.88%       96.71%       96.26%       96.48%        ---

  Exp 10       Optimized DataLoader       95.48%       97.40%       96.37%       96.88%   \~16 min
               workers                                                                  

  Exp 11       12 epochs                  95.65%       96.42%       97.66%       97.04%   \~21 min

  Exp 12       Reproducibility            95.48%       96.74%       97.08%       96.91%   \~18 min
               experiment                                                               

  Exp 13       Final configuration    **95.82%**   **96.97%**   **97.31%**   **97.14%**     **\~16
                                                                                             min**
  ------------------------------------------------------------------------------------------------

### Main Experimental Findings

The experiments showed that model performance was influenced by multiple
components of the training pipeline rather than by the CNN architecture
alone.

Key observations included:

-   Increasing batch size from **2 to 16** improved training efficiency
    and model performance.
-   Replacing the initial negative-log-likelihood setup with
    **CrossEntropyLoss using raw logits** produced a significant
    improvement.
-   Increasing training from **4 to 10 epochs** improved convergence.
-   Switching to the **Adam optimizer** improved optimization behavior.
-   Removing unnecessary **ColorJitter** augmentation improved the final
    configuration.
-   Moving data access to faster local SSD storage improved training
    efficiency.
-   Increasing DataLoader worker configuration reduced training time.
-   Class weighting and dropout were experimentally evaluated but were
    not retained in the final configuration.
-   A reproducibility experiment was performed to assess variation
    between runs.

## Final Model Performance

The final confusion matrix was:

``` text
                  Predicted
                NORMAL  PNEUMONIA
Actual NORMAL      291       26
Actual PNEUMONIA   23      832
```

Final classification metrics:

-   **Accuracy:** 95.82%
-   **Precision:** 96.97%
-   **Recall:** 97.31%
-   **F1-score:** 97.14%

The project also improved the NORMAL-class recall from approximately
**17% in the initial baseline to approximately 92%** in the improved
model while maintaining high pneumonia recall.

## Deep Learning Model

PneumoVision uses a **custom CNN architecture implemented in PyTorch**
rather than relying on a pretrained classification model.

The final architecture uses:

-   Convolution layers
-   ReLU activations
-   Batch Normalization
-   Max Pooling
-   Average Pooling
-   Final convolutional classification layer

The network produces **two raw logits**, corresponding to the two
classification classes.

### Final CNN Structure

``` text
Input Image
    ↓
Conv 3 → 8
    ↓
ReLU + BatchNorm + MaxPool
    ↓
Conv 8 → 20
    ↓
ReLU + BatchNorm + MaxPool
    ↓
Conv 20 → 10
    ↓
ReLU + BatchNorm + MaxPool
    ↓
Conv 10 → 20
    ↓
ReLU + BatchNorm
    ↓
Conv 20 → 32
    ↓
ReLU + BatchNorm
    ↓
Conv 32 → 10
    ↓
ReLU + BatchNorm
    ↓
Conv 10 → 10
    ↓
ReLU + BatchNorm
    ↓
Conv 10 → 14
    ↓
ReLU + BatchNorm
    ↓
Conv 14 → 16
    ↓
ReLU + BatchNorm
    ↓
Average Pooling
    ↓
Conv 16 → 2
    ↓
Raw Class Logits
```

## Custom CNN architecture

<img width="1405" height="787" alt="image" src="https://github.com/user-attachments/assets/92bb5f6d-bd87-4aba-86bc-68f3e41f19a9" />

## Web interface

<img width="1895" height="879" alt="image" src="https://github.com/user-attachments/assets/e1e68a79-44e9-4edc-ac91-8ae95bfcbc6f" />


<img width="1892" height="867" alt="image" src="https://github.com/user-attachments/assets/60d42f2f-fe4b-4752-b042-1ca3f26d312d" />


<img width="1894" height="819" alt="image" src="https://github.com/user-attachments/assets/1e162759-1455-4782-a2a8-394e77f5a037" />

   
<img width="1893" height="874" alt="image" src="https://github.com/user-attachments/assets/b3e59cc3-4101-4932-b5ed-7d63ab67e455" />


<img width="1900" height="873" alt="image" src="https://github.com/user-attachments/assets/34f190d5-eb9d-4c70-b750-de66781a455c" />


## Training Configuration

The final training configuration uses:

``` text
Framework        : PyTorch
Optimizer        : Adam
Learning Rate    : 0.001
Loss Function    : CrossEntropyLoss
Epochs           : 10
Batch Size       : 16
Scheduler        : StepLR
Step Size        : 6
Gamma            : 0.5
Input Size       : 224 × 224
```

A fixed seed of **42** was used as part of the reproducibility
experiments.

## Image Preprocessing

### Training Transform

The training pipeline uses:

-   Resize to 224
-   Center Crop to 224
-   Random Horizontal Flip
-   Random Rotation up to 10 degrees
-   Tensor conversion
-   ImageNet normalization

### Inference/Test Transform

Inference uses deterministic preprocessing:

-   Resize to 224
-   Center Crop to 224
-   Tensor conversion
-   ImageNet normalization

Training augmentation is intentionally not used during inference so that
the same input image receives deterministic preprocessing.

## Dataset

The dataset used for the project was shared by **Apollo Diagnostic
Center for research purposes**. The project was developed as a
proof-of-concept using the provided chest X-ray data.

The classification task contains two categories:

``` text
NORMAL
PNEUMONIA
```

The dataset was used for experimentation, model development, evaluation,
and inference testing.

## From Google Colab to Production-Oriented Code

The initial model-development stage was performed in **Google Colab**,
where a T4 GPU was used for repeated CNN experiments.

After the final model configuration was established, the workflow was
reorganized into a modular Python project structure instead of keeping
the complete process inside a single experimentation notebook.

The production-oriented workflow follows:

``` text
Configuration
     ↓
Component
     ↓
Artifact
     ↓
Next Component
```

The main pipeline is organized into:

``` text
Data Ingestion
      ↓
Data Transformation
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Packaging / Serving
```

This separation makes the workflow easier to maintain, debug, test, and
extend.

## Production Pipeline Architecture

``` mermaid
flowchart LR
    A[Data Ingestion] --> B[Data Transformation]
    B --> C[Model Training]
    C --> D[Model Evaluation]
    D --> E[BentoML Packaging]
    E --> F[FastAPI Inference]
    F --> G[Docker Container]
    G --> H[Render]
```

The production-oriented structure separates configuration, components,
artifacts, constants, logging, exception handling, model training,
evaluation, and serving responsibilities.

## Project Architecture

``` text
xray/
├── constant/
│   └── training_pipeline/
│
├── configuration/
│
├── components/
│   ├── data_ingestion.py
│   ├── data_transformation.py
│   ├── model_trainer.py
│   ├── model_evaluation.py
│   └── model_pusher.py
│
├── entity/
│   ├── config_entity.py
│   └── artifacts_entity.py
│
├── ml/
│   └── model/
│       └── model_service.py
│
├── pipeline/
│   └── training_pipeline.py
│
├── exception/
│
└── logger/
│
app.py
requirements.txt
Dockerfile
bentofile.yaml
.github/
└── workflows/
    └── ci-cd.yml
```

## Configuration and Artifact Flow

The modular pipeline uses configuration objects and artifact objects to
pass information between components.

``` text
DataIngestionConfig
        ↓
DataIngestion
        ↓
DataIngestionArtifact
        ↓
DataTransformationConfig
        ↓
DataTransformation
        ↓
DataTransformationArtifact
        ↓
ModelTrainerConfig
        ↓
ModelTrainer
        ↓
ModelTrainerArtifact
        ↓
ModelEvaluationConfig
        ↓
ModelEvaluation
        ↓
ModelEvaluationArtifact
        ↓
ModelPusher
        ↓
ModelPusherArtifact
```

This structure provides clear boundaries between pipeline stages and
makes debugging easier because each component produces a defined
artifact for the next stage.

## Model Packaging with BentoML

The trained PyTorch model is packaged using **BentoML** for model
management and serving integration.

The inference preprocessing transform is stored alongside the BentoML
model as a custom object so that the serving layer can apply the correct
deterministic preprocessing before inference.

The serving workflow is:

``` text
Trained PyTorch Model
        ↓
BentoML Model Store
        ↓
Inference Transform
        ↓
BentoML Service
        ↓
Inference Runner
        ↓
Prediction
```

The model returns two raw logits, and the inference service selects the
predicted class using the maximum logit.

## FastAPI Inference Layer

FastAPI provides the HTTP application layer for PneumoVision.

The inference flow is:

1.  Receive an X-ray image.
2.  Convert the image to RGB.
3.  Apply the stored inference transformation.
4.  Add the batch dimension.
5.  Send the tensor to the model runner.
6.  Obtain the model output.
7.  Select the predicted class.
8.  Return the prediction to the client.

The API also exposes the generated OpenAPI specification.

## Docker Containerization

The application is containerized using **Docker** to provide a
consistent runtime environment across development, testing, and
deployment.

Docker packages:

-   Application code
-   Python runtime
-   FastAPI
-   PyTorch
-   BentoML
-   Required dependencies
-   Model-serving configuration

The Dockerized application was tested locally before deployment to
Render.

### Build Docker Image

``` bash
docker build -t pneumovision .
```

### Run Docker Container

``` bash
docker run -d -p 8000:8000 pneumovision
```

### Local Application

Open:

``` text
http://localhost:8000/
```

The FastAPI OpenAPI specification can be accessed through:

``` text
http://localhost:8000/openapi.json
```

## Continuous Integration with GitHub Actions

PneumoVision uses **GitHub Actions** for automated continuous
integration.

The current workflow is defined in:

``` text
.github/workflows/ci-cd.yml
```

The workflow is triggered by pushes to `main` and pull requests
targeting `main`.

The current CI process is:

``` text
Git Push / Pull Request
        ↓
Checkout Repository
        ↓
Python Syntax Validation
        ↓
Docker Image Build
        ↓
Start Docker Container
        ↓
Wait for Application Startup
        ↓
Test Home Page
        ↓
Test FastAPI OpenAPI Endpoint
        ↓
Test Static CSS
        ↓
Cleanup Test Container
```

### CI Checks

The workflow validates:

-   Python syntax using `compileall`
-   Docker image build
-   Container startup
-   Home page availability
-   FastAPI OpenAPI endpoint availability
-   Static CSS availability

These are **smoke/integration checks**. They verify that the application
can build and start correctly inside the Docker environment.

The OpenAPI check does **not** evaluate model prediction accuracy. It
verifies that the FastAPI application starts successfully and exposes
its API schema.

## Continuous Deployment with Render

The Dockerized PneumoVision application is deployed using **Render**.

The deployment flow is:

``` text
GitHub Repository
        ↓
GitHub Actions CI
        ↓
CI Validation
        ↓
Render Automatic Deployment
        ↓
Dockerized PneumoVision Service
        ↓
Live Application
```

Render provides the cloud hosting layer for the Dockerized FastAPI
application.

The deployed service is publicly accessible through a Render URL.

## Health Check

The Render service uses the application root endpoint:

``` text
/
```

as the health-check endpoint.

This allows Render to periodically verify that the deployed application
is responding.

## Final CI/CD Architecture

The final deployment workflow separates validation from hosting:

``` mermaid
flowchart TD
    A[Developer] --> B[Git Push to main]
    B --> C[GitHub Actions]
    C --> D[Python Syntax Validation]
    D --> E[Docker Build]
    E --> F[Run Container]
    F --> G[Application Smoke Tests]
    G --> H{CI Passed?}
    H -->|Yes| I[Render Automatic Deployment]
    H -->|No| J[Deployment Not Promoted]
    I --> K[Dockerized PneumoVision]
    K --> L[Health Check /]
    L --> M[Live Application]
```

Responsibility is separated as follows:

-   **GitHub:** source control
-   **GitHub Actions:** continuous integration and automated validation
-   **Docker:** reproducible application runtime
-   **BentoML:** model packaging and serving integration
-   **Render:** cloud deployment and hosting

## Technology Stack

  Technology             Purpose
  ---------------------- ----------------------------------------------
  Python                 Core programming language
  PyTorch                Deep learning framework
  Custom CNN             Chest X-ray classification model
  Scikit-learn           Evaluation and ML utilities
  Google Colab           GPU-based experimentation and model training
  FastAPI                HTTP inference/application API
  BentoML                Model packaging and serving integration
  Docker                 Application containerization
  GitHub                 Source control
  GitHub Actions         Continuous integration automation
  Render                 Cloud deployment and hosting
  Pillow / Torchvision   Image processing and transformations

## Why These Technologies?

### PyTorch

Used to build, train, evaluate, and execute the custom CNN model.

### FastAPI

Provides the HTTP interface for image inference and application serving.

### BentoML

Provides model packaging and serving integration between the trained
PyTorch model and the production inference layer.

### Docker

Provides a consistent runtime environment and reduces differences
between local development, CI, and deployment.

### GitHub Actions

Automates validation whenever code changes are pushed or submitted
through a pull request.

### Render

Provides the deployment environment for the Dockerized application and
exposes the application as a publicly accessible web service.

## Logging and Exception Handling

The production-oriented project structure includes custom logging and
exception-handling components.

These components help:

-   Track pipeline execution
-   Identify failures
-   Debug component-level issues
-   Provide structured error information
-   Separate application logic from debugging infrastructure

## Important Engineering Lessons

### 1. Training and inference transformations must be different

Training uses random augmentation, while inference requires
deterministic preprocessing.

### 2. Loss function and model outputs must match

The final model uses:

``` text
Raw logits → CrossEntropyLoss
```

rather than applying an additional softmax before `CrossEntropyLoss`.

### 3. Model artifacts must include required inference information

The inference transformation is packaged alongside the model so that the
serving layer uses the correct preprocessing.

### 4. Container build success does not guarantee runtime success

The Docker image was tested by actually starting the container and
making HTTP requests.

### 5. Dependency versions matter

PyTorch, Torchvision, NumPy, Setuptools, BentoML, and related packages
were aligned to maintain compatibility with the model-serving
environment.

### 6. Local container testing should happen before deployment

The Dockerized application was validated locally before relying on the
cloud deployment environment.

### 7. Data-loading performance can affect training time

Moving data access to faster local storage and optimizing DataLoader
workers reduced training time significantly.

## Current Project Status

The project has progressed through the following stages:

``` text
✓ Dataset preparation
✓ CNN experimentation
✓ 13 documented experiments
✓ Final model selection
✓ Model evaluation
✓ Production-oriented Python pipeline
✓ Configuration and artifact architecture
✓ BentoML model packaging
✓ FastAPI inference
✓ Docker containerization
✓ Local Docker testing
✓ GitHub Actions CI
✓ Render deployment
✓ Render health check
✓ Automated deployment
```

The final deployed application uses the trained model checkpoint for
inference rather than retraining the model during every deployment.

Heavy model training and experimentation were performed in Google Colab
using GPU acceleration, while the production application focuses on
inference and serving.

## How to Run the Project Locally

### 1. Clone the Repository

Clone the repository to your local machine.

### 2. Create a Virtual Environment

``` bash
python -m venv .venv
```

On Windows:

``` powershell
.venv\Scripts\activate
```

### 3. Install Dependencies

``` bash
pip install -r requirements.txt
```

### 4. Run the Application

``` bash
python app.py
```

Open the application in your browser using the configured application
port.

### 5. Run with Docker

Build the image:

``` bash
docker build -t pneumovision .
```

Run the container:

``` bash
docker run -d -p 8000:8000 pneumovision
```

Then open:

``` text
http://localhost:8000/
```

## Repository Structure

``` text
PneumoVision/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── xray/
│   ├── constant/
│   ├── configuration/
│   ├── components/
│   ├── entity/
│   ├── exception/
│   ├── logger/
│   ├── ml/
│   │   └── model/
│   └── pipeline/
│
├── app.py
├── Dockerfile
├── bentofile.yaml
├── requirements.txt
│
├── progress.md
├── concepts.md
├── mistakes.md
├── interview.md
└── resume.md
```

## Documentation

The repository also maintains project-specific documentation:

-   **`progress.md`** --- Experiment history, metrics, configurations,
    and observations.
-   **`concepts.md`** --- Deep learning, computer vision, deployment,
    and production concepts used in the project.
-   **`mistakes.md`** --- Development and deployment issues encountered
    and their resolutions.
-   **`interview.md`** --- Project-specific interview questions and
    explanations.
-   **`resume.md`** --- Resume-oriented project achievements and
    technical highlights.

## Future Improvements

Potential future improvements include:

-   Introducing a dedicated validation set for more rigorous model
    selection.
-   Expanding the automated test suite beyond application smoke tests.
-   Adding automated model-performance evaluation to CI.
-   Adding dedicated experiment tracking.
-   Monitoring production inference behavior and application
    performance.
-   Improving model interpretability using techniques such as Grad-CAM.
-   Evaluating the model on additional and more diverse chest X-ray
    datasets.
-   Comparing the custom CNN against established pretrained
    architectures.

## Limitations

PneumoVision is a research and educational proof of concept.

Important limitations include:

-   The dataset size and composition limit generalization claims.
-   The model has not been clinically validated.
-   The model should not be interpreted as providing a medical
    diagnosis.
-   Performance on the experimental dataset does not guarantee
    equivalent performance on unseen clinical populations.
-   The current CI pipeline validates application and container health
    rather than clinical model correctness.
-   Model selection was performed within the experimental train/test
    workflow, so a dedicated validation split would provide a stronger
    evaluation methodology in a future iteration.


## Live-Link

   https://pneumovision-yvhn.onrender.com/

## Final architecture

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/c49a8df6-de13-414b-9502-9b8a3b21a96e" />


## Conclusion

PneumoVision demonstrates the progression of a deep learning project
from **experimentation to a production-oriented application**.

The project began with a custom CNN trained and optimized through **13
experiments in Google Colab**, resulting in a final accuracy of
**95.82%** and an **F1-score of 97.14%**.

The finalized model was then moved beyond notebook experimentation into
a modular Python architecture containing configuration, components,
artifacts, training, evaluation, and model-serving layers.

The trained PyTorch model was packaged with BentoML, exposed through
FastAPI, containerized with Docker, validated automatically using GitHub
Actions, and deployed as a Docker-based service on Render.

The resulting system demonstrates practical experience across:

-   Deep learning
-   Computer vision
-   Model experimentation
-   ML pipeline architecture
-   Model packaging
-   API development
-   Docker containerization
-   CI/CD
-   Cloud deployment
-   Debugging and dependency management
-   Production-oriented ML engineering

PneumoVision therefore represents a complete journey from **model
experimentation to a deployable ML application**, while clearly
distinguishing research results from production and clinical claims.


# Copyright © 2026 Pranav Patel. All Rights Reserved.
This repository is publicly available for portfolio, educational, and evaluation purposes. Unauthorized copying, redistribution, modification, or presentation of this project or its source code as one's own work is prohibited.
