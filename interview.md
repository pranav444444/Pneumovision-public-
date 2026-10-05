
---

# Section 1 : Project Overview

## Q1. What is the objective of this project?

### Answer

The objective of this project is to build a Deep Learning model that classifies Chest X-ray images into two classes:

- NORMAL
- PNEUMONIA

The model is implemented using PyTorch and will later be integrated with a FastAPI application, containerized using Docker, and deployed on AWS.

---

## Q2. What type of Machine Learning problem is this?

### Answer

This is a Supervised Learning problem because every X-ray image has a known label (NORMAL or PNEUMONIA).

---

## Q3. What Computer Vision task does this project solve?

### Answer

It is an Image Classification problem.

Input:
Chest X-ray

Output:
NORMAL or PNEUMONIA

It is not Object Detection or Image Segmentation.

---

## Q4. Why is CNN suitable for this project?

### Answer

CNNs automatically learn visual features such as edges, textures, and infection patterns from images without manual feature engineering, making them ideal for image classification tasks.

---

# Section 2 : Dataset

## Q5. What is the folder structure required by ImageFolder?

### Answer

```
train/
    NORMAL/
    PNEUMONIA/

test/
    NORMAL/
    PNEUMONIA/
```

ImageFolder automatically assigns labels based on folder names.

---

## Q6. How does ImageFolder assign labels?

### Answer

ImageFolder scans the folder names alphabetically and assigns integer labels automatically.

Example:

NORMAL â†’ 0

PNEUMONIA â†’ 1

---

## Q7. Is your dataset balanced?

### Answer

No.

Training Dataset

- NORMAL : 1266
- PNEUMONIA : 3418

Testing Dataset

- NORMAL : 317
- PNEUMONIA : 855

The dataset contains significantly more Pneumonia images than Normal images.

---

## Q8. Why is dataset imbalance important?

### Answer

An imbalanced dataset may bias the model toward predicting the majority class.

Therefore, evaluation should not rely only on Accuracy.

Metrics like Precision, Recall, F1-score, and Confusion Matrix become more important.

---

# Section 3 : Image Exploration

## Q9. Why did you inspect random images before training?

### Answer

To verify

- image quality
- image dimensions
- dataset correctness
- class labels

before building the model.

---

## Q10. Why do images need resizing?

### Answer

Deep Learning models require all images to have a fixed input size.

Since the dataset contains images of different resolutions, resizing standardizes the input dimensions.

---

# Section 4 : Data Augmentation

## Q11. Why do we use Data Augmentation?

### Answer

Data augmentation artificially increases training data diversity by applying random transformations such as rotation and flipping.

It helps reduce overfitting and improves model generalization.

---

## Q12. Why is Data Augmentation applied only to the training dataset?

### Answer

Training images are augmented to improve generalization.

Test images must remain unchanged so that evaluation reflects real-world performance.

---

## Q13. Why do we convert images to tensors?

### Answer

PyTorch models operate on tensors rather than NumPy arrays or image objects.

ToTensor converts images into PyTorch tensors.

---

## Q14. Why is Normalize used?

### Answer

Normalization scales pixel values using predefined mean and standard deviation.

It stabilizes training and usually helps the optimizer converge faster.

---

## Q15. Why are ImageNet mean and standard deviation values used?

### Answer

The values

Mean

[0.485, 0.456, 0.406]

Standard Deviation

[0.229, 0.224, 0.225]

are standard ImageNet statistics.

They are commonly used in PyTorch computer vision pipelines and are especially important when using pretrained models.

---

## Q16. Why is RandomRotation useful?

### Answer

Patients are not always positioned perfectly during X-ray acquisition.

Small rotations help the model become robust to slight orientation changes.

---

## Q17. Why might ColorJitter not always be ideal for Chest X-rays?

### Answer

Chest X-rays are grayscale medical images.

Brightness adjustments may still be reasonable, but changes to saturation and hue are generally less meaningful than for natural RGB images.

This is something that can be evaluated experimentally.

---

## Q18. Why might RandomHorizontalFlip be questionable in medical imaging?

### Answer

Horizontal flipping changes left-right anatomical orientation.

For some medical imaging tasks this may not be clinically appropriate.

Whether it helps should be verified experimentally.

---

# Section 5 : DataLoader

## Q19. What is ImageFolder?

### Answer

ImageFolder is a built-in PyTorch dataset class that automatically reads images from folders and assigns labels based on directory names.

---

## Q20. What is DataLoader?

### Answer

DataLoader loads data in batches, supports shuffling, and efficiently feeds images to the neural network during training.

---

## Q21. Why do we use batches instead of loading the entire dataset?

### Answer

Loading the entire dataset at once consumes large amounts of memory.

Batch processing is memory efficient and enables scalable training.

---

## Q22. Why is shuffle=True used during training?

### Answer

Shuffling prevents the model from learning the order of the dataset and improves generalization.

---

## Q23. Why is shuffle=False used during testing?

### Answer

Testing should be deterministic and reproducible.

Random shuffling is unnecessary during evaluation.

---

## Q24. What does pin_memory=True do?

### Answer

It speeds up transferring batches from CPU memory to GPU memory when CUDA is available.

---

## Q25. Why did the tensor shape become (2, 3, 224, 224) when Chest X-rays are grayscale?

### Answer

Although Chest X-rays are visually grayscale, PIL loads them as RGB by default.

The three channels contain identical grayscale information, resulting in a tensor with shape

(batch_size, 3, 224, 224).

This keeps the data compatible with standard computer vision pipelines.

---

## Q26. What do the tensor dimensions (2, 3, 224, 224) represent?

### Answer

2 â†’ Batch Size

3 â†’ Number of Channels

224 â†’ Height

224 â†’ Width

---

## Q27. Why was batch_size set to 2?

### Answer

Batch size is a hyperparameter.

A small value like 2 reduces memory usage and works on systems with limited resources.

Later, larger batch sizes can be experimented with to improve training speed.








# Started Production .py file questions:


## Production Pipeline & Data Ingestion Interview Questions

### Q1. Explain the architecture of your ML training pipeline.

The pipeline follows:

Constants â†’ Config â†’ Component â†’ Artifact â†’ Next Component.

Constants store shared project values. Config entities organize the settings required by each component. Components perform the actual operations, and artifact entities represent the outputs passed to downstream stages.

---

### Q2. What is the difference between a Config Entity and an Artifact Entity?

A Config Entity contains the inputs, parameters, and paths required by a component before it runs.

An Artifact Entity represents the output generated by that component after execution.

For example, `DataIngestionConfig` contains the S3 bucket and local dataset paths, while `DataIngestionArtifact` contains the final train and test paths passed to Data Transformation.

---

### Q3. Explain your Data Ingestion pipeline.

The chest X-ray dataset is stored in AWS S3 with separate train and test directories.

`DataIngestionConfig` provides the S3 bucket information and local artifact paths. `DataIngestion` calls a reusable `S3Operation` utility, which uses AWS CLI sync to download the dataset into a timestamped artifact directory.

After downloading, Data Ingestion returns a `DataIngestionArtifact` containing the train and test dataset paths for the Data Transformation stage.

---

### Q4. Why did you use AWS S3 in this project?

AWS S3 provides centralized cloud storage for the dataset, so the training pipeline does not depend on a dataset being manually present on a particular machine.

The ingestion component can retrieve the required data from the configured S3 location whenever the pipeline is executed.

---

### Q5. Why did you create a separate `s3_operations.py` file?

It separates cloud-storage operations from the Data Ingestion logic.

`DataIngestion` decides when data should be retrieved, while `S3Operation` handles how files are synchronized with AWS S3. This also allows the same S3 functionality to be reused by other pipeline components.

---

### Q6. Why don't you directly use constants inside every component?

Centralizing values avoids repeated hard-coded values and makes configuration easier to maintain.

Config objects also group only the settings required by a particular component, keeping component code cleaner.

---

### Q7. What is an artifact in an ML pipeline?

An artifact is an output produced by a pipeline stage and used by later stages.

For example:
- Data Ingestion â†’ train/test paths
- Data Transformation â†’ DataLoaders and transform files
- Model Training â†’ trained model path
- Model Evaluation â†’ evaluation results

---

### Q8. Why are timestamps used in the artifact directory?

Timestamps create a separate folder for each pipeline execution, preventing outputs from different runs from overwriting each other and making runs easier to track.

---

### Q9. Why does Data Ingestion return file paths instead of the images themselves?

The next component only needs to know where the train and test datasets are located.

Passing paths keeps the components independent and avoids unnecessarily moving the complete dataset between pipeline objects.

---

### Q10. What is the role of `initiate_data_ingestion()`?

It is the main method controlling the Data Ingestion stage.

It calls the method that retrieves data from S3, creates the `DataIngestionArtifact`, and returns that artifact to the next pipeline stage.

---

### Q11. Why was AWS CLI required for this project?

The current `S3Operation` implementation constructs commands such as `aws s3 sync` and executes them using `os.system()`.

Therefore, AWS CLI must be installed and authenticated on the machine running this implementation.

---

### Q12. Is there any limitation in the current S3 implementation?

Yes. The current implementation uses `os.system()` to execute AWS CLI commands. A command can fail and return a non-zero status without necessarily raising a Python exception.

A more robust production implementation could use `subprocess.run(..., check=True)` or an AWS SDK such as Boto3 with explicit error handling.








## Data Transformation Interview Questions

### Q1. What is the purpose of the Data Transformation component?

The Data Transformation component converts the ingested chest X-ray folders into model-ready PyTorch datasets and DataLoaders. It applies preprocessing and augmentation, creates batches, and passes the resulting DataLoaders to the model-training stage.

---

### Q2. Why do training and testing datasets use different transformations?

Training data uses random augmentation such as horizontal flipping and small rotations to improve model robustness.

Testing data uses only deterministic preprocessing such as resize, crop, tensor conversion, and normalization so evaluation remains consistent across runs.

---

### Q3. Why did you remove ColorJitter from the final pipeline?

ColorJitter was part of the original preprocessing pipeline, but it was removed during experimentation because it was not suitable for the final chest X-ray preprocessing strategy and did not improve the final model performance.

---

### Q4. What does `ImageFolder` do in PyTorch?

`ImageFolder` loads images from a folder structure where each class has its own subdirectory. It automatically assigns class indices based on the class folder names.

For this project:

NORMAL â†’ 0
PNEUMONIA â†’ 1

---

### Q5. What is the difference between Dataset and DataLoader?

A Dataset represents the individual samples, labels, and transformations.

A DataLoader controls how those samples are delivered to the model, including batch size, shuffling, and parallel data loading.

---

### Q6. Why did you use a batch size of 16?

Batch size 16 was selected during experimentation and provided better training behavior than the original batch size of 2 while remaining computationally manageable.

---

### Q7. Why is `shuffle=True` used for training but `shuffle=False` for testing?

Training data is shuffled so the model sees samples in different orders across epochs and does not depend on a fixed dataset ordering.

Testing does not require shuffling because no model parameters are updated and keeping a fixed order makes evaluation more consistent.

---

### Q8. What is the role of `num_workers` in a PyTorch DataLoader?

`num_workers` controls how many CPU worker processes are used to load and preprocess data in parallel.

Using multiple workers can reduce the time the GPU waits for the next batch.

---

### Q9. Does increasing `num_workers` make GPU computation faster?

No. It does not increase GPU computation speed directly.

It improves the data-loading pipeline by preparing batches in parallel, which can reduce GPU idle time.

---

### Q10. What does `persistent_workers=True` do?

It keeps DataLoader worker processes alive between epochs instead of recreating them every epoch, reducing worker startup overhead.

---

### Q11. What does `pin_memory=True` do?

Pinned memory can improve the efficiency of transferring batches from CPU memory to a CUDA GPU during training.

---

### Q12. Why do you save the train and test transformation objects?

The transformation objects are saved so the preprocessing logic can be preserved and reused later, especially during inference or deployment.

---

### Q13. What does `DataTransformationArtifact` contain?

It contains the transformed training DataLoader, transformed testing DataLoader, and the file paths of the saved train and test transformation objects.

These outputs are then passed to the downstream model-training stage.

---

### Q14. How does Data Transformation connect to the previous pipeline stage?

Data Ingestion returns a `DataIngestionArtifact` containing the train and test folder paths.

Data Transformation receives those paths, loads the images through `ImageFolder`, applies preprocessing, creates DataLoaders, and produces a `DataTransformationArtifact`.

---

### Q15. How can this project work on a laptop without a GPU?

The code dynamically selects CUDA when available and otherwise falls back to CPU.

The production pipeline can therefore run locally for tasks such as preprocessing, inference, and lightweight testing, while computationally expensive CNN training can be performed using a cloud GPU environment such as Google Colab.

---

### Q16. Will the deployed application retrain the CNN every time a user uploads an X-ray?

No.

Training and inference are separate stages. The CNN is trained beforehand and its best weights are saved in a `.pth` checkpoint. During deployment, the application loads the saved model and preprocessing pipeline and only performs inference on the uploaded X-ray.

---

### Q17. Why keep a Model Trainer component if a trained `.pth` model already exists?

The Model Trainer component makes the project reproducible and provides a complete end-to-end training pipeline.

The saved `.pth` model is used for inference and deployment, while the training component allows the model to be retrained later if the dataset, architecture, or training configuration changes.




## Model Training Interview Questions

### Q1. Explain the Model Training stage in your project.

The Model Training component receives the transformed training and testing DataLoaders from the Data Transformation stage.

It trains a custom CNN using CrossEntropyLoss and Adam optimization for 10 epochs, monitors test performance after each epoch, applies a StepLR learning-rate scheduler, saves the trained model, and returns the trained model path through `ModelTrainerArtifact`.

---

### Q2. What is the input to the ModelTrainer component?

The main input is `DataTransformationArtifact`, which contains:

- Training DataLoader
- Testing DataLoader
- Saved train transform path
- Saved test transform path

The trainer also receives `ModelTrainerConfig`, which provides epochs, optimizer parameters, scheduler parameters, device, and model-save paths.

---

### Q3. What does your CNN output?

The CNN outputs two raw logits:

`[logit_NORMAL, logit_PNEUMONIA]`

The model does not apply Sigmoid, Softmax, or LogSoftmax before returning the output.

---

### Q4. Why does the model return raw logits?

The final model uses `CrossEntropyLoss`, which expects raw logits.

CrossEntropyLoss internally performs the required log-softmax operation in a numerically stable way, so applying Softmax or Sigmoid before the loss is unnecessary.

---

### Q5. Why did you use CrossEntropyLoss?

This is a single-label, two-class classification problem where each image belongs to either NORMAL or PNEUMONIA.

The model outputs two logits and the targets are integer class indices, so CrossEntropyLoss is suitable for this setup.

---

### Q6. What was the original loss function and why did you change it?

The original implementation used NLLLoss with a probability/log-probability based output.

During experimentation, the pipeline was changed to raw logits with CrossEntropyLoss, which produced better optimization behavior and became part of the final model configuration.

---

### Q7. Which optimizer did you use in the final model?

The final model uses Adam with:

`learning_rate = 0.001`

The original implementation used SGD, but experiments showed that Adam provided better final performance for this project.

---

### Q8. What does the optimizer do?

The optimizer updates the CNN's trainable parameters using the gradients calculated during backpropagation.

In this project, Adam uses those gradients to update the convolutional and BatchNorm-related trainable parameters after each training batch.

---

### Q9. Explain one complete training iteration.

For every training batch:

1. Move images and labels to the selected device.
2. Clear old gradients using `optimizer.zero_grad()`.
3. Perform a forward pass through the CNN.
4. Calculate CrossEntropyLoss.
5. Run `loss.backward()` to calculate gradients.
6. Run `optimizer.step()` to update model parameters.
7. Calculate running training accuracy.

The core sequence is:

`zero_grad â†’ forward pass â†’ loss â†’ backward â†’ optimizer.step`

---

### Q10. Why is `optimizer.zero_grad()` required?

PyTorch accumulates gradients by default.

Therefore, the gradients from the previous batch must normally be cleared before calculating gradients for the current batch.

---

### Q11. What does `loss.backward()` do?

`loss.backward()` performs backpropagation.

It calculates the gradient of the loss with respect to every trainable model parameter so the optimizer knows how the parameters should be updated.

---

### Q12. What does `optimizer.step()` do?

`optimizer.step()` updates the model parameters using the gradients generated by `loss.backward()`.

In this pipeline, it is called once for every training batch.

---

### Q13. Why was the extra `optimizer.step()` after each epoch removed?

The original production code called `optimizer.step()` inside every training batch and then called it again after the epoch.

The second call was unnecessary because no new forward pass or backpropagation had occurred between those calls.

The final pipeline updates model parameters only after valid gradient calculations.

---

### Q14. What is the difference between `model.train()` and `model.eval()`?

`model.train()` puts the network into training mode.

`model.eval()` puts it into evaluation mode.

This is especially important because the CNN contains Batch Normalization layers, whose behavior differs between training and evaluation.

---

### Q15. Why do you use `torch.no_grad()` during testing?

During testing, model parameters are not updated.

`torch.no_grad()` disables gradient tracking, which reduces unnecessary memory usage and computation during evaluation.

---

### Q16. How is the predicted class obtained from raw logits?

The predicted class is selected using:

`argmax(dim=1)`

For example:

`[2.8, 0.4] â†’ class 0 â†’ NORMAL`

`[-0.2, 2.1] â†’ class 1 â†’ PNEUMONIA`

Softmax is not required when only the class with the largest score is needed.

---

### Q17. Why did you use StepLR?

StepLR gradually reduces the learning rate during training.

The final configuration is:

- `step_size = 6`
- `gamma = 0.5`

This means the learning rate is multiplied by 0.5 after the configured step interval, allowing smaller parameter updates during later training.

---

### Q18. What is the difference between `optimizer.step()` and `scheduler.step()`?

`optimizer.step()` updates the model weights using gradients.

`scheduler.step()` changes the optimizer's learning rate.

In this pipeline:

- `optimizer.step()` runs for every training batch.
- `scheduler.step()` runs once per epoch.

---

### Q19. How many epochs do you train for and why?

The final configuration uses 10 epochs.

Experiments with additional epochs did not provide meaningful improvement, so 10 epochs were retained as the final configuration.

---

### Q20. Why do you test the model after every epoch?

Testing after each epoch helps monitor how the model generalizes as training progresses.

In the ModelTrainer component, only test loss and accuracy are monitored after each epoch, while detailed metrics are calculated later by the separate Model Evaluation component.

---

### Q21. Why separate Model Training and Model Evaluation?

Model Training is responsible for learning model parameters and monitoring basic training behavior.

Model Evaluation is responsible for the detailed final assessment using metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

This keeps the pipeline responsibilities separate and easier to maintain.

---

### Q22. How does your code choose between CPU and GPU?

The project uses:

`torch.device("cuda" if torch.cuda.is_available() else "cpu")`

If CUDA is available, training can run on the GPU. Otherwise, the same code falls back to CPU.

---

### Q23. How can you build this project locally if your laptop has no GPU?

The production pipeline and source code are maintained locally in VS Code.

Lightweight tasks such as preprocessing, inference, debugging, and code validation can run on CPU.

Heavy CNN training can be performed using a cloud GPU environment such as Google Colab.

---

### Q24. Do you retrain the model whenever the deployed application receives an X-ray?

No.

Training and inference are separate.

The model is trained beforehand and saved. During deployment, the saved trained model is loaded and only a forward pass is performed on the uploaded X-ray.

---

### Q25. Why is the test transformation stored with the BentoML model instead of the training transformation?

The training transformation contains random augmentation such as horizontal flipping and rotation.

These random transformations should not be applied during inference.

The test transformation is deterministic and contains only the preprocessing required for prediction:

`Resize â†’ CenterCrop â†’ ToTensor â†’ Normalize`

Therefore, it is more appropriate for deployment.

---

### Q26. What does BentoML do in the Model Trainer stage?

After training, the PyTorch model is saved into the BentoML model store.

The deterministic inference transformation is also attached as a custom object so that later serving code can use compatible preprocessing.

---

### Q27. What is `ModelTrainerArtifact`?

`ModelTrainerArtifact` is the structured output of the Model Training component.

It currently contains the trained model path, which is passed to the Model Evaluation stage.

---

### Q28. How does Model Training connect to the complete pipeline?

The flow is:

`Data Ingestion â†’ Data Transformation â†’ Model Training â†’ Model Evaluation â†’ Model Pushing / Deployment`

Data Transformation provides the DataLoaders, Model Training produces the trained model, and Model Evaluation measures the final predictive performance.






## Model Evaluation Interview Questions

### Q1. What is the purpose of the Model Evaluation component?

The Model Evaluation component measures how well the trained CNN performs on unseen test data.

It loads the saved trained model, evaluates it using the test DataLoader, calculates test loss and classification metrics, and returns the final model accuracy through `ModelEvaluationArtifact`.

---

### Q2. What are the inputs to the Model Evaluation component?

It receives:

- `DataTransformationArtifact` â†’ provides the test DataLoader
- `ModelTrainerArtifact` â†’ provides the trained model path
- `ModelEvaluationConfig` â†’ provides the computation device

---

### Q3. Why is Model Evaluation kept separate from Model Training?

Model Training is responsible for learning and updating the model parameters.

Model Evaluation is responsible for measuring the final performance of the already-trained model using detailed metrics.

This separation keeps the pipeline modular and each component focused on one responsibility.

---

### Q4. Why do you call `model.eval()` before evaluation?

`model.eval()` switches the CNN into evaluation mode.

This is important because layers such as Batch Normalization behave differently during training and evaluation.

---

### Q5. Why do you use `torch.no_grad()` during evaluation?

No parameter updates are performed during evaluation.

`torch.no_grad()` disables gradient tracking, which reduces unnecessary memory usage and computation.

---

### Q6. Why is there no optimizer in the Model Evaluation component?

An optimizer is only required when model parameters are being updated.

During evaluation, the trained weights remain fixed, so there is no need for backpropagation or an optimizer.

---

### Q7. How is the trained model loaded?

The Model Trainer saves the complete PyTorch model to the configured model path.

The Model Evaluation component loads it using `torch.load()` and then moves it to the configured CPU or GPU device.

---

### Q8. What does the model output during evaluation?

The custom CNN outputs two raw logits:

`[logit_NORMAL, logit_PNEUMONIA]`

These are raw class scores rather than probabilities.

---

### Q9. How do you convert logits into a predicted class?

I use:

`torch.argmax(outputs, dim=1)`

It returns the index of the class with the highest logit.

For this project:

- `0 â†’ NORMAL`
- `1 â†’ PNEUMONIA`

---

### Q10. Why don't you apply Softmax before `argmax()`?

Softmax changes the logits into probabilities but preserves their ordering.

Therefore, the class with the maximum logit is also the class with the maximum Softmax probability, so Softmax is unnecessary when only the predicted class is required.

---

### Q11. Why do you use CrossEntropyLoss during evaluation?

CrossEntropyLoss measures how well the raw model logits match the actual class labels.

Although no backpropagation is performed during evaluation, the loss is still useful for measuring prediction quality.

---

### Q12. Why is `reduction="sum"` used for the evaluation loss?

The loss of every test image is accumulated across all batches.

After evaluating the complete dataset, the summed loss is divided by the total number of test samples to obtain the average loss per image.

This also handles the final smaller batch correctly.

---

### Q13. Why are actual labels and predictions stored for the entire test dataset?

Metrics such as precision, recall, F1-score, confusion matrix, and classification report require the complete set of true and predicted labels.

Therefore, predictions from every test batch are collected before calculating the final metrics.

---

### Q14. Why do you call `.cpu().numpy()` on predictions and labels?

Predictions and labels may be stored as PyTorch tensors on the GPU.

Scikit-learn metrics operate on CPU-based arrays, so the tensors are moved to the CPU and converted to NumPy arrays before metric calculation.

---

### Q15. What metrics do you use to evaluate the pneumonia model?

The final evaluation includes:

- Average Test Loss
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

These provide a more complete assessment than accuracy alone.

---

### Q16. Why is accuracy alone not enough for this project?

The dataset has an unequal class distribution, and a model may achieve reasonable accuracy while still performing poorly on one class.

For pneumonia classification, it is important to inspect false positives, false negatives, class-specific recall, precision, and F1-score as well.

---

### Q17. What does precision mean in your project?

Pneumonia precision answers:

"Of all X-rays predicted as PNEUMONIA, how many were actually PNEUMONIA?"

Formula:

`Precision = TP / (TP + FP)`

Higher precision means fewer NORMAL X-rays are incorrectly classified as pneumonia.

---

### Q18. What does recall mean in your project?

Pneumonia recall answers:

"Of all actual PNEUMONIA X-rays, how many were correctly detected?"

Formula:

`Recall = TP / (TP + FN)`

A false negative means a pneumonia X-ray was incorrectly classified as NORMAL, so recall is particularly important to monitor.

---

### Q19. What is the F1-score?

F1-score combines precision and recall into a single metric.

Formula:

`F1 = 2 Ã— (Precision Ã— Recall) / (Precision + Recall)`

It is useful when both false positives and false negatives matter.

---

### Q20. What does the confusion matrix show?

For this project, with:

- `0 = NORMAL`
- `1 = PNEUMONIA`

the matrix is interpreted as:

`[[TN, FP],`
` [FN, TP]]`

It shows the exact counts of:

- correctly classified NORMAL images
- NORMAL images wrongly predicted as PNEUMONIA
- PNEUMONIA images wrongly predicted as NORMAL
- correctly detected PNEUMONIA images

---

### Q21. Why is the confusion matrix important for a medical imaging classifier?

It shows the actual error types rather than only an overall percentage.

For example, it lets us separately examine false negatives, where pneumonia cases are missed, and false positives, where normal X-rays are incorrectly flagged as pneumonia.

---

### Q22. What does the classification report provide?

The classification report gives:

- Precision
- Recall
- F1-score
- Support

for each class separately.

This allows me to compare model performance on NORMAL and PNEUMONIA instead of relying only on overall metrics.

---

### Q23. Which class is treated as the positive class?

`ImageFolder` maps:

- `NORMAL â†’ 0`
- `PNEUMONIA â†’ 1`

Therefore, the default binary precision, recall, and F1 calculations treat PNEUMONIA as the positive class.

---

### Q24. What is `zero_division=0` used for?

If a metric becomes undefined, for example when the model predicts no samples of a particular class, division by zero may occur.

`zero_division=0` safely reports the metric as 0 instead of producing an undefined result or warning.

---

### Q25. How does Model Evaluation connect to the previous pipeline stages?

The flow is:

`Data Transformation`
â†’ provides test DataLoader

`Model Training`
â†’ provides trained model path

`Model Evaluation`
â†’ loads both and calculates final evaluation metrics

`ModelEvaluationArtifact`
â†’ passes the evaluation result to the next stage

---

### Q26. What does `ModelEvaluationArtifact` currently contain?

It currently contains:

`model_accuracy`

The detailed precision, recall, F1-score, confusion matrix, and classification report are calculated, printed, and logged, while the artifact interface is kept simple to maintain compatibility with downstream pipeline components.

---

### Q27. What were the final evaluation results of your selected model?

The final selected experiment achieved approximately:

- Accuracy: 95.82%
- Precision: 96.97%
- Recall: 97.31%
- F1-score: 97.14%

These values belong to the finalized notebook experiment and should only be quoted when discussing that specific trained model.

---

### Q28. Why is recall particularly important in pneumonia detection?

A false negative means an actual pneumonia case is classified as NORMAL.

Because such missed cases can be more concerning than simply reporting overall accuracy, pneumonia recall is an important class-specific metric to monitor.

---

### Q29. What is the difference between testing inside ModelTrainer and the separate ModelEvaluation stage?

`ModelTrainer.test()` is used for quick monitoring during training and mainly reports test loss and accuracy after each epoch.

`ModelEvaluation` performs the final detailed assessment using accuracy, precision, recall, F1-score, confusion matrix, and classification report.

---

### Q30. Explain the complete Model Evaluation flow.

The trained model is loaded from `ModelTrainerArtifact`, and the test DataLoader is received from `DataTransformationArtifact`.

The model is moved to the configured device, switched to evaluation mode, and evaluated inside `torch.no_grad()`.

For every test batch, the model produces raw logits, CrossEntropyLoss is calculated, `argmax` generates predicted classes, and the true and predicted labels are collected.

Finally, accuracy, precision, recall, F1-score, confusion matrix, and classification report are calculated and logged.








## Model Pusher / BentoML / Docker / AWS ECR Interview Questions

### Q1. What is the purpose of the Model Pusher component?

The Model Pusher packages the trained inference service into a Docker image and pushes that image to AWS ECR so that it can later be deployed on a cloud platform.

---

### Q2. Why can't you deploy only the `.pth` model file?

A model file alone does not contain the complete deployment environment.

Deployment also requires:

- model architecture
- preprocessing
- inference logic
- API/service
- Python dependencies
- runtime environment

Docker packages all of these together.

---

### Q3. What role does BentoML play in your project?

BentoML acts as the model-serving layer.

It stores the trained PyTorch model, exposes it through an inference service, creates the Bento package, and provides the `/predict` API.

---

### Q4. What is the difference between a BentoML model and a BentoML service?

The BentoML model is the stored trained CNN.

In my project:

`xray_model`

The BentoML service defines how users interact with that model.

In my project:

`xray_service`

---

### Q5. What is a Bento?

A Bento is a packaged machine learning service artifact created by BentoML.

It includes the service code, dependencies, API definition, configuration, and references to the required model.

---

### Q6. What is the role of `bentofile.yaml`?

`bentofile.yaml` defines how BentoML should package the service.

It specifies:

- service entry point
- files to include
- Python dependencies
- metadata labels
- package sources

---

### Q7. What is the service entry point in your project?

The service entry point is:

`xray.ml.model.model_service:svc`

This means BentoML loads the `svc` object from:

`xray/ml/model/model_service.py`

---

### Q8. What does `model_service.py` do?

It defines the inference API.

It receives a chest X-ray image, preprocesses it, runs the trained CNN through the BentoML runner, gets raw logits, selects the class using `argmax()`, and returns NORMAL or PNEUMONIA.

---

### Q9. Why do you use the test/inference transform instead of the training transform during deployment?

The training transform contains random augmentation such as rotation and horizontal flipping.

Inference should be deterministic, so I use the test/inference transform without random augmentation.

---

### Q10. Why do you convert the uploaded image to RGB?

The CNN was trained to accept three-channel images.

Therefore uploaded chest X-rays are converted to RGB before preprocessing.

---

### Q11. Why is `unsqueeze(0)` used before inference?

A transformed image has shape:

`[3, 224, 224]`

The CNN expects batched input:

`[batch, channels, height, width]`

So `unsqueeze(0)` changes it to:

`[1, 3, 224, 224]`

---

### Q12. What does the BentoML runner do?

The runner executes model inference.

The service sends the preprocessed tensor to the runner, which runs the PyTorch CNN and returns the output logits.

---

### Q13. How do you obtain the predicted class?

The CNN outputs two logits:

`[logit_NORMAL, logit_PNEUMONIA]`

I use:

`torch.argmax(..., dim=1)`

to select the class with the highest logit.

---

### Q14. Why was the prediction-label mapping corrected?

The original mapping mixed string and integer keys:

`"0"` and `1`

However, `argmax()` returns an integer class index.

Therefore the correct mapping is:

`{0: NORMAL, 1: PNEUMONIA}`

---

### Q15. Why was NumPy removed from `model_service.py`?

The inference transform already returns a PyTorch tensor because it contains `ToTensor()`.

The original implementation unnecessarily converted:

Tensor
â†’ NumPy
â†’ Tensor

So I removed that redundant conversion.

---

### Q16. What is Docker and why did you use it?

Docker packages the complete model-serving application and its dependencies into a portable image.

This helps ensure consistent behavior across different environments.

---

### Q17. What does your Docker image contain?

It contains:

- Python
- BentoML
- PyTorch
- torchvision
- Pillow
- CNN architecture
- trained model
- inference transform
- API code
- required project files

---

### Q18. Why did you test the Docker image locally before pushing it to AWS?

A Docker image can build successfully but still fail at runtime.

Local testing verifies that:

- the container starts
- the model loads
- the API is available
- the prediction endpoint works

before pushing the image to ECR.

---

### Q19. How did you test the container locally?

I ran the container with port mapping:

`3000:3000`

Then I opened:

`http://localhost:3000`

and tested the `/predict` endpoint with a real chest X-ray image.

The service returned the expected class.

---

### Q20. What is Amazon ECR?

Amazon ECR stands for Elastic Container Registry.

It is an AWS service used to store and manage Docker/container images.

---

### Q21. What is the difference between S3 and ECR in your project?

S3 stores object data such as the chest X-ray dataset.

ECR stores the deployable Docker image.

---

### Q22. Why does Docker need to authenticate with ECR?

ECR is a private registry.

Docker must authenticate before it can push or pull private images.

AWS CLI provides a temporary ECR login password, which is passed to `docker login`.

---

### Q23. What does `docker push` do?

`docker push` uploads the locally built Docker image layers to the AWS ECR repository.

---

### Q24. What does the `latest` tag mean?

`latest` is a mutable, human-readable image tag.

It can later point to a newer image.

---

### Q25. What is an image digest?

A digest is an immutable SHA-256 identifier for the exact image content.

A tag can change, but the digest identifies one exact version.

---

### Q26. Why did ECR show an Image Index and multiple manifests?

Docker BuildKit can push an OCI Image Index containing the runtime image plus additional metadata or provenance manifests.

The actual runtime image is Linux/AMD64.

---

### Q27. Why was the ECR image size smaller than the local Docker image?

Docker Desktop may show expanded/uncompressed layers.

ECR stores compressed layers.

Therefore the remote stored size can be much smaller.

---

### Q28. Why did you use `subprocess.run()` instead of `os.system()`?

`subprocess.run(..., check=True)` raises an exception when a command fails.

This provides more reliable error handling for BentoML, Docker, and AWS commands.

---

### Q29. Why did you stop using `bentoml containerize()`?

In my Windows environment with the older BentoML 1.0.x stack, `bentoml containerize` failed internally with a `NotImplementedError`.

I therefore used BentoML's generated Dockerfile and built the image directly with Docker.

---

### Q30. Why did you have to manually include the BentoML model store in the Docker image?

Because direct Docker building bypassed some BentoML automatic model-packaging behavior.

Inside the container, BentoML expected the model under:

`/home/bentoml/models`

So the registered `xray_model` was explicitly copied there.

---

### Q31. What dependency compatibility issues did you encounter?

The working environment was pinned to:

- BentoML 1.0.25
- PyTorch 1.13.1
- torchvision 0.14.1
- NumPy 1.26.4
- setuptools 80.9.0

This avoided compatibility problems discovered during deployment.

---

### Q32. What PyTorch issue occurred during deployment?

A newer PyTorch version changed `torch.load()` so that safer weights-only loading became the default.

BentoML was loading a full serialized custom `Net` object, so the custom class was rejected.

Using a compatible older PyTorch version allowed the model to load correctly.

---

### Q33. What NumPy issue occurred?

NumPy 2.x produced compatibility problems with components compiled against the NumPy 1.x API.

I pinned NumPy to version 1.26.4.

---

### Q34. What setuptools issue occurred?

The old BentoML dependency stack expected `pkg_resources`.

A newer setuptools version caused the container to fail because `pkg_resources` was unavailable.

I pinned setuptools to a compatible version.

---

### Q35. What did you learn from debugging this deployment?

I learned that ML deployment involves much more than saving a model.

It also requires understanding:

- serialization
- dependency compatibility
- API serving
- Docker
- model-store paths
- container runtime behavior
- AWS authentication
- cloud registries

I also learned to validate each stage independently instead of debugging the entire system at once.

---

### Q36. Explain the complete Model Pusher flow.

The trained PyTorch model is registered in BentoML along with the deterministic inference transform.

`model_service.py` defines the `/predict` API.

`bentoml build` creates the Bento.

The registered model is added to the Docker build context and copied into BentoML's model store inside the container.

Docker builds the final image.

The image is tested locally with a real X-ray.

Docker authenticates with AWS ECR using AWS CLI credentials.

Finally, the tested Docker image is pushed to the private ECR repository.
