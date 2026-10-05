# Lung Disease Detection Project



# Experiment Summary

|   Exp | Change                                     | Test Accuracy | Precision | Recall | F1 | Training Time | Result |
| ----: | ------------------------------------------ | ------------: | --------: | ------: | ------: | ------------: | ------------------------------------- |
| 1 | Baseline | 79.27 | - | - | - | 40 min | Baseline |
| 2 | Better Evaluation | 74.06% | 75.58% | 95.20% | 84.27% | 12 min | Better evaluation |
| 3 | Batch Size 2 → 16 | 89.85 | 90.17 | 96.61 | 93.28 | 11 min | Better recall |
| 4 | CrossEntropyLoss + Raw Logits | 93.77 | 96.55 | 94.85 | 95.69 | 11 min | Better optimization |
| 5 | Epochs 4 → 10 | 95.73 | 96.21 | **98.01** | 97.10 | 27 min | Best SGD model ⭐ |
| 6 | Optimizer: SGD → Adam | **95.82** | **96.64** | 97.66 | 97.15 | 22 min | Better optimizer ⭐ |
| 7 | Remove ColorJitter + Local SSD Dataset | **95.82** | 96.11 | **98.25** | **97.17** | ≈23–24 min | Highest Recall & F1 ⭐ |
| 8 | Class Weighted Loss | 95.39 | **97.06** | 96.61 | 96.83 | ≈23–24 min | Not adopted |
| 9 | CNN + Dropout Regularization | 94.88 | 96.71 | 96.26 | 96.48 | ≈20 min | Not adopted |
| 10 | DataLoader Optimization (`num_workers`) | 95.48 | 97.40 | 96.37 | 96.88 | **≈16 min** | Training efficiency improved 🚀 |
| 11 | Epochs 10 → 12 | **95.65** | 96.42 | 97.66 | 97.04 | ≈21 min | Extra epochs not beneficial |
| 12 | Reproducibility Check | 95.48 | 96.74 | 97.08 | 96.91 | ≈18 min | Stable performance |
| **13** | **Final Optimized Model (Re-trained)** | **95.82** | **96.97** | **97.31** | **97.14** | **≈16 min** | **Final Production Model ✅** |





## Experiment Log

---

### Experiment 0

Date:
31 July 2026

Status:
Project Initialized

Completed

- Created virtual environment
- Installed dependencies
- Moved dataset into project
- Opened project in VS Code

Observations

- Using PyTorch
- Will start implementation in notebook




# Experiment 0

Date: 31 July 2026

## Completed

- Set up Google Colab with GPU runtime.
- Mounted Google Drive.
- Configured dataset path.
- Imported all required PyTorch and image-processing libraries.
- Loaded dataset directory structure.
- Counted training and testing images.

## Dataset Statistics

Training
- Normal: 1266
- Pneumonia: 3418

Testing
- Normal: 317
- Pneumonia: 855

## Observation

- Dataset is imbalanced, with significantly more Pneumonia images than Normal images.
- This may bias the model toward predicting the majority class.
- Evaluation will emphasize Precision, Recall, F1-score, and Confusion Matrix rather than relying only on accuracy.


## Lessons Learned

- Google Colab is being used for GPU-based model training.
- VS Code remains the main project workspace for documentation, version control, and deployment.
- Dataset path issues were resolved by correctly pointing to the Data folder.
- Initial dataset inspection revealed class imbalance, which will influence evaluation and possibly training strategy.



## Completed

- Explored random Normal and Pneumonia X-ray images.
- Observed varying image dimensions across the dataset.
- Designed preprocessing pipelines for training and testing.
- Added data augmentation for the training set.
- Configured normalization and tensor conversion for model input.

## Observation

- Training images use augmentation to improve generalization.
- Test images are only preprocessed to ensure fair evaluation.
- All images will be standardized to 224×224 before training.



## Dataset Loading

# Completed
- Loaded training and testing datasets using `torchvision.datasets.ImageFolder`.
- Applied separate preprocessing pipelines for training and testing.
- Created PyTorch `DataLoader` objects for efficient batch loading.
- Configured training data with shuffling enabled and test data without shuffling.

# Hyperparameters
- Batch Size: 2 (Baseline)
- Shuffle (Train): True
- Shuffle (Test): False
- Pin Memory: True

# Observation
- Dataset loaded successfully.
- Class labels detected automatically: `['NORMAL', 'PNEUMONIA']`.
- Total Training Images: 4684
- Total Testing Images: 1172






# Experiment 1 - Baseline Custom CNN

## Date
31 July 2026

## Objective
Train the  original custom CNN architecture without any modifications to establish a baseline.

---

## Dataset

Training Images : 4684

Testing Images : 1172

Classes

- NORMAL
- PNEUMONIA

---

## Image Size

224 × 224

---

## Data Augmentation

Training

- Resize(224)
- CenterCrop(224)
- ColorJitter
- RandomHorizontalFlip
- RandomRotation(10)
- ToTensor
- Normalize

Testing

- Resize(224)
- CenterCrop(224)
- ToTensor
- Normalize

---

## Model

Custom CNN

9 Convolution Layers

Batch Normalization

Global Average Pooling

Output Convolution Layer

---

## Optimizer

SGD

Learning Rate = 0.01

Momentum = 0.8

---

## Scheduler

StepLR

step_size = 6

gamma = 0.5

(Note: Scheduler did not reduce the learning rate because only 4 epochs were trained.)

---

## Batch Size

2

---

## Epochs

4

---

## Device

Tesla T4 GPU

---

## Results

Epoch 0

Train Accuracy = 76.92%

Test Accuracy = 78.84%

---

Epoch 1

Train Accuracy = 75.88%

Test Accuracy = 77.90%

---

Epoch 2

Train Accuracy = 74.89%

Test Accuracy = 76.11%

---

Epoch 3

Train Accuracy = 75.79%

Test Accuracy = 79.27%

---

## Best Test Accuracy

79.27%

---

## Observations

• Baseline model established successfully.

• Batch size of 2 resulted in very slow first epoch due to many iterations.

• Learning rate scheduler remained inactive because training ended before step_size=6.

• Evaluation currently reports only Accuracy and Loss.

• Precision, Recall, F1-score and Confusion Matrix are not yet implemented.

---

## Future Experiments

- Add Precision, Recall and F1-score.
- Tune Batch Size.
- Tune Epochs.
- Improve output activation and loss pairing.
- Experiment with optimizer.
- Handle class imbalance.




## Experiment 2 - Enhanced Evaluation

### Objective
Improve the model evaluation pipeline without modifying the CNN architecture, loss function, optimizer, or training procedure.

### Changes
- Added Precision
- Added Recall
- Added F1-score
- Added Confusion Matrix
- Added Classification Report

### Model Changes
None

### Best Metrics
- Accuracy: **74.06%**
- Precision: **0.7558**
- Recall: **0.9520**
- F1-score: **0.8427**

### Observation
The enhanced evaluation pipeline provided a much clearer understanding of model performance on the imbalanced pneumonia dataset. Although the overall accuracy was relatively low, the model achieved a high Recall (95.20%), indicating that it correctly identified most pneumonia cases. This experiment served as the baseline for all subsequent optimization experiments.

---

### Notes

- Experiment 1 and Experiment 2 were trained in separate runs with different random weight initializations.

- The difference in accuracy between Experiment 1 and Experiment 2 should **not** be attributed to the enhanced evaluation metrics.

- Experiment 2 introduced **only evaluation improvements**; the CNN architecture, optimizer, loss function, hyperparameters, and training procedure remained unchanged.

- The results obtained in this experiment serve as the **reference baseline for evaluating the impact of all subsequent optimization experiments.**

## Experiment 3 - Batch Size Optimization

### Objective

Evaluate whether increasing batch size from 2 to 16 improves training efficiency and model performance.

### Hypothesis

A larger batch size will improve GPU utilization and maintain or improve evaluation metrics.

### Changes

Batch Size:
2 → 16

No other hyperparameters were modified.

### Best Results

Training Accuracy: 84.78%

Test Accuracy: 89.85%

Precision: 90.17%

Recall: 96.61%

F1-score: 93.28%

Training Time: ~11 minutes

### Observation

Compared with the previous run, this experiment achieved higher Accuracy, Recall, and F1-score. Recall improved substantially, reducing the number of missed pneumonia cases from 77 to 29 in this run.

### Conclusion

Batch size of 16 appears promising and will be retained for subsequent experiments, while noting that some variation may also be due to random weight initialization.






## Experiment 4 - Standard Output Layer and Loss Function

### Objective

Replace the non-standard Sigmoid + NLLLoss combination with the standard Raw Logits + CrossEntropyLoss configuration.

### Hypothesis

Using CrossEntropyLoss with raw logits will improve optimization and overall classification performance.

### Changes

- Removed final Sigmoid activation.
- Returned raw logits from the model.
- Replaced NLLLoss with CrossEntropyLoss.
- Added fixed random seed (SEED = 42) for improved reproducibility.

### Best Results

Training Accuracy: 94.09%

Test Accuracy: 93.77%

Precision: 96.55%

Recall: 94.85%

F1-score: 95.69%

Training Time: ~11 minutes

### Observation

The standard logits + CrossEntropyLoss configuration produced the best overall performance so far. Compared to the previous experiment, Accuracy, Precision, and F1-score improved substantially while Recall remained high.

### Conclusion

The CrossEntropyLoss configuration will be retained for future experiments.






## Experiment 5 - Epoch Tuning (4 → 10)

### Date
31 July 2026

### Objective

Evaluate whether increasing the number of training epochs from 4 to 10 improves the model's learning capability and overall classification performance.

---

### Hypothesis

The model had not fully converged after 4 epochs. Increasing the number of epochs should allow the model to learn better feature representations and improve evaluation metrics without causing significant overfitting.

---

### Changes

- Epochs:
  - Before: 4
  - After: 10

No other hyperparameters were changed.

- Batch Size = 16
- Optimizer = SGD
- Learning Rate = 0.01
- Loss Function = CrossEntropyLoss
- Output Layer = Raw Logits
- Random Seed = 42

---

### Best Results (Epoch 7)

Training Accuracy:
95.35%

Test Accuracy:
95.73%

Precision:
96.21%

Recall:
98.01%

F1-score:
97.10%

Training Time:
~27 minutes

---

### Observations

- The learning rate scheduler (StepLR) became active after Epoch 6, reducing the learning rate from 0.01 to 0.005.
- Accuracy improved from 93.77% to 95.73%.
- Recall increased from 94.85% to 98.01%.
- F1-score increased from 95.69% to 97.10%.
- False negatives (Pneumonia predicted as Normal) decreased significantly.
- No obvious signs of overfitting were observed within 10 epochs.

---

### Conclusion

Increasing the number of epochs improved overall model performance. The model achieved its best performance at Epoch 7, after which improvements became marginal. Therefore, 10 epochs is currently preferred over 4 epochs for this project.






## Experiment 6 - Optimizer Comparison (SGD → Adam)

### Date
31 July 2026

### Objective

Evaluate whether replacing the SGD optimizer with Adam improves convergence and overall model performance while keeping all other hyperparameters unchanged.

---

### Hypothesis

Adam uses adaptive learning rates for each parameter and may converge faster than SGD while maintaining or improving classification performance.

---

### Changes

Optimizer:
- Before: SGD (lr = 0.01, momentum = 0.8)
- After: Adam (lr = 0.001)

No other hyperparameters were modified.

- Batch Size = 16
- Epochs = 10
- CrossEntropyLoss
- Raw Logits
- Random Seed = 42
- StepLR(step_size=6, gamma=0.5)

---

### Best Results (Epoch 8)

Training Accuracy:
95.86%

Test Accuracy:
95.82%

Precision:
96.64%

Recall:
97.66%

F1-score:
97.15%

Training Time:
~25 minutes

---

### Observations

- Adam converged smoothly throughout training.
- Accuracy improved slightly compared to SGD.
- Precision and F1-score improved marginally.
- Recall decreased slightly but remained very high.
- Training completed faster than the previous SGD experiment.

---

### Conclusion

Adam provided the best overall balance of Accuracy, Precision, and F1-score while maintaining excellent Recall. Adam will be selected as the optimizer for subsequent experiments.




# Engineering Improvements

## Automatic Best Model Checkpointing

### Objective

Improve the training pipeline by automatically saving the best-performing model instead of the last epoch.

### Implementation

- Added model checkpointing during training.
- After every epoch, evaluation metrics are calculated.
- If the current accuracy exceeds the previous best accuracy:
  - Save the model checkpoint.
  - Store model weights.
  - Store optimizer state.
  - Store epoch number.
  - Store Accuracy, Precision, Recall and F1-score.

### Result

- Successfully saved the best-performing model automatically.
- Best model saved from Epoch 7.
- Prevents accidentally deploying a model from the final epoch if it is not the best-performing one.

### Importance

This approach follows standard deep learning practice and ensures that deployment always uses the highest-performing model observed during training.




## Engineering Improvement - Model Checkpoint Verification

### Objective

Verify that the saved checkpoint can be loaded successfully and reproduces the same evaluation metrics.

### Procedure

- Loaded the saved checkpoint (`best_model_checkpoint.pth`)
- Restored model weights using `load_state_dict()`
- Evaluated the restored model on the test dataset

### Result

The restored model produced identical evaluation metrics:

- Accuracy: 95.65%
- Precision: 96.53%
- Recall: 97.54%
- F1-score: 97.03%

The confusion matrix and classification report were identical to the saved checkpoint.

### Conclusion

Checkpoint saving and loading were successfully verified. The saved model is ready for inference and deployment.





## Experiment 7 – Remove ColorJitter

### Objective
Evaluate whether removing `ColorJitter` from the training data augmentation pipeline improves model generalization and classification performance on chest X-ray images.

### Changes Made
- Removed `transforms.ColorJitter()` from `train_transform`.
- Kept the following augmentations unchanged:
  - Resize(224)
  - CenterCrop(224)
  - RandomHorizontalFlip()
  - RandomRotation(10)
  - Normalize()
- Continued using:
  - Adam Optimizer (lr = 0.001)
  - StepLR Scheduler
  - Batch Size = 16
  - CrossEntropyLoss
  - 10 Epochs
  - Model Checkpointing
- Copied dataset from Google Drive to `/content/Data` before training to improve data loading speed.

### Results (Best Epoch = 8)

| Metric | Value |
|--------|-------:|
| Test Accuracy | **95.82%** |
| Precision | **96.11%** |
| Recall | **98.25%** |
| F1 Score | **97.17%** |
| Training Time | ~23–24 minutes |

### Observations
- Test accuracy remained unchanged compared to Experiment 6.
- Recall increased from **97.66% → 98.25%**, indicating fewer missed pneumonia cases.
- F1-score improved slightly.
- Precision decreased slightly but remained above 96%.
- Loading the dataset from local Colab storage (`/content/Data`) improved data loading efficiency compared to reading directly from Google Drive.

### Conclusion
Removing `ColorJitter` produced a small improvement in recall and F1-score without affecting overall accuracy. Since chest X-ray images are grayscale, color-based augmentation did not provide additional benefit for this dataset. Therefore, the simpler augmentation pipeline is preferred for subsequent experiments.






## Experiment 8 – Class Imbalance Handling (Class Weights)

### Objective
Evaluate whether assigning class weights to CrossEntropyLoss improves model performance on the minority class (NORMAL).

### Changes Made
- Added class weights to CrossEntropyLoss.
- Weight calculation:
  - NORMAL = 3875 / 1341 ≈ 2.89
  - PNEUMONIA = 1.0
- Kept all remaining settings unchanged:
  - Adam Optimizer
  - Batch Size = 16
  - 10 Epochs
  - StepLR Scheduler
  - Checkpoint Saving
  - Local SSD Dataset (/content/Data)

### Results (Best Epoch = 9)

| Metric | Value |
|--------|-------:|
| Test Accuracy | **95.39%** |
| Precision | **97.06%** |
| Recall | **96.61%** |
| F1 Score | **96.83%** |
| Training Time | ~23–24 min |

### Observations
- Precision increased slightly.
- Overall accuracy decreased.
- Recall decreased compared to Experiment 7.
- F1-score also decreased.
- Improvement in minority-class weighting did not translate into better overall classification performance.

### Conclusion
Adding class weights did not improve the model compared to Experiment 7. Therefore, class-weighted CrossEntropyLoss was not adopted for the final model, and the previous configuration was retained.





## Experiment 9 – CNN Architecture Improvement (Dropout)

### Objective
Evaluate whether introducing Dropout regularization into the CNN architecture improves model generalization and reduces overfitting.

### Changes Made
- Added `nn.Dropout2d(0.2)` after selected convolutional blocks.
- Kept all remaining configurations unchanged:
  - Adam Optimizer
  - Batch Size = 16
  - CrossEntropyLoss
  - StepLR Scheduler
  - 10 Epochs
  - Model Checkpointing
  - Dataset loaded from `/content/Data`

### Results (Best Epoch = 8)

| Metric | Value |
|--------|-------:|
| Test Accuracy | **94.88%** |
| Precision | **96.71%** |
| Recall | **96.26%** |
| F1 Score | **96.48%** |
| Training Time | ~20 minutes |

### Observations
- Precision increased slightly.
- Overall accuracy decreased.
- Recall and F1-score were lower than Experiment 7.
- No evidence that Dropout improved generalization for this CNN architecture.

### Conclusion
Adding Dropout did not outperform the previous architecture. The original CNN architecture (Experiment 7) was retained as the final architecture.





## Experiment 10 – DataLoader Optimization

### Objective
Improve training efficiency without modifying the CNN architecture or affecting model performance.

### Changes Made
- Added `num_workers=2` to both training and test DataLoaders.
- Enabled `persistent_workers=True`.
- Kept all remaining settings unchanged:
  - Adam Optimizer
  - Batch Size = 16
  - CrossEntropyLoss
  - StepLR Scheduler
  - 10 Epochs
  - Checkpoint Saving
  - Dataset loaded from `/content/Data`

### Results (Best Epoch = 9)

| Metric | Value |
|--------|-------:|
| Test Accuracy | **95.48%** |
| Precision | **97.40%** |
| Recall | **96.37%** |
| F1 Score | **96.88%** |
| Training Time | **≈16 min** |

### Observations
- Training time reduced by approximately **32%** compared to Experiment 7.
- Overall accuracy remained above 95%.
- Model performance remained stable despite significantly faster training.

### Conclusion
Optimizing the DataLoader substantially reduced training time while maintaining comparable classification performance. This optimization was adopted as an engineering improvement to the training pipeline.






## Experiment 11 – Increase Training Epochs (10 → 12)

### Objective
Evaluate whether extending training beyond 10 epochs improves model performance.

### Changes Made
- Increased total epochs from **10** to **12**.
- All other hyperparameters remained unchanged.

### Results (Best Epoch = 8)

| Metric | Value |
|--------|-------:|
| Test Accuracy | **95.65%** |
| Precision | **96.42%** |
| Recall | **97.66%** |
| F1 Score | **97.04%** |
| Training Time | **≈21 min** |

### Observations
- Best performance occurred at **Epoch 8**.
- Additional epochs (9–11) did not improve accuracy.
- Epoch 10 showed a noticeable performance drop, indicating the beginning of overfitting.

### Conclusion
Increasing the number of epochs beyond 10 did not improve generalization performance. The original 10-epoch configuration remains the preferred choice.







# Experiment 12 – Final Reproducibility Run

## Objective
Verify the reproducibility of the final selected model by retraining with the same optimal configuration.

---

## Changes Made

- Same configuration as Experiment 7
- Adam Optimizer
- Batch Size = 16
- CrossEntropyLoss
- No ColorJitter
- Local SSD Dataset
- num_workers Optimization
- EPOCHS = 10

---

## Results (Best Epoch = 7)

| Metric | Value |
|--------|------:|
| Test Accuracy | **95.48%** |
| Precision | **96.74%** |
| Recall | **97.08%** |
| F1 Score | **96.91%** |
| Training Time | **≈18 min** |

---

## Observations

- The model achieved its best performance at **Epoch 7**.
- Results remained very close to previous runs (95.48% vs 95.82% best).
- Small metric variations are expected due to random initialization and data shuffling.
- Faster training time (≈18 min) demonstrates a stable and optimized training pipeline.

---

## Conclusion

The retraining experiment confirms that the model consistently converges around **95.5–95.8%** accuracy, demonstrating good reproducibility. No further accuracy improvements were observed, so the current architecture and hyperparameters are considered finalized.






# Experiment 13: Final Optimized Model (Reproducibility)

## Objective

Re-train the finalized CNN architecture using all validated optimizations to verify reproducibility before freezing the model for deployment.

---

## Changes

- CNN architecture finalized
- Adam optimizer
- CrossEntropyLoss
- Batch size = 16
- Removed ColorJitter
- Dataset loaded from Local SSD
- Multi-worker DataLoader
- Best model checkpoint saved to Google Drive
- Trained for 10 epochs

---

## Results

- Best Epoch: 9
- Test Accuracy: **95.82%**
- Precision: **96.97%**
- Recall: **97.31%**
- F1-score: **97.14%**
- Training Time: **≈16 minutes**

---

## Conclusion

The final training successfully reproduced the best test accuracy (**95.82%**) while maintaining the optimized training time of approximately **16 minutes**. This confirms that the proposed CNN architecture and optimized training pipeline provide stable, reproducible, and deployment-ready performance.

This checkpoint is selected as the **final production model** for the end-to-end Lung Disease Detection project.