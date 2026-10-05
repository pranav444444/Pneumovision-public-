# Resume Project Highlights

## Project
**Lung Disease Detection using Custom CNN (PyTorch)**

### Resume Bullet Points

- Developed and optimized a custom **Convolutional Neural Network (CNN)** for automated lung disease (pneumonia) detection from chest X-ray images using **PyTorch**, achieving **95.82% test accuracy**, **96.97% precision**, **97.31% recall**, and **97.14% F1-score**.

- Improved model performance through **13 systematic experiments**, including optimizer tuning, loss function optimization, data augmentation refinement, hyperparameter tuning, checkpointing, and reproducibility validation, increasing test accuracy from **79.27%** to **95.82%** (**≈20.87% relative improvement**).

- Optimized the training pipeline by implementing **local SSD dataset loading** and a **multi-worker DataLoader**, reducing end-to-end training time from **27 minutes** to **16 minutes** (**≈41% reduction**) while maintaining comparable predictive performance.

- Implemented automated **best-model checkpointing**, enabling reproducible experiments and automatic saving of the highest-performing model for deployment.

- Performed comprehensive model evaluation using **Precision, Recall, F1-score, Confusion Matrix, and Classification Report** to validate model performance on an imbalanced medical imaging dataset.

- Conducted extensive experiment tracking and performance benchmarking across multiple model configurations, selecting the final production model based on reproducibility, predictive performance, and computational efficiency.

### Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab
- CNN
- Deep Learning
- Computer Vision





-"Although the dataset was imbalanced (approximately 1:2.8), the imbalance was moderate rather than extreme. The CNN learned discriminative radiographic features effectively because pneumonia and normal chest X-rays have distinct visual characteristics. Through systematic optimization—including Adam optimization, Batch Normalization, appropriate learning rate scheduling, removal of unsuitable ColorJitter augmentation, and careful hyperparameter tuning—we achieved balanced performance with approximately 96.97% precision and 97.31% recall. We also evaluated class-weighted loss, but empirical experiments showed that it did not improve performance, so it was excluded from the final model. This demonstrates that architectural and training optimizations were sufficient to handle the dataset imbalance without additional weighting."




### Clinical Interpretation

The baseline CNN model exhibited a strong bias toward the majority (PNEUMONIA) class, correctly identifying only **17% of NORMAL** chest X-rays while misclassifying **83%** as pneumonia. Through systematic optimization of the training pipeline, optimizer, hyperparameters, and data preprocessing, the final model correctly classified **92% of NORMAL** images while maintaining **97.31% Recall** for pneumonia cases. This substantially reduced false pneumonia diagnoses and resulted in a more balanced and clinically reliable model.