import torch
import bentoml
from torchvision import transforms

from xray.ml.model.arch import Net
from xray.constant.training_pipeline import (
    BENTOML_MODEL_NAME,
    INFERENCE_TRANSFORMS_KEY,
    DEVICE,
)


# -------------------------------------------------
# 1. Path to your final Colab checkpoint
# -------------------------------------------------
CHECKPOINT_PATH = "lung_disease_cnn_final.pth"


# -------------------------------------------------
# 2. Recreate the exact model architecture
# -------------------------------------------------
model = Net()


# -------------------------------------------------
# 3. Load the checkpoint
# -------------------------------------------------
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)


# -------------------------------------------------
# 4. Restore trained weights
# -------------------------------------------------
model.load_state_dict(
    checkpoint["model_state_dict"]
)


# -------------------------------------------------
# 5. Move model to device and switch to eval mode
# -------------------------------------------------
model = model.to(DEVICE)

model.eval()


# -------------------------------------------------
# 6. Recreate deterministic inference transform
# -------------------------------------------------
inference_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)


# -------------------------------------------------
# 7. Save model into BentoML Model Store
# -------------------------------------------------
bento_model = bentoml.pytorch.save_model(
    name=BENTOML_MODEL_NAME,
    model=model,
    custom_objects={
        INFERENCE_TRANSFORMS_KEY:
            inference_transform
    },
)


print("\nBentoML model registered successfully!")
print(f"Model tag: {bento_model.tag}")