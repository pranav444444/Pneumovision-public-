import bentoml
import torch
from bentoml.io import Image, Text

from xray.constant.training_pipeline import *


# Load the trained model from BentoML Model Store
bento_model = bentoml.pytorch.get(BENTOML_MODEL_NAME)

# Create runner for inference
runner = bento_model.to_runner()

# Create BentoML service
svc = bentoml.Service(
    name=BENTOML_SERVICE_NAME,
    runners=[runner]
)


@svc.api(
    input=Image(
        allowed_mime_types=[
            "image/jpeg",
            "image/png",
        ]
    ),
    output=Text(),
)
async def predict(img):

    # Convert uploaded image to RGB because
    # the CNN expects 3-channel input
    image = img.convert("RGB")

    # Load deterministic inference transform
    inference_transform = (
        bento_model.custom_objects.get(
            INFERENCE_TRANSFORMS_KEY
        )
    )

    # Apply preprocessing
    # Output shape:
    # [3, 224, 224]
    image = inference_transform(image)

    # Add batch dimension
    # [3, 224, 224]
    #        ↓
    # [1, 3, 224, 224]
    image = image.unsqueeze(0)

    # Run inference
    # Output contains raw logits:
    # [logit_NORMAL, logit_PNEUMONIA]
    batch_ret = await runner.async_run(image)

    # Select class having the highest logit
    predicted_class = torch.argmax(
        batch_ret,
        dim=1
    ).item()

    # Convert class index into class label
    pred = PREDICTION_LABEL[
        predicted_class
    ]

    return pred