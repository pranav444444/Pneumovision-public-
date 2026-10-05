# PneumoVision — Mistakes, Errors and Debugging Notes

This file records important errors encountered during development and deployment, their root causes, and the fixes applied.

---

## 1. Using Training Transform During Inference

### Problem

The original deployment flow used the training transform for prediction.

The training transform contains random augmentation such as:

- RandomHorizontalFlip
- RandomRotation

This means the same uploaded X-ray could be randomly modified during prediction.

### Fix

Use the deterministic test/inference transform:

Resize(224)
→ CenterCrop(224)
→ ToTensor()
→ Normalize()

### Lesson

Training augmentation should not normally be used during inference.

Inference preprocessing should be deterministic and consistent with the model's expected input.

---

## 2. BentoML Transform Key Mismatch

### Problem

The model was saved using:

`"transform"`

but `model_service.py` tried to retrieve:

`TRAIN_TRANSFORMS_KEY`

These keys did not match.

This could cause:

`bento_model.custom_objects.get(...)`

to return `None`.

### Fix

Created:

`INFERENCE_TRANSFORMS_KEY = "xray_inference_transforms"`

and used the same key consistently while:

- saving the BentoML model
- loading the transform during inference

### Lesson

Keys used to store and retrieve custom objects must match exactly.

---

## 3. Incorrect Prediction Label Dictionary

### Problem

The original mapping was:

`{"0": NORMAL, 1: PNEUMONIA}`

The key `"0"` was a string while `1` was an integer.

However, `torch.argmax()` returns an integer class index.

### Fix

Changed the mapping to:

`{0: NORMAL, 1: PNEUMONIA}`

### Lesson

Prediction output types must match dictionary key types.

---

## 4. Unnecessary Tensor → NumPy → Tensor Conversion

### Problem

The original inference code converted:

PyTorch Tensor
→ NumPy Array
→ PyTorch Tensor

even though `ToTensor()` had already created a valid PyTorch tensor.

### Fix

Simplified inference to:

`image = inference_transform(image)`

followed by:

`image = image.unsqueeze(0)`

### Lesson

Avoid unnecessary conversions between frameworks because they add complexity and potential bugs.

---

## 5. BentoML 1.0.10 Starlette Compatibility Error

### Error

`ImportError: cannot import name 'MultiPartMessage' from starlette.formparsers`

### Cause

The older BentoML version was incompatible with the installed Starlette version.

### Fix

Upgraded BentoML from:

`1.0.10`

to:

`1.0.25`

### Lesson

Deployment frameworks often depend on specific web-framework versions.

A working ML model can still fail because of unrelated dependency incompatibility.

---

## 6. NumPy 2.x Compatibility Error

### Error

A warning/error appeared stating that a module compiled against NumPy 1.x could not safely run with NumPy 2.x.

### Cause

The PyTorch/deployment stack was built for the older NumPy 1.x ABI.

### Fix

Pinned:

`numpy==1.26.4`

### Lesson

Major library upgrades can introduce ABI compatibility problems.

For production environments, known-working versions should be pinned.

---

## 7. Bento Metadata Missing `name` and `version`

### Error

Commands such as:

`bentoml list`

and:

`bentoml get`

failed with:

`KeyError: 'name'`

### Cause

The generated `bento.yaml` did not contain expected metadata fields:

`name`

and:

`version`

### Temporary Fix

Added:

`name: xray_service`

and the corresponding Bento version manually to the generated `bento.yaml`.

### Lesson

Generated artifacts can also contain framework-specific bugs.

Inspecting the generated configuration can help identify problems that are not caused by application code.

---

## 8. `bentoml containerize` Failed

### Error

`NotImplementedError`

inside BentoML's Docker containerization implementation.

### Cause

The BentoML 1.0.x containerization path did not work correctly in the current Windows environment.

### Fix

Bypassed:

`bentoml containerize`

and used the generated Dockerfile directly with:

`docker build`

### Lesson

Framework convenience commands are not mandatory.

If the lower-level tool works, it is acceptable to use Docker directly.

---

## 9. Docker Container Could Not Find `xray_model`

### Error

`no Models with name 'xray_model' exist in BentoML store`

### Cause

The Docker image contained the Bento service files but not the local BentoML model store.

Inside the container, BentoML expected models under:

`/home/bentoml/models`

### Fix

Copied the local registered model:

`xray_model`

into the Docker build context and then copied it into:

`/home/bentoml/models`

inside the image.

### Lesson

A model-serving framework may rely on its own model-store structure.

Packaging application code alone is not sufficient if the model artifacts are stored separately.

---

## 10. Custom Dockerfile Initially Removed Important COPY Command

### Error

Docker build failed with:

`entrypoint.sh: No such file or directory`

### Cause

While modifying the generated Dockerfile, the line:

`COPY --chown=bentoml:bentoml . ./`

was accidentally removed.

Therefore the Bento application files were never copied into the image.

### Fix

Restored:

`COPY --chown=bentoml:bentoml . ./`

and added the model-store copy separately.

### Lesson

When modifying generated Dockerfiles, preserve the original application-copy steps and add new steps instead of replacing existing ones.

---

## 11. `pkg_resources` Missing Inside Docker Container

### Error

`ModuleNotFoundError: No module named 'pkg_resources'`

### Cause

The older BentoML dependency stack expected `pkg_resources`, but a newer setuptools version no longer provided the expected behavior.

### Fix

Pinned:

`setuptools==80.9.0`

### Lesson

Even packaging utilities such as setuptools can affect application runtime.

Dependency versions should be reproducible.

---

## 12. PyTorch `weights_only=True` Loading Error

### Error

The BentoML runner failed with:

`_pickle.UnpicklingError: Weights only load failed`

and reported that the custom class:

`xray.ml.model.arch.Net`

was not allowlisted.

### Cause

Newer PyTorch versions changed the default behavior of `torch.load()` to safer weights-only loading.

The BentoML model had been stored as a full serialized PyTorch model object.

### Fix

Pinned the deployment environment to:

`torch==1.13.1`

and:

`torchvision==0.14.1`

which matches the older BentoML serialization behavior.

### Lesson

Model serialization format and framework version must be compatible.

Saving a full model object creates stronger dependency on the exact framework/environment than saving only a `state_dict`.

---

## 13. Docker Image Built Successfully but Failed at Runtime

### Problem

Several Docker images built successfully but later failed when started.

### Examples

- missing `pkg_resources`
- missing BentoML model
- PyTorch model deserialization error

### Lesson

A successful:

`docker build`

does not prove that the application works.

Always validate:

Docker build
→ Docker run
→ API startup
→ real prediction

before pushing the image to a cloud registry.

---

## 14. Tested Docker Locally Before AWS Push

### Correct Practice

The final image was tested locally using:

`docker run --rm -p 3000:3000 ...`

The BentoML API was opened at:

`http://localhost:3000`

A real chest X-ray was uploaded to:

`POST /predict`

and the model returned the expected class.

### Lesson

Cloud deployment should happen only after local runtime and inference are verified.

---

## 15. AWS ECR Push Verification

### Result

The final tested Docker image was successfully pushed to the private ECR repository:

`xray_bento_image`

with tag:

`latest`

### Verification

AWS ECR showed:

- active image
- image digest
- Linux/AMD64 manifest
- image size
- push timestamp

### Lesson

A deployment pipeline should verify the final cloud artifact after pushing instead of assuming the command succeeded.

---

## 16. Final Working Deployment Stack

The final working deployment environment uses:

- BentoML 1.0.25
- PyTorch 1.13.1
- torchvision 0.14.1
- NumPy 1.26.4
- setuptools 80.9.0

These versions should remain pinned until the deployment stack is intentionally upgraded and retested.

---

## Must Remember

Model training success does not guarantee deployment success.

Deployment problems can come from:

- dependency versions
- serialization
- Docker configuration
- model-store paths
- API framework compatibility
- cloud authentication

Debug deployment one layer at a time:

BentoML model
→ Bento service
→ Docker build
→ Docker runtime
→ API prediction
→ ECR push

Do not change multiple layers at once unless necessary.