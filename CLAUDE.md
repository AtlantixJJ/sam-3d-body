# SAM 3D Body Architecture Documentation

## Overview

SAM 3D Body is a promptable model for single-image full-body 3D human mesh recovery (HMR). It employs an encoder-decoder architecture with support for auxiliary prompts (2D keypoints and masks), enabling user-guided inference similar to the SAM family of models. The model estimates human pose using the Momentum Human Rig (MHR) representation, which decouples skeletal structure and surface shape.

## Architecture Components

### 1. Backbone (Image Encoder)

**Location:** `sam_3d_body/models/backbones/`

The backbone extracts visual features from input images. Two main implementations:

- **ViT (Vision Transformer)** (`vit.py`):
  - Variants: ViT-H (1280 dim), ViT-L (1024 dim), ViT-B (768 dim)
  - Patch size: 16x16
  - Input size: 256x192 (or 256x256, 512x384 variants)
  - Supports flash attention for efficiency
  - Configurable frozen stages for transfer learning

- **DINOv3** (`dinov3.py`):
  - Loads pre-trained DINOv3 models from torch.hub
  - Returns intermediate layer features
  - Patch-based representation with positional encoding

**Key Features:**
- Absolute positional embeddings with interpolation for varying input sizes
- Support for gradient checkpointing
- Layer-wise depth tracking for learning rate scaling

### 2. Prompt Encoder

**Location:** `sam_3d_body/models/decoders/prompt_encoder.py`

Encodes user prompts into embeddings for the decoder:

**Components:**
- **Keypoint Prompts:**
  - Per-joint learnable embeddings (70 body joints)
  - Position encoding using random spatial frequencies
  - Special embeddings for invalid (-2) and incorrect (-1) points

- **Mask Prompts:**
  - Downscaling CNN (v1: 4x+4x striding, v2: 2x+2x+2x+2x striding)
  - LayerNorm2d + GELU activation
  - Zero-initialized final layer for gating
  - "No mask" embedding for cases without mask input

**Positional Encoding:**
- Random Fourier features with Gaussian matrix
- Normalizes coordinates to [0,1] then encodes with sin/cos
- Supports both dense (grid) and sparse (point) encoding

### 3. Promptable Decoder

**Location:** `sam_3d_body/models/decoders/promptable_decoder.py`

Cross-attention Transformer decoder that processes pose tokens with image context:

**Architecture:**
- Multiple `TransformerDecoderLayer` blocks (configurable depth)
- Each layer has:
  - Self-attention on tokens
  - Cross-attention to image features
  - Feed-forward network (FFN)
  - Layer normalization
  - Optional layer scale and drop path

**Key Features:**
- **Two-way attention:** Optional bidirectional attention (SAM-style)
- **Repeat positional encoding:** Re-adds PE at each layer
- **Intermediate predictions:** Can output pose at each layer for iterative refinement
- **Keypoint token updates:** Dynamic token updates based on intermediate predictions
- **Hand embeddings:** Optional hand-specific feature integration

**Token Flow:**
1. Pose tokens (initialized or from previous iteration)
2. Prompt tokens (from keypoint/mask encoder)
3. Keypoint query tokens (per-joint queries)
4. Optional 3D keypoint tokens

### 4. Prediction Heads

#### MHR Head
**Location:** `sam_3d_body/models/heads/mhr_head.py`

Predicts MHR (Momentum Human Rig) parameters:

**Output Parameters:**
- Global rotation (6D representation): 6 dims
- Body pose (continuous representation): 260 dims
- Shape parameters: 45 dims
- Scale parameters: 28 dims
- Hand pose (PCA): 54 dims × 2 (left/right)
- Face expression: 72 dims
- **Total:** 407 dimensions

**MHR Forward:**
1. Projects decoder output to parameter space
2. Converts 6D rotation to rotation matrix, then to Euler angles
3. Converts continuous body pose to Euler angles
4. Combines hand PCA coefficients with mean/components
5. Computes scale from PCA components
6. Runs MHR model to get:
   - Skinned vertices (18,439 vertices)
   - Joint coordinates (127 joints)
   - Joint rotations
   - Keypoints (70 from 308 Sapiens keypoints)

**Coordinate System:** Flips Y and Z axes for camera compatibility

#### Camera Head
**Location:** `sam_3d_body/models/heads/camera_head.py`

Predicts camera parameters for perspective projection:

**Output:** 3 parameters (s, tx, ty)
- s: scale factor
- tx, ty: translation in normalized space

**Projection:**
1. Computes camera translation: `pred_cam_t = [tx + cx, ty + cy, tz]`
   - tz (depth) = `2 * focal_length / (bbox_size * s)`
2. Projects 3D keypoints to 2D using perspective projection
3. Returns 2D keypoints in original image space

### 5. Supporting Modules

#### Camera Embedding
**Location:** `sam_3d_body/models/modules/camera_embed.py`

Encodes camera ray information into image features for perspective-aware processing.

#### Geometry Utils
**Location:** `sam_3d_body/models/modules/geometry_utils.py`

Utilities for:
- Perspective projection
- Rotation representations (6D, rotation matrix, Euler angles)
- Camera intrinsic matrix computation

#### MHR Utils
**Location:** `sam_3d_body/models/modules/mhr_utils.py`

Utilities for MHR parameter conversion:
- Continuous to model parameters (body/hand)
- Euler angle fixing for wrist joints
- Rotation angle differences

## Data Flow

### Inference Pipeline

**Entry Point:** `SAM3DBodyEstimator.process_one_image()`

1. **Preprocessing:**
   - Human detection (optional, using ViTDet)
   - Mask generation (optional, using SAM2)
   - FOV estimation (optional, using MOGE2)
   - Image cropping and affine transformation
   - Normalization to [0,1]

2. **Feature Extraction:**
   - Backbone processes cropped image → `image_embeddings` (B, C, H, W)
   - Camera ray conditioning added to embeddings

3. **Token Initialization:**
   - Pose token: learnable initialization to zero-pose
   - Camera token: zero-initialized
   - Condition info: CLIFF-style bbox/focal encoding (cx/f, cy/f, b/f)
   - Combined into initial token embedding

4. **Prompt Processing (if enabled):**
   - Keypoint prompts encoded with position embeddings
   - Mask prompts downscaled and embedded
   - Previous estimate embedded (for iterative refinement)
   - Tokens concatenated: [pose_token, prev_token, prompt_tokens, keypoint_query_tokens]

5. **Decoder Processing:**
   - Multi-layer cross-attention between tokens and image features
   - Intermediate predictions at each layer (optional)
   - Keypoint tokens updated with predicted 2D/3D locations

6. **Prediction:**
   - MHR Head: outputs pose/shape/scale parameters
   - Camera Head: outputs camera translation
   - MHR model: generates 3D mesh and keypoints
   - Perspective projection: projects to 2D

7. **Hand Refinement (full mode):**
   - Extract hand bounding boxes from body pose
   - Flip left hand image horizontally
   - Run hand decoder on each hand crop
   - Merge hand predictions with body

### Main Model Class

**Location:** `sam_3d_body/models/meta_arch/sam3d_body.py`

**Class:** `SAM3DBody(BaseModel)`

**Key Methods:**

- `_initialize_model()`: Builds all components
- `forward_decoder()`: Runs decoder with prompts
- `forward_step()`: Single forward pass (body or hand)
- `run_inference()`: Full inference with optional hand refinement
- `camera_project()`: Projects 3D to 2D
- `_get_hand_box()`: Extracts hand regions from body pose

## Inference Types

1. **Body Only:** Uses body decoder for full-body prediction (fast)
2. **Hand Only:** Uses hand decoder for hand-specific prediction
3. **Full:** Sequential body → hand refinement (highest quality)
   - Detects wrist angle to determine if hand refinement needed
   - Processes each hand separately with dedicated decoder
   - Merges refined hand pose with body prediction

## Key Design Choices

### 1. Iterative Refinement
- Decoder supports intermediate predictions
- Keypoint tokens update based on current predictions
- Previous estimates guide next iteration

### 2. Dual Decoder Architecture
- Separate decoders for body and hands
- Hand decoder uses wrist-centric coordinate frame
- Enables high-resolution hand detail

### 3. Promptable Design
- Keypoint prompts: guide specific joint locations
- Mask prompts: focus on person region
- Previous estimates: enable iterative correction
- Follows SAM philosophy of flexible user guidance

### 4. Perspective-Aware
- Full perspective projection (not weak-perspective)
- Camera intrinsics from FOV estimator or defaults
- CLIFF-style conditioning with bbox/focal normalization

### 5. MHR Representation
- Decoupled skeleton (pose) and surface (shape/scale)
- Hierarchical joint structure (127 joints)
- PCA-compressed hand pose (54 dims per hand)
- Expression parameters for face (72 dims)

## Model Building

**Entry Point:** `sam_3d_body/build_models.py`

**Functions:**
- `load_sam_3d_body(checkpoint_path, device, mhr_path)`: Loads from local checkpoint
- `load_sam_3d_body_hf(repo_id)`: Loads from HuggingFace Hub

**Configuration:** YAML files define architecture hyperparameters:
- Backbone type and settings
- Decoder depth and dimensions
- Head configurations
- Training settings (FP16, frozen stages)

## External Dependencies

### Required Models:
1. **MHR Model** (`mhr_model.pt`): Parametric body mesh model
2. **Human Detector** (optional): ViTDet for person detection
3. **Segmentor** (optional): SAM2 for mask generation
4. **FOV Estimator** (optional): MOGE2 for camera intrinsics

### Key Libraries:
- PyTorch: Core framework
- roma: Rotation mathematics
- timm: Vision model layers
- flash-attn: Efficient attention (optional)

## Output Format

**Per-person predictions:**
- `pred_vertices`: 3D mesh vertices (18,439 × 3)
- `pred_keypoints_3d`: 3D joint locations (70 × 3)
- `pred_keypoints_2d`: 2D projected joints (70 × 2)
- `pred_cam_t`: Camera translation (3,)
- `focal_length`: Focal length (scalar)
- `global_rot`: Global rotation Euler angles (3,)
- `body_pose_params`: Body pose parameters (133,)
- `hand_pose_params`: Hand pose PCA (108,)
- `shape_params`: Shape PCA (45,)
- `scale_params`: Scale PCA (28,)
- `expr_params`: Expression PCA (72,)

## File Organization

```
sam_3d_body/
├── models/
│   ├── backbones/          # Feature extractors (ViT, DINOv3)
│   ├── decoders/           # Prompt encoder, decoder
│   ├── heads/              # MHR head, camera head
│   ├── meta_arch/          # Main model (SAM3DBody)
│   ├── modules/            # Utilities (transformers, geometry)
│   └── optim/              # FP16 utilities
├── data/
│   ├── transforms/         # Image preprocessing
│   └── utils/              # Batch preparation, I/O
├── utils/                  # Config, checkpoint, logging
├── visualization/          # Rendering utilities
├── build_models.py         # Model loading
└── sam_3d_body_estimator.py  # Inference wrapper
```

## Summary

SAM 3D Body is a sophisticated full-body mesh recovery system that combines:
- Vision Transformer backbones for robust feature extraction
- Promptable cross-attention decoder for flexible inference
- MHR parametric model for accurate body representation
- Dual-decoder architecture for high-quality hand details
- Perspective camera model for metric reconstruction

The architecture supports both automatic (detector-based) and manual (prompt-based) workflows, making it suitable for diverse applications from batch processing to interactive editing.
