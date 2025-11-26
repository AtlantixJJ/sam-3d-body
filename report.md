# SAM 3D Body Architecture Documentation

## Executive Summary

**SAM 3D Body** is a promptable 3D human mesh recovery model that adapts SAM's architecture for pose estimation. It combines ViT backbones, SAM-style promptable decoders, and the MHR (Momentum Human Rig) body model to enable interactive, high-fidelity human mesh reconstruction from single images.

**Key Features:**
- **Promptable design**: Interactive keypoint correction similar to SAM's mask refinement
- **Dynamic keypoint tokens**: Tokens update during decoding based on predictions
- **Ray-conditioned encoding**: Camera-aware features for better depth estimation
- **Dual decoder system**: Separate decoders for body and hand refinement
- **High-fidelity output**: 18,439 vertices, 127 joints, 70 keypoints

---

## Architecture Overview

### High-Level Pipeline

```
Input Image (B, 3, H, W)
    ↓
Backbone (ViT-L / DINOv3)
    → (B, 1280, 18, 13) features
    ↓
Ray Condition Encoder (camera-aware)
    → (B, 1280, 18, 13) conditioned features
    ↓
Prompt Encoder (optional keypoint/mask prompts)
    → (B, N_prompts, 1280) embeddings
    ↓
Token Construction
    [pose_token | prev_token | prompts | kp_tokens_2d | kp_tokens_3d]
    → (B, 73-145, 1024)
    ↓
Promptable Decoder (2 layers, SAM-style)
    - Layer 1: Predict → Update keypoint tokens
    - Layer 2: Refine → Final prediction
    → (B, N_tokens, 1024)
    ↓
┌─────────────┴──────────────┐
↓                            ↓
MHR Head                  Camera Head
(B, 519) params           (B, 3) [s, tx, ty]
↓                            ↓
MHR Model (TorchScript)      Perspective Projection
→ Vertices (B, 18439, 3)     → 2D Keypoints (B, 70, 2)
→ Keypoints (B, 70, 3)
→ Joints (B, 127, 3)
```

### Dual Decoder System

For full inference mode:
1. **Body Decoder**: Processes full-body crop → full-body pose + hand bounding boxes
2. **Hand Detector**: Extracts hand regions from predictions
3. **Hand Decoder** (×2): Refines left/right hands separately
4. **Merge**: Combines body and refined hand predictions

---

## Core Components

### 1. Backbone (Image Encoder)

**Location:** `sam_3d_body/models/backbones/`

**Variants:**
- **ViT-L**: 1280 dim, 24 layers, 16 heads, ~300M params
- **ViT-B**: 768 dim, 12 layers, 12 heads
- **DINOv3**: Pretrained from torch.hub

**Input/Output:**
- Input: (B×N, 3, 192, 256) normalized with ImageNet stats
- Output: (B×N, 1280, 18, 13) for ViT-L with 16×16 patches
- Patch size: 14×14 (ViT-L) or 16×16 (ViT-B)

**Features:**
- Absolute positional embeddings with interpolation
- Optional frozen stages for transfer learning
- Flash attention support for efficiency

### 2. Ray Condition Encoder

**Location:** `sam_3d_body/models/modules/camera_embed.py`

**Purpose:** Encode camera intrinsics into image features for camera-aware processing.

**Processing:**
```python
# Compute ray directions for each pixel
ray_x = (x - cx) / fx
ray_y = (y - cy) / fy
rays = [ray_x, ray_y]  # (B, 2, H, W)

# Fourier encoding (16 frequency bands)
freq_bands = linspace(1.0, max_res/2, 16)
encoding = [rays, sin(π*rays*freq), cos(π*rays*freq)]  # (B, 99, H, W)

# Concatenate and project back
combined = concat([image_features, ray_encoding], dim=1)  # (B, 1379, H, W)
output = LayerNorm(Conv1x1(combined))  # (B, 1280, H, W)
```

**Parameters:** ~1.77M (conv + norm)

**Benefits:**
- Camera-specific features (vs camera-agnostic ViT)
- Better depth estimation
- Handles varying focal lengths and principal points

### 3. Prompt Encoder

**Location:** `sam_3d_body/models/decoders/prompt_encoder.py`

**Keypoint Prompts:**
- Input: (B, N_clicks, 3) - normalized coords [x, y] + label
- Labels: -2 (invalid), -1 (negative), 0-69 (joint index)
- Output: (B, N_clicks, 1280) embeddings + validity mask

**Encoding Process:**
```python
# Position encoding via Fourier features
pos_embed = sin_cos_encoding(coords)  # (B, N, 1280)

# Add semantic embedding based on label
if label == -2: embed = invalid_point_embed
elif label == -1: embed = not_a_point_embed
elif 0 <= label < 70: embed = point_embeddings[label]

output = pos_embed + embed
```

**Parameters:**
- 70 × learnable joint embeddings: 89,600 params
- 2 × special embeddings (invalid, negative): 2,560 params
- Total: ~92K params

**Mask Prompts (Optional):**
- Downscaling CNN with LayerNorm2d + GELU
- Variants: v1 (4×4 striding) or v2 (2×2×2×2 striding)
- Output: (B, 1280, H_feat, W_feat) dense embedding
- Parameters: ~400K (v2 variant)

### 4. Token Embeddings

**Location:** `sam_3d_body/models/meta_arch/sam3d_body.py:65-243`

**Initial Tokens:**
```python
init_pose: (1, 519)      # Zero-pose in 6D rot + continuous representation
init_camera: (1, 3)      # Zeros [0, 0, 0]
```

**Projection Layers:**
| Layer | Input | Output | Params | Purpose |
|-------|-------|--------|--------|---------|
| init_to_token_mhr | 525 | 1024 | ~537K | Init pose+cam+cond → token |
| prev_to_token_mhr | 522 | 1024 | ~533K | Prev pose+cam → token |
| prompt_to_token | 1280 | 1024 | 1.31M | Prompt embeddings → token |

**Keypoint Tokens:**
- `keypoint_embedding`: (70, 1024) learnable base embeddings
- `keypoint_feat_linear`: Projects sampled image features (1280→1024)
- `keypoint_posemb_linear`: FFN encodes 2D positions (2→1024)
- `keypoint3d_embedding`: (70, 1024) for 3D keypoint tokens
- `keypoint3d_posemb_linear`: FFN encodes 3D positions (3→1024)
- Total: ~3.6M params

**Token Count:**
- Minimal: 73 (pose + prev + prompt + 70 kp_2d tokens)
- Full: 145 (+ 70 kp_3d tokens + 2 hand detection tokens)

### 5. Promptable Decoder

**Location:** `sam_3d_body/models/decoders/promptable_decoder.py`

**Configuration:**
```yaml
dims: 1024              # Token dimension
context_dims: 1280      # Image feature dimension
depth: 2                # Number of layers
num_heads: 8
head_dims: 64
mlp_dims: 1024
enable_twoway: false    # SAM's bidirectional attention (disabled)
repeat_pe: true         # Add PE at each layer (LaPE)
do_interm_preds: true   # Predict at each layer
keypoint_token_update: true  # Dynamic token updates
```

**Architecture:**

Each TransformerDecoderLayer contains:
1. **Self-Attention**: Tokens attend to each other
2. **Cross-Attention**: Tokens query image features
3. **FFN**: Feed-forward network
4. **LayerNorm**: Before each sub-layer

**Per-layer parameters:** ~6.6M → **Total (depth=2): ~13.2M**

**Dynamic Keypoint Token Update:**

Novel mechanism that updates tokens between layers:

```python
# After decoder layer (except last):
# 1. Make intermediate prediction
pose_output = mhr_head(pose_token, init_estimate)
cam_output = camera_head(pose_token, init_camera)

# 2. Project 3D to 2D keypoints
pred_kps_2d = project(pose_output['pred_keypoints_3d'], cam_output)

# 3. Sample image features at predicted locations
kp_feats = grid_sample(image_embeddings, pred_kps_2d)  # (B, 70, 1280)
kp_feats = keypoint_feat_linear(kp_feats)  # (B, 70, 1024)

# 4. Encode positions
kp_pos = keypoint_posemb_linear(pred_kps_2d)  # (B, 70, 1024)

# 5. Update tokens
keypoint_tokens = learnable_embedding + kp_feats + kp_pos
```

**Benefits:**
- Tokens become spatially grounded to predicted locations
- Gather relevant image context at joint positions
- Enable iterative refinement within single forward pass

### 6. MHR Head (Pose Regression)

**Location:** `sam_3d_body/models/heads/mhr_head.py`

**Network:**
```python
FFN:
  Input: 1024 (pose token)
  Hidden: 128 (1024 / 8)
  Output: 519
  Params: ~200K
```

**Output Dimensions (npose = 519):**

| Component | Indices | Dims | Representation |
|-----------|---------|------|----------------|
| Global rotation | 0:6 | 6 | 6D rotation (first 2 cols of R) |
| Body pose | 6:266 | 260 | Compact continuous (130 joints × 2) |
| Shape | 266:311 | 45 | PCA coefficients |
| Scale | 311:339 | 28 | Skeletal bone length PCA |
| Left hand | 339:393 | 54 | Hand pose PCA |
| Right hand | 393:447 | 54 | Hand pose PCA |
| Face expression | 447:519 | 72 | Expression parameters |

**Rotation Parametrization:**

**6D Rotation** (more stable than axis-angle):
```python
# Network predicts 6 unconstrained values
global_rot_6d = pred[:, 0:6]

# Convert to rotation matrix via Gram-Schmidt
a1, a2 = global_rot_6d[:, :3], global_rot_6d[:, 3:]
b1 = normalize(a1)
b2 = normalize(a2 - (a2·b1)*b1)
b3 = cross(b1, b2)
R = [b1 | b2 | b3]  # (B, 3, 3)

# Convert to Euler for MHR model
global_rot_euler = rotmat_to_euler("ZYX", R)
```

**MHR Model (Non-trainable TorchScript):**
- Inputs: shape (B, 45), model_params (B, 275), expr (B, 72)
- Outputs:
  - skinned_verts: (B, 18439, 3) - mesh vertices
  - joint_coords: (B, 127, 3) - joint positions
  - joint_quats: (B, 127, 4) - joint orientations
  - keypoints: (B, 70, 3) - from Sapiens-308 regressor

**Output Dictionary:**
```python
{
    'pred_pose_raw': (B, 266),          # 6D rot + continuous pose
    'global_rot': (B, 3),               # Euler angles (ZYX)
    'body_pose': (B, 133),              # Joint Eulers
    'shape': (B, 45),                   # Shape PCA
    'scale': (B, 28),                   # Scale PCA
    'hand': (B, 108),                   # Hand PCA (54 L + 54 R)
    'face': (B, 72),                    # Expression (zeroed)
    'pred_keypoints_3d': (B, 70, 3),    # 3D keypoints
    'pred_vertices': (B, 18439, 3),     # Mesh vertices
    'pred_joint_coords': (B, 127, 3),   # Joint positions
    'joint_global_rots': (B, 127, 3, 3) # Joint rotation matrices
}
```

### 7. Camera Head

**Location:** `sam_3d_body/models/heads/camera_head.py`

**Network:**
```python
FFN:
  Input: 1024
  Hidden: 128
  Output: 3  # [scale, tx, ty]
  Params: ~131K
```

**Perspective Projection (CLIFF-style):**
```python
# Predicted parameters
s, tx, ty = pred_cam

# Compute depth from scale
bs = bbox_size * s * default_scale_factor
depth = 2 * focal_length / bs

# Compute translation offsets
dx = 2 * (bbox_center_x - img_w/2) / bs
dy = 2 * (bbox_center_y - img_h/2) / bs

# Final camera translation
cam_trans = [tx + dx, ty + dy, depth]

# Full perspective projection
j3d_cam = j3d + cam_trans.unsqueeze(1)
j2d_x = fx * j3d_cam[:,:,0] / j3d_cam[:,:,2] + cx
j2d_y = fy * j3d_cam[:,:,1] / j3d_cam[:,:,2] + cy
```

**CLIFF Conditioning:**
```python
# Normalized bbox info concatenated with tokens
cx_norm = (bbox_center_x - img_w/2) / focal_length
cy_norm = (bbox_center_y - img_h/2) / focal_length
b_norm = bbox_scale / focal_length
condition_info = [cx_norm, cy_norm, b_norm]  # (B, 3)
```

---

## Architectural Innovations

### 1. Promptable Interaction

**Similar to SAM's interactive refinement:**
```
SAM (Segmentation):
1. Encode image → User clicks → Decoder → Mask
2. User adds correction → Refined mask

SAM3DBody (Pose):
1. Encode image → Decoder → Initial pose
2. User corrects joint location → Refined pose
3. Keypoint tokens update → Further refinement
```

**Key Difference:** SAM3DBody updates keypoint tokens *dynamically during decoding*, not just between user iterations.

### 2. Camera-Aware Features

**Ray Conditioning** (applied *before* transformer):
- Encodes camera rays via Fourier features
- Makes all attention operations camera-aware
- Better than late fusion (CLIFF concatenates after encoding)

**vs Standard Approaches:**
| Method | Camera Handling | When Applied |
|--------|-----------------|--------------|
| HMR/SPIN | ❌ None | N/A |
| CLIFF | ✅ Bbox conditioning | Late (concat to features) |
| CameraHMR | ✅ Extrinsic | Late (MLP) |
| **SAM3DBody** | ✅ **Ray + Bbox** | **Early (before transformer)** |

### 3. Continuous Representations

**6D Rotation** (vs axis-angle):
- ✅ Continuous (any 6D vector → valid rotation)
- ✅ No discontinuities (vs axis-angle at θ=0,2π)
- ✅ Efficient (6 params vs 9 for full matrix)
- ✅ Unique representation

**Compact Continuous Pose** (260D):
- Represents 130 joints in continuous space
- Converted to Euler angles for MHR model
- More stable for gradient-based optimization

### 4. High-Fidelity MHR Model

**vs SMPL:**
| Feature | SMPL | MHR |
|---------|------|-----|
| Vertices | 6,890 | **18,439** (2.7× more) |
| Joints | 24 | **127** (5.3× more) |
| Shape | 10D PCA | **45D** PCA |
| Scale | ❌ Not modeled | ✅ **28D** skeletal scaling |
| Hands | ❌ Fixed | ✅ **54D** per hand |
| Face | ❌ Not modeled | ✅ **72D** expression |
| Keypoints | 24-45 | **70** (Sapiens) |

---

## Data Flow

### Inference Pipeline

**Entry:** `SAM3DBodyEstimator.process_one_image()`

**Steps:**
1. **Preprocessing:**
   - Human detection (optional, ViTDet)
   - Mask generation (optional, SAM2)
   - FOV estimation (optional, MOGE2)
   - Affine transformation to 192×256 crop
   - Normalization

2. **Encoding:**
   - Backbone: image → features (B, 1280, 18, 13)
   - Ray encoder: add camera conditioning
   - Prompt encoder: encode keypoint/mask prompts (if provided)

3. **Token Construction:**
   ```python
   # Initial tokens
   pose_token = init_to_token_mhr([condition_info, init_pose, init_camera])

   # Add prompts (if interactive)
   if keypoints is not None:
       prev_token = prev_to_token_mhr([prev_pose, prev_camera])
       prompt_tokens = prompt_to_token(prompt_encoder(keypoints))
       tokens = concat([pose_token, prev_token, prompt_tokens])

   # Add keypoint query tokens
   tokens = concat([tokens, keypoint_embedding.weight])  # 70 tokens

   # Optional: Add 3D keypoint tokens
   tokens = concat([tokens, keypoint3d_embedding.weight])  # 70 tokens
   ```

4. **Decoder (2 layers):**
   ```python
   for layer in [0, 1]:
       # Self-attention + Cross-attention + FFN
       tokens = decoder_layer(tokens, image_features)

       # Intermediate prediction (layer 0 only)
       if layer == 0:
           pose_output = mhr_head(tokens[:,0])
           cam_output = camera_head(tokens[:,0])

           # Update keypoint tokens
           pred_kps_2d = project(pose_output, cam_output)
           tokens[:, 3:73] = update_kp_tokens(pred_kps_2d)
   ```

5. **Prediction:**
   ```python
   # Final tokens
   pose_token_final = tokens[:, 0]

   # Regress parameters
   pose_params = mhr_head(pose_token_final)  # (B, 519)
   cam_params = camera_head(pose_token_final)  # (B, 3)

   # Generate mesh
   vertices, keypoints_3d, joints = mhr_model(
       pose_params['shape'],
       pose_params['model_params'],
       pose_params['face']
   )

   # Project to 2D
   keypoints_2d = perspective_project(keypoints_3d, cam_params)
   ```

6. **Hand Refinement (full mode):**
   ```python
   # Extract hand boxes from body prediction
   left_box, right_box = get_hand_boxes(pose_output)

   # Flip left hand (to make it right-hand-like)
   left_img = flip_horizontal(crop(img, left_box))

   # Run hand decoder on each hand
   left_hand_pose = hand_decoder(left_img)
   right_hand_pose = hand_decoder(crop(img, right_box))

   # Merge with body
   final_pose['hand'][:,:54] = left_hand_pose['hand'][:,54:]  # Left
   final_pose['hand'][:,54:] = right_hand_pose['hand'][:,54:]  # Right
   ```

### Inference Modes

| Mode | Description | Speed | Use Case |
|------|-------------|-------|----------|
| **body** | Body decoder only | Fast (22 FPS) | Quick pose estimation |
| **hand** | Hand decoder only | Fast | Hand-specific tasks |
| **full** | Body + 2× hand decoders | Slow (8 FPS) | High-quality full-body |

---

## Model Building

**Location:** `sam_3d_body/build_models.py`

**Loading:**
```python
# From local checkpoint
model, cfg = load_sam_3d_body(
    checkpoint_path="path/to/model.ckpt",
    device="cuda",
    mhr_path="path/to/mhr_model.pt"
)

# From HuggingFace
model, cfg = load_sam_3d_body_hf(
    repo_id="facebook/sam-3d-body-dinov3"
)
```

**Configuration:** YAML files define:
- Backbone type and settings
- Decoder depth and dimensions
- Head configurations
- Training settings (FP16, frozen stages)

---

## Parameter Count & Computational Cost

### Parameter Breakdown

```
Component                      Parameters
─────────────────────────────  ───────────
Backbone (ViT-L)               ~300.0 M
Ray Condition Encoder            1.8 M
Prompt Encoder                   0.5 M
  ├─ Keypoint embeddings         0.09 M
  └─ Mask encoder (optional)     0.4 M
Token Embeddings                 7.2 M
  ├─ Init embeddings             0.001 M
  ├─ Linear projections          3.5 M
  ├─ Keypoint tokens             3.6 M
  └─ Hand detect (optional)      0.1 M
Promptable Decoder (×2)         13.2 M
  ├─ Layer 1                     6.6 M
  └─ Layer 2                     6.6 M
MHR Head                         0.2 M
Camera Head                      0.13 M
─────────────────────────────  ───────────
Body Decoder Total              322.0 M
Hand Decoder Total              ~15.0 M
─────────────────────────────  ───────────
GRAND TOTAL (trainable)         337.0 M

Non-trainable:
  MHR Model (TorchScript)       ~100-200 MB
```

### Computational Cost

```
Component                      GFLOPs
─────────────────────────────  ────────
Backbone (ViT-L)               ~15.0
Ray Encoding                     0.5
Prompt Encoder                   0.1
Token Projections                0.2
Decoder (2 layers)               2.2
  ├─ Self-attention              0.6
  ├─ Cross-attention             1.0
  └─ FFN                         0.6
MHR Head                         0.3
Camera Head                      0.1
MHR Model                        2.0
Keypoint Token Update            0.5
─────────────────────────────  ────────
Total (single person)          ~20.9
Full inference (body+2×hand)   ~45.0
```

---

## File Organization

```
sam_3d_body/
├── models/
│   ├── backbones/              # ViT, DINOv3 encoders
│   │   ├── vit.py
│   │   └── dinov3.py
│   ├── decoders/               # Promptable decoder, prompt encoder
│   │   ├── promptable_decoder.py
│   │   ├── prompt_encoder.py
│   │   └── keypoint_prompt_sampler.py
│   ├── heads/                  # Regression heads
│   │   ├── mhr_head.py         # Pose/shape/hand parameters
│   │   └── camera_head.py      # Camera translation
│   ├── meta_arch/              # Main model
│   │   ├── sam3d_body.py       # SAM3DBody class
│   │   └── base_model.py
│   ├── modules/                # Utilities
│   │   ├── transformer.py      # Attention layers
│   │   ├── camera_embed.py     # Ray conditioning
│   │   ├── geometry_utils.py   # Projection, rotations
│   │   └── mhr_utils.py        # MHR conversions
│   └── optim/                  # FP16 utilities
├── data/
│   ├── transforms/             # Image preprocessing
│   └── utils/                  # Batch preparation, I/O
├── utils/                      # Config, checkpoint, logging
├── visualization/              # Rendering utilities
├── build_models.py             # Model loading functions
└── sam_3d_body_estimator.py   # High-level inference wrapper

demo.py                         # Demo script
tools/
├── build_detector.py           # Human detector (ViTDet)
├── build_sam.py                # Segmentor (SAM2)
└── build_fov_estimator.py      # FOV estimator (MOGE2)
```

---

## Comparison with Baselines

### Architecture Comparison

| Component | HMR/SPIN | CLIFF | TokenPose | **SAM3DBody** |
|-----------|----------|-------|-----------|---------------|
| **Backbone** | ResNet-50 | ViT | ViT | **ViT-L** |
| **Camera conditioning** | ❌ | ✅ Bbox | ✅ Bbox | ✅ **Bbox + Ray** |
| **Decoder** | MLP | MLP | Cross-Attn | **SAM Decoder** |
| **Iterative refinement** | ✅ IEF | ❌ | ❌ | ✅ **Per-layer** |
| **Promptable** | ❌ | ❌ | ❌ | ✅ **Yes** |
| **Keypoint tokens** | ❌ | ❌ | ✅ Static | ✅ **Dynamic** |
| **Body model** | SMPL | SMPL | SMPL | **MHR** |
| **Pose params** | 72D axis-angle | 72D | 72D | **266D** (6D+cont) |
| **Hand modeling** | ❌ | Basic | Basic | ✅ **Dual decoder** |

### Key Advantages

1. **Promptable**: First interactive 3D pose estimation system
2. **Dynamic tokens**: Tokens update based on predictions during decoding
3. **Camera-aware**: Ray conditioning applied early in pipeline
4. **High-fidelity**: MHR model with 2.7× more vertices than SMPL
5. **Stable optimization**: 6D rotation + continuous pose representation

### Trade-offs

| Advantage | Limitation |
|-----------|------------|
| ✅ Interactive refinement | ⚠️ Large model (337M params) |
| ✅ Detailed mesh (18K verts) | ⚠️ Slower inference (8 FPS full) |
| ✅ Camera-aware features | ⚠️ Requires MHR model (not SMPL) |
| ✅ Separate hand decoder | ⚠️ More complex pipeline |

---

## Summary

SAM 3D Body successfully adapts the Segment Anything Model's promptable architecture to 3D human pose and mesh estimation. The model combines:

- **ViT-L backbone** for robust feature extraction
- **Ray-conditioned encoding** for camera awareness
- **SAM-style promptable decoder** with dynamic keypoint tokens
- **MHR parametric model** for high-fidelity output
- **Dual decoder system** for body and hand specialization

The architecture balances capability (promptable interaction, detailed mesh) with efficiency (lightweight decoder, shared backbone), making it suitable for both automatic pose estimation and interactive refinement applications.

**Best use cases:** Applications requiring high accuracy and user refinement (e.g., VFX, animation, AR/VR, motion capture).
