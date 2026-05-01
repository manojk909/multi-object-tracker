# Technical Report: Multi-Object Detection & Tracking Pipeline

**Project**: Production-grade Multi-Object Detection & Tracking  
**Version**: 1.0.0

---

## 1. System Overview

This pipeline performs real-time multi-object detection and tracking on video input. It is designed for robust performance across challenging real-world scenarios including occlusion, motion blur, scale variation, camera movement, and similarly-appearing subjects.

---

## 2. Detector: YOLOv8 (Ultralytics)

### 2.1 Architecture

YOLOv8 is a single-stage anchor-free detector based on CSPDarknet backbone with a PAN neck and decoupled detection head. It predicts bounding boxes and class probabilities in a single forward pass, making it extremely fast compared to two-stage detectors.

Key improvements over predecessors:
- Anchor-free predictions eliminate grid-size dependency
- Decoupled head separates classification and regression, improving gradient flow
- C2f bottleneck replaces C3 for richer gradient information

### 2.2 Model Variants

| Variant | Params | mAP@50-95 | Use Case |
|---------|--------|-----------|----------|
| n (nano) | 3.2M | 37.3 | Edge / real-time |
| s (small) | 11.2M | 44.9 | Balanced |
| m (medium) | 25.9M | 50.2 | Quality |
| l (large) | 43.7M | 52.9 | High accuracy |
| x (extra) | 68.2M | 53.9 | Maximum accuracy |

---

## 3. Tracker: ByteTrack

### 3.1 Algorithm

ByteTrack (Zhang et al., 2022) uses EVERY detection box in association, including low-confidence ones. Two-stage association:

1. High-confidence detections matched to active tracks via IoU + Hungarian algorithm
2. Low-confidence detections used to recover occluded tracks in a second pass

This is ByteTrack's key innovation: low-confidence detections that would normally be discarded still recover partially occluded tracks, dramatically reducing ID switches.

### 3.2 Kalman Filter

ByteTrack uses a constant-velocity Kalman filter to predict object positions when detections are missing, bridging short gaps from occlusion or motion blur.

State vector: [cx, cy, aspect_ratio, height, vx, vy, va, vh]

---

## 4. Speed vs. Accuracy Tradeoff

| Config | Speed (CPU) | Speed (GPU) | Re-ID | Best For |
|--------|-------------|-------------|-------|----------|
| YOLOv8n + ByteTrack | ~12 FPS | ~85 FPS | Moderate | Most use cases |
| YOLOv8s + ByteTrack | ~8 FPS | ~60 FPS | Moderate | Better accuracy |
| YOLOv8m + DeepSORT | ~4 FPS | ~40 FPS | Strong | Heavy occlusion |

Rationale: ByteTrack with YOLOv8n provides near-real-time performance on consumer hardware without requiring appearance feature extraction. For most scenarios (surveillance, sports, traffic), IoU-based tracking suffices.

---

## 5. ID Consistency Strategy

Four mechanisms maintain stable IDs:

1. **Kalman Prediction**: Predicts location when detection is missing; predicted box joins matching
2. **Two-stage Byte Association**: Low-confidence detections recover occluded tracks
3. **Track Buffer**: Lost tracks kept alive for N frames; ID restored if object reappears
4. **Color Consistency**: Golden-ratio hue-stepping maps each ID to a unique perceptual color

---

## 6. Challenges

### Occlusion
Handled via two-stage ByteTrack association and Kalman bridging. The `track_buffer` prevents premature deletion.

### Similar Appearance
ByteTrack relies on spatial IoU rather than appearance, robust when objects move predictably. For severe cases, DeepSORT with appearance embeddings is recommended.

### Motion Blur
Lowering confidence threshold captures blurred detections. ByteTrack's low-confidence stage specifically addresses this.

### Scale Variation
YOLO's FPN multi-scale head handles this natively. Increasing `--img-size` helps detect small distant objects.

### Camera Movement
Kalman velocity prediction adapts to smooth camera motion. For extreme movement, affine motion compensation can be added as a pre-processing stage.

---

## 7. Known Failure Cases

1. **Long re-appearances**: Objects missing >track_buffer frames receive new IDs on return
2. **Extreme density**: 50+ overlapping objects cause sub-optimal Hungarian matching
3. **Stationary objects**: Without motion, Kalman predictions diverge from actual position
4. **Very small objects**: Sub-8px objects after YOLO resize are rarely detected reliably
5. **Fisheye / 360° cameras**: Radial distortion breaks rectangular bounding box assumptions

---

## 8. Future Improvements

### 8.1 Appearance Re-Identification
Integrate OSNet, Fast-ReID, or ViT-based re-ID to recover original IDs after long disappearances.

### 8.2 Pose Estimation
Run YOLOv8-pose on tracked persons for skeleton overlay, activity recognition, and discriminative features.

### 8.3 Multi-Camera Tracking
Cross-camera re-ID with shared appearance embeddings and homography calibration for unified IDs across camera networks.

### 8.4 StrongSORT / OC-SORT
Adopt StrongSORT (EMA appearance updates) or OC-SORT (Observation-Centric SORT) for non-linear motion scenarios.

### 8.5 Camera Motion Compensation
Sparse optical flow (SIFT/ORB) to estimate global affine motion between frames, then compensate Kalman predictions accordingly.

### 8.6 Real-time Streaming
Add RTSP/webcam source support with async frame producer thread decoupled from inference consumer thread.

### 8.7 Model Quantization
INT8 quantization via TensorRT or ONNX Runtime for 2-4x speedup on edge GPUs with minimal accuracy loss.

---

## 9. References

1. Jocher, G. et al. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
2. Zhang, Y. et al. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. ECCV 2022.
3. Wojke, N., Bewley, A., Paulus, D. (2017). Simple Online and Realtime Tracking with a Deep Association Metric. ICIP 2017.
4. Du, Y. et al. (2023). StrongSORT: Make DeepSORT Great Again. IEEE TCSVT.
5. Cao, J. et al. (2023). Observation-Centric SORT. CVPR 2023.
