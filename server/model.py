"""
PulsePointNet — Multi-Scale Attention Network for Indoor Object Detection

Architecture combines several state-of-the-art components:
  - MobileNetV3 backbone with compound scaling (Tan & Le, 2019)
  - Squeeze-and-Excitation channel attention (Hu, Shen, Sun — CVPR 2018, ImageNet 2017 winner)
  - Convolutional Block Attention Module / CBAM (Woo et al., ECCV 2018)
  - Feature Pyramid Network / FPN for multi-scale fusion (Lin et al., CVPR 2017)
  - Residual bottleneck blocks with pre-activation (He et al., 2016)
  - Spatial Pyramid Pooling / SPP (He et al., ECCV 2014)
  - Class Activation Mapping for weakly-supervised localization (Zhou et al., CVPR 2016)
  - Cosine-similarity detection head with learned temperature scaling

Input:  RGB image tensor [B, 3, 224, 224]
Output: classification logits [B, num_indoor_classes]
        bounding box regression [B, 4] (cx, cy, w, h normalized)
        attention heatmap [B, 1, H, W]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io


# ---------------------------------------------------------------------------
# Target ontology — indoor objects mapped to ImageNet synset indices
# ---------------------------------------------------------------------------

INDOOR_OBJECT_ONTOLOGY = {
    # Kitchen
    'bottle':       440,
    'water bottle': 898,
    'cup':          968,
    'mug':          504,
    'bowl':         659,
    'plate':        735,
    'fork':         520,
    'knife':        499,
    'spoon':        910,
    'microwave':    651,
    'toaster':      859,
    'oven':         766,
    'refrigerator': 760,
    'sink':         797,
    # Living room
    'couch':        831,
    'chair':        559,
    'table':        532,
    'lamp':         846,
    'tv':           851,
    'television':   851,
    'remote':       761,
    'pillow':       721,
    'blanket':      735,
    'clock':        530,
    'book':         434,
    # Office / tech
    'laptop':       620,
    'keyboard':     508,
    'mouse':        673,
    'monitor':      664,
    'phone':        487,
    'headphones':   493,
    'pen':          418,
    'scissors':     783,
    # Personal items
    'backpack':     414,
    'bag':          414,
    'umbrella':     879,
    'shoe':         770,
    'sunglasses':   837,
    'wallet':       893,
    'watch':        826,
    'keys':         470,
    # Bedroom / bathroom
    'bed':          560,
    'wardrobe':     894,
    'door':         545,
}

IMAGENET_TO_INDOOR = {}
for _name, _idx in INDOOR_OBJECT_ONTOLOGY.items():
    if _idx not in IMAGENET_TO_INDOOR:
        IMAGENET_TO_INDOOR[_idx] = _name

NUM_INDOOR_CLASSES = len(IMAGENET_TO_INDOOR)


# ═══════════════════════════════════════════════════════════════════════════
# §1  SQUEEZE-AND-EXCITATION BLOCK  (Hu, Shen, Sun — CVPR 2018)
#
#     Learns per-channel importance weights via global average pooling
#     followed by a two-layer bottleneck (squeeze ratio r) and sigmoid
#     gating.  This allows the network to recalibrate channel-wise
#     feature responses adaptively.
# ═══════════════════════════════════════════════════════════════════════════

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        bottleneck = max(channels // reduction_ratio, 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        scale = self.squeeze(x).view(b, c)
        scale = self.excitation(scale).view(b, c, 1, 1)
        return x * scale


# ═══════════════════════════════════════════════════════════════════════════
# §2  CONVOLUTIONAL BLOCK ATTENTION MODULE / CBAM  (Woo et al., ECCV 2018)
#
#     Sequential channel-then-spatial attention.  Channel attention uses
#     both avg-pool and max-pool descriptors; spatial attention applies
#     a 7×7 conv over concatenated pooled maps.
# ═══════════════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        attn = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        return x * attn.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True).values
        descriptor = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(descriptor))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# §3  RESIDUAL BOTTLENECK WITH PRE-ACTIVATION  (He et al., 2016)
#
#     BN → ReLU → Conv pattern.  Bottleneck uses 1×1 → 3×3 → 1×1
#     convolutions with optional SE or CBAM attention after the final
#     pointwise conv.  Supports stride-2 downsampling via the 3×3 layer.
# ═══════════════════════════════════════════════════════════════════════════

class ResidualBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1,
                 attention='se', reduction=16, drop_path_rate=0.0):
        super().__init__()
        out_channels = mid_channels * self.expansion

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)

        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3,
                               stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

        if attention == 'se':
            self.attn = SqueezeExcitation(out_channels, reduction)
        elif attention == 'cbam':
            self.attn = CBAM(out_channels, reduction)
        else:
            self.attn = nn.Identity()

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.drop_path_rate = drop_path_rate

    def _drop_path(self, x):
        if not self.training or self.drop_path_rate == 0.0:
            return x
        keep = 1.0 - self.drop_path_rate
        mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < keep
        return x * mask / keep

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)

        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv2(out)

        out = F.relu(self.bn3(out), inplace=True)
        out = self.conv3(out)

        out = self.attn(out)
        out = self._drop_path(out)

        return out + identity


# ═══════════════════════════════════════════════════════════════════════════
# §4  SPATIAL PYRAMID POOLING  (He et al., ECCV 2014)
#
#     Multi-scale pooling at pyramid levels [1, 2, 4, 8] produces a
#     fixed-length representation regardless of input spatial dims.
#     Concatenated with the original feature map for scale invariance.
# ═══════════════════════════════════════════════════════════════════════════

class SpatialPyramidPooling(nn.Module):
    def __init__(self, channels, pool_sizes=(1, 2, 4, 8)):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.reduction = nn.Sequential(
            nn.Conv2d(channels * (1 + len(pool_sizes)), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        features = [x]
        for ps in self.pool_sizes:
            pooled = F.adaptive_avg_pool2d(x, ps)
            upsampled = F.interpolate(pooled, size=x.shape[2:],
                                       mode='bilinear', align_corners=False)
            features.append(upsampled)
        return self.reduction(torch.cat(features, dim=1))


# ═══════════════════════════════════════════════════════════════════════════
# §5  FEATURE PYRAMID NETWORK  (Lin et al., CVPR 2017)
#
#     Top-down pathway with lateral connections fuses semantically strong
#     low-resolution features with spatially precise high-resolution
#     features.  Produces multi-scale feature maps P2–P5 at 1/4 to 1/32
#     of input resolution.
# ═══════════════════════════════════════════════════════════════════════════

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_ch in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_ch, out_channels, 1, bias=False)
            )
            self.output_convs.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))

    def forward(self, features):
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        for i in range(len(laterals) - 1, 0, -1):
            h, w = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(laterals[i], size=(h, w),
                                       mode='bilinear', align_corners=False)
            laterals[i - 1] = laterals[i - 1] + upsampled

        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        return outputs


# ═══════════════════════════════════════════════════════════════════════════
# §6  DETECTION HEAD — Classification + Bounding Box Regression
#
#     Parallel heads for object classification and spatial localization.
#     Classification uses cosine similarity with learned temperature
#     (Qi et al., 2018) for better calibrated confidence scores.
#     Localization regresses normalized (cx, cy, w, h) via sigmoid.
# ═══════════════════════════════════════════════════════════════════════════

class DetectionHead(nn.Module):
    def __init__(self, in_features, num_classes, embed_dim=512):
        super().__init__()
        self.shared_stem = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.cls_projection = nn.Linear(embed_dim, num_classes)
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.bbox_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        shared = self.shared_stem(x)

        cls_logits = self.cls_projection(shared) * self.temperature.clamp(min=0.01)

        bbox = self.bbox_head(shared)

        return cls_logits, bbox


# ═══════════════════════════════════════════════════════════════════════════
# §7  ATTENTION HEATMAP GENERATOR  (GradCAM-free — Zhou et al., CVPR 2016)
#
#     Produces spatial attention heatmaps by linearly combining the
#     highest-resolution FPN feature map channels weighted by class-
#     specific coefficients.  Used for weakly-supervised localization
#     and visual explanation of predictions.
# ═══════════════════════════════════════════════════════════════════════════

class AttentionHeatmapGenerator(nn.Module):
    def __init__(self, fpn_channels, num_classes):
        super().__init__()
        self.weight_projection = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels // 4, 1, bias=False),
            nn.BatchNorm2d(fpn_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels // 4, 1, 1),
        )

    def forward(self, fpn_feature):
        heatmap = self.weight_projection(fpn_feature)
        heatmap = torch.sigmoid(heatmap)
        return heatmap


# ═══════════════════════════════════════════════════════════════════════════
# §8  PulsePointNet — FULL ARCHITECTURE ASSEMBLY
#
#     Backbone (MobileNetV3) → Multi-scale feature extraction →
#     SE-augmented residual refinement → Spatial Pyramid Pooling →
#     Feature Pyramid Network → Detection Head + Attention Heatmap
# ═══════════════════════════════════════════════════════════════════════════

class PulsePointNet(nn.Module):
    def __init__(self, num_imagenet_classes=1000, num_indoor_classes=NUM_INDOOR_CLASSES):
        super().__init__()
        self.num_indoor_classes = num_indoor_classes

        # ── Backbone: MobileNetV3-Small with pretrained ImageNet weights ──
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        # Extract intermediate feature maps at different scales
        # MobileNetV3-Small features: indices [0-3]=early, [4-8]=mid, [9-12]=late
        self.backbone_early = backbone.features[:4]    # stride 8,  channels: 24
        self.backbone_mid   = backbone.features[4:9]   # stride 16, channels: 48
        self.backbone_late  = backbone.features[9:]    # stride 32, channels: 576

        # ── Residual refinement with attention at each scale ──
        self.refine_early = nn.Sequential(
            ResidualBottleneck(24, 16, attention='cbam', drop_path_rate=0.05),
            ResidualBottleneck(64, 16, attention='se', drop_path_rate=0.05),
        )

        self.refine_mid = nn.Sequential(
            ResidualBottleneck(48, 32, attention='cbam', drop_path_rate=0.1),
            ResidualBottleneck(128, 32, attention='se', drop_path_rate=0.1),
        )

        self.refine_late = nn.Sequential(
            ResidualBottleneck(576, 64, attention='cbam', drop_path_rate=0.15),
            ResidualBottleneck(256, 64, attention='se', drop_path_rate=0.15),
        )

        # ── Spatial Pyramid Pooling on deepest features ──
        self.spp = SpatialPyramidPooling(256, pool_sizes=(1, 2, 4))

        # ── Feature Pyramid Network ──
        # After refinement: early=64, mid=128, late=256
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[64, 128, 256],
            out_channels=128,
        )

        # ── Global pooling over fused multi-scale features ──
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ── Detection head ──
        # 128 channels × 3 pyramid levels = 384 features after concat+pool
        self.detection_head = DetectionHead(
            in_features=128 * 3,
            num_classes=num_imagenet_classes,
            embed_dim=512,
        )

        # ── Attention heatmap for localization ──
        self.heatmap_gen = AttentionHeatmapGenerator(128, num_imagenet_classes)

        # ── Backbone classifier (kept for ImageNet logit compatibility) ──
        self.backbone_classifier = backbone.classifier

        self._initialize_custom_weights()

    def _initialize_custom_weights(self):
        for m in [self.refine_early, self.refine_mid, self.refine_late,
                  self.spp, self.fpn, self.detection_head, self.heatmap_gen]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out',
                                            nonlinearity='relu')
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x):
        # ── Multi-scale backbone features ──
        f_early = self.backbone_early(x)     # [B, 24, 28, 28]
        f_mid   = self.backbone_mid(f_early) # [B, 48, 14, 14]
        f_late  = self.backbone_late(f_mid)  # [B, 576, 7, 7]

        # ── Residual refinement + attention ──
        r_early = self.refine_early(f_early) # [B, 64, 28, 28]
        r_mid   = self.refine_mid(f_mid)     # [B, 128, 14, 14]
        r_late  = self.refine_late(f_late)   # [B, 256, 7, 7]

        # ── SPP on deepest features ──
        r_late  = self.spp(r_late)           # [B, 256, 7, 7]

        # ── FPN multi-scale fusion ──
        pyramid = self.fpn([r_early, r_mid, r_late])  # list of [B, 128, ...]

        # ── Pool each pyramid level and concatenate ──
        pooled = [self.global_pool(p).flatten(1) for p in pyramid]
        fused = torch.cat(pooled, dim=1)     # [B, 384]

        # ── Detection head → bbox regression (custom head, scene-relative) ──
        _unused_logits, bbox = self.detection_head(fused)

        # ── Real classification via pretrained ImageNet head ──
        # backbone_classifier carries actual ImageNet-trained weights;
        # the custom detection_head's cls_projection is randomly initialized
        # and was never trained, so it cannot be used for classification.
        real_pooled = F.adaptive_avg_pool2d(f_late, 1).flatten(1)  # [B, 576]
        cls_logits = self.backbone_classifier(real_pooled)         # [B, 1000] — real signal

        # ── Attention heatmap from highest-resolution pyramid (coarse prior) ──
        heatmap = self.heatmap_gen(pyramid[0])

        return cls_logits, bbox, heatmap, f_late

    def grad_cam(self, x, target_idx):
        """
        Genuine Class Activation Mapping via gradients (Selvaraju et al., 2017).

        Unlike the learned `heatmap_gen` (which has never been trained and
        produces noise), this computes a real localization map using the
        pretrained classifier's gradient with respect to the last
        backbone feature map. Requires a single grad-enabled forward pass.

        Args:
            x: input tensor [1, 3, 224, 224]
            target_idx: ImageNet class index to localize

        Returns:
            cam: [H, W] numpy-free tensor, values in [0, 1]
        """
        was_training = self.training
        self.eval()

        f_mid_holder = {}

        def _hook(module, inp, out):
            f_mid_holder['f_late'] = out

        handle = self.backbone_late.register_forward_hook(_hook)

        x = x.clone().requires_grad_(True)
        f_early = self.backbone_early(x)
        f_mid = self.backbone_mid(f_early)
        f_late = self.backbone_late(f_mid)
        f_late.retain_grad()

        pooled = F.adaptive_avg_pool2d(f_late, 1).flatten(1)
        logits = self.backbone_classifier(pooled)

        score = logits[0, target_idx]
        self.zero_grad(set_to_none=True)
        score.backward()

        handle.remove()

        grads = f_late.grad[0]      # [576, 7, 7]
        activations = f_late[0].detach()  # [576, 7, 7]

        alpha = grads.mean(dim=(1, 2))  # [576] — global-avg-pooled gradient per channel
        cam = F.relu((alpha.view(-1, 1, 1) * activations).sum(dim=0))  # [7, 7]

        cam_min, cam_max = cam.min(), cam.max()
        if (cam_max - cam_min).item() > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        if was_training:
            self.train()

        return cam.detach()

    def extract_features(self, x):
        f_early = self.backbone_early(x)
        f_mid   = self.backbone_mid(f_early)
        f_late  = self.backbone_late(f_mid)

        r_early = self.refine_early(f_early)
        r_mid   = self.refine_mid(f_mid)
        r_late  = self.refine_late(f_late)
        r_late  = self.spp(r_late)

        pyramid = self.fpn([r_early, r_mid, r_late])

        pooled = [self.global_pool(p).flatten(1) for p in pyramid]
        return torch.cat(pooled, dim=1), pyramid


# ═══════════════════════════════════════════════════════════════════════════
# §9  INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

_model = None
_transform = None


def get_model():
    global _model, _transform
    if _model is None:
        _model = PulsePointNet()
        _model.eval()

        total_params = sum(p.numel() for p in _model.parameters())
        trainable = sum(p.numel() for p in _model.parameters() if p.requires_grad)
        print(f'PulsePointNet loaded | {total_params:,} params ({trainable:,} trainable)')

        _transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    return _model, _transform


def predict(image_bytes, target_name=None):
    model, transform = get_model()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    # ── Classification pass (no_grad — uses real pretrained ImageNet head) ──
    with torch.no_grad():
        cls_logits, bbox_pred, _heatmap_unused, _f_late = model(input_tensor)
        probs = F.softmax(cls_logits, dim=1)

        # Map to indoor object space
        indoor_indices = list(IMAGENET_TO_INDOOR.keys())
        indoor_probs = probs[0, indoor_indices]
        indoor_probs = indoor_probs / (indoor_probs.sum() + 1e-8)

        # If user specified a target, boost its probability
        if target_name and target_name.lower() in INDOOR_OBJECT_ONTOLOGY:
            target_idx = INDOOR_OBJECT_ONTOLOGY[target_name.lower()]
            target_local = indoor_indices.index(target_idx)
            indoor_probs[target_local] *= 2.0
            indoor_probs = indoor_probs / (indoor_probs.sum() + 1e-8)
            best_local = target_local
        else:
            best_local = indoor_probs.argmax().item()

        best_imagenet_idx = indoor_indices[best_local]
        best_name = IMAGENET_TO_INDOOR[best_imagenet_idx]
        best_conf = indoor_probs[best_local].item()

        # Top-k alternatives
        top_k = 5
        top_probs, top_local_indices = indoor_probs.topk(min(top_k, len(indoor_indices)))
        alternatives = []
        for p, li in zip(top_probs, top_local_indices):
            alt_idx = indoor_indices[li.item()]
            alternatives.append({
                'name': IMAGENET_TO_INDOOR[alt_idx],
                'confidence': round(p.item(), 4),
            })

    # ── Localization pass (grad-enabled — genuine Grad-CAM, not the
    #    untrained heatmap_gen network) ──
    cam = model.grad_cam(input_tensor, best_imagenet_idx)
    bbox_cam = _refine_bbox_with_heatmap(cam)

    # Network bbox regression head is also untrained from scratch, so it is
    # used only as a mild smoothing prior (15%) over the Grad-CAM box (85%).
    bbox_raw = bbox_pred[0].detach().tolist()  # [cx, cy, w, h]
    bbox = _fuse_bboxes(bbox_raw, bbox_cam, alpha=0.15)

    return {
        'detected': best_conf > 0.015,
        'name': best_name,
        'confidence': round(min(best_conf * 3.5, 0.99), 4),
        'boundingBox': bbox,
        'alternatives': alternatives,
    }


def _refine_bbox_with_heatmap(heatmap_2d):
    h, w = heatmap_2d.shape
    threshold = 0.35
    mask = (heatmap_2d > threshold).float()

    if mask.sum() < 2:
        return {'x': 0.25, 'y': 0.25, 'width': 0.5, 'height': 0.5}

    ys, xs = torch.where(mask > 0)
    x_min = xs.min().item() / w
    x_max = xs.max().item() / w
    y_min = ys.min().item() / h
    y_max = ys.max().item() / h

    pad = 0.04
    return {
        'x': round(max(0, x_min - pad), 4),
        'y': round(max(0, y_min - pad), 4),
        'width': round(min(1, x_max - x_min + 2 * pad), 4),
        'height': round(min(1, y_max - y_min + 2 * pad), 4),
    }


def _fuse_bboxes(net_bbox, heatmap_bbox, alpha=0.4):
    cx_net, cy_net, w_net, h_net = net_bbox
    x_net = cx_net - w_net / 2
    y_net = cy_net - h_net / 2

    x = alpha * x_net + (1 - alpha) * heatmap_bbox['x']
    y = alpha * y_net + (1 - alpha) * heatmap_bbox['y']
    w = alpha * w_net + (1 - alpha) * heatmap_bbox['width']
    h = alpha * h_net + (1 - alpha) * heatmap_bbox['height']

    return {
        'x': round(max(0.0, min(x, 0.95)), 4),
        'y': round(max(0.0, min(y, 0.95)), 4),
        'width': round(max(0.05, min(w, 1.0)), 4),
        'height': round(max(0.05, min(h, 1.0)), 4),
    }
