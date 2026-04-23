import warnings
from typing import Dict, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
#  Preprocessing helpers
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_XCEPTION_MEAN = [0.5, 0.5, 0.5]
_XCEPTION_STD  = [0.5, 0.5, 0.5]


def _build_transform(model, img_size=(224, 224)):
    name = type(model).__name__.lower()

    for module in model.modules():
        if isinstance(module, nn.Linear):
            n = module.in_features
            side = int((n / 3) ** 0.5)
            if side * side * 3 == n:
                img_size = (side, side)
            break

    if "xception" in name:
        norm = transforms.Normalize(_XCEPTION_MEAN, _XCEPTION_STD)
    else:
        norm = transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)

    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        norm,
    ])


def _load_image(
    source: Union[str, torch.Tensor, np.ndarray, Image.Image],
    model: nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Returns
    -------
    tensor   : (1, C, H, W) preprocessed tensor on *device*
    img_rgb  : (H, W, 3) uint8 numpy array for display
    """
    if isinstance(source, torch.Tensor):
        # Already a tensor — assume it's preprocessed; just ensure batch dim
        tensor = source.unsqueeze(0) if source.dim() == 3 else source
        tensor = tensor.to(device)
        # Build a displayable image by undoing imagenet normalisation
        t = tensor.squeeze(0).cpu()
        mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(_IMAGENET_STD ).view(3, 1, 1)
        img_rgb = ((t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return tensor, img_rgb

    if isinstance(source, np.ndarray):
        pil = Image.fromarray(source).convert("RGB")
    elif isinstance(source, str):
        pil = Image.open(source).convert("RGB")
    elif isinstance(source, Image.Image):
        pil = source.convert("RGB")
    else:
        raise TypeError(f"Unsupported image source type: {type(source)}")

    img_rgb = np.array(pil.resize((224, 224)))
    tf      = _build_transform(model)
    tensor  = tf(pil).unsqueeze(0).to(device)
    return tensor, img_rgb


# ---------------------------------------------------------------------------
#  Target-layer auto-detection
# ---------------------------------------------------------------------------

def _is_vit(model: nn.Module) -> bool:
    """True if the model contains transformer encoder layers (ViT family)."""
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            return True
    return False


def get_target_layer(model: nn.Module) -> Optional[nn.Module]:
    """
    Heuristically returns the best conv layer for Grad-CAM.
    Returns None for ViT (use Attention Rollout instead).

    Priority order per architecture:
      ResNet   → layer4 (last residual block)
      Xception → last SeparableConv / Conv2d in the exit flow
      EfficientNet → features[-1] block
      Custom NN    → last Conv2d found in the model
    """
    if _is_vit(model):
        return None  # signal to use AttentionRollout

    name = type(model).__name__.lower()

    # ── ResNet ──────────────────────────────────────────────────────────────
    # Look for a named sub-module called "layer4"
    for attr in ("layer4", "model.layer4", "resnet.layer4"):
        layer = _get_nested_attr(model, attr)
        if layer is not None:
            return layer

    # ── EfficientNet ────────────────────────────────────────────────────────
    for attr in ("features", "model.features", "efficientnet.features"):
        features = _get_nested_attr(model, attr)
        if features is not None:
            return features[-1]

    # ── Xception / any CNN ──────────────────────────────────────────────────
    # Fall back: return the last Conv2d layer (wrapped in its parent module)
    last_conv_parent = _find_last_conv_parent(model)
    if last_conv_parent is not None:
        return last_conv_parent

    warnings.warn(
        "Could not auto-detect a target layer. "
        "Pass `target_layer=...` explicitly to `visualize_gradcam`."
    )
    return None


def _get_nested_attr(obj, dotted_path: str):
    parts = dotted_path.split(".")
    try:
        for p in parts:
            obj = getattr(obj, p)
        return obj
    except AttributeError:
        return None


def _find_last_conv_parent(model: nn.Module) -> Optional[nn.Module]:
    """Walk named children and return the deepest parent that contains a Conv2d."""
    last = None
    for name, module in model.named_modules():
        for child in module.children():
            if isinstance(child, nn.Conv2d):
                last = module  # keep updating — we want the *last* one
    return last


# ---------------------------------------------------------------------------
#  Grad-CAM (CNN / NN)
# ---------------------------------------------------------------------------

class _GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._features    = None
        self._gradients   = None
        self._handles     = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd(module, inp, out):
            self._features = out.detach()

        def bwd(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._handles.append(self.target_layer.register_forward_hook(fwd))
        self._handles.append(self.target_layer.register_full_backward_hook(bwd))

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        """Returns a (H_feat, W_feat) float32 CAM in [0, 1]."""
        self.model.eval()
        x = x.detach().clone().requires_grad_(True)

        logits = self.model(x)           # (1, 1) — binary
        self.model.zero_grad()
        logits.sum().backward()          # scalar, so sum() is fine for (1,1)

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self._features).sum(dim=1).squeeze()  # (H, W)
        cam     = torch.relu(cam).cpu().numpy()
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        return cam.astype(np.float32)


# ---------------------------------------------------------------------------
#  Attention Rollout (ViT)
# ---------------------------------------------------------------------------

class _AttentionRollout:
    def __init__(self, model: nn.Module, discard_ratio: float = 0.9):
        self.model         = model
        self.discard_ratio = discard_ratio
        self._attentions   = []
        self._handles      = []
        self._register_hooks()

    def _register_hooks(self):
        def make_hook():
            def hook(module, inp, out):
                # torchvision ViT returns (attn_output, attn_weights)
                if isinstance(out, tuple) and len(out) == 2 and out[1] is not None:
                    self._attentions.append(out[1].detach())
            return hook

        for _, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                self._handles.append(module.register_forward_hook(make_hook()))

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        """Returns a (grid, grid) float32 attention map in [0, 1]."""
        self._attentions = []
        self.model.eval()
        with torch.no_grad():
            _ = self.model(x)

        if not self._attentions:
            warnings.warn(
                "Attention Rollout: no attention weights captured. "
                "torchvision ViT may need `need_weights=True` (default). "
                "Returning blank map."
            )
            return np.zeros((14, 14), dtype=np.float32)

        result = None
        I      = None  # identity, lazy-init

        for attn in self._attentions:
            # attn: (batch, heads, seq, seq)  — seq includes CLS token
            attn_avg = attn.mean(dim=1)[0]  # (seq, seq)

            # Discard lowest `discard_ratio` fraction of attention
            flat = attn_avg.flatten()
            k    = int(self.discard_ratio * flat.size(0))
            if k > 0:
                threshold = flat.kthvalue(k).values
                attn_avg  = attn_avg.masked_fill(attn_avg < threshold, 0.0)

            # Residual connection (A + I) / 2 normalised
            if I is None:
                I = torch.eye(attn_avg.size(0), device=attn_avg.device)
            attn_aug = (attn_avg + I) / 2.0
            attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            result = attn_aug if result is None else attn_aug @ result

        # CLS → patch tokens
        mask      = result[0, 1:]              # (num_patches,)
        num_patch = mask.size(0)
        grid      = int(round(num_patch ** 0.5))
        cam       = mask.cpu().numpy().reshape(grid, grid).astype(np.float32)
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        return cam


# ---------------------------------------------------------------------------
#  Shared overlay helper
# ---------------------------------------------------------------------------

def _make_overlay(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Resize cam to img_rgb, apply jet colormap, blend with original."""
    h, w    = img_rgb.shape[:2]
    cam_up  = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_up), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.clip(
        (1 - alpha) * img_rgb.astype(np.float32) + alpha * heatmap.astype(np.float32),
        0, 255,
    ).astype(np.uint8)
    return overlay


# ---------------------------------------------------------------------------
#  Public API — single image
# ---------------------------------------------------------------------------

def visualize_gradcam(
    model        : nn.Module,
    image_source : Union[str, torch.Tensor, np.ndarray, Image.Image],
    device       : torch.device,
    target_layer : Optional[nn.Module] = None,
    alpha        : float               = 0.45,
    title        : str                 = "",
    ax           : Optional[plt.Axes]  = None,
    show         : bool                = True,
    discard_ratio: float               = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute and (optionally) plot a Grad-CAM or Attention-Rollout overlay.

    Parameters
    ----------
    model         : trained PyTorch model (any of the 5 architectures)
    image_source  : file path, PIL Image, numpy (H,W,3) uint8, or preprocessed tensor
    device        : torch.device
    target_layer  : override the auto-detected layer (CNN only)
    alpha         : heatmap blend strength (0=invisible, 1=full heatmap)
    title         : subplot title
    ax            : if given, draws into this Axes instead of creating a new figure
    show          : call plt.show() when done (set False when embedding in a grid)
    discard_ratio : Attention Rollout — fraction of lowest weights to zero out

    Returns
    -------
    cam     : raw (H_feat, W_feat) float32 map
    overlay : (224, 224, 3) uint8 blended image
    """
    tensor, img_rgb = _load_image(image_source, model, device)

    use_vit = _is_vit(model)

    if use_vit:
        explainer = _AttentionRollout(model, discard_ratio=discard_ratio)
    else:
        layer = target_layer if target_layer is not None else get_target_layer(model)
        if layer is None:
            raise ValueError(
                "Could not find a target layer. Pass `target_layer=` explicitly."
            )
        explainer = _GradCAM(model, layer)

    try:
        cam = explainer(tensor)
    finally:
        explainer.remove_hooks()

    overlay = _make_overlay(img_rgb, cam, alpha=alpha)

    method = "Attention Rollout" if use_vit else "Grad-CAM"

    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        _plot_triplet(axes, img_rgb, cam, overlay, title or method)
        if show:
            plt.tight_layout()
            plt.show()
    else:
        # When called inside a grid, just show the overlay in the given Axes
        ax.imshow(overlay)
        ax.set_title(title or method, fontsize=9)
        ax.axis("off")

    return cam, overlay


# ---------------------------------------------------------------------------
#  Public API — grid across the 4 experiment models
# ---------------------------------------------------------------------------

# Type alias for the images argument
_ImageSources = Union[
    Dict[str, Union[str, torch.Tensor, np.ndarray, Image.Image]],
    str, torch.Tensor, np.ndarray, Image.Image,
]


def visualize_gradcam_grid(
        models_dict: Dict[str, nn.Module],
        images: _ImageSources,
        device: torch.device,
        target_layer: Optional[nn.Module] = None,
        alpha: float = 0.45,
        title: str = "",
        discard_ratio: float = 0.9,
        figsize: Tuple[int, int] = (20, 5),
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Run Grad-CAM / Attention-Rollout on every model in *models_dict* and
    display the results side-by-side.

    Parameters
    ----------
    models_dict : {"inpainting": model, "insight": model, ...}

    images : one of
        • dict  — one image per experiment, keys must match models_dict
                  e.g. {"inpainting": "data/inpainting/img.jpg",
                         "insight":    "data/insight/img.jpg",  ...}
        • single image (path / PIL / tensor / ndarray)
                  — the same image is used for every model
                  (useful when comparing how different models read the same input)

    Returns
    -------
    results : {experiment_name: (cam_array, overlay_array)}
    """
    exp_names = list(models_dict.keys())

    # Resolve per-experiment image sources
    if isinstance(images, dict):
        missing = set(exp_names) - set(images.keys())
        if missing:
            raise ValueError(
                f"images dict is missing keys: {missing}. "
                f"Keys must match models_dict: {exp_names}"
            )
        image_map = images
    else:
        # Broadcast the single image to every experiment
        image_map = {name: images for name in exp_names}

    # ── Layout: original images on top row, overlays on bottom ──────────────
    # Rows: 0 = original, 1 = overlay
    # Cols: one per experiment
    n = len(exp_names)
    fig, axes = plt.subplots(2, n, figsize=figsize)
    if n == 1:
        axes = axes.reshape(2, 1)  # keep 2-D indexing consistent

    results = {}
    for col, exp_name in enumerate(exp_names):
        model = models_dict[exp_name]
        src = image_map[exp_name]

        # Top row — original image
        _, img_rgb = _load_image(src, model, device)
        axes[0, col].imshow(img_rgb)
        axes[0, col].set_title(exp_name, fontsize=10, fontweight="bold")
        axes[0, col].axis("off")

        # Bottom row — Grad-CAM overlay
        cam, overlay = visualize_gradcam(
            model=model,
            image_source=src,
            device=device,
            target_layer=target_layer,
            alpha=alpha,
            title="",
            ax=axes[1, col],
            show=False,
            discard_ratio=discard_ratio,
        )
        axes[1, col].set_title("overlay", fontsize=8)
        results[exp_name] = (cam, overlay)

    method = "Attention Rollout" if _is_vit(next(iter(models_dict.values()))) else "Grad-CAM"
    sup = f"{title}  [{method}]" if title else method
    fig.suptitle(sup, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return results


# ---------------------------------------------------------------------------
#  Internal: triplet plot (original | heatmap | overlay)
# ---------------------------------------------------------------------------

def _plot_triplet(
    axes   : np.ndarray,
    img_rgb: np.ndarray,
    cam    : np.ndarray,
    overlay: np.ndarray,
    title  : str,
):
    h, w = img_rgb.shape[:2]
    cam_up = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

    axes[0].imshow(img_rgb);           axes[0].set_title("Original");  axes[0].axis("off")
    axes[1].imshow(cam_up, cmap="jet"); axes[1].set_title("Heat map"); axes[1].axis("off")
    axes[2].imshow(overlay);            axes[2].set_title("Overlay");  axes[2].axis("off")

    axes[0].figure.suptitle(title, fontsize=12, fontweight="bold")