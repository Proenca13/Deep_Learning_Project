import warnings
from typing import Dict, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# --- NEW IMPORT ---
from pytorch_grad_cam import GradCAM

# ---------------------------------------------------------------------------
#  Preprocessing helpers
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_XCEPTION_MEAN = [0.5, 0.5, 0.5]
_XCEPTION_STD = [0.5, 0.5, 0.5]


def _build_transform(model, img_size=(224, 224)):
    name = type(model).__name__.lower()

    if not _is_vit(model):
        for module in model.modules():
            if isinstance(module, nn.Linear):
                n = module.in_features
                side = int((n / 3) ** 0.5)
                if side * side * 3 == n and side >= 32:
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
        std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
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
    tf = _build_transform(model)
    tensor = tf(pil).unsqueeze(0).to(device)
    return tensor, img_rgb


# ---------------------------------------------------------------------------
#  Target-layer auto-detection & ViT Reshaping
# ---------------------------------------------------------------------------

def _is_vit(model: nn.Module) -> bool:
    """True if the model contains transformer encoder layers (ViT family)."""
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            return True
    return False


def _vit_reshape_transform(tensor, height=14, width=14):
    """
    Reshapes the ViT sequence into a 2D feature map for the grad-cam library.
    Assumes standard 224x224 input with 16x16 patches -> 14x14 grid.
    """
    # Remove the CLS token (the first token)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # PyTorch-grad-cam expects channels to be the first dimension: (batch, channels, H, W)
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def get_target_layer(model: nn.Module) -> Optional[nn.Module]:
    """
    Heuristically returns the best conv layer for Grad-CAM.
    Returns None for ViT.
    """
    if _is_vit(model):
        return None

    name = type(model).__name__.lower()

    # ── ResNet ──────────────────────────────────────────────────────────────
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
#  Custom Grad-CAM (For CNNs only)
# ---------------------------------------------------------------------------

class _GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._features = None
        self._gradients = None
        self._handles = []
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

        logits = self.model(x)  # (1, 1) — binary
        self.model.zero_grad()
        logits.sum().backward()  # scalar, so sum() is fine for (1,1)

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._features).sum(dim=1).squeeze()  # (H, W)
        cam = torch.relu(cam).cpu().numpy()
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        return cam.astype(np.float32)


# ---------------------------------------------------------------------------
#  Shared overlay helper
# ---------------------------------------------------------------------------

def _make_overlay(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Resize cam to img_rgb, apply jet colormap, blend with original."""
    h, w = img_rgb.shape[:2]
    cam_up = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
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
        model: nn.Module,
        image_source: Union[str, torch.Tensor, np.ndarray, Image.Image],
        device: torch.device,
        target_layer: Optional[nn.Module] = None,
        alpha: float = 0.45,
        title: str = "",
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        discard_ratio: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    tensor, img_rgb = _load_image(image_source, model, device)
    use_vit = _is_vit(model)

    if use_vit:
        if target_layer is None:
            # --- Robust Auto-Detection for Wrapped ViTs ---
            t_layer = None
            # Look for the encoder in the model or its sub-modules (wrappers)
            for module in model.modules():
                if hasattr(module, "encoder") and hasattr(module.encoder, "layers"):
                    t_layer = module.encoder.layers[-1].ln_1
                    break

            if t_layer is None:
                raise ValueError(
                    "Could not auto-detect ViT target layer in 'DeepFakeViT'. "
                    "Please pass 'target_layer=model.model.encoder.layers[-1].ln_1' "
                    "(or whatever your internal attribute is named) explicitly."
                )
        else:
            t_layer = target_layer

        target_layers = [t_layer]
        explainer = GradCAM(
            model=model,
            target_layers=target_layers,
            reshape_transform=_vit_reshape_transform
        )
        cam = explainer(input_tensor=tensor, targets=None)[0]
        method = "Grad-CAM (ViT)"

    else:
        # --- CNN Logic (Unchanged) ---
        layer = target_layer if target_layer is not None else get_target_layer(model)
        if layer is None:
            raise ValueError("Could not find a target layer for CNN.")
        explainer = _GradCAM(model, layer)
        try:
            cam = explainer(tensor)
        finally:
            explainer.remove_hooks()
        method = "Grad-CAM (CNN)"

    overlay = _make_overlay(img_rgb, cam, alpha=alpha)

    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        _plot_triplet(axes, img_rgb, cam, overlay, title or method)
        if show:
            plt.tight_layout()
            plt.show()
    else:
        ax.imshow(overlay)
        ax.set_title(title or method, fontsize=9)
        ax.axis("off")

    return cam, overlay


# ---------------------------------------------------------------------------
#  Public API — grid across the 4 experiment models
# ---------------------------------------------------------------------------

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
    Run Grad-CAM on every model in *models_dict* and display results side-by-side.
    """
    exp_names = list(models_dict.keys())

    if isinstance(images, dict):
        missing = set(exp_names) - set(images.keys())
        if missing:
            raise ValueError(
                f"images dict is missing keys: {missing}. "
                f"Keys must match models_dict: {exp_names}"
            )
        image_map = images
    else:
        image_map = {name: images for name in exp_names}

    n = len(exp_names)
    fig, axes = plt.subplots(2, n, figsize=figsize)
    if n == 1:
        axes = axes.reshape(2, 1)

    results = {}
    for col, exp_name in enumerate(exp_names):
        model = models_dict[exp_name]
        src = image_map[exp_name]

        tensor, img_rgb = _load_image(src, model, device)

        with torch.no_grad():
            prob = torch.sigmoid(model(tensor)).item()
        pred_label = "FAKE" if prob >= 0.5 else "REAL"

        axes[0, col].imshow(img_rgb)
        axes[0, col].set_title(f"{exp_name}\n{pred_label} ({prob:.1%})", fontsize=10, fontweight="bold")
        axes[0, col].axis("off")

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

    method = "Grad-CAM (ViT)" if _is_vit(next(iter(models_dict.values()))) else "Grad-CAM"
    sup = f"{title}  [{method}]" if title else method
    fig.suptitle(sup, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return results


# ---------------------------------------------------------------------------
#  Internal: triplet plot (original | heatmap | overlay)
# ---------------------------------------------------------------------------

def _plot_triplet(
        axes: np.ndarray,
        img_rgb: np.ndarray,
        cam: np.ndarray,
        overlay: np.ndarray,
        title: str,
):
    h, w = img_rgb.shape[:2]
    cam_up = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

    axes[0].imshow(img_rgb);
    axes[0].set_title("Original");
    axes[0].axis("off")
    axes[1].imshow(cam_up, cmap="jet");
    axes[1].set_title("Heat map");
    axes[1].axis("off")
    axes[2].imshow(overlay);
    axes[2].set_title("Overlay");
    axes[2].axis("off")
    axes[0].figure.suptitle(title, fontsize=12, fontweight="bold")