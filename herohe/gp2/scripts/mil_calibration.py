"""Post-hoc temperature scaling for MIL baseline models (ABMIL / CLAM / TransMIL)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def forward_logits(
    model: nn.Module,
    batch: dict,
    device: torch.device,
    aggregator: str,
) -> torch.Tensor:
    """Return slide logits (1, C) for one bag."""
    x = batch["features"].to(device)
    agg = aggregator.lower()
    if agg == "clam":
        h = x.squeeze(0)
        logits, _, _, _, _ = model(h, label=None, instance_eval=False)
        return logits
    if agg in ("abmil", "attnmisl"):
        return model(x)["logits"]
    c = batch["coords"].to(device) if batch["coords"] is not None else None
    return model(x, coords=c)["logits"]


@torch.no_grad()
def collect_logits_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    aggregator: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    for batch in loader:
        logits_list.append(forward_logits(model, batch, device, aggregator))
        labels_list.append(batch["label"].to(device))
    if not logits_list:
        return torch.empty(0, 0, device=device), torch.empty(0, dtype=torch.long, device=device)
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0).view(-1)


def fit_temperature_mil(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    aggregator: str,
    max_iter: int = 50,
    lr: float = 0.01,
) -> float:
    """Fit scalar T on validation NLL; returns T (>= 1e-3)."""
    logits, y = collect_logits_labels(model, val_loader, device, aggregator)
    if logits.numel() == 0:
        return 1.0
    T = nn.Parameter(torch.ones(1, device=device))
    optim = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optim.zero_grad()
        loss = F.cross_entropy(logits / T.clamp_min(1e-3), y)
        loss.backward()
        return loss

    optim.step(closure)
    return float(T.detach().clamp_min(1e-3).item())


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    T = max(float(temperature), 1e-3)
    return logits / T
