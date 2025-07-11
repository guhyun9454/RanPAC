import torch
from typing import Tuple, List

from .ebo_postprocessor import EBOPostprocessor
from .gen_postprocessor import GENPostprocessor
from .pro_gen_postprocessor import PRO_GENPostprocessor
from .pro2_msp_postprocessor import PROv2_MSP_Postprocessor
from .pro2_ent_postprocessor import PROv2_ENT_Postprocessor
from .pro2_tempscale_postprocessor import PROv2_TEMPSCALE_Postprocessor
from .maxlogit_postprocessor import MaxLogitPostprocessor
from .base_postprocessor import BasePostprocessor
from .pseudo_adv_postprocessor import PseudoADVPostprocessor

__all__ = ["SUPPORTED_METHODS", "compute_ood_scores"]

# ---------------------------------------------------------------------------
# 1. Method registry & default hyper-parameters
# ---------------------------------------------------------------------------
SUPPORTED_METHODS: List[str] = [
    "MSP",
    "ENERGY",
    "GEN",
    "RPO_MSP",
    "PRO_MSP_T",
    "PRO_ENT",
    "PRO_GEN",
    "MAXLOGIT",
    "PSEUDO",  # 추가
]

_DEFAULT_PARAMS = {
    "ENERGY": {"temperature": 1.0},
    "GEN": {"gamma": 0.1, "M": 100},
    "PRO_GEN": {"gamma": 0.1, "M": 100, "noise_level": 1e-4, "gd_steps": 3},
    "RPO_MSP": {"temperature": 1.0, "noise_level": 0.003, "gd_steps": 1},
    "PRO_ENT": {"noise_level": 0.0014, "gd_steps": 2},
    "PRO_MSP_T": {"temperature": 1.0, "noise_level": 0.003, "gd_steps": 1},
    "PSEUDO": {"epsilon": 0.003},
}

_POSTPROCESSOR_REGISTRY = {
    "MSP": BasePostprocessor,
    "ENERGY": EBOPostprocessor,
    "GEN": GENPostprocessor,
    "PRO_GEN": PRO_GENPostprocessor,
    "RPO_MSP": PROv2_MSP_Postprocessor,
    "PRO_ENT": PROv2_ENT_Postprocessor,
    "PRO_MSP_T": PROv2_TEMPSCALE_Postprocessor,
    "MAXLOGIT": MaxLogitPostprocessor,
    "PSEUDO": PseudoADVPostprocessor,
}

# ---------------------------------------------------------------------------
# 2. Public API
# ---------------------------------------------------------------------------

def compute_ood_scores(
    method: str,
    model: torch.nn.Module,
    id_loader: torch.utils.data.DataLoader,
    ood_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """해당 OOD 방법으로부터 (ID, OOD) 점수를 반환"""

    method = method.upper()
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"[ood_adapter] 지원하지 않는 방법: {method}")

    params = _DEFAULT_PARAMS.get(method, {})

    processor_cls = _POSTPROCESSOR_REGISTRY[method]
    processor = processor_cls(**params)
    # Postprocessor 가 학습이 필요하면 setup()을 호출 (id_loader 정보 사용)
    if hasattr(processor, "setup"):
        try:
            processor.setup(model, id_loader, ood_loader, device)
        except TypeError:
            # 이전 버전 호환
            processor.setup(model, id_loader, ood_loader)

    def _gather_scores(loader):
        scores = []
        model.eval()
        for inputs, _ in loader:
            inputs = inputs.to(device)
            model.zero_grad(set_to_none=True)
            _, conf = processor.postprocess(model, inputs)
            scores.append(conf.detach().cpu())
        return torch.cat(scores, dim=0)

    id_scores = _gather_scores(id_loader)
    ood_scores = _gather_scores(ood_loader)
    return id_scores, ood_scores 