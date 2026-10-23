"""Explicit CPU runner for Macs where MMEngine otherwise selects MPS.

The full MMCV build used by this experiment provides CPU operators. Inherit all
training/evaluation behavior and change only the native runner's device choice.
"""
from mmengine.model import is_model_wrapper, revert_sync_batchnorm
from mmengine.registry import RUNNERS
from mmengine.runner import Runner


@RUNNERS.register_module()
class CPURunner(Runner):
    def wrap_model(self, model_wrapper_cfg, model):
        if self.distributed or model_wrapper_cfg is not None:
            raise ValueError("CPURunner supports single-process local training only")
        if is_model_wrapper(model):
            raise ValueError("CPURunner expects an unwrapped model")
        self.logger.info("Local RTMDet: CPU selected explicitly; SyncBN -> BatchNorm")
        return revert_sync_batchnorm(model.to("cpu"))
