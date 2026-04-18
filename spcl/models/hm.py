import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, autograd


class HM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class HybridMemory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())

    def forward(self, inputs, indexes):
        # inputs: B*2048, features: L*2048
        # Normalize inputs to unit sphere — backbone returns unnormalized features
        # in training mode, causing exp() overflow (norms ~35 → post-temp ~300 → inf)
        inputs = F.normalize(inputs, dim=1)

        if not hasattr(self, '_diag_count'): self._diag_count = 0
        _d = self._diag_count < 3  # log first 3 batches only

        if _d:
            _in = inputs.detach()
            print(f"  [HM] raw_inputs: shape={tuple(_in.shape)} min={_in.min().item():.4f} max={_in.max().item():.4f} "
                  f"nan={torch.isnan(_in).any().item()} inf={torch.isinf(_in).any().item()} "
                  f"row_norms: min={_in.norm(dim=1).min().item():.4f} max={_in.norm(dim=1).max().item():.4f}")
            _mf = self.features.detach()
            print(f"  [HM] memory_features: nan={torch.isnan(_mf).any().item()} inf={torch.isinf(_mf).any().item()} "
                  f"row_norms: min={_mf.norm(dim=1).min().item():.4f} max={_mf.norm(dim=1).max().item():.4f}")

        inputs = hm(inputs, indexes, self.features, self.momentum)

        if _d:
            print(f"  [HM] post-hm (similarity): min={inputs.min().item():.4f} max={inputs.max().item():.4f} "
                  f"nan={torch.isnan(inputs).any().item()} inf={torch.isinf(inputs).any().item()}")

        inputs /= self.temp

        if _d:
            print(f"  [HM] post-temp (÷{self.temp}): min={inputs.min().item():.4f} max={inputs.max().item():.4f} "
                  f"nan={torch.isnan(inputs).any().item()} inf={torch.isinf(inputs).any().item()}")

        B = inputs.size(0)

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            if _d:
                print(f"  [HM] exp(vec): min={exps.min().item():.6f} max={exps.max().item():.6f} "
                      f"nan={torch.isnan(exps).any().item()} inf={torch.isinf(exps).any().item()}")
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return (masked_exps/masked_sums)

        targets = self.labels[indexes].clone()
        labels = self.labels.clone()

        if _d:
            n_classes = labels.max().item() + 1
            n_unique = len(torch.unique(labels))
            print(f"  [HM] labels: n_classes={n_classes} n_unique={n_unique} "
                  f"targets_range=[{targets.min().item()}, {targets.max().item()}]")

        sim = torch.zeros(labels.max()+1, B).float().cuda()
        sim.index_add_(0, labels, inputs.t().contiguous())
        nums = torch.zeros(labels.max()+1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(self.num_samples,1).float().cuda())
        mask = (nums>0).float()
        sim /= (mask*nums+(1-mask)).clone().expand_as(sim)

        if _d:
            print(f"  [HM] sim_avg: min={sim.min().item():.4f} max={sim.max().item():.4f} "
                  f"nan={torch.isnan(sim).any().item()} inf={torch.isinf(sim).any().item()}")

        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())

        if _d:
            print(f"  [HM] masked_sim: min={masked_sim.min().item():.6f} max={masked_sim.max().item():.6f} "
                  f"nan={torch.isnan(masked_sim).any().item()}")

        loss = F.nll_loss(torch.log(masked_sim+1e-6), targets)

        if _d:
            print(f"  [HM] loss={loss.item():.6f} nan={torch.isnan(loss).item()}")
            self._diag_count += 1

        return loss
