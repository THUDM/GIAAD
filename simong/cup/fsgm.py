from copy import deepcopy
import random

from tqdm.auto import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch_sparse


class FSGM():

    def __init__(self,
                 model: nn.Module,
                 features: torch.Tensor,
                 edge_index: torch.Tensor,
                 edge_weight: torch.Tensor,
                 n: int,
                 d: int,
                 labels_test: torch.Tensor,
                 idx_test: np.ndarray,
                 display_step: int = 2,
                 node_budget: int = 500,
                 edge_budget: int = 100,
                 edge_step_size: int = 3,
                 do_only_connect_test=False,
                 eps: float = 1e-30,
                 feature_lr: float = 1e2,  # 1e-3,
                 feature_init_std: float = 10,
                 feature_max_abs: float = 2,
                 monitor_time: bool = False,
                 feature_do_use_seeds: bool = False,
                 feature_dedicated_iterations: int = None,
                 stop_optimizing_if_label_flipped: bool = False,
                 edge_with_random_reverse: bool = False):
        super().__init__()
        self.device = features.device
        self.model = deepcopy(model).to(self.device)
        self.model.eval()
        self.features = features
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.n = n
        self.d = d
        self.labels_test = labels_test
        self.idx_test = idx_test
        self.display_step = display_step
        self.node_budget = node_budget
        self.edge_budget = edge_budget
        self.edge_step_size = edge_step_size
        self.do_only_connect_test = do_only_connect_test
        self.eps = eps
        self.feature_lr = feature_lr
        self.feature_init_std = feature_init_std
        self.feature_max_abs = feature_max_abs
        self.monitor_time = monitor_time
        self.feature_do_use_seeds = feature_do_use_seeds
        self.feature_dedicated_iterations = feature_dedicated_iterations
        self.stop_optimizing_if_label_flipped = stop_optimizing_if_label_flipped
        self.edge_with_random_reverse = edge_with_random_reverse
        assert self.edge_budget % self.edge_step_size == 0

    def attack(self):
        features = self.features
        edge_index = self.edge_index
        edge_weight = self.edge_weight

        if self.feature_do_use_seeds:
            probs = F.softmax(self.model(features, edge_index, edge_weight), dim=-1)
            feature_seeds = [
                ((probs[:, i][:, None] * features).sum(0) / probs[:, i].sum()).detach()
                for i
                in range(probs.shape[1])
            ]

        new_features = None
        for i in tqdm(range(self.node_budget), desc='Adding edges'):
            next_node = self.n + i + 1
            if self.do_only_connect_test:
                new_edge_weight = self.eps * torch.ones(len(self.idx_test)).cuda()
                new_edge_idx = torch.stack([torch.arange(self.n - len(self.idx_test), self.n),
                                            (next_node - 1) * torch.ones(len(self.idx_test)).long()]).cuda()
            else:
                new_edge_weight = self.eps * torch.ones(self.n).cuda()
                new_edge_idx = torch.stack([torch.arange(self.n),
                                            (next_node - 1) * torch.ones(self.n).long()]).cuda()

            if self.feature_do_use_seeds:
                seed_id = random.choice(range(probs.shape[1]))
                next_new_features = deepcopy(feature_seeds[seed_id])[None, :]
            else:
                next_new_features = self.feature_init_std * torch.randn((1, self.d)).cuda()

            if new_features is not None:
                new_features = torch.cat([new_features, next_new_features])
            else:
                new_features = next_new_features
            new_features = torch.clamp(new_features, -self.feature_max_abs, self.feature_max_abs)

            new_edge_weight.requires_grad = True
            new_features.requires_grad = True

            n_steps = self.edge_budget // self.edge_step_size
            if self.edge_with_random_reverse:
                n_steps += 2

            for j in range(n_steps):

                if self.monitor_time:
                    time_start = torch.cuda.Event(enable_timing=True)
                    time_symm = torch.cuda.Event(enable_timing=True)
                    time_forward = torch.cuda.Event(enable_timing=True)
                    time_edge_update = torch.cuda.Event(enable_timing=True)
                    time_feature_update = torch.cuda.Event(enable_timing=True)

                    time_start.record()

                combined_edge_index = torch.cat((edge_index, new_edge_idx), dim=-1)
                combined_edge_weight = torch.cat((edge_weight, new_edge_weight))
                combined_features = torch.cat((features, new_features))

                symmetric_edge_index = torch.cat(
                    (combined_edge_index, torch.flip(combined_edge_index, dims=[1, 0])), dim=-1
                )
                symmetric_edge_weight = torch.cat([combined_edge_weight, torch.flip(combined_edge_weight, dims=[0])])
                symmetric_edge_index, symmetric_edge_weight = torch_sparse.coalesce(
                    symmetric_edge_index,
                    symmetric_edge_weight,
                    m=next_node,
                    n=next_node,
                    op='mean'
                )

                if self.monitor_time:
                    time_symm.record()

                logits = self.model(combined_features, symmetric_edge_index, symmetric_edge_weight)
                if self.stop_optimizing_if_label_flipped:
                    not_yet_flipped_mask = logits[self.idx_test].argmax(-1) == self.labels_test - 1
                    if not_yet_flipped_mask.sum() > 0:
                        loss = F.cross_entropy(logits[self.idx_test][not_yet_flipped_mask],
                                               self.labels_test[not_yet_flipped_mask] - 1)
                    else:
                        loss = F.cross_entropy(logits[self.idx_test], self.labels_test - 1)
                else:
                    loss = F.cross_entropy(logits[self.idx_test], self.labels_test - 1)

                if self.monitor_time:
                    time_forward.record()

                gradient_edge, gradient_feature = torch.autograd.grad(
                    loss,
                    [new_edge_weight, new_features]
                )

                if self.edge_with_random_reverse and j == n_steps - 1:
                    edge_step_size = self.edge_step_size + random.choice(range(self.edge_step_size))
                    topk_idx = torch.topk(
                        (self.eps - new_edge_weight) * gradient_edge - (1 - new_edge_weight) * 1e8, edge_step_size
                    )[1]
                    with torch.no_grad():
                        new_edge_weight.index_put_((topk_idx,), torch.tensor(self.eps).float())

                else:
                    topk_idx = torch.topk(
                        (1 - new_edge_weight) * gradient_edge, self.edge_step_size
                    )[1]
                    with torch.no_grad():
                        new_edge_weight.index_put_((topk_idx,), torch.tensor(1).float())

                if self.monitor_time:
                    time_edge_update.record()

                if j > 0 and self.feature_dedicated_iterations is None:
                    with torch.no_grad():
                        new_features = new_features + self.feature_lr * gradient_feature
                        new_features = torch.clamp(new_features, -self.feature_max_abs, self.feature_max_abs)
                    new_features.requires_grad = True

                if self.monitor_time:
                    time_feature_update.record()
                    torch.cuda.synchronize()

                    print(f'Symmetrize took: {time_start.elapsed_time(time_symm)}')
                    print(f'Forward took: {time_symm.elapsed_time(time_forward)}')
                    print(f'Edge update took: {time_forward.elapsed_time(time_edge_update)}')
                    print(f'Feature update took: {time_edge_update.elapsed_time(time_feature_update)}')

                if j % self.display_step == 0:
                    ratio_class_labels_flipped = (
                        logits.argmax(-1)[self.idx_test] + 1 == self.labels_test
                    ).float().mean()
                    print(f'Adversarial node {i} after adding {j*self.edge_step_size} edges, we changed '
                          f'the prediction for {100*(1 - ratio_class_labels_flipped):.3f} %')

            new_edge_idx = new_edge_idx[:, new_edge_weight == 1]
            new_edge_weight = new_edge_weight[new_edge_weight == 1]

            combined_edge_index = torch.cat((edge_index, new_edge_idx), dim=-1)
            combined_edge_weight = torch.cat((edge_weight, new_edge_weight)).detach()

            symmetric_edge_index = torch.cat(
                (combined_edge_index, torch.flip(combined_edge_index, dims=[1, 0])), dim=-1
            )
            symmetric_edge_weight = torch.cat([combined_edge_weight, torch.flip(combined_edge_weight, dims=[0])])
            symmetric_edge_index, symmetric_edge_weight = torch_sparse.coalesce(
                symmetric_edge_index,
                symmetric_edge_weight,
                m=next_node,
                n=next_node,
                op='max'
            )

            if self.feature_dedicated_iterations is not None:
                optimizer = torch.optim.Adam((new_features,), lr=self.feature_lr)
                ratio_class_labels_flipped = (
                    logits.argmax(-1)[self.idx_test] + 1 == self.labels_test
                ).float().mean()
                print(f'Adversarial node {i} before optimizing the features, we changed '
                      f'the prediction for {100*(1 - ratio_class_labels_flipped):.3f} %')
                for j in range(self.feature_dedicated_iterations):
                    combined_features = torch.cat((features, new_features))
                    logits = self.model(combined_features, symmetric_edge_index, symmetric_edge_weight)
                    loss = F.cross_entropy(logits[self.idx_test], self.labels_test - 1)

                    optimizer.zero_grad()
                    (-loss).backward()
                    optimizer.step()
                    with torch.no_grad():
                        new_features = torch.clamp(new_features, -self.feature_max_abs, self.feature_max_abs)
                ratio_class_labels_flipped = (
                    logits.argmax(-1)[self.idx_test] + 1 == self.labels_test
                ).float().mean()
                print(f'Adversarial node {i} after optimizing the features, we changed '
                      f'the prediction for {100*(1 - ratio_class_labels_flipped):.3f} %')

            edge_index = symmetric_edge_index
            edge_weight = symmetric_edge_weight

            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return edge_index, edge_weight, new_features.detach()
