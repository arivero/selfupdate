import copy

import torch

from selfupdate.config import ExperimentConfig
from selfupdate.data.dataset import Item, LengthBucketBatchSampler, collate_padded_items
from selfupdate.train.layerwise import _summed_batch
from selfupdate.train.losses import HiddenLoss


def _summed_b1(cfg, stack, loss_fn, it, device="cpu"):
    """The historical item path: one example per walk, now expressed as a
    B=1 collated batch through the unified _summed_batch (bit-exact: no pad
    rows, gather == slice)."""
    b1 = collate_padded_items([it])
    targets = {L: b1.hidden[L] for L in range(1, stack.n_layers + 1)}
    return _summed_batch(cfg, stack, loss_fn, b1, targets, device)


class TinyStack(torch.nn.Module):
    def __init__(self, vocab=64, hidden=8, n_layers=4):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, hidden)
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.Tanh())
             for _ in range(n_layers)]
        )
        self.final_norm = torch.nn.LayerNorm(hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)
        self.n_layers = n_layers

    def freeze_non_blocks(self):
        self.embed_tokens.requires_grad_(False)
        self.final_norm.requires_grad_(False)
        self.lm_head.requires_grad_(False)

    def embed(self, input_ids):
        with torch.no_grad():
            return self.embed_tokens(input_ids)

    def rope(self, hidden, position_ids):
        return None

    def run_block(self, L, hidden, position_embeddings, position_ids=None):
        return self.blocks[L - 1](hidden)

    def loss_view(self, L, block_out):
        return self.final_norm(block_out) if L == self.n_layers else block_out

    def block_params(self, L):
        return list(self.blocks[L - 1].parameters())


def _items(stack):
    torch.manual_seed(12)
    out = []
    for i, (T, s0, A, ans0) in enumerate([(7, 2, 4, 4), (9, 3, 5, 5)]):
        ids = torch.randint(3, 50, (T,), dtype=torch.long)
        hidden = {
            L: torch.randn(A, stack.embed_tokens.embedding_dim)
            for L in range(1, stack.n_layers + 1)
        }
        out.append(Item(
            example_id=f"ex{i}",
            student_ids=ids,
            position_ids=torch.arange(T),
            s0=s0,
            A=A,
            ans0=ans0,
            hidden=hidden,
        ))
    return out


def _block_grads(stack):
    return [
        [p.grad.detach().clone() for p in stack.block_params(L)]
        for L in range(1, stack.n_layers + 1)
    ]


def _assert_same_grads(a, b, *, atol=6e-3, rtol=1e-2):
    for L, (ga, gb) in enumerate(zip(a, b), start=1):
        for pa, pb in zip(ga, gb):
            assert torch.allclose(pa, pb, atol=atol, rtol=rtol), f"L{L} grad mismatch"


def test_padded_summed_batch_matches_item_loop_for_local_blocks():
    base = TinyStack()
    base.freeze_non_blocks()
    items = _items(base)
    cfg = ExperimentConfig()
    cfg.model.device = "cpu"
    cfg.train.hidden_loss = "nmse"

    item_stack = copy.deepcopy(base)
    batch_stack = copy.deepcopy(base)
    loss_fn_item = HiddenLoss("nmse", item_stack.final_norm, item_stack.lm_head)
    loss_fn_batch = HiddenLoss("nmse", batch_stack.final_norm, batch_stack.lm_head)

    for it in items:
        _summed_b1(cfg, item_stack, loss_fn_item, it)
    batch = collate_padded_items(items)
    targets = {L: batch.hidden[L] for L in range(1, batch_stack.n_layers + 1)}
    _summed_batch(cfg, batch_stack, loss_fn_batch, batch, targets, "cpu")

    _assert_same_grads(_block_grads(item_stack), _block_grads(batch_stack))


def test_padded_summed_batch_matches_item_loop_for_sliding_readout():
    base = TinyStack()
    base.freeze_non_blocks()
    items = _items(base)
    cfg = ExperimentConfig()
    cfg.model.device = "cpu"
    cfg.train.hidden_loss = "nmse"
    cfg.train.conn_window = 2
    cfg.train.conn_stride = 1
    cfg.train.readout_window_blocks = 2
    cfg.train.readout_weight = 0.25
    cfg.train.readout_source = "teacher_kl"

    item_stack = copy.deepcopy(base)
    batch_stack = copy.deepcopy(base)
    loss_fn_item = HiddenLoss("nmse", item_stack.final_norm, item_stack.lm_head)
    loss_fn_batch = HiddenLoss("nmse", batch_stack.final_norm, batch_stack.lm_head)

    for it in items:
        _summed_b1(cfg, item_stack, loss_fn_item, it)
    batch = collate_padded_items(items)
    targets = {L: batch.hidden[L] for L in range(1, batch_stack.n_layers + 1)}
    _summed_batch(cfg, batch_stack, loss_fn_batch, batch, targets, "cpu")

    _assert_same_grads(_block_grads(item_stack), _block_grads(batch_stack))


def test_length_bucket_sampler_randomizes_without_global_sorting():
    lengths = [10, 260, 12, 270, 18, 520, 530, 20]
    sampler = LengthBucketBatchSampler(lengths, batch_size=2, bucket_width=128, seed=7)
    first_epoch = list(iter(sampler))
    flat = [idx for batch in first_epoch for idx in batch]
    assert sorted(flat) == list(range(len(lengths)))
    assert flat != sorted(flat, key=lambda i: lengths[i])
    for batch in first_epoch:
        assert len(batch) <= 2
