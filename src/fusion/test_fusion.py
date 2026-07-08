"""Unit tests for Q2E inverse-entropy rank fusion.

Validates the *official* implementation (external/q2e_official/src/eval/fusion_score.py)
against the behaviour described in the paper:
  P_i = softmax(S_i, dim=0)              # pre-softmax over queries (done in infer.py)
  H(P_i) = -sum_v P_i log2(P_i + eps)    # per-query (row) entropy
  S_fused = sum_i (1 / (H(P_i)+eps)) * P_i
  final = min_max_normalize(S_fused)     # per-row

Run:  python -m src.fusion.test_fusion   (from external/q2e_official, with .venv-eval)
or:   PYTHONPATH=external/q2e_official python src/fusion/test_fusion.py
"""
import sys, os
import torch
import torch.nn.functional as F

# import the official implementation
HERE = os.path.dirname(__file__)
OFFICIAL = os.path.abspath(os.path.join(HERE, "..", "..", "external", "q2e_official"))
sys.path.insert(0, OFFICIAL)
from src.eval.fusion_score import fusion_inverse_entropy, entropy  # noqa: E402
from src.eval.utils import min_max_normalize  # noqa: E402


def ref_inverse_entropy(pre_softmaxed):
    """Independent re-implementation from the paper equations."""
    Q, V = pre_softmaxed[0].shape
    acc = torch.zeros(Q, V)
    for P in pre_softmaxed:
        H = -torch.sum(P * torch.log2(P + 1e-6), dim=-1, keepdim=True)  # Q x 1
        acc += (1.0 / (H + 1e-6)) * P
    return min_max_normalize(acc)


def test_matches_reference():
    torch.manual_seed(0)
    Q, V = 7, 11
    raw = [torch.randn(Q, V) for _ in range(5)]
    pre = [F.softmax(s, dim=0) for s in raw]  # infer.py pre-softmax over queries (dim=0)
    got = fusion_inverse_entropy(None, pre, [None] * 5, list(range(Q)), list(range(V)))
    exp = ref_inverse_entropy(pre)
    assert torch.allclose(got, exp, atol=1e-5), (got - exp).abs().max()
    print("[ok] official fusion == paper-equation reference")


def test_low_entropy_dominates():
    """A confident (low-entropy) component should dominate the fused ranking."""
    Q, V = 3, 6
    # confident component: near one-hot per row -> low entropy -> high weight
    confident = torch.full((Q, V), 0.01)
    for q in range(Q):
        confident[q, q] = 0.95
    confident = confident / confident.sum(dim=1, keepdim=True)
    # diffuse component: uniform -> max entropy -> ~zero weight
    diffuse = torch.full((Q, V), 1.0 / V)
    fused = fusion_inverse_entropy(None, [confident, diffuse], [None, None],
                                   list(range(Q)), list(range(V)))
    # argmax of fused should follow the confident component's diagonal
    for q in range(Q):
        assert fused[q].argmax().item() == q, (q, fused[q])
    print("[ok] low-entropy component dominates fusion")


def test_entropy_monotonic():
    """Sharper distribution -> lower entropy."""
    sharp = torch.tensor([[0.97, 0.01, 0.01, 0.01]])
    flat = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    assert entropy(sharp).item() < entropy(flat).item()
    # uniform over V=4 has entropy log2(4)=2
    assert abs(entropy(flat).item() - 2.0) < 1e-2
    print("[ok] entropy is monotonic and matches log2(V) for uniform")


def test_minmax_normalize_range():
    x = torch.randn(4, 9)
    n = min_max_normalize(x)
    assert n.min() >= -1e-4 and n.max() <= 1.0 + 1e-4
    # each row's min ~0 and max ~1
    assert torch.allclose(n.min(dim=1).values, torch.zeros(4), atol=1e-4)
    assert torch.allclose(n.max(dim=1).values, torch.ones(4), atol=1e-3)
    print("[ok] min-max normalize maps each row to [0,1]")


if __name__ == "__main__":
    test_entropy_monotonic()
    test_minmax_normalize_range()
    test_matches_reference()
    test_low_entropy_dominates()
    print("\nALL FUSION TESTS PASSED")
