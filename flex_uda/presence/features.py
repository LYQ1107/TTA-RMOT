"""The original query-separable decode, with its final PCD representation."""
import torch


def margins_and_features(model, prepared, texts):
    """Return [candidates, queries] and [candidates, queries, 768].

    Deliberately not decorated with no_grad: the same operation also supports
    the slow adapter step. Frozen exports must enter torch.no_grad themselves.
    """
    captured = {}
    old = model.sample_expression_num
    handle = model.output_ca_qkv[-1].register_forward_hook(
        lambda _module, _inputs, output: captured.update(pcd=output))
    try:
        model.sample_expression_num = len(texts)
        n = len(prepared['track_ids'])
        text = torch.cat([h for h, _ in texts], dim=1).expand(n, -1, -1, -1).contiguous()
        mask = torch.cat([m for _, m in texts], dim=1).expand(n, -1, -1).contiguous()
        logits, _ = model.decode(prepared['outputs'], text, mask,
                                 prepared['positions'], prepared['boxes'])
        margin = logits[:, -1, :, 1] - logits[:, -1, :, 0]
        pcd = captured['pcd']
        if pcd.shape != (*margin.shape, 768):
            raise ValueError(f'Unexpected final PCD shape: {pcd.shape}')
        return margin, pcd
    finally:
        handle.remove()
        model.sample_expression_num = old


def sentence_vector(text):
    hidden, mask = text
    weight = mask.to(hidden.dtype).unsqueeze(-1)
    return ((hidden * weight).sum(-2) / weight.sum(-2).clamp_min(1)).reshape(-1, 768)[0]
