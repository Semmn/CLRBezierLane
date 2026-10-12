"""Learned NMS on top of a one-to-many head (bbox_head.dedup_cfg).

The one-to-many head scores every query that sits on a lane, so several queries
fire per lane and NMS keeps one. This module learns that choice instead. Each
prediction looks at its nearest higher-scored predictions, by the NMS distance
(mean |dx| over the rows both cover), and outputs a correction to its o2m logit;
the corrected score is trained with a one-to-one target. A per-query classifier
cannot do this, because duplicates of one lane have nearly the same features;
the decision needs the other predictions.

The inputs (final-stage predictions, their RoI features, the o2m scores) are
detached, so the o2m head trains exactly as it does without this module, and one
checkpoint evaluates both ways: o2m scores + NMS, or these scores without NMS.
Prior art: Relation Networks (Hu et al., 2018, learned duplicate removal),
YOLOv10 (dual o2m / o2o heads), Polar R-CNN (graph-based o2o head for lanes).
"""
import torch
import torch.nn as nn

# per-neighbour features, see LearnedLaneNMS._edges
EDGE_FEATURES = 9


@torch.no_grad()
def lane_rows(pred, n_strips):
    """Row extent of each prediction as the CULane NMS kernel computes it.

    pred [..., 6+R] (official layout: logits, start_y, start_x, theta, length,
    xs). Returns (start, end), row indices counted from the bottom, inclusive;
    end < start means the prediction covers no row.
    """
    n = float(n_strips)
    start = torch.trunc((1.0 - pred[..., 2].float()) * n + 0.5).clamp(0, n_strips)
    rows = pred[..., 5].float() * n
    end = torch.trunc(start + rows - 1.0 + 0.5 - (rows < 1.0).float()).clamp(max=n_strips)
    return start.long(), end.long()


@torch.no_grad()
def pairwise_geometry(pred, n_strips, x_scale):
    """NMS geometry between the predictions of one image, pred [Q, 6+R].

    Returns [Q, Q] tensors: mean |dx| (px) over the common rows, |dx| on the
    lowest and the highest common row, the number of common rows; and the row
    extents (start, end) [Q].
    """
    start, end = lane_rows(pred, n_strips)
    xs = pred[:, 6:].float() * x_scale
    num_rows = xs.shape[-1]
    r = torch.arange(num_rows, device=pred.device).view(1, 1, -1)
    lo = torch.maximum(start[:, None], start[None, :])
    hi = torch.minimum(end[:, None], end[None, :]).clamp(max=num_rows - 1)
    count = (hi - lo + 1).clamp(min=0)
    inside = (r >= lo[..., None]) & (r <= hi[..., None])
    diff = (xs[:, None, :] - xs[None, :, :]).abs()
    mean = (diff * inside).sum(-1) / count.clamp(min=1)
    d_bottom = diff.gather(-1, lo.clamp(0, num_rows - 1)[..., None]).squeeze(-1)
    d_top = diff.gather(-1, hi.clamp(0, num_rows - 1)[..., None]).squeeze(-1)
    return mean, d_bottom, d_top, count, start, end


class LearnedLaneNMS(nn.Module):
    """o2o score correction from each prediction's higher-scored neighbours.

    Node i: its RoI feature and o2m score. Edges j -> i: the ``num_neighbors``
    predictions that outrank i (higher o2m score) and share rows with it,
    nearest first by the NMS distance. Messages are max-pooled, as NMS
    suppresses on the strongest overlap. The output layer starts at zero, so
    the corrected score starts equal to the o2m score.
    """

    def __init__(self, feat_dim, n_strips, img_w, hidden=32, num_neighbors=16,
                 dist_scale=50.0, max_dist=10.0):
        super().__init__()
        self.n_strips = int(n_strips)
        self.x_scale = float(img_w - 1)
        self.num_neighbors = int(num_neighbors)
        self.dist_scale = float(dist_scale)   # NMS threshold (px): distances in its units
        self.max_dist = float(max_dist)
        if self.num_neighbors < 1:
            raise ValueError("dedup_cfg.num_neighbors must be >= 1")
        self.node = nn.Sequential(nn.Linear(feat_dim + 2, hidden), nn.ReLU(inplace=True))
        self.edge = nn.Linear(EDGE_FEATURES, hidden)
        self.from_j = nn.Linear(hidden, hidden, bias=False)
        self.from_i = nn.Linear(hidden, hidden, bias=False)
        self.message = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(hidden, hidden),
                                     nn.ReLU(inplace=True))
        self.out = nn.Sequential(nn.Linear(2 * hidden + 1, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, 1))
        self.zero_init()

    def zero_init(self):
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    @torch.no_grad()
    def _edges(self, pred, prob):
        """Neighbour indices [B, Q, k], their validity, edge features [B, Q, k, F]
        and each prediction's distance to its nearest valid neighbour [B, Q]."""
        batch_size, num_q = prob.shape
        k = min(self.num_neighbors, max(num_q - 1, 1))
        dev = prob.device
        nbr = torch.zeros((batch_size, num_q, k), dtype=torch.long, device=dev)
        valid = torch.zeros((batch_size, num_q, k), dtype=torch.bool, device=dev)
        feats = torch.zeros((batch_size, num_q, k, EDGE_FEATURES), device=dev)
        nn_dist = torch.full((batch_size, num_q), self.max_dist, device=dev)
        idx = torch.arange(num_q, device=dev)
        n = float(self.n_strips)
        for b in range(batch_size):
            mean, d_bot, d_top, count, start, end = pairwise_geometry(
                pred[b], self.n_strips, self.x_scale)
            s = prob[b]
            # higher[i, j]: j outranks i (ties: the lower index outranks)
            higher = (s[None, :] > s[:, None]) | ((s[None, :] == s[:, None])
                                                   & (idx[None, :] < idx[:, None]))
            dist = torch.where(higher & (count > 0), mean / self.dist_scale,
                               torch.full_like(mean, float("inf")))
            d, j = dist.topk(k, dim=1, largest=False)
            ok = torch.isfinite(d)
            span = (end - start + 1).clamp(min=1).float()
            cnt = count.gather(1, j).float()
            f = torch.stack([
                d,
                d_bot.gather(1, j) / self.dist_scale,
                d_top.gather(1, j) / self.dist_scale,
                cnt / n,
                cnt / torch.minimum(span[:, None], span[j]),
                (start[j] - start[:, None]).float() / n,
                (span[j] - span[:, None]) / n,
                s[j],
                s[j] - s[:, None],
            ], dim=-1)
            f[..., :3] = f[..., :3].clamp(max=self.max_dist)
            feats[b] = torch.where(ok[..., None], f, torch.zeros_like(f))
            nbr[b] = j
            valid[b] = ok
            nn_dist[b] = torch.where(ok[:, 0], d[:, 0].clamp(max=self.max_dist),
                                     torch.full_like(d[:, 0], self.max_dist))
        return nbr, valid, feats, nn_dist

    def forward(self, pred, feat):
        """pred [B, Q, 6+R]: final-stage predictions; feat [B, Q, C]: their RoI
        features. Returns the correction to the o2m logit (class 1), [B, Q]."""
        pred = pred.detach().float()
        feat = feat.detach().float()
        logits = pred[..., :2]
        prob = torch.softmax(logits, dim=-1)[..., 1]
        margin = (logits[..., 1] - logits[..., 0]).clamp(-20.0, 20.0) / 10.0
        h = self.node(torch.cat([feat, prob[..., None], margin[..., None]], dim=-1))
        batch_size, num_q, hid = h.shape
        if num_q < 2:
            agg = h.new_zeros(batch_size, num_q, hid)
            nn_dist = h.new_full((batch_size, num_q), self.max_dist)
        else:
            nbr, valid, edge_f, nn_dist = self._edges(pred, prob)
            k = nbr.shape[-1]
            hj = torch.gather(self.from_j(h), 1, nbr.view(batch_size, num_q * k, 1)
                              .expand(-1, -1, hid)).view(batch_size, num_q, k, hid)
            msg = self.message(self.edge(edge_f) + hj + self.from_i(h)[:, :, None, :])
            # messages are >= 0 (ReLU), so 0 stands for "no neighbour"
            agg = msg.masked_fill(~valid[..., None], 0.0).max(dim=2).values
        out = self.out(torch.cat([h, agg, nn_dist[..., None] / self.max_dist], dim=-1))
        return out.squeeze(-1)
