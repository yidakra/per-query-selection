"""Two-stage retrieve-then-rerank cost cascade for MultiVENT 2.0.

Stage 1 (cheap, all videos): tier A dense cosine over video embeddings, then tier B adds a caption
score over the top-K candidates. Stage 2 (expensive, escalated queries only): score the LLM event
decomposition over the same top-K. The router escalates the top-f fraction of queries by predicted
gain, using only stage-1 features.

Tier A is implemented here (numpy cosine). The caption scorer, event scorer, and gain predictor are
injected so the real models (SigLIP text tower / ColBERT captions / LLaMA event decomposition / the
ridge router) drop in without touching the control flow. `smoke_test.py` runs the whole thing on
synthetic data to check the flow.
"""
import numpy as np


def conf_features(scores):
    """Cheap-tier confidence features from one query's score vector over the gallery (numpy version
    of router_cascade_exp.conf_feats)."""
    s = np.asarray(scores, float)
    order = np.argsort(-s)
    ss = s[order]
    e = np.exp(ss - ss.max()); p = e / e.sum()
    ent = float(-(p * np.log(p + 1e-12)).sum())
    return {"top1": float(ss[0]), "margin12": float(ss[0] - ss[1]),
            "margin13": float(ss[0] - ss[2]),
            "z1": float((ss[0] - s.mean()) / (s.std() + 1e-9)),
            "entropy": ent, "maxp": float(p[0]), "top5mass": float(p[:5].sum()),
            "std": float(s.std())}


FEATURE_ORDER = ["margin12", "margin13", "z1", "entropy", "maxp", "top5mass", "std", "top1"]


class TwoStageRetriever:
    def __init__(self, video_ids, video_emb, caption_scorer, event_scorer, gain_predictor, k=1000):
        """
        video_ids     : list[str], length N
        video_emb     : np.ndarray [N, d], L2-normalized recommended
        caption_scorer: (query_text, cand_ids) -> {vid: score}   (tier B, over candidates)
        event_scorer  : (query_text, cand_ids) -> {vid: score}   (Full, over candidates)
        gain_predictor: (feature_matrix [Q, F]) -> np.ndarray [Q] predicted A->B gain (the router)
        k             : rerank depth
        """
        self.ids = list(video_ids)
        self.emb = video_emb
        self.caption_scorer = caption_scorer
        self.event_scorer = event_scorer
        self.gain_predictor = gain_predictor
        self.k = k

    def stage1(self, query_vec, query_text):
        """Tier A over all videos + tier B over the top-K. Returns (tierA scores, tierB run over
        candidates, candidate ids, stage-1 confidence features)."""
        a = self.emb @ np.asarray(query_vec, float)                 # cosine if inputs normalized
        topk = np.argsort(-a)[:self.k]
        cand = [self.ids[i] for i in topk]
        feats = conf_features(a)
        b_extra = self.caption_scorer(query_text, cand)             # tier-B caption contribution
        b_run = {vid: float(a[topk[j]]) + b_extra.get(vid, 0.0) for j, vid in enumerate(cand)}
        return a, b_run, cand, feats

    def stage2(self, query_text, cand, b_run):
        """Full rerank: add the event-decomposition score over the candidates."""
        ev = self.event_scorer(query_text, cand)
        return {vid: b_run[vid] + ev.get(vid, 0.0) for vid in cand}

    def run_all(self, queries, query_vecs, f_escalate=0.5):
        """queries: {qid: text}; query_vecs: {qid: vec}. Escalate the top-f by predicted gain.
        Returns (run {qid: {vid: score}}, info per qid)."""
        qids = list(queries)
        s1 = {}
        feat_mat = []
        for qid in qids:
            a, b_run, cand, feats = self.stage1(query_vecs[qid], queries[qid])
            s1[qid] = (b_run, cand)
            feat_mat.append([feats[k] for k in FEATURE_ORDER])
        gains = self.gain_predictor(np.array(feat_mat))             # router: predicted A->B gain
        n_esc = int(round(f_escalate * len(qids)))
        escalate = set(np.array(qids)[np.argsort(-gains)[:n_esc]])

        run, info = {}, {}
        for qid in qids:
            b_run, cand = s1[qid]
            if qid in escalate:
                run[qid] = self.stage2(queries[qid], cand, b_run)
            else:
                run[qid] = b_run
            info[qid] = {"escalated": qid in escalate, "n_cand": len(cand)}
        return run, info
