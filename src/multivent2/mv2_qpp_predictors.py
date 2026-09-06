"""QPP predictors implemented to match github.com/Narabzad/QPP-4-RAG (querygym/qpp/run_qpp_querygym.py)
so the RQ2 table is an apples-to-apples comparison with Arabzadeh et al. (2026).

Two families, and the split matters for the multimodal argument:

  SCORE-ONLY predictors need nothing but the retrieved score list (plus the query's token count).
  They transfer to any retrieval system, including a visual channel with no text at all:
  WIG, NQC, SMV, RSD, sigma_max, sigma_x.

  DOCUMENT-STATISTICS predictors need a lexical index over the corpus: term counts, document
  frequencies, collection frequencies. That includes every pre-retrieval predictor (IDF, ICTF, SCQ,
  SCS) and also the post-retrieval Clarity, which builds an RM1 language model from the top-k
  documents and compares it to the collection model. For a channel whose documents are video frames
  these are undefined, not merely inconvenient.

Adaptation for similarity-based retrieval: QPP-4-RAG runs on BM25 and Cohere scores, which are
positive. Our channels emit cosine similarities that can be negative, and SMV/RSD take log(s/mu).
Following iQPP (Poesina et al. 2023) on adapting QPP to cosine retrieval, the score list is shifted to
be strictly positive before the log-based predictors only (s - min(s) + eps); every other predictor uses
the raw scores. The shift is monotone, so it leaves the ranking untouched.
"""
import numpy as np

EPS = 1e-9


def _shift_positive(s):
    """Monotone shift into (0, inf) so log-ratio predictors are defined on cosine scores."""
    m = s.min()
    return s - m + 1e-6 if m <= 0 else s


def wig(s, k, nq):
    """QPP-4-RAG WIG: (mean(top-k) - mean(all)) / sqrt(|q|); no-norm variant drops the corpus mean."""
    corpus = s.mean()
    return ((s[:k].mean() - corpus) / np.sqrt(max(1, nq)),
            s[:k].mean() / np.sqrt(max(1, nq)))


def nqc(s, k):
    """QPP-4-RAG NQC: std(top-k) / mean(all); no-norm variant is the bare std."""
    corpus = s.mean()
    return (s[:k].std() / (corpus if abs(corpus) > EPS else EPS), s[:k].std())


def smv(s, k):
    """QPP-4-RAG SMV: mean(s_i * |log(s_i / mean(top-k))|) over the top-k, optionally / mean(all)."""
    sp = _shift_positive(s)
    top = sp[:k]
    mu = top.mean()
    val = float(np.mean(top * np.abs(np.log(top / (mu if mu > EPS else EPS)))))
    corpus = sp.mean()
    return (val / (corpus if corpus > EPS else EPS), val)


def sigma_max(s):
    """QPP-4-RAG sigma-max: the largest std over growing prefixes of the ranked score list."""
    # cumulative std without an O(n^2) loop: std over prefix i = sqrt(E[x^2] - E[x]^2)
    n = np.arange(1, len(s) + 1)
    c1 = np.cumsum(s)
    c2 = np.cumsum(s * s)
    var = np.maximum(0.0, c2 / n - (c1 / n) ** 2)
    return float(np.sqrt(var).max())


def sigma_x(s, x, nq):
    """QPP-4-RAG sigma-x: std of the scores at least x times the top score, / sqrt(|q|)."""
    keep = s[s >= s[0] * x] if s[0] > 0 else s[s >= s[0] / max(x, EPS)]
    if len(keep) < 2:
        return 0.0
    return float(keep.std() / np.sqrt(max(1, nq)))


def score_only_suite(scores, nq, k=100, k_rsd=1000, depth=None):
    """Every predictor computable from scores alone. `scores` need not be sorted.

    `depth`, when given, is the shallow-evidence window: the predictor sees only the top-`depth`
    results, so every statistic here, including RSD, sigma_max, sigma_x0.5 and max, is computed on
    that truncated list. With depth None the behaviour is unchanged and Table 1 is reproduced."""
    s = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    if depth is not None:
        s = s[:max(1, int(depth))]
    k = min(k, len(s))
    kr = min(k_rsd, len(s))
    wig_n, wig_nn = wig(s, k, nq)
    nqc_n, nqc_nn = nqc(s, k)
    smv_n, smv_nn = smv(s, k)
    _, rsd = smv(s, kr)                      # QPP-4-RAG defines RSD as SMV-no-norm at k=1000
    return {
        "WIG_norm": float(wig_n),
        "WIG": float(wig_nn),
        "NQC_norm": float(nqc_n),
        "NQC": float(nqc_nn),
        "SMV_norm": float(smv_n),
        "SMV": float(smv_nn),
        "RSD": float(rsd),
        "sigma_max": sigma_max(s),
        "sigma_x0.5": sigma_x(s, 0.5, nq),
        "max": float(s[0]),
    }


SCORE_ONLY = ["WIG_norm", "WIG", "NQC_norm", "NQC", "SMV_norm", "SMV", "RSD",
              "sigma_max", "sigma_x0.5", "max"]

# Needs a lexical index over the documents, so it exists only for text-bearing channels.
NEEDS_DOC_TEXT = ["clarity"]


def pre_retrieval_suite(qtokens, stats):
    """Pre-retrieval predictors exactly as QPP-4-RAG defines them, over a lexical index built from a
    text channel's documents. `stats` is an Index (below).

    These need document-side statistics, so a channel must carry text for them to exist at all. The
    visual channel does not, which is why the pre-retrieval block of the RQ2 table is per-channel
    rather than global.
    """
    n_docs, total_terms = stats.n_docs, stats.total_terms
    idf, ictf_v, scq = [], [], []
    for t in qtokens:
        df, cf = stats.counts(t)
        idf.append(np.log2(n_docs / df) if df else 0.0)
        ictf_v.append(np.log2(total_terms / cf) if cf else 0.0)
        scq.append((1 + np.log2(cf)) * (np.log2(n_docs / df) if df else 0.0) if cf else 0.0)
    ql = max(1, len(qtokens))
    avg_ictf = float(np.mean(ictf_v)) if ictf_v else 0.0

    # SCS-2: KL between the query's ML model and the collection model
    scs2 = 0.0
    for t in set(qtokens):
        pml = qtokens.count(t) / ql
        _, cf = stats.counts(t)
        pcoll = cf / max(1, total_terms)
        if pcoll > 0:
            scs2 += pml * np.log2(pml / pcoll)

    return {
        "IDF_avg": float(np.mean(idf)) if idf else 0.0,
        "IDF_max": float(np.max(idf)) if idf else 0.0,
        "IDF_sum": float(np.sum(idf)) if idf else 0.0,
        "IDF_std": float(np.std(idf)) if idf else 0.0,
        "SCQ_avg": float(np.mean(scq)) if scq else 0.0,
        "SCQ_max": float(np.max(scq)) if scq else 0.0,
        "SCQ_sum": float(np.sum(scq)) if scq else 0.0,
        "avgICTF": avg_ictf,
        "SCS_1": float(np.log2(1.0 / ql) + avg_ictf),
        "SCS_2": float(scs2),
        # QPP-4-RAG's QL is literally len(qtokens). It reads no index at all, so unlike every other
        # predictor here it is identical across channels by construction -- the same number is asked to
        # choose between speech, text and frames. Included for coverage against their table, and it
        # doubles as the cleanest illustration of why the family cannot express a source choice.
        "QL": float(len(qtokens)),
    }


PRE_RETRIEVAL = ["IDF_avg", "IDF_max", "IDF_sum", "IDF_std", "SCQ_avg", "SCQ_max", "SCQ_sum",
                 "avgICTF", "SCS_1", "SCS_2", "QL"]


class Index:
    """Minimal lexical index over a text channel: document frequency, collection frequency, totals.
    Stands in for Pyserini's IndexReader, which QPP-4-RAG uses over MS MARCO."""

    def __init__(self, texts, tokenizer=None):
        from collections import Counter
        self.tok = tokenizer or (lambda s: s.lower().split())
        self.df, self.cf = Counter(), Counter()
        self.n_docs, self.total_terms = 0, 0
        for t in texts:
            toks = self.tok(t)
            if not toks:
                continue
            self.n_docs += 1
            self.total_terms += len(toks)
            self.cf.update(toks)
            self.df.update(set(toks))

    def counts(self, term):
        return self.df.get(term, 0), self.cf.get(term, 0)


def clarity(top_doc_texts, collection_df, total_terms, term_num=100):
    """Clarity (Cronen-Townsend 2002) as QPP-4-RAG computes it: build an RM1-style language model from
    the top-k retrieved documents, keep its `term_num` heaviest terms, and take the KL divergence
    against the collection model. Requires document text, so it is undefined for a visual channel.

    top_doc_texts : list[list[str]] tokenized top-k documents
    collection_df : dict token -> collection term count
    total_terms   : total tokens in the collection
    """
    tf = {}
    for toks in top_doc_texts:
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
    if not tf:
        return 0.0
    top = sorted(tf.items(), key=lambda kv: -kv[1])[:term_num]
    mass = sum(v for _, v in top)
    out = 0.0
    for t, v in top:
        p_wq = v / mass
        p_tD = collection_df.get(t, 0) / max(1, total_terms)
        if p_tD <= 0:
            continue
        out += p_wq * np.log(p_wq / p_tD)
    return float(out)
