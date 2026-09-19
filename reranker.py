from sentence_transformers import CrossEncoder

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Cross-encoder re-ranking over hybrid search's already-narrowed candidate list.

    A cross-encoder scores a (query, chunk) pair by running both through the
    model TOGETHER, letting it directly compare meaning rather than comparing
    two independently-computed vectors (what dense embedding search does). This
    makes it more accurate, but too slow to run against an entire corpus — which
    is exactly why it only re-scores hybrid search's top ~15-20 candidates
    rather than every chunk.

    Using cross-encoder/ms-marco-MiniLM-L-6-v2 rather than bge-reranker-large:
    same fundamental mechanism and story (real cross-encoder re-ranking), but
    ~22M params vs ~560M — realistically usable on CPU (well under a second for
    ~20 candidates, vs. 5-15+ seconds for the larger model).
    """

    def __init__(self, model_name=DEFAULT_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, top_k=3):
        """
        candidates: list of (score, chunk) tuples, e.g. HybridSearch.search()'s output.
                    The incoming `score` is ignored — it's on a different scale
                    (RRF score) than what the cross-encoder produces, so re-ranking
                    replaces it rather than combining with it.
        Returns up to top_k (cross_encoder_score, chunk) tuples, highest first.
        """
        if not candidates:
            return []

        chunks = [chunk for _score, chunk in candidates]
        pairs = [(query, chunk["text"]) for chunk in chunks]
        ce_scores = self.model.predict(pairs)

        reranked = sorted(zip(ce_scores, chunks), key=lambda x: x[0], reverse=True)
        return [(float(score), chunk) for score, chunk in reranked[:top_k]]
