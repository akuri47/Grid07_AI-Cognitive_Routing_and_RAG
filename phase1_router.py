"""
Phase 1: Vector-Based Persona Matching (The Router)
Uses FAISS + sentence-transformers to find bots that "care" about a given post.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple

# ─── Bot Persona Definitions ────────────────────────────────────────────────

BOT_PERSONAS = {
    "Bot_A": (
        "I believe AI and crypto will solve all human problems. I am highly optimistic "
        "about technology, Elon Musk, and space exploration. I dismiss regulatory concerns."
    ),
    "Bot_B": (
        "I believe late-stage capitalism and tech monopolies are destroying society. "
        "I am highly critical of AI, social media, and billionaires. I value privacy and nature."
    ),
    "Bot_C": (
        "I strictly care about markets, interest rates, trading algorithms, and making money. "
        "I speak in finance jargon and view everything through the lens of ROI."
    ),
}

# ─── Router Class ────────────────────────────────────────────────────────────

class PersonaRouter:
    """
    Embeds bot personas into a FAISS index and routes incoming posts
    to the bots with the highest cosine similarity scores.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[Router] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.bot_ids: List[str] = []
        self.index = None
        self._build_index()

    def _build_index(self):
        """Embed all personas and store them in a FAISS flat index (cosine similarity)."""
        texts = list(BOT_PERSONAS.values())
        self.bot_ids = list(BOT_PERSONAS.keys())

        print("[Router] Generating persona embeddings …")
        embeddings = self.model.encode(texts, normalize_embeddings=True)  # L2-normalised → dot product = cosine

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)          # Inner Product on unit vectors == cosine similarity
        self.index.add(embeddings.astype(np.float32))
        print(f"[Router] FAISS index built with {self.index.ntotal} persona vectors (dim={dim})\n")

    def route_post_to_bots(
        self,
        post_content: str,
        threshold: float = 0.30,          # adjust per embedding model; 0.30 works well for MiniLM
    ) -> List[Tuple[str, float]]:
        """
        Embed the post and return bots whose persona cosine-similarity exceeds `threshold`.

        Args:
            post_content: The incoming social-media post text.
            threshold:    Minimum cosine similarity to include a bot (0-1).

        Returns:
            List of (bot_id, similarity_score) tuples, sorted descending.
        """
        print(f"[Router] Routing post: \"{post_content}\"")
        query_vec = self.model.encode([post_content], normalize_embeddings=True).astype(np.float32)

        # Query all stored vectors
        scores, indices = self.index.search(query_vec, k=len(self.bot_ids))
        scores, indices = scores[0], indices[0]      # unwrap batch dimension

        matched: List[Tuple[str, float]] = []
        print("[Router] Similarity scores:")
        for idx, score in zip(indices, scores):
            bot_id = self.bot_ids[idx]
            print(f"         {bot_id}: {score:.4f}")
            if score >= threshold:
                matched.append((bot_id, float(score)))

        matched.sort(key=lambda x: x[1], reverse=True)
        print(f"[Router] Matched bots (threshold={threshold}): {[b for b, _ in matched]}\n")
        return matched


# ─── Quick Demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    router = PersonaRouter()

    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "The Fed raised interest rates again. Bond yields spike as traders scramble.",
        "Big Tech is spying on us. We need open-source alternatives and digital privacy.",
        "Bitcoin hits new all-time high amid regulatory ETF approvals.",
        "SpaceX Starship successfully completed its orbital test flight.",
    ]

    for post in test_posts:
        results = router.route_post_to_bots(post, threshold=0.30)
        if results:
            print(f"  → Post routed to: {[r[0] for r in results]}")
        else:
            print("  → No bots matched (below threshold).")
        print("-" * 70)
