"""Token counting utilities."""

try:
    import tiktoken
except ImportError:
    tiktoken = None


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in text using tiktoken.

    Falls back to a rough character-based estimate if tiktoken is unavailable.
    """
    if tiktoken is not None:
        try:
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))

    # Fallback: ~4 chars per token for English text
    return max(1, len(text) // 4)


def count_messages_tokens(
    messages: list[dict[str, str]],
    model: str = "gpt-4o",
) -> int:
    """Count tokens in a list of chat messages."""
    total = 0
    for msg in messages:
        for value in msg.values():
            total += count_tokens(value, model)
    # Overhead: ~3 tokens per message + 1 per key
    total += len(messages) * 4
    return total
