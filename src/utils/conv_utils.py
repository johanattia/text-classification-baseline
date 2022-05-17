"""Utils function for text classification CNN"""


def normalize_backbone_size(value: str):
    backbone_size = value.lower()
    if backbone_size not in {"small", "large"}:
        raise ValueError(
            "The `backbone_size` argument must be one of "
            f'"small", "large". Received: {value}'
        )
    return backbone_size


def check_character_embedding(char_embedding: str, embed_dim: int):
    if char_embedding and embed_dim is None:
        raise ValueError("`embed_dim` must be given when `char_embedding` is `True`")
    return char_embedding, embed_dim
