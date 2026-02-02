def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    step = chunk_size - overlap

    text_chunk = []

    for i in range(0, len(tokens), step):
        start, stop = i, i+chunk_size
        chunk = tokens[start:stop]

        text_chunk.append(chunk)

        if stop >= len(tokens):
            break
    
    return text_chunk