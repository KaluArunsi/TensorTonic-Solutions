def percent_change(series):
    """
    Compute the fractional change between consecutive values.
    """
    pct = []

    n = len(series) - 1
    i = 1

    while i <= n:
        prev = series[i-1]
        curr = series[i]

        if prev == 0:
            pct.append(prev)
        else:
            op = (curr - prev) / prev
            op = float(op)
            pct.append(op)
        i += 1

    return pct