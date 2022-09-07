
def relative_error(a, b):
    """
    get the relative error between a and b
    """
    maxVal = max(abs(a), abs(b))
    if maxVal > 1.e-9:
        return abs(a - b) / maxVal  # return relative error
    else:
        return abs(a - b)  # return absolute error if a and b are almost 0