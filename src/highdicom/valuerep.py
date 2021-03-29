from math import log10, floor

from numpy import isfinite


def get_ds_string(f: float) -> str:
    """Get a string representation of a float suitable for DS value types

    Returns a string representation of a floating point number that follows
    the constraints that apply to decimal strings (DS) value representations.
    These include that the string must be a maximum of 16 characters and
    contain only digits, and '+', '-', '.' and 'e' characters.

    Parameters
    ----------
    f: float
        Floating point number whose string representation is required

    Returns
    -------
    str:
        String representation of f, following the decimal string constraints

    Raises
    ------
    ValueError:
        If the float is not representable as a decimal string (for example
        because it has value ``nan`` or ``inf``)

    """
    if not isfinite(f):
        raise ValueError(
            "Cannot encode non-finite floats as DICOM decimal strings. "
            f"Got {f}"
        )

    fstr = str(f)
    # In the simple case, the built-in python string representation
    # will do
    if len(fstr) <= 16:
        return fstr

    # Decide whether to use scientific notation
    # (follow convention of python's standard float to string conversion)
    logf = log10(abs(f))
    use_scientific = logf < -4 or logf >= 13

    # Characters needed for '-' at start
    sign_chars = 1 if f < 0.0 else 0

    if use_scientific:
        # How many chars are taken by the exponent at the end.
        # In principle, we could have number where the exponent
        # needs three digits represent (bigger than this cannot be
        # represented by floats)
        if logf >= 100 or logf <= -100:
            exp_chars = 5  # e.g. 'e-123'
        else:
            exp_chars = 4  # e.g. 'e+08'
        remaining_chars = 14 - sign_chars - exp_chars
        return f'%.{remaining_chars}e' % f
    else:
        if logf >= 1.0:
            # chars remaining for digits after sign, digits left of '.' and '.'
            remaining_chars = 14 - sign_chars - int(floor(logf))
        else:
            remaining_chars = 14 - sign_chars
        return f'%.{remaining_chars}f' % f
