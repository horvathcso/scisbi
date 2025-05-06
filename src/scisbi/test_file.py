def fibonacci(n):
    """
    Calculate the n-th Fibonacci number.

    The Fibonacci sequence is defined as:
        F(0) = 0, F(1) = 1,
        F(n) = F(n-1) + F(n-2) for n >= 2.

    Args:
        n (int): The index of the Fibonacci number to calculate. Must be a non-negative integer.

    Returns:
        int: The n-th Fibonacci number.

    Raises:
        ValueError: If n is a negative integer.
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer.")

    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
