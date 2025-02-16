"""
Dummy Authentication
"""
from functools import wraps


def auth_required(f):
    """Decorator to require authentication"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        return f(*args, **kwargs)

    #decorated_function.__name__ = f.__name__
    return decorated_function
