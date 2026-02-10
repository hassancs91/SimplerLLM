"""
Vector Database Exceptions.

Custom exception classes for vector database operations, providing
specific error types for different failure scenarios.

Exception Hierarchy:
    VectorDBError (base)
    ├── VectorNotFoundError
    ├── DimensionMismatchError
    ├── VectorDBConnectionError
    └── VectorDBOperationError

Example:
    >>> from SimplerLLM.vectors import VectorDB, VectorProvider
    >>> from SimplerLLM.vectors.exceptions import VectorNotFoundError
    >>>
    >>> db = VectorDB.create(provider=VectorProvider.LOCAL, db_folder="./vectors")
    >>> try:
    ...     result = db.get_vector_by_id("non-existent-id")
    ... except VectorNotFoundError as e:
    ...     print(f"Vector not found: {e}")
"""


class VectorDBError(Exception):
    """
    Base exception for all VectorDB operations.

    All vector database exceptions inherit from this class,
    making it easy to catch any vector-related error.

    Example:
        >>> try:
        ...     db.add_vector(vector, meta)
        ... except VectorDBError as e:
        ...     print(f"Database error: {e}")
    """
    pass


class VectorNotFoundError(VectorDBError):
    """
    Raised when a vector ID is not found in the database.

    This exception is raised when attempting to retrieve, update,
    or delete a vector using an ID that doesn't exist.

    Example:
        >>> try:
        ...     result = db.get_vector_by_id("missing-id")
        ... except VectorNotFoundError:
        ...     print("Vector does not exist")
    """
    pass


class DimensionMismatchError(VectorDBError):
    """
    Raised when vector dimensions don't match the database dimension.

    Once a database's dimension is set (either explicitly or by the
    first vector added), all subsequent vectors must have the same
    dimension.

    Example:
        >>> db = VectorDB.create(provider=VectorProvider.LOCAL, db_folder="./vectors")
        >>> db.add_vector([0.1, 0.2, 0.3], {"text": "3D vector"})  # Sets dimension to 3
        >>> try:
        ...     db.add_vector([0.1, 0.2], {"text": "2D vector"})  # Wrong dimension!
        ... except DimensionMismatchError as e:
        ...     print(f"Dimension error: {e}")
    """
    pass


class VectorDBConnectionError(VectorDBError):
    """
    Raised when connection to a vector database fails.

    This exception is specific to remote database providers like Qdrant
    and indicates network or authentication issues.

    Example:
        >>> try:
        ...     db = VectorDB.create(
        ...         provider=VectorProvider.QDRANT,
        ...         url="invalid-host",
        ...         port=6333
        ...     )
        ... except VectorDBConnectionError as e:
        ...     print(f"Connection failed: {e}")
    """
    pass


class VectorDBOperationError(VectorDBError):
    """
    Raised when a database operation fails.

    This is a general exception for operations that fail for reasons
    other than connection issues or missing vectors.

    Example:
        >>> try:
        ...     db.clear_database()
        ... except VectorDBOperationError as e:
        ...     print(f"Operation failed: {e}")
    """
    pass
