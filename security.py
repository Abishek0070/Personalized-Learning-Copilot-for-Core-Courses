import hashlib
from passlib.context import CryptContext

# Set up the encryption context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def _get_prehash(password: str) -> str:
    """Internal helper to bypass the 72-byte bcrypt limit."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def hash_password(password: str) -> str:
    """Hashes a password using SHA-256 followed by Bcrypt."""
    prehashed = _get_prehash(password)
    return pwd_context.hash(prehashed)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain text password against a hashed version."""
    prehashed_plain = _get_prehash(plain_password)
    return pwd_context.verify(prehashed_plain, hashed_password)