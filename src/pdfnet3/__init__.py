"""pDFNet3: Personalized DeepFilterNet3 for Real-Time Neural Speech Isolation."""

__version__ = "1.0.0"

from pdfnet3.film import HIDDEN, SITES, add_film, verify_film_graph
from pdfnet3.pipeline import (
    PROVIDER_ALIASES,
    SUPPORTED_PROVIDERS,
    PDFNet3Session,
    auto_select_provider,
    get_available_execution_providers,
    get_available_providers,
    get_recommended_provider,
    resolve_execution_providers,
)

__all__ = [
    "HIDDEN",
    "PROVIDER_ALIASES",
    "SITES",
    "SUPPORTED_PROVIDERS",
    "PDFNet3Session",
    "__version__",
    "add_film",
    "auto_select_provider",
    "get_available_execution_providers",
    "get_available_providers",
    "get_recommended_provider",
    "resolve_execution_providers",
    "verify_film_graph",
]
