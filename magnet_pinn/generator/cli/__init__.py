"""
    CLI interface for the generator module.
"""
from .cli import parse_arguments
from .helpers import print_report, validate_arguments

__all__ = ["parse_arguments", "print_report", "validate_arguments"]
