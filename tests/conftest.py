"""
Pytest shared fixtures and configuration.
"""

import os
import sys
import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from src.api.app import create_app


@pytest.fixture(scope="session")
def app():
    return create_app()


@pytest.fixture(scope="session")
def client(app):
    return TestClient(app)


@pytest.fixture
def sample_invoice_text():
    return """
    TAX INVOICE
    Invoice Number: INV/2024/001
    Date: 15-01-2024
    Due Date: 30-01-2024

    Bill To:
    Customer: Sharma Enterprises
    Email: rajesh.sharma@sharmaenterprises.com
    Phone: +91 98765 43210
    GSTIN: 27AAAAA0000A1Z5

    From: Tech Solutions India Pvt. Ltd.

    Services:
    Software Development   4   25000   100000
    Cloud Hosting          1   15000    15000
    Support Services      12    2000    24000

    Total Amount: ₹1,39,000.00
    Grand Total: ₹1,63,820.00 (incl. GST)
    """


@pytest.fixture
def minimal_invoice_text():
    return """
    INVOICE
    Date: 16-01-2024
    Total: ₹75,000.00
    """


@pytest.fixture
def international_invoice_text():
    return """
    COMMERCIAL INVOICE
    Invoice #: INT/2024/007
    Client: Global Imports Inc.
    Amount: $15,000.00
    Contact: info@globalimports.com
    """
