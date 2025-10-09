"""
utils.py
---------
Utility functions for AI Mail Insight Assistant
Author: Harshitha

Description:
------------
This module contains reusable helper functions for text preprocessing,
category-based AI reply generation, and summary formatting.
These are used by main.py (Streamlit app) and notebooks to keep the
codebase modular and clean.
"""

import re

# ---------------------------
# 🧹 Text Preprocessing
# ---------------------------
def clean_text(text: str) -> str:
    """
    Cleans and normalizes input email text for ML classification.

    Steps:
    - Lowercases text
    - Removes email addresses, URLs, and non-alphabetic characters
    - Strips extra whitespace

    Args:
        text (str): Raw email body or subject.

    Returns:
        str: Cleaned and normalized text.
    """
    text = str(text).lower()
    text = re.sub(r'\S+@\S+', '', text)  # remove email addresses
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters/spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------
# 💌 Polite AI Reply Generator
# ---------------------------
def generate_reply(category: str) -> str:
    """
    Generates a polite, professional reply message
    based on the predicted email category.

    Args:
        category (str): The predicted intent label (HR, IT Support, etc.)

    Returns:
        str: A short, polite reply.
    """
    replies = {
        "IT Support": (
            "Hi, thanks for reaching out. "
            "We’ll look into your IT issue right away and keep you posted."
        ),
        "HR": (
            "Hello, we’ve received your HR-related request "
            "and will process it shortly. Thank you for your patience."
        ),
        "Finance": (
            "Hi, your finance query has been received. "
            "Our team will review and follow up soon."
        ),
        "Admin": (
            "Hello, your admin request has been logged. "
            "We’ll ensure it’s resolved promptly."
        ),
        "General": (
            "Hi, thank you for your message. "
            "We appreciate your update and will get back to you soon."
        ),
    }

    # Return default if category not recognized
    return replies.get(category, "Thank you for reaching out. We’ll get back to you soon.")


# ---------------------------
# 🧾 Summary Formatter
# ---------------------------
def format_summary(summary: str) -> str:
    """
    Cleans and capitalizes a generated summary for better presentation.

    Args:
        summary (str): The raw summary text.

    Returns:
        str: Nicely formatted summary string.
    """
    if not isinstance(summary, str):
        return ""
    summary = summary.strip()
    if not summary:
        return ""
    if not summary.endswith('.'):
        summary += '.'
    summary = summary[0].upper() + summary[1:]
    return summary
