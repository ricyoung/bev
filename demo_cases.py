"""Example decisions shared by the scorers."""
DEMO = [
    ("The store accepts returns within 30 days. This item was bought 12 days ago.",
     {"eligible": {"type": "boolean", "description": "Is this item within the store return window?"}}),
    ("Customer says the invoice total does not match the quote.",
     {"route": {"type": "enum", "description": "Which team should take this ticket?", "choices": ["billing", "support", "sales"],
                "choice_descriptions": {"billing": "an invoice, a charge or a refund", "support": "a product question", "sales": "a quote or a renewal"}}}),
    ("The essay has three well-argued paragraphs, no spelling errors, and a clear conclusion.",
     {"quality": {"type": "enum", "description": "Essay quality, 0 (unusable) to 3 (excellent).", "choices": ["0", "1", "2", "3"]}}),
]
