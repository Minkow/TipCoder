"""Shared code extraction, separate from generation and reward-model scoring."""


def sanitize(code, entrypoint=None):
    from evalplus.sanitize import sanitize as extract

    return extract(code, entrypoint=entrypoint)


def code_extract(text):
    from evalplus.sanitize import code_extract as extract

    return extract(text)
