"""
constants.py
------------
Shared constants and regular expressions used across the pipeline.
"""

import re

# Base speaker role terminology
INTERVIEWEE_TERMS = "Customer|Participant|Guest|User|Client|Interviewee|Respondent"
INTERVIEWER_TERMS = "Interviewer|Researcher|Host|Moderator"
ALL_SPEAKER_TERMS = f"{INTERVIEWEE_TERMS}|{INTERVIEWER_TERMS}"

# Pre-compiled Regex Patterns for quote retrieval
INTERVIEWEE_PATTERN = re.compile(rf"^({INTERVIEWEE_TERMS})\s*:", re.IGNORECASE)
INTERVIEWER_PATTERN = re.compile(rf"^({INTERVIEWER_TERMS})\s*:", re.IGNORECASE)

# Pre-compiled Regex Patterns for transcript ingestion and chunking
SPEAKER_PATTERN = re.compile(rf"^({ALL_SPEAKER_TERMS})\s*:", re.MULTILINE | re.IGNORECASE)
TURN_PATTERN = re.compile(rf"(?=^(?:{ALL_SPEAKER_TERMS})\s*:)", re.MULTILINE | re.IGNORECASE)