"""
prompts.py
----------
Centralized prompt templates for the insight agent.
Keeping prompts separate from logic makes iteration and A/B testing easy.
"""


THEME_EXTRACTION_PROMPT = """You are an expert qualitative researcher analyzing a customer interview transcript.

Your task is to extract structured insights from the transcript below.

Return ONLY a valid JSON object with this exact structure:
{{
  "themes": [
    {{
      "title": "Short theme name (3-5 words)",
      "description": "One sentence describing this theme",
      "frequency": "high | medium | low",
      "sentiment": "positive | negative | neutral | mixed"
    }}
  ],
  "pain_points": [
    {{
      "title": "Short pain point label",
      "description": "What the customer struggles with",
      "severity": "critical | moderate | minor",
      "quote": "A direct quote from the transcript illustrating this pain point"
    }}
  ],
  "feature_requests": [
    {{
      "title": "Feature name",
      "description": "What the customer wants",
      "quote": "Supporting quote if available"
    }}
  ],
  "overall_sentiment": "positive | negative | neutral | mixed",
  "sentiment_summary": "One sentence summarizing the customer's overall emotional tone",
  "recommendations": [
    {{
      "title": "Short recommendation title",
      "description": "Actionable product or UX recommendation based on the interview",
      "priority": "high | medium | low"
    }}
  ]
}}

Rules:
- Return ONLY the JSON. No preamble, no markdown fences, no explanation.
- All quotes must be verbatim from the transcript.
- Be specific. Avoid generic observations.
- Extract 3-6 themes, 2-5 pain points, 1-4 feature requests, 2-4 recommendations.

TRANSCRIPT:
{transcript}
"""


EVIDENCE_INSIGHT_PROMPT = """You are a senior product researcher. 
Given a customer insight and supporting quotes, write a crisp, evidence-backed insight card.

Insight topic: {topic}

Supporting quotes from the interview:
{quotes}

Return ONLY a valid JSON object with this structure:
{{
  "insight": "One clear sentence stating what you learned",
  "evidence": ["quote 1", "quote 2"],
  "recommendation": "One concrete product or design recommendation",
  "confidence": "high | medium | low"
}}

Rules:
- Return ONLY the JSON. No markdown, no explanation.
- The insight should be a synthesis, not a restatement of the quotes.
- The recommendation should be specific and actionable.
"""


AGGREGATE_INSIGHT_PROMPT = """You are an expert qualitative researcher analyzing a set of customer interview transcripts.

Each interview is labeled with metadata and the transcript content below.

Your task is to extract both the overall research findings and cross-session patterns.

Return ONLY a valid JSON object with this structure:
{{
  "themes": [
    {{
      "title": "Short theme name (3-5 words)",
      "description": "One sentence describing this theme",
      "frequency": "high | medium | low",
      "sentiment": "positive | negative | neutral | mixed"
    }}
  ],
  "pain_points": [
    {{
      "title": "Short pain point label",
      "description": "What customers struggle with",
      "severity": "critical | moderate | minor",
      "quote": "A direct quote from the transcripts illustrating this pain point"
    }}
  ],
  "feature_requests": [
    {{
      "title": "Feature name",
      "description": "What the customer wants",
      "quote": "Supporting quote if available"
    }}
  ],
  "overall_sentiment": "positive | negative | neutral | mixed",
  "sentiment_summary": "One sentence summarizing the group emotional tone",
  "recommendations": [
    {{
      "title": "Short recommendation title",
      "description": "Actionable product or UX recommendation based on the interviews",
      "priority": "high | medium | low"
    }}
  ],
  "cross_session_findings": [
    {{
      "title": "Insight title",
      "description": "What this pattern means across interviews",
      "supporting_sessions": ["Interview 1", "Interview 2"]
    }}
  ],
  "trend_signals": [
    {{
      "signal": "Short signal name",
      "description": "How this trend is evolving across the research set"
    }}
  ],
  "research_next_steps": [
    {{
      "title": "Next step",
      "description": "What the research or product team should explore or validate next"
    }}
  ]
}}

Rules:
- Return ONLY the JSON. No preamble, no markdown fences, no explanation.
- Use quotes verbatim from the transcripts when possible.
- Be specific and call out patterns that appear across multiple interviews.

INTERVIEWS:
{transcripts}
"""


RESEARCH_ASSISTANT_PROMPT = """You are a research assistant helping a product team interpret customer interview evidence.

Question: {query}

Relevant supporting quotes from the transcripts:
{quotes}

Return ONLY a valid JSON object with this structure:
{{
  "answer": "A concise, research-backed answer to the question",
  "supporting_quotes": ["quote 1", "quote 2"],
  "recommended_next_steps": "One or two specific next steps for the product/research team"
}}

Rules:
- Return ONLY JSON, no markdown fences or extra explanation.
- Base your answer on the evidence provided.
- If the evidence is weak, be transparent and suggest how to verify.
"""


QUERY_EXPANSION_PROMPT = """You are helping a researcher search through customer interview transcripts.

Original query: {query}

Generate 3 alternative phrasings of this query that might surface different but relevant passages.
Return ONLY a JSON array of strings. Example: ["phrase 1", "phrase 2", "phrase 3"]
"""


DIARIZE_PROMPT = """You are an expert audio transcription editor.
I am providing you with a raw transcript from a customer interview that currently lacks speaker labels.

Your task is to reconstruct the dialogue by logically inferring the speakers and adding "Interviewer:" and "Customer:" labels.

Rules:
- Maintain the exact original words. Do not summarize or paraphrase.
- Add "Interviewer:" for the person asking questions and driving the conversation.
- Add "Customer:" for the person answering and sharing their experience.
- Add a blank line between different speaker turns.
- Return ONLY the formatted transcript text. No preamble, no markdown code blocks.

RAW TRANSCRIPT:
{transcript}
"""
