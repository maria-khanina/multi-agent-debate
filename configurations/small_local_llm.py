MODEL = "qwen2.5:7b"
TEMPERATURE = 0.2
MAX_TOKENS_PER_TURN = 200


RESEARCH_INIT_PROMPT = """
    You search for scientific papers that can be useful in the debate.
    Debate: {motion}
    Stance: {stance}

    You have {max_tries} attempts to search for papers.
    Output EXACTLY ONE short search query.
    Do not include quotes or conversational text.
"""

RESEARCH_MIDDLE_PROMPT = """
    Result: {approved} papers approved and saved.
    Attempts left: {attempts_left}.
    Output your next DISTINCT search query.
"""

PREPARATION_TRIES = 3

OPENING_PROMPT = """You are a debater. 
Motion: {motion}.
Your Stance: {stance}.

Rule: Write a strong, one-paragraph opening statement laying out
 your core argument.
"""

DEBATOR_PROMPT = """You are a debater.
Motion: {motion}
Your Stance: {stance}

Rules:
1. Read the transcript.
2. Directly attack the last point made by your opponent.
3. Provide one strong piece of evidence for your stance.
4. Do not repeat previous arguments. If the debate is stuck,
 introduce a completely new angle.
4. Keep your answer strictly to one short paragraph (maximum 4 sentences).
5. Write organically, with no headers, use simple language.

Transcript:
{transcript}

Your Argument:"""


JUDGE_PROMPT = """You are a debate judge.
Motion: {motion}

Rules:
1. Read the transcript below.
2. Evaluate who won:
    - Penalize debaters who change the subject when losing an argument.
    - Penalize debaters who just repeat the same point in different words.
    - Reward debaters who directly dismantle the specific logic of their opponent.
    - Reward logical consistancy, clarity, politeness
3. Keep your verdict strictly to one paragraph. State the winner clearly,
    cite the exact points made by debaters that mostly affected your verdict.

Transcript:
{transcript}

Your Verdict:"""
