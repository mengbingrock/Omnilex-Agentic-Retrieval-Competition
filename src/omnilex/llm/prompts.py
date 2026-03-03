"""Prompt templates for legal citation retrieval.

Contains prompts for:
1. Direct generation baseline - prompting LLM to generate citations directly
2. Agentic retrieval baseline - ReAct-style agent with search tools
"""

import re

# =============================================================================
# DIRECT GENERATION PROMPTS
# =============================================================================

DIRECT_GENERATION_PROMPT = """\
You are a Swiss legal expert. Given a legal query, identify and list ALL \
relevant Swiss legal citations.

Citation formats to use:
- Federal laws: Art. [article] [Abs. paragraph] [LAW]
  (e.g., Art. 1 ZGB, Art. 11 Abs. 2 OR, Art. 117 StGB)
- Court decisions: BGE [volume] [section] [page] E. [consideration]
  (e.g., BGE 116 Ia 56 E. 2b) or docket-style (e.g., 5A_800/2019 E. 2)

Common law abbreviations: ZGB (Civil Code), OR (Code of Obligations), \
StGB (Criminal Code), BV (Constitution)

Important:
- Only list citations that are directly relevant to the query
- Include both statutory law and case law (BGE) when applicable
- Be precise with citation formats
- Do not include explanations, only the citations

Query: {query}

List only the citations, one per line:"""


DIRECT_GENERATION_PROMPT_DE = """\
Du bist ein Schweizer Rechtsexperte. Für die gegebene rechtliche Frage, \
identifiziere und liste ALLE relevanten Schweizer Rechtszitate auf.

Zitierformate:
- Bundesgesetze: Art. [Artikel] [Abs. Absatz] [GESETZ]
  (z.B. Art. 1 ZGB, Art. 11 Abs. 2 OR, Art. 117 StGB)
- Bundesgerichtsentscheide: BGE [Band] [Abteilung] [Seite] E. [Erwägung]
  (z.B. BGE 116 Ia 56 E. 2b) oder Dossiernummer (z.B. 5A_800/2019 E. 2)

Übliche Gesetzesabkürzungen: ZGB (Zivilgesetzbuch), OR (Obligationenrecht), \
StGB (Strafgesetzbuch), BV (Bundesverfassung)

Wichtig:
- Nur direkt relevante Zitate auflisten
- Sowohl Gesetzesrecht als auch Rechtsprechung (BGE) einbeziehen
- Präzise Zitierformate verwenden
- Keine Erklärungen, nur die Zitate

Frage: {query}

Liste nur die Zitate auf, eines pro Zeile:"""


# =============================================================================
# AGENTIC RETRIEVAL PROMPTS
# =============================================================================

AGENT_SYSTEM_PROMPT = """\
You are a Swiss legal research assistant with access to two search tools:

1. search_laws(query): Search Swiss federal laws by keywords
   - Returns relevant law provisions with citations and text excerpts
   - Use for finding statutory law: codes, acts, ordinances

2. search_courts(query): Search Swiss Federal Court decisions by keywords
   - Returns relevant case law with citations and excerpts
   - Use for finding judicial interpretations and precedents

Your task is to find ALL relevant Swiss legal citations for the given query.

Citation formats:
- Federal laws: Art. X [Abs. Y] LAW (e.g., Art. 1 ZGB, Art. 11 Abs. 2 OR)
- Court decisions: BGE XXX YY ZZZ E. [consideration] or docket-style (e.g., 5A_800/2019 E. 2)

Instructions:
- Search BOTH laws AND court decisions for comprehensive results
- Use multiple search queries if needed (different terms, German/English)
- Extract citations in standard format
- Continue searching until you have found all relevant sources

Format your response as:
Thought: [Your reasoning about what to search next]
Action: [tool_name]
Action Input: [search query]

After receiving results, either continue searching or provide final answer:
Final Answer: [List of all found citations, one per line]

Remember: Always search both laws AND court decisions before giving your final answer."""


AGENT_SYSTEM_PROMPT_DE = """\
Du bist ein Schweizer Rechtsrecherche-Assistent mit Zugang zu zwei Such-Tools:

1. search_laws(query): Durchsuche Schweizer Bundesgesetze nach Stichwörtern
   - Gibt relevante Gesetzesbestimmungen mit Zitaten und Textauszügen zurück
   - Verwende für Gesetzesrecht: Kodizes, Gesetze, Verordnungen

2. search_courts(query): Durchsuche Schweizer Bundesgerichtsentscheide nach Stichwörtern
   - Gibt relevante Rechtsprechung mit Zitaten und Auszügen zurück
   - Verwende für gerichtliche Auslegungen und Präzedenzfälle

Deine Aufgabe ist es, ALLE relevanten Schweizer Rechtszitate zur gegebenen Frage zu finden.

Zitierformate:
- Bundesgesetze: Art. X [Abs. Y] GESETZ (z.B. Art. 1 ZGB, Art. 11 Abs. 2 OR)
- Gerichtsentscheide: BGE XXX YY ZZZ E. [Erwägung] oder Dossiernummer (z.B. 5A_800/2019 E. 2)

Anleitung:
- Durchsuche SOWOHL Gesetze ALS AUCH Gerichtsentscheide für umfassende Ergebnisse
- Verwende bei Bedarf mehrere Suchanfragen (verschiedene Begriffe, Deutsch/Englisch)
- Zitate im Standardformat ausgeben
- Suche weiter, bis alle relevanten Quellen gefunden sind

Antwortformat:
Thought: [Deine Überlegung, was als Nächstes zu suchen ist]
Action: [tool_name]
Action Input: [Suchanfrage]

Nach Erhalt der Ergebnisse entweder weiter suchen oder Schlussantwort geben:
Final Answer: [Liste aller gefundenen Zitate, eines pro Zeile]

Merke: Suche immer sowohl Gesetze ALS AUCH Gerichtsentscheide, bevor du deine Schlussantwort gibst."""


AGENT_REACT_TEMPLATE = """\
You are a Swiss legal research assistant. Given a legal query, \
use the available tools to find relevant citations.

Available tools:
{tools}

Use this format:

Question: the legal query you must research
Thought: consider what to search for
Action: the tool to use (search_laws or search_courts)
Action Input: your search query
Observation: the search results
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I have gathered enough citations
Final Answer: list of citations, one per line

Begin!

Question: {query}
Thought:"""


# =============================================================================
# CITATION EXTRACTION PROMPTS
# =============================================================================

CITATION_EXTRACTION_PROMPT = """\
Extract all Swiss legal citations from the following text.

Citation formats to identify:
- Federal law citations: Art. X [Abs. Y] LAW
  (e.g., Art. 1 ZGB, Art. 41 OR, Art. 11 Abs. 2 OR)
- Court citations: BGE or docket-style followed by E. consideration
  (e.g., BGE 116 Ia 56 E. 2b, 5A_800/2019 E. 2)

Common law abbreviations: ZGB, OR, StGB, BV, SchKG, BGG

Text:
{text}

List each citation on a separate line. Only output the citations, nothing else:"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def format_direct_generation_prompt(query: str, language: str = "en") -> str:
    """Format direct generation prompt with query.

    Args:
        query: Legal query text
        language: Language code ("en" or "de")

    Returns:
        Formatted prompt string
    """
    if language == "de":
        return DIRECT_GENERATION_PROMPT_DE.format(query=query)
    return DIRECT_GENERATION_PROMPT.format(query=query)


def format_agent_prompt(
    query: str, tools_description: str = "", language: str = "en"
) -> str:
    """Format agent prompt with query and tool descriptions.

    Args:
        query: Legal query text
        tools_description: Description of available tools
        language: Language code ("en" or "de")

    Returns:
        Formatted prompt string
    """
    if tools_description:
        return AGENT_REACT_TEMPLATE.format(tools=tools_description, query=query)
    prompt = AGENT_SYSTEM_PROMPT_DE if language == "de" else AGENT_SYSTEM_PROMPT
    return prompt + f"\n\nQuery: {query}\n\nThought:"


def parse_citations_from_output(output: str) -> list[str]:
    """Parse citations from LLM output.

    Args:
        output: Raw LLM output text

    Returns:
        List of citation strings
    """
    citations = []

    for line in output.split("\n"):
        line = line.strip()

        # Skip empty lines and common non-citation prefixes
        if not line:
            continue
        if line.lower().startswith(("thought:", "action:", "observation:", "final answer:")):
            continue

        # Clean up common list markers
        line = line.lstrip("-•*0123456789.) ")

        # Check for law citations (Art., SR) or court citations (BGE, docket-style)
        if "SR" in line or "BGE" in line or "Art." in line or re.search(r"\d+[A-Z]_", line):
            citations.append(line)

    return citations


def parse_agent_action(response: str) -> tuple[str, str] | None:
    """Parse action and input from agent response.

    Args:
        response: Agent response text

    Returns:
        Tuple of (action_name, action_input) or None if no action found
    """
    import re

    action_match = re.search(r"Action:\s*(\w+)", response, re.IGNORECASE)
    input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)

    if action_match and input_match:
        return action_match.group(1).strip(), input_match.group(1).strip()

    return None


def extract_final_answer(response: str) -> str | None:
    """Extract final answer from agent response.

    Args:
        response: Agent response text

    Returns:
        Final answer text or None if not found
    """
    import re

    match = re.search(r"Final Answer:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return None
