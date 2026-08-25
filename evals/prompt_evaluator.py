"""
Prompt Evaluation Framework
Evaluates agent system prompts (personas) to assess whether they effectively
guide the LLM toward the intended behavior.

Scores prompts on:
  - Role Clarity: Is the agent's role/persona clearly defined?
  - Constraint Effectiveness: Are guardrails (no hallucination, stay on catalog) well-defined?
  - Task Alignment: Does the prompt align with the agent's actual responsibility?
  - Tone Guidance: Does it establish the right conversational tone?
  - Completeness: Does it cover edge cases and expected behavior?
  - Conciseness: Is it focused without unnecessary verbosity?

Each dimension is scored 1-5, with an overall weighted average.
"""
import json
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from utils.llm_setup import get_llm


PROMPT_EVAL_SYSTEM = """You are an expert prompt engineer evaluating system prompts used for AI agents in a multi-agent guitar shopping assistant.

You will be given:
- The agent's NAME and intended ROLE
- The agent's SYSTEM PROMPT (the persona/instructions given to the LLM)

Your job is to evaluate how well the system prompt will guide the LLM to fulfill its intended role.

Score each dimension from 1 to 5:

**Role Clarity** (1-5): Is the agent's identity and role clearly communicated?
- 5: Crystal clear who the agent is, what it does, and its boundaries
- 3: Role is implied but could be confused with other agents
- 1: No clear role definition, could be any generic assistant

**Constraint Effectiveness** (1-5): Are guardrails well-defined to prevent bad behavior?
- 5: Strong, explicit constraints (no hallucination, stay on data, handle edge cases)
- 3: Some constraints but with loopholes or ambiguity
- 1: No meaningful constraints, agent could easily go off-rails

**Task Alignment** (1-5): Does the prompt match what the agent is supposed to do?
- 5: Every instruction directly supports the agent's core function
- 3: Mostly aligned but includes irrelevant guidance or misses key tasks
- 1: Prompt describes a different role than what the agent actually does

**Tone Guidance** (1-5): Does it establish appropriate conversational style?
- 5: Clear tone direction (friendly, professional, etc.) with examples/specifics
- 3: General tone mentioned but not well-specified
- 1: No tone guidance, agent will default to generic LLM style

**Completeness** (1-5): Does it handle edge cases and expected behaviors?
- 5: Covers normal flow, edge cases, what to do when uncertain, and fallback behavior
- 3: Covers happy path but not edge cases
- 1: Minimal instructions, leaves too much to interpretation

**Conciseness** (1-5): Is it focused and not bloated?
- 5: Every sentence adds value, no redundancy, optimal length
- 3: Somewhat verbose, some repetition but still usable
- 1: Excessively long, contradictory, or full of filler

Respond ONLY with valid JSON in this exact format:
{{
    "role_clarity": {{"score": <1-5>, "reason": "<one sentence>"}},
    "constraint_effectiveness": {{"score": <1-5>, "reason": "<one sentence>"}},
    "task_alignment": {{"score": <1-5>, "reason": "<one sentence>"}},
    "tone_guidance": {{"score": <1-5>, "reason": "<one sentence>"}},
    "completeness": {{"score": <1-5>, "reason": "<one sentence>"}},
    "conciseness": {{"score": <1-5>, "reason": "<one sentence>"}},
    "suggestions": ["<improvement suggestion 1>", "<improvement suggestion 2>", "..."],
    "overall_assessment": "<one paragraph summary>"
}}"""


class PromptEvaluator:
    """Evaluates agent system prompts for effectiveness."""

    WEIGHTS = {
        "role_clarity": 0.20,
        "constraint_effectiveness": 0.25,
        "task_alignment": 0.20,
        "tone_guidance": 0.10,
        "completeness": 0.15,
        "conciseness": 0.10,
    }

    def __init__(self):
        self.llm = get_llm()

    def evaluate_prompt(
        self,
        agent_name: str,
        agent_role: str,
        system_prompt: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a single agent's system prompt.

        Args:
            agent_name: Name of the agent (e.g., "Information Agent")
            agent_role: Brief description of what the agent should do
            system_prompt: The actual system prompt being evaluated

        Returns:
            Dict with dimension scores, suggestions, and overall score (0-100).
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_EVAL_SYSTEM),
            ("user", "Agent Name: {agent_name}\nIntended Role: {agent_role}\n\nSystem Prompt to Evaluate:\n---\n{system_prompt}\n---"),
        ])

        chain = prompt | self.llm
        raw = chain.invoke({
            "agent_name": agent_name,
            "agent_role": agent_role,
            "system_prompt": system_prompt,
        })

        try:
            result = json.loads(raw.content)
        except json.JSONDecodeError:
            content = raw.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(content[start:end])
            else:
                return {
                    "error": "Failed to parse evaluation response",
                    "raw_response": raw.content,
                }

        # Compute overall weighted score
        total = 0.0
        for dim, weight in self.WEIGHTS.items():
            score = result.get(dim, {}).get("score", 3)
            total += score * weight
        overall_score = round((total / 5) * 100, 1)

        return {
            "agent_name": agent_name,
            "dimensions": {k: result[k] for k in self.WEIGHTS if k in result},
            "overall_score": overall_score,
            "suggestions": result.get("suggestions", []),
            "overall_assessment": result.get("overall_assessment", ""),
        }

    def evaluate_all_agents(self) -> Dict[str, Any]:
        """
        Evaluate all agent prompts in the system.
        Imports agents and extracts their system prompts automatically.
        """
        from agents.information_agent import InformationAgent
        from agents.recommendation_agent import RecommendationAgent
        from agents.negotiator_agent import NegotiatorAgent

        agents_config = [
            {
                "name": "Information Agent",
                "role": "Retrieves and presents guitar specs, brand info, and features from the catalog using RAG. Should only discuss guitars present in the knowledge base.",
                "prompt": InformationAgent().system_prompt,
            },
            {
                "name": "Recommendation Agent",
                "role": "Analyzes customer preferences (skill level, budget, genre, playing style) and recommends matching guitars from the catalog. Should ask clarifying questions when needed.",
                "prompt": RecommendationAgent().system_prompt,
            },
            {
                "name": "Negotiator Agent",
                "role": "Handles pricing inquiries, suggests deals and bundles, and helps customers understand the value of guitars. Should be persuasive but honest.",
                "prompt": NegotiatorAgent().system_prompt,
            },
        ]

        results = []
        for agent in agents_config:
            result = self.evaluate_prompt(
                agent_name=agent["name"],
                agent_role=agent["role"],
                system_prompt=agent["prompt"],
            )
            results.append(result)

        # Aggregate
        valid = [r for r in results if "error" not in r]
        avg_score = sum(r["overall_score"] for r in valid) / len(valid) if valid else 0

        return {
            "results": results,
            "aggregate": {
                "overall_avg_score": round(avg_score, 1),
                "total_evaluated": len(valid),
                "strongest_agent": max(valid, key=lambda x: x["overall_score"])["agent_name"] if valid else None,
                "weakest_agent": min(valid, key=lambda x: x["overall_score"])["agent_name"] if valid else None,
            },
        }
