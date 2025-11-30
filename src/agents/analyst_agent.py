"""
Analyst Agent

This agent generates structured fraud investigation briefs using Gemini LLM.
It analyzes enriched fraud events and produces actionable recommendations.
"""

import json
import logging
from typing import Dict, Any, List
from datetime import datetime

try:
    from google.adk import Agent, Runner
    from google.adk.sessions import Session
    from google.genai import types
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    print("Warning: Google ADK not available. Running in mock mode.")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalystAgent:
    """
    Agent for generating fraud investigation briefs.

    This agent:
    1. Analyzes enriched fraud detection events
    2. Generates structured investigation briefs
    3. Provides evidence-based reasoning
    4. Suggests specific next actions
    5. Assigns priority levels
    6. Explains model and rule decisions (explainability)
    """

    def __init__(self, model_name: str = "gemini-2.0-flash-exp", temperature: float = 0.1):
        """
        Initialize the Analyst Agent.

        Args:
            model_name: Name of the Gemini model to use
            temperature: Sampling temperature (lower = more deterministic)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.agent = None

        if ADK_AVAILABLE:
            self._init_adk_agent()
        else:
            logger.warning("ADK not available. Agent will run in limited mode.")

    def _init_adk_agent(self):
        """Initialize the ADK agent."""
        # Analyst agent doesn't need custom tools, just uses LLM for analysis
        # Create generation config
        gen_config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=2048,
            top_p=0.95,
            top_k=40
        )

        self.agent = Agent(
            name="AnalystAgent",
            model=self.model_name,
            instruction=self._get_system_instruction(),
            generate_content_config=gen_config
        )

        logger.info(f"Analyst Agent initialized with model: {self.model_name}")

    def _get_system_instruction(self) -> str:
        """Get the system instruction for the analyst agent."""
        return """You are a Senior Fraud Investigation Analyst with expertise in financial crime detection.

Your role is to analyze enriched fraud detection events and produce actionable investigation briefs.

CRITICAL SAFETY RULES:
1. NEVER make irreversible decisions (like blocking accounts permanently)
2. ONLY use information provided in the event data - DO NOT hallucinate or make assumptions
3. If data is missing, explicitly state what's missing
4. Always recommend human review for high-stakes decisions
5. Be precise with numbers and evidence
6. Cite specific data points from the provided event

Your responsibilities:
1. Analyze the complete fraud detection event including:
   - Fraud detection scores (rules + ML)
   - Customer profile and history
   - Merchant information
   - Geographic context
   - Dispute history

2. Generate a structured investigation brief with:
   - Executive Summary (2-3 sentences)
   - Evidence Bullets (specific facts from the data)
   - Priority Level (low/medium/high/critical)
   - Suggested Next Actions (specific, actionable steps)
   - Explainability (why was this flagged? which features/rules contributed?)

3. Provide explainability:
   - Identify top contributing features
   - Explain which rules were triggered and why
   - Show ML model contribution to the decision
   - Make technical results understandable to non-technical investigators

4. Suggest appropriate actions based on priority:
   - Critical: Immediate action required (block, contact customer urgently)
   - High: Review within 4 hours, may require transaction hold
   - Medium: Review within 24 hours, monitor closely
   - Low: Standard monitoring, no immediate action

OUTPUT FORMAT:
Your output MUST be valid JSON with this exact structure:
{
  "investigation_brief": {
    "case_id": "string",
    "timestamp": "ISO timestamp",
    "priority": "critical|high|medium|low",
    "summary": "2-3 sentence executive summary",
    "evidence": [
      "Evidence point 1",
      "Evidence point 2",
      "Evidence point 3"
    ],
    "suggested_actions": [
      "Specific action 1",
      "Specific action 2"
    ],
    "explainability": {
      "fraud_score_breakdown": {
        "combined_score": 0.0,
        "rule_contribution": 0.0,
        "ml_contribution": 0.0
      },
      "top_features": [
        {"feature": "name", "value": "value", "impact": "description"}
      ],
      "triggered_rules": [
        {"rule": "name", "reason": "why it triggered"}
      ]
    },
    "risk_factors": [
      "Risk factor 1",
      "Risk factor 2"
    ],
    "confidence": "high|medium|low",
    "analyst_notes": "Additional context or observations"
  }
}

Remember:
- Be factual and precise
- Use only provided data
- Recommend human oversight for critical decisions
- Make complex ML/rule results understandable
- Focus on actionable insights
"""

    def analyze_event(
        self,
        enriched_event: Dict[str, Any],
        session: 'Session' = None
    ) -> Dict[str, Any]:
        """
        Analyze an enriched fraud event and generate investigation brief.

        Args:
            enriched_event: Enriched transaction event with fraud detection and context
            session: Optional session for context

        Returns:
            Dict with investigation brief
        """
        logger.info(f"Analyzing event {enriched_event.get('transaction_id')}")

        # NOTE: Using direct analysis for now
        # Full ADK Runner integration requires additional refactoring
        return self._analyze_direct(enriched_event)

    def _analyze_with_adk(
        self,
        enriched_event: Dict[str, Any],
        session: 'Session' = None
    ) -> Dict[str, Any]:
        """Run analysis using ADK agent with Gemini."""
        # Create analysis prompt
        prompt = f"""Analyze this enriched fraud detection event and generate a structured investigation brief.

ENRICHED EVENT DATA:
{json.dumps(enriched_event, indent=2)}

Generate a complete investigation brief in the required JSON format.
Focus on actionable insights and clear explanations."""

        # Run the agent
        runner = Runner()

        if session:
            response = runner.run(
                agent=self.agent,
                user_message=prompt,
                session=session
            )
        else:
            response = runner.run(
                agent=self.agent,
                user_message=prompt
            )

        # Parse the response
        try:
            # Extract JSON from response
            response_text = str(response)
            # Try to find JSON in the response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text

            result = json.loads(json_text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from agent response, using structured fallback")
            result = self._create_fallback_brief(enriched_event, response_text)

        return result

    def _analyze_direct(self, enriched_event: Dict[str, Any]) -> Dict[str, Any]:
        """Generate investigation brief directly without ADK (fallback mode)."""
        transaction_id = enriched_event.get("transaction_id")
        fraud_detection = enriched_event.get("fraud_detection", {})
        context = enriched_event.get("context", {})
        risk_summary = enriched_event.get("risk_summary", {})

        # Extract scores
        combined_assessment = fraud_detection.get("combined_assessment", {})
        combined_score = combined_assessment.get("combined_score", 0)
        priority = combined_assessment.get("priority", "low")

        # Extract rule and ML analysis
        rule_analysis = fraud_detection.get("rule_analysis", {})
        ml_analysis = fraud_detection.get("ml_analysis", {})

        triggered_rules = rule_analysis.get("triggered_rules", [])

        # Build evidence list
        evidence = []
        evidence.append(f"Transaction amount: ${enriched_event['original_transaction'].get('amount', 0):.2f}")
        evidence.append(f"Combined fraud score: {combined_score:.3f}")

        for rule in triggered_rules:
            evidence.append(rule.get("details", ""))

        if context.get("customer", {}).get("fraud_history", 0) > 0:
            evidence.append(f"Customer has {context['customer']['fraud_history']} previous fraud cases")

        if context.get("merchant", {}).get("risk_level") == "high":
            evidence.append(f"Merchant is high-risk with {context['merchant']['fraud_rate']}% fraud rate")

        # Build suggested actions
        suggested_actions = []
        if combined_score >= 0.85:
            suggested_actions.extend([
                "IMMEDIATE: Block transaction pending investigation",
                "Contact customer via verified phone/email within 1 hour",
                "Escalate to senior fraud analyst",
                "Review all recent transactions from this customer"
            ])
        elif combined_score >= 0.7:
            suggested_actions.extend([
                "Place transaction on hold pending review",
                "Request additional verification from customer",
                "Review customer's recent transaction history",
                "Investigate merchant if multiple flags present"
            ])
        elif combined_score >= 0.5:
            suggested_actions.extend([
                "Monitor transaction closely",
                "Flag for review within 24 hours",
                "Verify with customer if amount is unusual"
            ])
        else:
            suggested_actions.append("Process normally with standard monitoring")

        # Build explainability
        top_features = []
        features = ml_analysis.get("features", {})
        for feat_name in ["amount", "geo_distance_km", "velocity_1h", "merchant_risk_score"]:
            if feat_name in features:
                top_features.append({
                    "feature": feat_name,
                    "value": features[feat_name],
                    "impact": self._explain_feature_impact(feat_name, features[feat_name])
                })

        # Create investigation brief
        brief = {
            "investigation_brief": {
                "case_id": f"CASE-{transaction_id}",
                "timestamp": datetime.now().isoformat(),
                "priority": priority,
                "summary": self._generate_summary(enriched_event, combined_score, triggered_rules),
                "evidence": evidence[:10],  # Top 10 evidence points
                "suggested_actions": suggested_actions,
                "explainability": {
                    "fraud_score_breakdown": {
                        "combined_score": combined_score,
                        "rule_contribution": rule_analysis.get("rule_score", 0),
                        "ml_contribution": ml_analysis.get("ml_score", 0)
                    },
                    "top_features": top_features,
                    "triggered_rules": [
                        {"rule": r["rule"], "reason": r["details"]}
                        for r in triggered_rules
                    ]
                },
                "risk_factors": risk_summary.get("risk_factors", []),
                "confidence": "high" if len(triggered_rules) >= 2 else "medium" if len(triggered_rules) == 1 else "low",
                "analyst_notes": f"Transaction flagged by {len(triggered_rules)} rule(s) and ML anomaly detection. "
                                f"Total risk indicators: {risk_summary.get('total_risk_indicators', 0)}"
            }
        }

        logger.info(f"Investigation brief generated for {transaction_id}: Priority={priority}")

        return brief

    def _generate_summary(
        self,
        enriched_event: Dict[str, Any],
        combined_score: float,
        triggered_rules: List[Dict[str, Any]]
    ) -> str:
        """Generate executive summary."""
        transaction = enriched_event.get("original_transaction", {})
        amount = transaction.get("amount", 0)
        customer_id = transaction.get("customer_id")

        if combined_score >= 0.85:
            severity = "Critical fraud alert"
        elif combined_score >= 0.7:
            severity = "High-risk transaction detected"
        elif combined_score >= 0.5:
            severity = "Medium-risk transaction identified"
        else:
            severity = "Low-risk transaction flagged"

        summary = (
            f"{severity} for customer {customer_id}. "
            f"${amount:.2f} transaction scored {combined_score:.2%} fraud probability "
            f"with {len(triggered_rules)} rule violations. "
            "Immediate investigation recommended." if combined_score >= 0.7
            else "Review recommended within standard timeframe."
        )

        return summary

    def _explain_feature_impact(self, feature_name: str, value: Any) -> str:
        """Explain the impact of a feature value."""
        explanations = {
            "amount": f"Transaction amount of ${value:.2f} is {'unusually high' if value > 1000 else 'within normal range'}",
            "geo_distance_km": f"Location is {value:.1f}km from home ({'suspicious' if value > 100 else 'normal'})",
            "velocity_1h": f"{value} transactions in last hour ({'high velocity' if value >= 5 else 'normal'})",
            "merchant_risk_score": f"Merchant risk score {value:.2f} ({'high risk' if value >= 0.5 else 'low risk'})",
            "amount_z_score": f"Amount is {abs(value):.1f} standard deviations from average ({'unusual' if abs(value) > 2 else 'typical'})"
        }
        return explanations.get(feature_name, f"Value: {value}")

    def _create_fallback_brief(self, enriched_event: Dict[str, Any], agent_response: str) -> Dict[str, Any]:
        """Create a fallback brief when JSON parsing fails."""
        return {
            "investigation_brief": {
                "case_id": f"CASE-{enriched_event.get('transaction_id')}",
                "timestamp": datetime.now().isoformat(),
                "priority": "medium",
                "summary": "Investigation brief generation encountered parsing issues. Manual review required.",
                "evidence": ["See agent response below"],
                "suggested_actions": ["Review raw agent output", "Manual investigation required"],
                "explainability": {
                    "fraud_score_breakdown": {"combined_score": 0, "rule_contribution": 0, "ml_contribution": 0},
                    "top_features": [],
                    "triggered_rules": []
                },
                "risk_factors": [],
                "confidence": "low",
                "analyst_notes": f"Raw agent response: {agent_response}"
            }
        }


def create_analyst_agent(
    model_name: str = "gemini-2.0-flash-exp",
    temperature: float = 0.1
) -> AnalystAgent:
    """
    Factory function to create an Analyst Agent.

    Args:
        model_name: Name of the Gemini model to use
        temperature: Sampling temperature

    Returns:
        Initialized AnalystAgent instance
    """
    return AnalystAgent(model_name=model_name, temperature=temperature)