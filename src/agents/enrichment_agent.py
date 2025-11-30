"""
Enrichment Agent

This agent enriches flagged transactions with additional context including
customer profile, dispute history, merchant information, and geographic data.
"""

import json
import logging
from typing import Dict, Any
from datetime import datetime

try:
    from google.adk import Agent, Runner
    from google.adk.tools import FunctionTool
    from google.adk.sessions import Session
    from google.genai import types
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    print("Warning: Google ADK not available. Running in mock mode.")

from ..tools.enrichment_tools import (
    get_customer_profile,
    get_dispute_history,
    get_merchant_info,
    get_transaction_context,
    get_geo_location_info
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnrichmentAgent:
    """
    Agent for enriching fraud detection events with contextual data.

    This agent:
    1. Retrieves customer profile and history
    2. Fetches merchant risk information
    3. Gathers dispute history
    4. Collects geographic/location context
    5. Produces a comprehensive enriched event
    """

    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize the Enrichment Agent.

        Args:
            model_name: Name of the Gemini model to use
        """
        self.model_name = model_name
        self.agent = None

        if ADK_AVAILABLE:
            self._init_adk_agent()
        else:
            logger.warning("ADK not available. Agent will run in limited mode.")

    def _init_adk_agent(self):
        """Initialize the ADK agent with enrichment tools."""
        # Import tool functions
        from ..tools.enrichment_tools import (
            get_customer_profile,
            get_dispute_history,
            get_merchant_info,
            get_transaction_context,
            get_geo_location_info
        )

        # Create FunctionTools by wrapping Python functions
        tools = [
            FunctionTool(get_customer_profile),
            FunctionTool(get_dispute_history),
            FunctionTool(get_merchant_info),
            FunctionTool(get_transaction_context),
            FunctionTool(get_geo_location_info)
        ]

        self.agent = Agent(
            name="EnrichmentAgent",
            model=self.model_name,
            tools=tools,
            instruction=self._get_system_instruction()
        )

        logger.info(f"Enrichment Agent initialized with model: {self.model_name}")

    def _get_system_instruction(self) -> str:
        """Get the system instruction for the agent."""
        return """You are an Enrichment Agent specialized in gathering contextual data for fraud investigations.

Your responsibilities:
1. Enrich flagged transactions with comprehensive contextual data
2. Retrieve customer profiles and history
3. Gather merchant risk information
4. Collect dispute and chargeback history
5. Assess geographic/location risk factors
6. Produce a complete enriched event for fraud analysts

Enrichment workflow:
- Always use get_transaction_context first to get comprehensive data
- If more detail needed, use specific tools:
  * get_customer_profile for detailed customer info
  * get_dispute_history for dispute patterns
  * get_merchant_info for merchant risk assessment
  * get_geo_location_info for location-based risks

Important guidelines:
- Gather ALL available context - investigators need complete information
- Identify and highlight key risk factors
- Look for patterns in dispute history
- Assess merchant reliability and risk
- Note any unusual geographic patterns
- Be thorough but organized in presenting data

Output format:
Provide a well-structured enriched event with:
- Transaction summary
- Customer context (profile, history, disputes)
- Merchant context (risk level, fraud rate, history)
- Geographic context (location, risk factors)
- Key risk indicators identified during enrichment
- Overall contextual risk assessment

Remember: Your job is to provide complete context, not to make fraud decisions.
Gather and present all relevant data for the Analyst Agent to review.
"""

    def enrich_transaction(
        self,
        transaction: Dict[str, Any],
        fraud_detection_result: Dict[str, Any] = None,
        session: 'Session' = None
    ) -> Dict[str, Any]:
        """
        Enrich a transaction with contextual data.

        Args:
            transaction: Transaction data
            fraud_detection_result: Results from fraud detection (optional)
            session: Optional session for context

        Returns:
            Dict with enriched transaction data
        """
        logger.info(f"Enriching transaction {transaction.get('transaction_id')}")

        # NOTE: Using direct tool execution for now
        # Full ADK Runner integration requires additional refactoring
        return self._enrich_direct(transaction, fraud_detection_result)

    def _enrich_with_adk(
        self,
        transaction: Dict[str, Any],
        fraud_detection_result: Dict[str, Any],
        session: 'Session' = None
    ) -> Dict[str, Any]:
        """Run enrichment using ADK agent."""
        # Create enrichment prompt
        prompt = f"""Enrich this flagged transaction with all available contextual data:

Transaction:
{json.dumps(transaction, indent=2)}

Fraud Detection Result:
{json.dumps(fraud_detection_result, indent=2) if fraud_detection_result else 'Not provided'}

Please gather comprehensive context using all available tools."""

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

        result = {
            "transaction_id": transaction.get("transaction_id"),
            "enrichment_timestamp": datetime.now().isoformat(),
            "agent_response": str(response),
            "session_id": session.id if session else None
        }

        return result

    def _enrich_direct(
        self,
        transaction: Dict[str, Any],
        fraud_detection_result: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Run enrichment directly without ADK (fallback mode)."""
        customer_id = transaction.get("customer_id")
        merchant_id = transaction.get("merchant_id")
        transaction_id = transaction.get("transaction_id")

        # Get comprehensive transaction context
        transaction_context = json.loads(get_transaction_context(
            transaction_id, customer_id, merchant_id
        ))

        # Get geographic info
        geo_info = json.loads(get_geo_location_info(
            transaction.get("latitude", 37.7749),
            transaction.get("longitude", -122.4194)
        ))

        # Compile enriched event
        enriched_event = {
            "transaction_id": transaction_id,
            "enrichment_timestamp": datetime.now().isoformat(),
            "original_transaction": transaction,
            "fraud_detection": fraud_detection_result,
            "context": transaction_context,
            "geographic_info": geo_info,
            "risk_summary": {
                "contextual_risk_score": transaction_context["risk_assessment"]["combined_risk_score"],
                "risk_factors": transaction_context["risk_assessment"]["risk_factors"],
                "customer_risk": transaction_context["customer"]["fraud_history"] > 0,
                "merchant_risk": transaction_context["merchant"]["risk_level"] in ["medium", "high"],
                "location_risk": geo_info["location_risk"]["risk_level"] in ["medium", "high"],
                "total_risk_indicators": (
                    transaction_context["risk_assessment"]["risk_factor_count"] +
                    (1 if geo_info["location_risk"]["known_fraud_hotspot"] else 0)
                )
            },
            "enrichment_quality": {
                "data_completeness": "high",
                "sources_queried": ["customer_profile", "merchant_info", "dispute_history", "geo_location"],
                "missing_data": []
            }
        }

        logger.info(
            f"Transaction {transaction_id} enriched: "
            f"{enriched_event['risk_summary']['total_risk_indicators']} risk indicators found"
        )

        return enriched_event


def create_enrichment_agent(model_name: str = "gemini-2.0-flash-exp") -> EnrichmentAgent:
    """
    Factory function to create an Enrichment Agent.

    Args:
        model_name: Name of the Gemini model to use

    Returns:
        Initialized EnrichmentAgent instance
    """
    return EnrichmentAgent(model_name=model_name)