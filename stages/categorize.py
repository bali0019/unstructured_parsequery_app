"""
Stage 3: Categorize - Classify document using AI Query

Uses Databricks AI Query API with configurable prompts to categorize documents.
Includes MLflow tracing for observability.
"""

import mlflow
from datetime import datetime
from typing import Dict, Any
import logging
import json
import os
from openai import OpenAI
from config import (
    CATEGORIZE_PROMPT,
    AI_QUERY_MODEL,
    AI_QUERY_TEMPERATURE,
    AI_QUERY_MAX_TOKENS,
    DATABRICKS_BASE_URL
)
from utils import get_databricks_token

logger = logging.getLogger(__name__)


def categorize_document(
    parsed_data: Dict[str, Any],
    prompt_template: str = None,
    model: str = None
) -> Dict[str, Any]:
    """
    Categorize document using Databricks AI Query API

    Args:
        parsed_data: Parsed document data from stage 2
        prompt_template: Custom prompt template (uses config default if None)
        model: Model to use for categorization (uses config default if None)

    Returns:
        Dictionary with categorization results
    """
    # Use config defaults if not provided
    prompt_template_used = prompt_template or CATEGORIZE_PROMPT
    model_used = model or AI_QUERY_MODEL

    # Extract text from parsed data
    document_text = ""
    if "pages" in parsed_data:
        document_text = "\n\n".join([
            page.get("text", "") for page in parsed_data["pages"]
        ])

    with mlflow.start_span(
        name="stage_3_categorize",
        span_type="LLM",
        attributes={"stage": "categorize", "model": model_used}
    ) as span:
        # Set inputs for trace
        span.set_inputs({
            "prompt_template": prompt_template_used[:200] + "..." if len(prompt_template_used) > 200 else prompt_template_used,
            "model": model_used,
            "document_text_length": len(document_text)
        })

        try:
            logger.info(f"Categorizing document with {len(document_text)} characters")

            # Format prompt with document text
            prompt = prompt_template_used.format(document_text=document_text[:5000])  # Limit to 5000 chars

            # Call Databricks AI Query using OpenAI client
            # Get OAuth token using client credentials (automatically injected by Databricks Apps)
            token = get_databricks_token()
            client = OpenAI(
                api_key=token,
                base_url=DATABRICKS_BASE_URL
            )

            response = client.chat.completions.create(
                model=model_used,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=AI_QUERY_TEMPERATURE,
                max_tokens=AI_QUERY_MAX_TOKENS
            )

            # Set LLM attributes on span for observability
            span.set_attribute("request_id", response.id)
            span.set_attribute("model", response.model)
            span.set_attribute("prompt_tokens", response.usage.prompt_tokens)
            span.set_attribute("completion_tokens", response.usage.completion_tokens)
            span.set_attribute("total_tokens", response.usage.total_tokens)
            span.set_attribute("finish_reason", response.choices[0].finish_reason)

            # Parse JSON response
            response_text = response.choices[0].message.content
            try:
                categorization_result = json.loads(response_text)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response, using as-is: {response_text}")
                # Fallback to mock structure if parsing fails
                categorization_result = {
                    "primary_category": "Unknown",
                    "primary_confidence": 0.0,
                    "primary_justification": response_text[:200],
                    "secondary_category": "Unknown",
                    "secondary_confidence": 0.0,
                    "secondary_justification": "Failed to parse response"
                }

            # Set outputs for trace (only categorization results, not pages/parsed_doc)
            span.set_outputs({
                "primary_category": categorization_result.get("primary_category"),
                "primary_confidence": categorization_result.get("primary_confidence"),
                "primary_justification": categorization_result.get("primary_justification"),
                "secondary_category": categorization_result.get("secondary_category"),
                "secondary_confidence": categorization_result.get("secondary_confidence")
            })

            result = {
                "status": "success",
                **parsed_data,  # Include parsed data
                "categorization": categorization_result,
                "model_used": model_used,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Categorization successful: {categorization_result['primary_category']}")
            return result

        except Exception as e:
            logger.error(f"Categorization failed: {str(e)}", exc_info=True)

            # Set error output for trace
            span.set_outputs({
                "error": str(e)
            })

            return {
                "status": "failed",
                **parsed_data,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
