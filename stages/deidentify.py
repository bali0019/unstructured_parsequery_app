"""
Stage 5: De-identify - Remove/mask PII using AI Query

Uses Databricks AI Query API with configurable prompts to identify and mask PII.
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
    DEIDENTIFY_PROMPT,
    AI_QUERY_MODEL,
    AI_QUERY_TEMPERATURE,
    AI_QUERY_MAX_TOKENS,
    DATABRICKS_BASE_URL
)
from utils import get_databricks_token

logger = logging.getLogger(__name__)


def deidentify_document(
    extracted_data: Dict[str, Any],
    prompt_template: str = None,
    model: str = None
) -> Dict[str, Any]:
    """
    De-identify document by masking PII using Databricks AI Query API

    Args:
        extracted_data: Extracted entity data from stage 4
        prompt_template: Custom prompt template (uses config default if None)
        model: Model to use for de-identification (uses config default if None)

    Returns:
        Dictionary with de-identified content and PII masking info
    """
    # Use config defaults if not provided
    prompt_template_used = prompt_template or DEIDENTIFY_PROMPT
    model_used = model or AI_QUERY_MODEL

    # Extract text from data
    document_text = ""
    if "pages" in extracted_data:
        document_text = "\n\n".join([
            page.get("text", "") for page in extracted_data["pages"]
        ])

    with mlflow.start_span(
        name="stage_5_deidentify",
        span_type="LLM",
        attributes={"stage": "deidentify", "model": model_used}
    ) as span:
        # Set inputs for trace
        span.set_inputs({
            "prompt_template": prompt_template_used[:200] + "..." if len(prompt_template_used) > 200 else prompt_template_used,
            "model": model_used,
            "document_text_length": len(document_text)
        })

        try:
            logger.info(f"De-identifying document with {len(document_text)} characters")

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
                deidentify_result = json.loads(response_text)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response, using as-is: {response_text}")
                # Fallback to empty PII items if parsing fails
                deidentify_result = {
                    "pii_items": [],
                    "raw_response": response_text[:200]
                }

            # Apply masking to extracted entities (if present)
            if "extraction" in extracted_data and "entities" in extracted_data["extraction"]:
                for entity in extracted_data["extraction"]["entities"]:
                    if entity["type"] in ["person", "organization", "email"]:
                        entity["value"] = "[REDACTED]"
                        entity["masked"] = True

            # Set outputs for trace (only PII masking results, not pages/parsed_doc/categorization/extraction)
            span.set_outputs({
                "pii_items_masked": len(deidentify_result.get("pii_items", [])),
                "pii_items": deidentify_result.get("pii_items", [])[:5]  # Only first 5 items to avoid bloat
            })

            result = {
                "status": "success",
                **extracted_data,  # Include previous data
                "deidentification": deidentify_result,
                "pii_items_masked": len(deidentify_result.get("pii_items", [])),
                "model_used": model_used,
                "final_output": True,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"De-identification successful: {len(deidentify_result.get('pii_items', []))} PII items masked")
            return result

        except Exception as e:
            logger.error(f"De-identification failed: {str(e)}", exc_info=True)

            # Set error output for trace
            span.set_outputs({
                "error": str(e)
            })

            return {
                "status": "failed",
                **extracted_data,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
