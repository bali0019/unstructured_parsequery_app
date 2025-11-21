"""
Stage 4: Extract Entities - Extract structured data using AI Query

Uses Databricks AI Query API with configurable prompts to extract entities.
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
    EXTRACT_PROMPT,
    AI_QUERY_MODEL,
    AI_QUERY_TEMPERATURE,
    AI_QUERY_MAX_TOKENS,
    DATABRICKS_BASE_URL
)
from utils import get_databricks_token

logger = logging.getLogger(__name__)


def extract_entities(
    categorized_data: Dict[str, Any],
    prompt_template: str = None,
    model: str = None
) -> Dict[str, Any]:
    """
    Extract entities from document using Databricks AI Query API

    Args:
        categorized_data: Categorized document data from stage 3
        prompt_template: Custom prompt template (uses config default if None)
        model: Model to use for extraction (uses config default if None)

    Returns:
        Dictionary with extracted entities
    """
    # Use config defaults if not provided
    prompt_template_used = prompt_template or EXTRACT_PROMPT
    model_used = model or AI_QUERY_MODEL

    # Extract text from data
    document_text = ""
    if "pages" in categorized_data:
        document_text = "\n\n".join([
            page.get("text", "") for page in categorized_data["pages"]
        ])

    with mlflow.start_span(
        name="stage_4_extract",
        span_type="RETRIEVER",
        attributes={"stage": "extract", "model": model_used}
    ) as span:
        # Set inputs for trace
        span.set_inputs({
            "prompt_template": prompt_template_used[:200] + "..." if len(prompt_template_used) > 200 else prompt_template_used,
            "model": model_used,
            "document_text_length": len(document_text)
        })

        try:
            logger.info(f"Extracting entities from document with {len(document_text)} characters")

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
                extraction_result = json.loads(response_text)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response, using as-is: {response_text}")
                # Fallback to empty entities if parsing fails
                extraction_result = {
                    "entities": [],
                    "raw_response": response_text[:200]
                }

            # Set outputs for trace (only extraction results, not pages/parsed_doc/categorization)
            span.set_outputs({
                "entities_count": len(extraction_result.get("entities", [])),
                "entities": extraction_result.get("entities", [])[:5]  # Only first 5 entities to avoid bloat
            })

            result = {
                "status": "success",
                **categorized_data,  # Include previous data
                "extraction": extraction_result,
                "entities_count": len(extraction_result.get("entities", [])),
                "entities_extracted": len(extraction_result.get("entities", [])),
                "model_used": model_used,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Extraction successful: {len(extraction_result.get('entities', []))} entities found")
            return result

        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}", exc_info=True)

            # Set error output for trace
            span.set_outputs({
                "error": str(e)
            })

            return {
                "status": "failed",
                **categorized_data,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
