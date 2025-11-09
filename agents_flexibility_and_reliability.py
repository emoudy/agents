"""
A specialized invoice extraction tool that provides more guarantees about the extracted data.  If we later need to handle purchase orders, receipts, or contracts, we’ll need to implement new tools for each.

This specialized approach offers several advantages:

- Data Consistency: The fixed schema ensures that invoice data is always structured the same way, making it easier to work with downstream systems like databases or accounting software.
- Required Fields: We can specify which fields are required, ensuring critical information is always extracted or an error is raised if it can’t be found.
- Field Validation: The schema can include format specifications (like ensuring dates are properly formatted) and field-specific constraints.
- Focused Prompting: We can provide detailed guidance to the LLM about where to look for specific information, improving extraction accuracy.

Use specialized tools when:
- Data consistency is critical
- You have a well-defined set of document types
- You need to enforce specific validation rules
- The extracted data feeds into other systems with strict requirements
"""

@register_tool(tags=["document_processing", "invoices"])
def extract_invoice_data(action_context: ActionContext, document_text: str) -> dict:
    """
    Extract standardized invoice data from document text. This tool enforces a consistent
    schema for invoice data extraction across all documents.
    
    Args:
        document_text: The text content of the invoice to process
        
    Returns:
        A dictionary containing extracted invoice data in a standardized format
    """
    # Define a fixed schema for invoice data
    invoice_schema = {
        "type": "object",
        "required": ["invoice_number", "date", "amount"],  # These fields must be present
        "properties": {
            "invoice_number": {"type": "string"},
            "date": {"type": "string", "format": "date"},
            "amount": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "currency": {"type": "string"}
                },
                "required": ["value", "currency"]
            },
            "vendor": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "tax_id": {"type": "string"},
                    "address": {"type": "string"}
                },
                "required": ["name"]
            },
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "number"},
                        "unit_price": {"type": "number"},
                        "total": {"type": "number"}
                    },
                    "required": ["description", "total"]
                }
            }
        }
    }

    # Create a focused prompt that guides the LLM in invoice extraction
    extraction_prompt = f"""
    Extract invoice information from the following document text. 
    Focus on identifying:
    - Invoice number (usually labeled as 'Invoice #', 'Reference', etc.)
    - Date (any dates labeled as 'Invoice Date', 'Issue Date', etc.)
    - Amount (total amount due, including currency)
    - Vendor information (company name, tax ID if present, address)
    - Line items (individual charges and their details)

    Document text:
    {document_text}
    """
    
    # Use our general extraction tool with the specialized schema and prompt
    return prompt_llm_for_json(
        action_context=action_context,
        schema=invoice_schema,
        prompt=extraction_prompt
    )