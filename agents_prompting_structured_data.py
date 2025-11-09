"""
Example of an invoice processing system that combines specialized extraction with a simple storage mechanism. The system will use the LLM’s capabilities to understand invoice content while maintaining strict data consistency through a fixed schema.

One of the most powerful aspects of this tool-based approach is how it enables horizontal scaling of agent capabilities. Rather than constantly expanding the core goals or system prompt of an agent—which can lead to prompt bloat and conflicting instructions—we can encapsulate specific functionality in well-defined tools that the agent can access as needed.

The first tool, extract_invoice_data, acts as our intelligent document analyzer. This function uses self-prompting to take raw document text and transform it into structured data following a consistent schema. It uses a prompt that guides the LLM to identify crucial invoice elements like invoice numbers, dates, and line items. By enforcing a fixed JSON schema with required fields, the tool ensures data consistency regardless of the original invoice format. It is still possible that the LLM may hallucinate, so other techniques could be needed for a production use case, but this demonstrates the basic functionality.

The second tool, store_invoice, provides a simple persistence mechanism in a dictionary. Once an invoice has been properly extracted and structured, this function saves it to our invoice database, using the invoice number as a unique identifier. The invoices are stored separate from the memory so that they can be persisted across runs of the agent.

This implementation provides several key benefits:
- Consistent Data Structure: The fixed schema in extract_invoice_data ensures all invoices are processed into a consistent format. The prompting / logic for how to extract invoice data is separate from the agent’s core reasoning, making it easier to modify and maintain.
- Modular Design: Each tool has a single, clear responsibility, making the system easy to maintain and extend. Details for how the tools are implemented are hidden from the overall Goals of the agent.
- Error Handling: Built-in validation ensures required fields are present and data is properly formatted.
- Persistent Storage: The simple dictionary-based storage can be easily replaced with a database or other persistence mechanism by modifying the storage tools. The work that the agent does can now be persisted across runs.

"""

@register_tool(tags=["document_processing", "invoices"])
def extract_invoice_data(action_context: ActionContext, document_text: str) -> dict:
    """
    Extract standardized invoice data from document text.

    This tool ensures consistent extraction of invoice information by using a fixed schema
    and specialized prompting for invoice understanding. It will identify key fields like
    invoice numbers, dates, amounts, and line items from any invoice format.

    Args:
        document_text: The text content of the invoice to process

    Returns:
        A dictionary containing the extracted invoice data in a standardized format
    """
    invoice_schema = {
        "type": "object",
        "required": ["invoice_number", "date", "total_amount"],
        "properties": {
            "invoice_number": {"type": "string"},
            "date": {"type": "string"},
            "total_amount": {"type": "number"},
            "vendor": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {"type": "string"}
                }
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
                    }
                }
            }
        }
    }

    # Create a focused prompt for invoice extraction
    extraction_prompt = f"""
            You are an expert invoice analyzer. Extract invoice information accurately and 
            thoroughly. Pay special attention to:
            - Invoice numbers (look for 'Invoice #', 'No.', 'Reference', etc.)
            - Dates (focus on invoice date or issue date)
            - Amounts (ensure you capture the total amount correctly)
            - Line items (capture all individual charges)
            
            Stop and think step by step. Then, extract the invoice data from:
            
            <invoice>
            {document_text}
            </invoice>
    """

    # Use prompt_llm_for_json with our specialized prompt
    return prompt_llm_for_json(
        action_context=action_context,
        schema=invoice_schema,
        prompt=extraction_prompt
    )

@register_tool(tags=["storage", "invoices"])
def store_invoice(action_context: ActionContext, invoice_data: dict) -> dict:
    """
    Store an invoice in our invoice database. If an invoice with the same number
    already exists, it will be updated.
    
    Args:
        invoice_data: The processed invoice data to store
        
    Returns:
        A dictionary containing the storage result and invoice number
    """
    # Get our invoice storage from context
    storage = action_context.get("invoice_storage", {})
    
    # Extract invoice number for reference
    invoice_number = invoice_data.get("invoice_number")
    if not invoice_number:
        raise ValueError("Invoice data must contain an invoice number")
    
    # Store the invoice
    storage[invoice_number] = invoice_data
    
    return {
        "status": "success",
        "message": f"Stored invoice {invoice_number}",
        "invoice_number": invoice_number
    }