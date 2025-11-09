"""
Through self-prompting, our agent can use an LLM as a universal parser that understands the natural structure of invoices, regardless of their format. The LLM can read an invoice like a human would, identifying key information based on context and common patterns, then output that information in a structured format our agent can process. We don’t need to write separate parsers for each vendor’s invoice format or maintain complex rules about where to find specific information - the LLM can handle all of that complexity for us.

This approach especially valuable is that it creates a reliable bridge between unstructured text and structured data processing. The agent can use this tool whenever it needs to convert natural language information into a format that other systems can process programmatically. This enables workflows where the agent can:

Receive unstructured text input from various sources
Use prompt_llm_for_json to extract relevant information in a structured format
Pass the structured data to other APIs or services for further processing
Make decisions based on the processed results

"""

@register_tool()
def prompt_llm_for_json(action_context: ActionContext, schema: dict, prompt: str):
    """
    Have the LLM generate JSON in response to a prompt. Always use this tool when you need structured data out of the LLM.
    This function takes a JSON schema that specifies the structure of the expected JSON response.
    
    Args:
        schema: JSON schema defining the expected structure
        prompt: The prompt to send to the LLM
        
    Returns:
        A dictionary matching the provided schema with extracted information
    """
    generate_response = action_context.get("llm")
    
    # Try up to 3 times to get valid JSON
    for i in range(3):
        try:
            # Send prompt with schema instruction and get response
            response = generate_response(Prompt(messages=[
                {"role": "system", 
                 "content": f"You MUST produce output that adheres to the following JSON schema:\n\n{json.dumps(schema, indent=4)}. Output your JSON in a ```json markdown block."},
                {"role": "user", "content": prompt}
            ]))

            # Check if the response has json inside of a markdown code block
            if "```json" in response:
                # Search from the front and then the back
                start = response.find("```json")
                end = response.rfind("```")
                response = response[start+7:end].strip()

            # Parse and validate the JSON response
            return json.loads(response)
            
        except Exception as e:
            if i == 2:  # On last try, raise the error
                raise e
            print(f"Error generating response: {e}")
            print("Retrying...")

invoice_schema = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "date": {"type": "string"},
        "amount": {"type": "number"},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "number"},
                    "unit_price": {"type": "number"}
                }
            }
        }
    }
}

extracted_data = prompt_llm_for_json(
    action_context=context,
    schema=invoice_schema,
    prompt="Extract invoice details from this text: 'INVOICE #1234...'"
)