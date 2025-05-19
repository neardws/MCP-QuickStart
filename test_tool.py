import asyncio
from agents.tool import Tool # Make sure this import path is correct for your setup

# 1. Simplest synchronous function
def simple_sync_func(name: str, age: int = 30):
    """A simple synchronous tool function."""
    print(f"simple_sync_func called with name='{name}', age={age}")
    return f"Hello, {name}! You are {age}."

# 2. Simplest asynchronous function
async def simple_async_func(item_id: str):
    """A simple asynchronous tool function."""
    print(f"simple_async_func called with item_id='{item_id}'")
    await asyncio.sleep(0.01) # Simulate async work
    return f"Processed item {item_id}"

async def main():
    print("--- Testing Tool with simple_sync_func (parameters=None) ---")
    try:
        tool1 = Tool(
            name="my_sync_tool",
            description="A very simple synchronous tool.",
            func=simple_sync_func,
            parameters=None # Let Pydantic infer from type hints
        )
        print("SUCCESS: Tool with simple_sync_func created.")
        # You can optionally try to see the inferred schema if Pydantic v2
        # from pydantic import TypeAdapter
        # if hasattr(tool1, 'parameters_schema'): # older agents might have this
        #     print("Inferred Schema (tool1.parameters_schema):", tool1.parameters_schema)
        # elif hasattr(tool1, 'model_fields') and 'parameters' in tool1.model_fields: # Pydantic v2 style
        #     # This is a bit more involved to get the actual schema for parameters if inferred
        #     pass


    except Exception as e:
        print(f"ERROR creating Tool with simple_sync_func: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing Tool with simple_async_func (parameters=None) ---")
    try:
        tool2 = Tool(
            name="my_async_tool",
            description="A very simple asynchronous tool.",
            func=simple_async_func,
            parameters=None # Let Pydantic infer
        )
        print("SUCCESS: Tool with simple_async_func created.")
    except Exception as e:
        print(f"ERROR creating Tool with simple_async_func: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing Tool with simple_sync_func and explicit minimal schema ---")
    minimal_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The name to greet."},
            "age": {"type": "integer", "description": "The age.", "default": 30}
        },
        "required": ["name"]
    }
    try:
        tool3 = Tool(
            name="my_sync_tool_with_schema",
            description="A simple synchronous tool with an explicit schema.",
            func=simple_sync_func,
            parameters=minimal_schema
        )
        print("SUCCESS: Tool with simple_sync_func and explicit schema created.")
    except Exception as e:
        print(f"ERROR creating Tool with simple_sync_func and explicit schema: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 