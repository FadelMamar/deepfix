"""
Example usage of the Cursor CLI wrapper.

This example demonstrates how to use the Cursor wrapper to send queries
and receive responses from Cursor CLI in non-interactive mode.
"""

from deepsight.integrations import Cursor
from deepsight.integrations.cursor.errors import CursorError, CursorNotFoundError, CursorTimeoutError


def main():
    """Demonstrate usage with custom configuration."""
    print("\n=== Custom Configuration Example ===")
    
    try:
        # Initialize with custom settings
        cursor = Cursor(
            model="auto",
            output_format="text",
            timeout=60,  # 1 minute timeout
        )
        
        # Send a code-related query
        prompt = """
        Describe what a python decorator is used for.
        """
        
        print(f"Prompt: {prompt.strip()}")
        print("Sending query to Cursor CLI...")
        
        response = cursor.query(prompt)
        print(f"Response:\n{response}")
        
    except CursorNotFoundError:
        print("Error: Cursor CLI not found. Please install cursor-agent first.")
    except CursorTimeoutError:
        print("Error: Query timed out.")
    except CursorError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
