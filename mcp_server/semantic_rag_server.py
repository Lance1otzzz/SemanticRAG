from typing import Any
import os
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("SemanticRAG")


@mcp.tool()
async def get_semantic_split_points(text_file_path: str) -> list[tuple[int, int]] | str:
    """
    Suggest semantic split points in the given text, identifying where topic shifts or
    section boundaries occur.

    Args:
        text_file_path (str): Path to the local text file.

    Returns:
        list of tuple: A list of suggested split points, where each split point is
                       represented as a tuple (line_number, character_index).
    """
    if not os.path.exists(text_file_path):
        return "Invalid text file path."

    # TODO: Read text content

    return "Not implemented."


@mcp.tool()
async def extract_named_entities(text_file_path: str) -> dict[str, list[Any]] | str:
    """
    Extract named entities from the given text, such as people, locations, and key concepts.

    Args:
        text_file_path (str): Path to the local text file.

    Returns:
        dict: A dictionary of named entity categories and their extracted values,
              e.g., {"person": [...], "location": [...], "concept": [...]}.
    """

    if not os.path.exists(text_file_path):
        return "Invalid text file path."

    # TODO: Read text content

    return "Not implemented."


if __name__ == "__main__":
    # Run the server using stdio transport
    mcp.run(transport='stdio')
