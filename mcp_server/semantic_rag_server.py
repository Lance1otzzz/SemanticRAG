from typing import Any
import os
from mcp.server.fastmcp import FastMCP
import subprocess


# Initialize FastMCP server
mcp = FastMCP("SemanticRAG")


def extract_markdown_from_pdf(pdf_path: str) -> str:
    """
    Extract Markdown content from a PDF file using the `magic-pdf` CLI tool.

    Args:
        pdf_path (str): The path to the PDF file to be processed.

    Returns:
        str: The extracted Markdown content if successful, an error message if the
             file path is invalid, or if `magic-pdf` fails during execution.
    """

    if not os.path.exists(pdf_path):
        return "Invalid PDF path"

    pdf_name = os.path.basename(pdf_path).rstrip(".pdf")

    cmd = ["magic-pdf", "-p", pdf_path, "-o", "output_temp", "-m", "auto"]
    try:
        subprocess.run(cmd, check=True)

        with open(f"output_temp/{pdf_name}/auto/{pdf_name}.md", "r") as f:
            md = f.read()

        return md

    except subprocess.CalledProcessError as e:
        return f"Error running magic-pdf: {e}"


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
