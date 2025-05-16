import argparse
import asyncio
import os

from dotenv import load_dotenv
from pyzerox import zerox

load_dotenv()


async def main(file_path: str, model: str, output_dir: str):
    result = await zerox(file_path=file_path, model=model, output_dir=output_dir)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR + markdown conversion using ZeroX")
    parser.add_argument("--model", type=str, required=True, help="Model to use (e.g., gemini/gemini-2.0-flash)")
    parser.add_argument("--file", type=str, required=True, help="Path to the PDF file to process")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for markdown file")

    args = parser.parse_args()

    # Run the asynchronous function
    result = asyncio.run(main(args.file, args.model, args.output_dir))

    # Print markdown result
    print(result)
