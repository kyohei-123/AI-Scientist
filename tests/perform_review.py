import argparse
import pathlib
import sys

sys.path.insert(0, pathlib.Path(__file__).parent.parent.as_posix())

from ai_scientist.perform_review_custom_v1 import run_review

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True, help="Folder where model files are located")
    parser.add_argument("--model", type=str, required=True, help="Model name to use (e.g., gemini-pro)")
    parser.add_argument("--paper", type=str, default="model_documentation.pdf", help="Model document file (PDF)")
    parser.add_argument("--note_output", type=str, default="review_note.txt", help="Output review note")
    parser.add_argument("--review_output", type=str, default="review_output.txt", help="Final review output")

    # def run_review(result_dir, model_name, pdf_filename, note_filename, review_output_filename, reference_docs=None):
    args = parser.parse_args()
    run_review(
        result_dir=args.result_dir,
        model_name=args.model,
        paper_filename=args.paper,
        note_filename=args.note_output,
        review_output_filename=args.review_output,
    )
