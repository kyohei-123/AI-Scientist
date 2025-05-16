import argparse
import glob
import os.path as osp

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model


def create_review_note(model_name, result_dir, reference_docs, note_path):
    prompt = """
You are a senior model reviewer at a regulated financial institution.
Your task is to extract a concise yet comprehensive list of review viewpoints that would be used to evaluate a credit risk model for internal risk governance.
You are given a set of reference documents. These may include:
- Internal model documentation (e.g., LaTeX files or PDFs)
- Regulatory guidelines (e.g., from the FSA, Basel Committee, or similar authorities)
- Historical review notes or internal best practice manuals
From these materials, identify key review viewpoints commonly considered by expert credit model validators. Your output must satisfy the following conditions:
- For each **review viewpoint**, include:
  1. A short **summary of the viewpoint** (e.g., "Model's variable signs should align with domain knowledge")
  2. A brief **justification** explaining why this viewpoint is important
  3. The **reference document name(s)** or source section from which the viewpoint was derived
- Avoid generic academic language. Frame each viewpoint in terms of practical risk governance (e.g., transparency, reasonableness, validation concerns).
- Output in bullet-point format as follows:
"""
    fnames = reference_docs + [note_path]
    io = InputOutput(yes=True, chat_history_file=osp.join(result_dir, "review_chat.txt"))
    coder = Coder.create(
        main_model=Model(model_name),
        fnames=fnames,
        io=io,
        stream=False,
        use_git=False,
        edit_format="diff",
    )
    coder.run(prompt)


def review(model_name, result_dir, paper_path, note_path, review_output_path):
    print("Running Aider for formal review (Step 2)...")
    fnames = [paper_path, note_path, review_output_path]
    io = InputOutput(yes=True, chat_history_file=osp.join(result_dir, "review_chat.txt"))
    coder = Coder.create(
        main_model=Model(model_name),
        fnames=fnames,
        io=io,
        stream=False,
        use_git=False,
        edit_format="diff",
    )
    review_prompt = """
You are a senior credit model validator at a regulated financial institution.

You are provided with:
1. A model documentation file (`template.tex`)
2. A set of reviewer notes (`review_note.txt`), where each note includes a review viewpoint and its associated reference document (e.g., regulatory guidance or internal governance policies).

Your task is to write a **formal model validation review report** based on these inputs.

The report must:
- Be structured into the following sections:
  - Summary
  - Strengths
  - Weaknesses
  - Questions for Model Owner
  - Recommendations
- For each issue raised, **clearly indicate the relevant reference document** that supports the concern or recommendation.
- Avoid simply listing the review questions — instead, **evaluate** the model documentation against each viewpoint and provide a conclusion (e.g., "adequate", "incomplete", "unclear").
- For weaknesses, include a suggested improvement or remediation path.
- Use professional, neutral, and precise language suitable for internal audit and governance reporting.

Avoid placeholders such as "To be determined." Your review should be as complete and specific as possible based on the available documentation.

Your output will be saved to `review_output.txt`.
"""
    coder.run(review_prompt)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True, help="Folder where model files are located")
    parser.add_argument("--model", type=str, required=True, help="Model name to use (e.g., gemini/gemini-2.0-flash)")
    parser.add_argument("--paper", type=str, default="template.tex", help="Model document tex file")
    parser.add_argument("--note_path", type=str, default="review_note.txt", help="Output review note")
    parser.add_argument("--review_output", type=str, default="review_output.txt", help="Final review output")
    return parser.parse_args()


def run_review(result_dir, model_name, paper_filename, note_filename, review_output_filename, reference_docs=None):
    paper_path = osp.join(result_dir, "latex", paper_filename)
    note_path = osp.join(result_dir, note_filename)
    review_output_path = osp.join(result_dir, review_output_filename)

    if not reference_docs:
        reference_docs = glob.glob("./data/references/*.md")

    create_review_note(model_name, result_dir, reference_docs, note_path)

    review(model_name, result_dir, paper_path, note_path, review_output_path)


def main():
    args = parse_arguments()
    run_review(
        result_dir=args.result_dir,
        model_name=args.model,
        paper_filename=args.paper,
        note_filename=args.note_path,
        review_output_filename=args.review_output,
    )


if __name__ == "__main__":
    main()
