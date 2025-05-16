import argparse
import json
import os
import os.path as osp
import pathlib
import sys

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model

sys.path.insert(0, pathlib.Path(__file__).parent.parent.as_posix())

from ai_scientist.llm import create_client
from ai_scientist.perform_writeup_custom_v1 import generate_latex


def test_generate_tex():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="run_0", help="実験結果の格納場所")
    args = parser.parse_args()
    exp_dir = args.exp_dir
    # 必要なパラメータを設定
    model = "gemini/gemini-2.0-flash"
    exp_file = osp.join(exp_dir, "experiment.py")
    writeup_file = osp.join(exp_dir, "latex/template.tex")
    notes = osp.join(exp_dir, "notes.txt")
    io = InputOutput(yes=True, chat_history_file=osp.join(exp_dir, "_aider.txt"))

    main_model = Model(model)
    fnames = [exp_file, writeup_file, notes]
    coder = Coder.create(
        main_model=main_model,
        fnames=fnames,
        io=io,
        stream=False,
        use_git=False,
        edit_format="diff",
    )

    generate_latex(coder, exp_dir, f"{exp_dir}/test.pdf")


if __name__ == "__main__":
    test_generate_tex()
