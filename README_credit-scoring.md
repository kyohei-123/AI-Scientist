# Automatic credit risk modeling and documentation

# Python environment

This study was conudcted by python3.11

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Dataset preparation for model development

You need kaggle api to download this dataset.

```bsah
cd data
kaggle datasets download utkarshx27/american-companies-bankruptcy-prediction-dataset
unzip american-companies-bankruptcy-prediction-dataset.zip -d american_bankruptcy
```

# Dataset preparation as the reference material for model review

Parse reference manuals to markdown which to be used by AI Scientist as the reference document for reviewing models

```bash
cd data/references
python prepare.py --model gemini/gemini-2.0-flash --file ./pdf_02.pdf
```

# Dry run to create run_0

```bash
cd templates/credit_scoring
python experiment.py --out_dir run_0 --data_dir ../../data/american_bankruptcy
```

# Run AI-Scientist customized for credit risk scoring

```bash
python launch_scientist_custom_v1.py --model "gemini/gemini-2.0-flash" --experiment credit_scoring --num-ideas 1 --skip-novelty-check --skip-idea-generation
```

# Perfor review

```bash
python ai_scientist/perform_review_custom_v1.py --model gemini/gemini-2.0-flash --result_dir "results/credit_scoring/20250516_154706_feature_interaction_credit_scoring" --paper template.tex
```

# Component tests

TEX ファイル の作成（perform_writeup の機能確認）

```bash
python tests/perform_writeup.py --exp_dir ./results/credit_scoring/test
```

TEX pdf の生成

```bash
python tests/generate_tex.py --exp_dir ./results/credit_scoring/test
```
