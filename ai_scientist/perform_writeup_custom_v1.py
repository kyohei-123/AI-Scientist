import argparse
import json
import os.path as osp
import re
import shutil
import subprocess

from ai_scientist.llm import AVAILABLE_LLMS, create_client

# 企業向けモデル記述書用のシステムメッセージを定義
model_documentation_system_msg = """
You are an experienced model risk management professional who creates model documentation for internal governance and regulatory review.
Your task is to help develop comprehensive, clear, and technically sound model documentation that follows industry best practices and meets regulatory expectations.
Focus on documenting the model in a practical business context, with emphasis on model purpose, methodology, implementation, validation, and governance.
Use a professional, technically precise but accessible style appropriate for both technical stakeholders and business executives.
"""


# GENERATE LATEX
def generate_latex(coder, folder_name, pdf_file, timeout=30, num_error_corrections=5):
    folder = osp.abspath(folder_name)
    cwd = osp.join(folder, "latex")  # Fixed potential issue with path
    writeup_file = osp.join(cwd, "template.tex")

    # Check all included figures are actually in the directory.
    with open(writeup_file, "r") as f:
        tex_text = f.read()
    figures = re.findall(r"\\includegraphics.*?{(.*?)}", tex_text)
    duplicates = []
    for figure in figures:
        if not osp.exists(osp.join(cwd, figure)):
            print(f"Figure {figure} not found in directory.")
            coder.run(
                f"""Figure {figure} not found in directory. Either remove the figure or add it to the directory."""
            )
        if figures.count(figure) > 1:
            duplicates.append(figure)
        if duplicates:
            coder.run(
                f"""The following figures are duplicated in the document: {duplicates}. Remove the duplicates."""
            )

    # Iteratively fix any LaTeX bugs
    for i in range(num_error_corrections):
        compile_latex(cwd, pdf_file, timeout=timeout)


def compile_latex(cwd, pdf_file, timeout=30):
    print("GENERATING MODEL DOCUMENTATION PDF")

    commands = [
        ["pdflatex", "template.tex"],
        ["pdflatex", "template.tex"],
    ]

    for command in commands:
        try:
            print(f"Running {' '.join(command)}")
            output = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            print("Standard Output:\n", result.stdout)
            print("Standard Error:\n", result.stderr)
        except subprocess.TimeoutExpired:
            print(f"Command {command} timed out")
            return False

    if osp.exists(osp.join(cwd, "template.pdf")):
        shutil.copy(osp.join(cwd, "template.pdf"), pdf_file)
        print(f"Copied to {pdf_file}")
        return True
    return False


# Define document section order
section_order = [
    "Cover Page",
    "Table of Contents",
    "Model Overview",
    "Model Development and Implementation",
    "Model Validation and Performance",
    "Model Governance",
    "Risk Assessment",
    "Conclusion",
    "Appendices",
]

# Section guidance tips
per_section_tips = {
    "Cover Page": """
Create a professional cover page for the model documentation that includes:
- Full model name and version number
- Business unit or department responsible for the model
- Model owner and primary contacts
- Date of document creation
- Classification (e.g., Confidential, Internal Use Only)
- Brief (1-2 sentence) description of model purpose
    """,
    "Table of Contents": """
Generate a comprehensive table of contents that lists all major sections and subsections with page numbers.
Include lists of figures and tables if applicable.
    """,
    "Model Overview": """
Provide a comprehensive overview of the model that includes:
- Clear statement of model purpose and objectives
- Business context and use cases
- Model classification (e.g., decision, estimation, forecasting)
- Inputs and outputs
- Key assumptions and constraints
- Regulatory context if applicable
- Relationship to other models in the ecosystem
- Stakeholders and user groups

This section should clearly establish WHY this model exists and its business value.
    """,
    "Model Development and Implementation": """
Document both the model development process and implementation details:

DEVELOPMENT:
- Data sources, quality, and preparation processes
- Feature engineering and selection methodology
- Model selection process and justification
- Model structure and algorithm details
- Parameter estimation techniques
- Handling of missing data
- Train/validation/test split methodology
- Hyperparameter tuning approach
- Development environment and tools

IMPLEMENTATION:
- Technical infrastructure and system architecture
- Deployment method and operational workflow
- Integration with existing systems
- Security controls and access management
- Monitoring framework
- Fallback procedures
- User interfaces and reporting capabilities
- Performance considerations and optimization
- Testing methodology and results
- Implementation timeline and deployment status

Focus on business relevance of choices while maintaining technical rigor.
    """,
    "Model Validation and Performance": """
Document the independent validation of the model and its performance metrics:

VALIDATION:
- Validation scope and objectives
- Conceptual soundness assessment
- Process verification
- Outcomes analysis and benchmarking
- Sensitivity analysis and stress testing
- Stability monitoring
- Limitations and boundary conditions
- Validation findings and recommendations
- Remediation plans for any issues identified

PERFORMANCE:
- Key performance indicators and metrics
- Performance across different segments
- Comparison to benchmark models or previous versions
- Stability over time
- Sensitivity to key inputs
- Economic value and business impact assessment
- Visualization of performance results
- Threshold setting and justification
- Performance monitoring framework

Include both quantitative metrics and qualitative business assessment.
    """,
    "Model Governance": """
Document the governance framework for the model:
- Roles and responsibilities
- Approval authorities and process
- Ongoing monitoring requirements
- Annual recertification process
- Change management procedures
- Issue escalation protocol
- Documentation standards
- Training requirements for users
- Audit trail requirements

Include any committee approvals obtained or required.
    """,
    "Risk Assessment": """
Provide a comprehensive risk assessment:
- Model risk categorization and rating
- Key risks and limitations
- Compensating controls
- Reporting and oversight mechanisms
- Testing of controls
- Contingency plans
- Compliance with regulatory requirements
- Potential business impact of model failure
- Risk acceptance statements

Use a systematic approach to risk identification and assessment.
    """,
    "Conclusion": """
Summarize the key points of the model documentation:
- Restatement of model purpose and value
- Summary of methodology and implementation
- Overview of performance and validation results
- Status of governance approvals
- Next steps and future enhancements

The conclusion should provide confidence in the model's fitness for purpose.
    """,
    "Appendices": """
Include relevant supporting materials:
- Detailed technical specifications
- Additional performance charts and tables
- Sample model outputs
- Data dictionaries
- Code samples or algorithms
- Testing results
- Alternative approaches considered
- Glossary of terms

Each appendix should be clearly labeled and referenced in the main text.
    """,
}

# Define primary purpose for each section
section_purpose = {
    "Cover Page": "provide clear identification of the model documentation",
    "Table of Contents": "allow readers to navigate efficiently through the document",
    "Model Overview": "explain the purpose, scope and key features of the model",
    "Model Development and Implementation": "detail the technical development process and implementation approach",
    "Model Validation and Performance": "document validation methods and performance metrics",
    "Model Governance": "clarify governance structure and responsibilities",
    "Risk Assessment": "evaluate risks associated with model usage and mitigating controls",
    "Conclusion": "summarize key points and confirm model's fitness for purpose",
    "Appendices": "provide detailed technical supporting information",
}


# 企業向けモデル記述書に特化したセクション生成プロンプト
def get_section_generation_prompt(section_name, notes, section_order, current_sections=None):
    primary_purpose = section_purpose.get(section_name, "provide key information for this section")
    section_tip = per_section_tips.get(section_name, "")

    if current_sections is None:
        current_sections = {}

    context = ""
    for sec in section_order:
        if sec in current_sections and sec != section_name:
            context += f"\n\n## {sec}\n{current_sections[sec]}"

    prompt = f"""
Create the "{section_name}" section. The primary purpose of this section is to {primary_purpose}.

DEVELOPMENT NOTES:
{notes}

GUIDANCE FOR THIS SECTION:
{section_tip}

CONTEXT FROM OTHER SECTIONS:
{context}

Please consider the following:
1. Focus primarily on achieving the section's main purpose
2. Organize information with appropriate headings and structure
3. Avoid academic paper style - this is a technical model documentation
4. Maintain technical accuracy while providing appropriate detail
5. Describe the model as a technical solution rather than a research contribution

Write only the section content without the section heading.
"""
    return prompt


# PERFORM WRITEUP
def perform_writeup(idea, folder_name, coder, client, client_model, num_cite_rounds=0, engine=None):
    """企業向けモデル記述書を作成する主関数"""

    print(f"Generating model documentation for {idea['Name']}")

    # 実験ノートを読み込む
    notes = ""
    notes_file = osp.join(folder_name, "notes.txt")
    if osp.exists(notes_file):
        with open(notes_file, "r") as f:
            notes = f.read()

    # 各セクションを生成
    current_sections = {}

    # 各セクションを順番に生成
    for section in section_order:
        print(f"Generating section: {section}")
        section_prompt = get_section_generation_prompt(section, notes, section_order, current_sections)
        section_content = coder.run(section_prompt).strip()
        current_sections[section] = section_content

        # セクションの内容を書き込む
        with open(osp.join(folder_name, f"section_{section.lower().replace(' ', '_')}.txt"), "w") as f:
            f.write(section_content)

    # 第2ラウンド：セクションの改善
    print("Starting enhancement round for each section")
    enhancement_prompt = """
Review the model documentation draft. Please enhance it by:
1. Removing any academic paper-style language or formatting
2. Adding clear section structure with descriptive subheadings
3. Ensuring content is technically precise
4. Checking for consistency across sections
5. IMPORTANT: Avoid presenting the model as a novel research approach or academic contribution
6. Focus on technical content rather than formal document elements
7. Ensure technical accuracy and clarity in all explanations
"""

    for section in section_order:
        print(f"Enhancing section: {section}")
        current_content = current_sections[section]
        section_enhancement_prompt = f"""
{enhancement_prompt}

CURRENT SECTION: {section}

{current_content}

Please provide an improved version of this section that maintains all technical content but eliminates any academic paper style presentation.
"""
        improved_content = coder.run(section_enhancement_prompt).strip()
        current_sections[section] = improved_content

    # LaTeX生成
    generate_latex(coder, folder_name, f"{folder_name}/{idea['Name']}.pdf")


if __name__ == "__main__":

    from aider.coders import Coder
    from aider.io import InputOutput
    from aider.models import Model

    parser = argparse.ArgumentParser(description="Perform writeup for a project")
    parser.add_argument("--folder", type=str)
    parser.add_argument("--no-writing", action="store_true", help="Only generate")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="semanticscholar",
        choices=["semanticscholar", "openalex"],
        help="Scholar engine to use.",
    )
    args = parser.parse_args()
    client, client_model = create_client(args.model)
    print("Make sure you cleaned the Aider logs if re-generating the writeup!")
    folder_name = args.folder
    idea_name = osp.basename(folder_name)
    exp_file = osp.join(folder_name, "experiment.py")
    vis_file = osp.join(folder_name, "plot.py")
    notes = osp.join(folder_name, "notes.txt")
    model = args.model
    writeup_file = osp.join(folder_name, "latex", "template.tex")
    ideas_file = osp.join(folder_name, "ideas.json")
    with open(ideas_file, "r") as f:
        ideas = json.load(f)
    for idea in ideas:
        if idea["Name"] in idea_name:
            print(f"Found idea: {idea['Name']}")
            break
    if idea["Name"] not in idea_name:
        raise ValueError(f"Idea {idea_name} not found")
    fnames = [exp_file, writeup_file, notes]
    io = InputOutput(yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt")
    if args.model == "deepseek-coder-v2-0724":
        main_model = Model("deepseek/deepseek-coder")
    elif args.model == "llama3.1-405b":
        main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
    else:
        main_model = Model(model)
    coder = Coder.create(
        main_model=main_model,
        fnames=fnames,
        io=io,
        stream=False,
        use_git=False,
        edit_format="diff",
    )
    if args.no_writing:
        generate_latex(coder, args.folder, f"{args.folder}/test.pdf")
    else:
        try:
            perform_writeup(idea, folder_name, coder, client, client_model)
        except Exception as e:
            print(f"Failed to perform writeup: {e}")
