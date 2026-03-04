# Academic Writing Guide V3

Richard J. Young — UNLV Department of Neuroscience
For use with the LaTeX paper template in this directory.

## Quick Start

From the repository root (`00_masters_papers/`):

```bash
# 1. Copy the template to a new directory
cp -r 00_template/ MyNewPaper/
cd MyNewPaper/

# 2. Edit main.tex:
#    - Set \title{...} to your paper title
#    - Set \fancyhead[LO]{...} to a short running header
#    - Update \author{...} if adding co-authors
#    - Update \keywords{...}

# 3. Fill in each sections/*.tex file following the embedded guidance

# 4. Add figures to figures/ as figN_descriptive_name.pdf

# 5. Add references to references.bib (uncomment examples as starting point)

# 6. Compile
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Paper Structure Overview

| Section | Target Words | Purpose | Key Question |
|---------|-------------|---------|--------------|
| Abstract | 150--250 | Standalone summary | What did you find? |
| Introduction | 600--800 | Motivate the work | Why should anyone care? |
| Related Work | 800--1200 | Situate in field | What has been done before? |
| Methods | 1000--1500 | Enable replication | How did you do it? |
| Results | 800--1200 | Report findings | What happened? |
| Discussion | 1000--1500 | Interpret + contextualize | What does it mean? |
| Conclusion | 150--300 | Synthesize takeaways | So what? What's next? |

---

## Abstract (150--250 words)

Write this LAST, after the full paper is complete.

### 5-Sentence Formula

| Sentence | Content | Example Starter |
|----------|---------|-----------------|
| 1--2 | Background/context | "Despite advances in X, Y remains poorly understood..." |
| 3 | Methods summary | "This study employed..." |
| 4 | Main results | "Results demonstrated that..." |
| 5 | Implications | "These findings suggest..." |
| FINAL | Key finding (most important takeaway) | "Collectively, this work establishes..." |

### Rules
- No citations in the abstract
- No abbreviations without definition
- Last sentence must be the single most important takeaway

---

## Introduction (4 Paragraphs)

### Paragraph 1: The Problem
**Purpose**: Establish why this topic matters.

Include:
- Broad context (1--2 sentences)
- Specific problem statement
- Real-world relevance or impact
- Hook to engage the reader

Example structure:
> "[Broad field] faces a critical challenge: [specific problem]. This issue affects [scope/population]. Without resolution, [consequence]."

### Paragraph 2: What Is Known
**Purpose**: Summarize existing literature.

Include:
- Key findings from prior research (cite 3--6 seminal works)
- Current theoretical frameworks
- Established methods or approaches
- Areas of consensus

Literature review standards:
- Critically evaluate studies (don't just list)
- Identify methodological strengths/weaknesses
- Trace intellectual progression of the field
- Synthesize — don't summarize sequentially

Example structure:
> "Previous research has established that [finding 1] (Author, Year). Building on this, [Author] demonstrated [finding 2]. The prevailing theoretical framework suggests [theory]. However, these studies share [common limitation]."

### Paragraph 3: The Gap (The Hook)
**Purpose**: Identify what's missing, broken, or unresolved.

Include:
- Specific gap in knowledge
- Unresolved debates or contradictions
- Methodological limitations of prior work
- Why this gap matters

Key phrases:
- "However, no study has yet examined..."
- "What remains unclear is..."
- "A critical limitation of prior work is..."
- "Conflicting findings regarding X suggest..."

### Paragraph 4: Your Solution
**Purpose**: State what you will do and predict outcomes.

Include (in order):
1. Primary purpose: "The primary aim of this study was to..."
2. Primary hypothesis (explicit, testable): "It was hypothesized that [IV] would [predicted relationship] [DV]."
3. Secondary purpose (if applicable)
4. Secondary hypothesis
5. Brief method preview (optional)

Hypothesis format:
- Must be falsifiable
- Specify direction (increase/decrease/differ)
- Name variables explicitly

---

## Related Work

Survey and critically evaluate existing studies:
- Identify gaps, unresolved debates, methodological flaws
- Position your work within the broader field
- Group by theme, not chronologically
- End by clearly stating how your work differs

---

## Methods

**Purpose**: Enable exact replication.

### Required Subsections

1. **Participants / Data Source**
   - Sample size (with justification / power analysis)
   - Demographics (age, sex, relevant characteristics)
   - Inclusion/exclusion criteria
   - Recruitment method
   - For secondary data: source, version, access date

2. **Study Design**
   - Design type (experimental, observational, longitudinal, etc.)
   - Groups/conditions
   - Randomization procedure (if applicable)
   - Blinding (if applicable)

3. **Measures / Variables**
   - Independent variables: operationalization, levels
   - Dependent variables: how measured, units
   - Covariates/confounds: what was controlled
   - Instruments: reliability, validity, citations

4. **Procedures**
   - Step-by-step protocol
   - Timeline
   - Setting/environment
   - Instructions given to participants

5. **Data Analysis**
   - Statistical tests used (with justification)
   - Software and version
   - Significance threshold (e.g., alpha = .05)
   - Handling of missing data
   - Assumptions tested

6. **Ethics**
   - IRB/ethics approval (protocol number)
   - Informed consent process
   - Data protection measures

7. **Preregistration** (if applicable)
   - Registry name and ID
   - Any deviations from preregistered plan

---

## Results

**Cardinal Rule**: Report only. No interpretation.

### Structure

**Opening** (1--2 sentences):
> "To examine whether [X affects Y], [analysis] was conducted."

**Body** — for each analysis/figure/table:
1. Reference the figure/table: "As shown in Figure 1..."
2. State the statistical test: "A two-way ANOVA revealed..."
3. Report the finding with statistics: "...a significant main effect of X, F(2, 87) = 4.32, p = .016, eta-squared = .09."
4. State the direction in plain language: "Participants in the treatment condition scored higher."

**Organization options**:
- By hypothesis (H1, then H2, etc.)
- By figure/table order
- Thematic (grouped by variable type)
- Chronological (if longitudinal)

**Closing** (1 paragraph):
> "In summary, the results supported H1 but not H2. These findings are examined in the following section."

### Formatting Rules
- All statistics in APA format
- Every figure/table must be referenced in text
- Present exact p-values (not just p < .05)
- Include effect sizes
- Report non-significant findings too

---

## Discussion

### Paragraph 1: Summary and Takeaway
- Restate primary purpose (1 sentence)
- State primary finding (1 sentence)
- Overall takeaway / "so what" statement
- Mirror abstract language

### Paragraphs 2--N: Detailed Findings (follow figure order)
For each major finding:
1. Restate the finding: "H1, that X would predict Y, was supported."
2. Interpret: "This suggests that [mechanism/explanation]."
3. Relate to prior literature: "This aligns with [Author, Year]..." OR "This contradicts [Author, Year], possibly due to..."
4. Novelty statement (where applicable): "To our knowledge, this is the first study to demonstrate..."

### Limitations Paragraph
Address:
- Alternative explanations for findings
- Methodological limitations
- Sampling limitations
- Measurement limitations
- Generalizability concerns

Frame constructively:
> "One limitation is [X]; however, [mitigation or why still valid]."

Do NOT:
- Undermine your entire study
- List every possible flaw
- Apologize excessively

### Integration Paragraph
- Theoretical implications
- Practical/clinical implications
- How this changes the field's understanding

### Final Paragraph: Conclusion and Future
1. One-sentence summary of contribution
2. Key implication
3. Specific future research directions (2--3)
4. Closing statement

---

## Tone and Style Rules

| Rule | Do | Don't |
|------|-----|-------|
| Person | Third person ("The study examined...") | First person ("I examined...") |
| Voice | Active preferred ("Results showed...") | Excessive passive ("It was shown that...") |
| Language | Formal, precise | Colloquial, vague |
| Claims | Supported by citations | Unsupported assertions |
| Hedging | Appropriate ("suggests," "indicates") | Overclaiming ("proves," "definitely") |
| Emotion | Neutral, objective | Personal opinions, enthusiasm |

---

## ML Research: Additional Standards

### Originality Requirements
- Identify a genuine knowledge gap (not a trivial variation)
- Articulate a clear, testable hypothesis
- Explain significance to the field
- Demonstrate that critical analysis drives the work (not just data collection)

### Experimental Rigor Checklist
- [ ] Document ALL models trained (including failed attempts)
- [ ] Justify model architecture choices
- [ ] Use appropriate baselines (state-of-the-art, not strawmen)
- [ ] Specify train/validation/test splits with rationale
- [ ] Document cross-validation scheme
- [ ] Confirm no data leakage (test data never seen during training)
- [ ] Report all preprocessing steps
- [ ] Include hyperparameter search details
- [ ] Report variance (standard deviation across runs/seeds)

### Reproducibility Requirements

| Level | Requirement |
|-------|-------------|
| Bronze | Raw data publicly available |
| Bronze | Trained model weights shared |
| Bronze | Source code archived (GitHub + Zenodo) |
| Silver | Environment specification (requirements.txt, Docker) |
| Silver | Random seeds documented |
| Gold | End-to-end reproduction script |
| Gold | Compute requirements documented |

**Repository requirements**:
- Data: Domain-specific repository OR generalist (Zenodo, Figshare, OSF)
- Code: GitHub with tagged release + archival DOI
- Models: Hugging Face Hub, Zenodo, or institutional repository

---

## LaTeX Conventions for This Template

### Figure Naming
All figures go in `figures/` subfolder. Name as:
```
figN_descriptive_name.pdf
```
Examples: `fig1_architecture.pdf`, `fig2_benign_vs_harmful.pdf`, `fig7_heatmap.pdf`

### Citation Key Format
Use `authorYEARkeyword` convention:
```
young2025tempest
vaswani2017attention
bai2022constitutional
```

### Section Comment Headers
Use `======` dividers for visual separation (see section stubs):
```latex
% ============================================================
% SECTION NAME
% ============================================================
```

### Hypothesis Labels
Label hypotheses H1, H2, H3 in the Introduction. Reference them by label in Results and Discussion.

---

## arXiv Submission Checklist

- [ ] `main.tex` compiles cleanly (zero errors; warnings are acceptable)
- [ ] `figures/` contains all PDFs referenced in `.tex`
- [ ] `PRIMEarxiv.sty` is in the root directory (not in a subdirectory)
- [ ] `references.bib` is present and bibtex ran successfully
- [ ] `main.bbl` is present (pre-compiled bibliography)
- [ ] `00README.json` is present (specifies pdflatex compiler)
- [ ] Abstract is under 250 words
- [ ] Keywords are listed
- [ ] No absolute file paths in `.tex` files
- [ ] No Unicode characters (use LaTeX equivalents: `$\geq$` not >=)

### Create Submission Zip
```bash
zip -r submission.zip \
  main.tex \
  sections/ \
  figures/ \
  PRIMEarxiv.sty \
  references.bib \
  main.bbl \
  00README.json
```

Upload `submission.zip` to https://arxiv.org/submit

---

## Overleaf Setup

1. **New Project** > **Upload Project** > upload the submission zip (or the whole template folder as a zip)
2. **Settings** (gear icon):
   - Compiler: **pdfLaTeX**
   - Main document: **main.tex**
3. Compile (should work immediately with zero configuration)

For collaborative editing:
- Share the Overleaf project link with co-authors
- Use Overleaf's track changes for review

---

## Pre-Submission Checklist

### Abstract
- [ ] 150--250 words
- [ ] Contains: background, methods, results, implications
- [ ] Final sentence = key finding

### Introduction
- [ ] 4-paragraph structure followed
- [ ] Gap clearly identified
- [ ] Explicit hypotheses stated

### Methods
- [ ] Replicable detail provided
- [ ] All subsections complete
- [ ] Ethics/IRB documented

### Results
- [ ] No interpretation (report only)
- [ ] All figures/tables referenced in text
- [ ] Statistics in proper format
- [ ] Effect sizes reported

### Discussion
- [ ] Follows figure/result order
- [ ] Limitations addressed constructively
- [ ] Future directions are specific
- [ ] Novelty stated

### ML-Specific
- [ ] Baselines are appropriate (not strawmen)
- [ ] No data leakage
- [ ] Code/data/models shared or path documented
- [ ] Variance reported across runs
