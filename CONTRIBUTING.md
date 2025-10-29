# Contribution Guidelines

Thank you for considering contributing to Awesome AI Models! This document outlines the guidelines for contributing to this curated list.

## Table of Contents

- [How to Contribute](#how-to-contribute)
- [Quality Standards](#quality-standards)
- [Adding Items](#adding-items)
- [Updating Items](#updating-items)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)

---

## How to Contribute

1. **Fork** this repository
2. **Create a branch** for your changes (`git checkout -b add-new-model`)
3. **Make your changes** following the guidelines below
4. **Commit** with a clear message (`git commit -m "Add XYZ model"`)
5. **Push** to your fork (`git push origin add-new-model`)
6. **Open a Pull Request** to the main repository

---

## Quality Standards

Items added to this list must meet the following criteria:

### For AI Models

- ✅ **Documented**: Official documentation or model card available
- ✅ **Accessible**: Publicly available (via API or open-source)
- ✅ **Verified**: Performance claims backed by official benchmarks or papers
- ✅ **Maintained**: Active development or stable release
- ✅ **Licensed**: Clear license terms specified

### For Tools & Resources

- ✅ **Functional**: Working and usable in current state
- ✅ **Documented**: Clear README with usage instructions
- ✅ **Active**: Updated within last 2 years
- ✅ **Quality**: Well-maintained with community adoption
- ✅ **Relevant**: Directly related to AI models

---

## Adding Items

### Item Format

Follow this exact format:

```markdown
* [Name](url) - Brief one-line description.
```

**Requirements**:
- Use asterisk (`*`) for bullet points
- Link to official source (GitHub repo, official website, or documentation)
- Description should be 5-20 words
- End with a period
- One line only (no line breaks)

### Good Examples

```markdown
* [Llama 3.1 70B](https://huggingface.co/meta-llama/Llama-3.1-70B) - Strong general-purpose model, 128K context, runs on 2x A100 (4-bit).
* [vLLM](https://docs.vllm.ai/) - Industry-leading inference server with PagedAttention, 24x faster than HF.
```

### Bad Examples

```markdown
* Llama 3.1 - great model
* [vLLM] - inference server
* [Tool](url) - This is an AMAZING tool that will revolutionize everything! It's super fast and everyone should use it!!!
```

### Where to Add

- Find the appropriate category in README.md
- Add your item in **alphabetical order** or logical grouping
- If no suitable category exists, suggest one in your PR

---

## Updating Items

### When to Update

- Pricing changes (for commercial models)
- New version releases with significant improvements
- Benchmark score updates from official sources
- Broken links or deprecated URLs
- License changes

### What to Update

- Include source for the update in PR description
- Update date in file if significant change
- Fix any broken links
- Remove deprecated/abandoned projects

---

## Pull Request Process

### Before Submitting

1. **Check for duplicates** - Search existing items
2. **Verify links** - Ensure all URLs work
3. **Test formatting** - Preview markdown rendering
4. **Follow style guide** - Match existing format exactly
5. **One item per PR** - Separate PRs for multiple unrelated items

### PR Description Template

```markdown
## Type of Change
- [ ] New model
- [ ] New tool/resource
- [ ] Update existing item
- [ ] Fix broken link
- [ ] Other: ___________

## Item Details
- **Name**:
- **Category**:
- **URL**:
- **Description**:

## Verification
- [ ] Link works
- [ ] Follows format guidelines
- [ ] Added to correct category
- [ ] Alphabetically ordered (if applicable)
- [ ] Official source linked

## Additional Context
[Any relevant information, sources, or reasoning]
```

### Review Process

- Maintainers will review within 1-2 weeks
- Changes may be requested for formatting or categorization
- Provide sources if asked for verification
- Be respectful and patient

---

## Style Guidelines

### Description Writing

**Do**:
- Be concise and factual
- Mention key features or differentiators
- Use present tense
- Use active voice
- State what it does, not opinions

**Don't**:
- Use marketing language ("best", "revolutionary", "amazing")
- Include version numbers (they become outdated)
- Add emojis (except in specific contexts)
- Write multi-line descriptions
- Express personal opinions

### Examples

✅ **Good**:
```markdown
* [Mixtral 8x7B](url) - MoE model (47B total, 13B active), runs on single RTX 4090 (4-bit), Apache 2.0.
```

❌ **Bad**:
```markdown
* [Mixtral 8x7B](url) - This is the BEST and most AMAZING open-source model! Everyone should use it! Version 0.1 is incredible!
```

### Formatting

- **Links**: Use `[Name](url)` format
- **Bold**: Use `**text**` for emphasis (sparingly)
- **Code**: Use backticks for technical terms when needed
- **Lists**: Use `*` for consistency
- **Headers**: Use `##` for main sections

---

## Updating Detailed Guides

If your contribution involves detailed information (benchmarks, cost analysis, etc.), consider updating the relevant supplementary files:

- **GUIDE.md** - Strategic guidance, decision frameworks
- **BENCHMARKS.md** - Performance comparisons
- **COST_ANALYSIS.md** - Pricing and TCO information
- **DEPLOYMENT.md** - Setup instructions and hardware requirements
- **CASE_STUDIES.md** - Real-world usage examples

Follow the same quality standards and include sources for all data.

---

## Sources and Citations

When adding performance claims or technical specifications:

- Link to official benchmarks or papers
- Use verified sources (official repos, papers, documentation)
- Avoid blog posts or unofficial sources
- Include retrieval date for time-sensitive information
- Be clear about "reported" vs "measured" vs "estimated" data

---

## Questions?

If you have questions about contributing:

1. Check existing issues and discussions
2. Review this guide thoroughly
3. Open an issue with your question
4. Tag maintainers if needed

---

## Code of Conduct

- Be respectful and constructive
- Focus on the content, not the person
- Accept constructive feedback gracefully
- Help others learn and improve
- Follow GitHub's Community Guidelines

---

Thank you for contributing to Awesome AI Models! Your efforts help the community make informed decisions about AI model selection and deployment.
