#!/usr/bin/env python3
"""Check and fix blog post frontmatter for Astro content collection."""

import os
import re
import sys
from pathlib import Path

BLOG_DIR = Path("src/content/blog")

# Required frontmatter fields
REQUIRED_FIELDS = ["title", "description", "date"]

# Tag detection keywords
TAG_KEYWORDS = {
    "Deep Learning": ["deep learning", "neural network", "transformer", "attention", "training", "pretrain", "finetune"],
    "Paper Summary": ["论文概述", "paper summary", "abstract", "arxiv"],
    "Attention": ["attention", "self-attention", "flash attention", "sparse attention"],
    "Sparse": ["sparse", "sparsity", "mixture of expert", "moe"],
    "Long Context": ["long context", "long-context", "context length", "context window"],
    "Inference": ["inference", "serving", "speculative decoding", "quantization", "kv cache"],
    "Distributed": ["distributed", "parallel", "pipeline", "tensor", "fsdp", "deepspeed", "megatron"],
    "MoE": ["mixture of expert", "moe", "routing", "expert"],
    "Video": ["video", "magi", "visual", "image generation"],
    "KV Cache": ["kv cache", "kv-cache", "key-value cache", "pageattention"],
    "Compression": ["quantization", "pruning", "distillation", "fp8", "int4", "low-bit"],
    "RL": ["reinforcement learning", "rlhf", "grpo", "reward model", "ppo", "rl "],
    "Architecture": ["architecture", "state space", "ssm", "linear attention", "mamba"],
    "Decoding": ["decoding", "speculative", "eagle", "medusa", "draft"],
}


def extract_title(content: str) -> str:
    """Extract title from first # heading."""
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def extract_date_from_table(content: str) -> str:
    """Extract date from overview table (发布/Date/日期)."""
    for line in content.split("\n"):
        # Standard format: 2025-01-15 or 2025/01/15
        m = re.search(r"\*\*(?:发布|Date|日期|Published)\*\*\s*\|\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})", line)
        if m:
            return m.group(1).replace("/", "-")
        # Chinese format: 2025年10月 or 2025年10月15日
        m = re.search(r"\*\*(?:发布|Date|日期|Published)\*\*\s*\|\s*(\d{4})年(\d{1,2})月(?:(\d{1,2})日)?", line)
        if m:
            year, month = m.group(1), m.group(2).zfill(2)
            day = m.group(3).zfill(2) if m.group(3) else "01"
            return f"{year}-{month}-{day}"
        # Year-month only: 2025-10 or 2025/10
        m = re.search(r"\*\*(?:发布|Date|日期|Published)\*\*\s*\|\s*(\d{4})[-/](\d{1,2})(?:\s|$)", line)
        if m:
            return f"{m.group(1)}-{m.group(2).zfill(2)}-01"
    return ""


def extract_date_from_filename(filepath: Path) -> str:
    """Try to extract date from arxiv-style filename like 2502.12345-xxx.md."""
    m = re.match(r"(\d{4})\.(\d{5})", filepath.stem)
    if m:
        arxiv_id = f"{m.group(1)}.{m.group(2)}"
        # arxiv IDs encode year.month, not exact date — use first of month
        year = m.group(1)
        month = m.group(2)[:2]
        if 1 <= int(month) <= 12:
            return f"20{year[:2] if len(year) == 4 else year}-{month.zfill(2)}-01"
    # Try date-like prefix like 260510730
    m = re.match(r"(\d{4})(\d{2})(\d{2})\d*", filepath.stem)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        if 2020 <= int(y) <= 2030 and 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            return f"{y}-{mo}-{d}"
    return ""


def extract_description(content: str) -> str:
    """Extract description from first non-heading, non-table paragraph."""
    lines = content.split("\n")
    in_table = False
    past_heading = False
    for line in lines:
        stripped = line.strip()
        if not past_heading:
            if stripped.startswith("# "):
                past_heading = True
            continue
        if stripped.startswith("## "):
            continue
        if stripped.startswith("|"):
            in_table = True
            continue
        if in_table and stripped == "":
            in_table = False
            continue
        if in_table:
            continue
        if stripped.startswith("#") or stripped.startswith("!") or stripped.startswith("["):
            continue
        if stripped == "" or stripped == "---":
            continue
        # Clean markdown formatting
        desc = re.sub(r"\*\*([^*]+)\*\*", r"\1", stripped)
        desc = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", desc)
        desc = re.sub(r"`([^`]+)`", r"\1", desc)
        if len(desc) > 150:
            desc = desc[:147] + "..."
        return desc
    return ""


def detect_tags(content: str) -> list[str]:
    """Detect tags based on content keywords."""
    content_lower = content.lower()
    tags = []
    for tag, keywords in TAG_KEYWORDS.items():
        for kw in keywords:
            if kw in content_lower:
                tags.append(tag)
                break
    if not tags:
        tags = ["Deep Learning", "Paper Summary"]
    return tags


def parse_frontmatter(content: str) -> tuple[dict, str, str]:
    """Parse existing frontmatter. Returns (fields, raw_fm, body)."""
    if not content.startswith("---"):
        return {}, "", content

    end = content.find("---", 3)
    if end == -1:
        return {}, "", content

    fm_text = content[3:end]
    body = content[end + 3:].lstrip("\n")

    fields = {}
    lines = fm_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        # Check for key: value
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()

            if val.startswith("["):
                # Inline array: [a, b, c]
                val = [v.strip().strip('"').strip("'") for v in val.strip("[]").split(",") if v.strip()]
                fields[key] = val
            elif val.startswith('"') and val.endswith('"'):
                fields[key] = val[1:-1]
            elif val == "":
                # Could be a multi-line array (next lines start with "  - ")
                items = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    stripped_next = next_line.strip()
                    if stripped_next.startswith("- "):
                        items.append(stripped_next[2:].strip().strip('"').strip("'"))
                        j += 1
                    elif stripped_next == "":
                        j += 1
                    else:
                        break
                if items:
                    fields[key] = items
                    i = j
                    continue
                else:
                    fields[key] = val
            else:
                fields[key] = val
        i += 1

    return fields, fm_text, body


def build_frontmatter(fields: dict) -> str:
    """Build frontmatter string from fields dict."""
    lines = ["---"]
    for key in ["title", "description", "date", "tags", "draft"]:
        if key not in fields:
            continue
        val = fields[key]
        if key == "tags" and isinstance(val, list):
            lines.append("tags:")
            for tag in val:
                lines.append(f"  - {tag}")
        elif isinstance(val, list):
            lines.append(f"{key}: [{', '.join(val)}]")
        elif isinstance(val, bool):
            lines.append(f"{key}: {str(val).lower()}")
        else:
            lines.append(f'{key}: "{val}"')
    lines.append("---")
    return "\n".join(lines)


def check_file(filepath: Path) -> tuple[bool, list[str]]:
    """Check if a file has valid frontmatter. Returns (is_valid, issues)."""
    content = filepath.read_text(encoding="utf-8")
    issues = []

    if not content.startswith("---"):
        return False, ["Missing frontmatter"]

    fields, _, _ = parse_frontmatter(content)

    for field in REQUIRED_FIELDS:
        if field not in fields:
            issues.append(f"Missing required field: {field}")
        elif field == "date":
            val = fields[field]
            if isinstance(val, str) and not re.match(r"\d{4}-\d{2}-\d{2}", val):
                issues.append(f"Invalid date format: {val} (expected YYYY-MM-DD)")

    if "tags" in fields and not isinstance(fields["tags"], list):
        issues.append(f"tags should be an array, got: {type(fields['tags']).__name__}")

    return len(issues) == 0, issues


def fix_file(filepath: Path, dry_run: bool = False) -> bool:
    """Fix frontmatter for a file. Returns True if fixed."""
    content = filepath.read_text(encoding="utf-8")
    fields, _, body = parse_frontmatter(content)

    changed = False

    # Extract title
    if "title" not in fields or not fields["title"]:
        title = extract_title(content)
        if title:
            fields["title"] = title
            changed = True

    # Extract description
    if "description" not in fields or not fields["description"]:
        desc = extract_description(content)
        if desc:
            fields["description"] = desc
            changed = True

    # Extract date
    if "date" not in fields or not fields["date"]:
        date = extract_date_from_table(content)
        if not date:
            date = extract_date_from_filename(filepath)
        if date:
            fields["date"] = date
            changed = True

    # Ensure tags is a list
    if "tags" not in fields or not fields["tags"]:
        tags = detect_tags(content)
        fields["tags"] = tags
        changed = True
    elif isinstance(fields["tags"], str):
        # Convert string tags to list
        fields["tags"] = [t.strip() for t in fields["tags"].split(",") if t.strip()]
        changed = True

    # Ensure draft field
    if "draft" not in fields:
        fields["draft"] = False
        changed = True

    if changed:
        new_content = build_frontmatter(fields) + "\n\n" + body
        if not dry_run:
            filepath.write_text(new_content, encoding="utf-8")
        return True

    return False


def fix_missing_images(blog_dir: Path, dry_run: bool = False) -> int:
    """Remove references to missing image files. Returns count of fixes."""
    fixed = 0
    for md_file in sorted(blog_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        lines = content.split("\n")
        new_lines = []
        changed = False

        for line in lines:
            # Find markdown image references: ![alt](figures/...)
            m = re.search(r"!\[([^\]]*)\]\((figures/[^)]+)\)", line)
            if m:
                img_path = blog_dir / m.group(2)
                if not img_path.exists():
                    new_lines.append(f"<!-- MISSING IMAGE: {m.group(2)} -->")
                    changed = True
                    fixed += 1
                    continue
            new_lines.append(line)

        if changed and not dry_run:
            md_file.write_text("\n".join(new_lines), encoding="utf-8")
            print(f"  IMG  {md_file.name}: removed missing image reference(s)")

    return fixed


def main():
    """
    Usage:
      python3 fix-frontmatter.py            # Check and fix all issues
      python3 fix-frontmatter.py --check    # Check only, don't fix
      python3 fix-frontmatter.py --images   # Fix missing image references only
      python3 fix-frontmatter.py -v         # Verbose output
    """
    dry_run = "--check" in sys.argv
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    images_only = "--images" in sys.argv

    if "--help" in sys.argv or "-h" in sys.argv:
        print(main.__doc__)
        sys.exit(0)

    if not BLOG_DIR.exists():
        print(f"Error: {BLOG_DIR} does not exist")
        sys.exit(1)

    if images_only:
        print("Fixing missing image references...\n")
        fixed = fix_missing_images(BLOG_DIR, dry_run)
        print(f"\nFixed {fixed} missing image reference(s)")
        sys.exit(0)

    md_files = sorted(BLOG_DIR.glob("*.md"))
    total = len(md_files)
    ok_count = 0
    fixed_count = 0
    error_count = 0

    print(f"Checking {total} blog files in {BLOG_DIR}...\n")

    for filepath in md_files:
        is_valid, issues = check_file(filepath)

        if is_valid:
            ok_count += 1
            if verbose:
                print(f"  OK   {filepath.name}")
        else:
            if dry_run:
                print(f"  FAIL {filepath.name}: {'; '.join(issues)}")
                error_count += 1
            else:
                was_fixed = fix_file(filepath)
                if was_fixed:
                    print(f"  FIX  {filepath.name}")
                    fixed_count += 1
                else:
                    print(f"  SKIP {filepath.name}: could not fix")
                    error_count += 1

    # Fix missing image references
    if not dry_run:
        img_fixed = fix_missing_images(BLOG_DIR, dry_run)
        if img_fixed:
            print(f"\n  Fixed {img_fixed} missing image reference(s)")

    print(f"\n{'=' * 40}")
    print(f"Total:  {total}")
    print(f"OK:     {ok_count}")

    if dry_run:
        print(f"Failed: {error_count}")
        if error_count > 0:
            print(f"\nRun without --check to fix automatically.")
            sys.exit(1)
    else:
        print(f"Fixed:  {fixed_count}")
        if error_count > 0:
            print(f"Errors: {error_count}")

    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
