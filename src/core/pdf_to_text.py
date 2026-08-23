import os
import re
import logging
import fitz  

logger = logging.getLogger(__name__)


def _clean_text_block(text: str) -> str:

    if not text:
        return ""
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    
    return " ".join(lines)


def extract_pdf_to_markdown(pdf_path: str, output_path: str = None, include_page_markers: bool = True) -> str:

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open PDF file '{pdf_path}': {e}")
        raise ValueError(f"Could not read PDF file '{pdf_path}': {e}") from e

    if len(doc) == 0:
        logger.warning(f"PDF file '{pdf_path}' has 0 pages.")
        return ""

    font_sizes = []
    for page in doc:
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            font_sizes.append(round(span.get("size", 10), 1))

    body_font_size = 10.0
    if font_sizes:
        from collections import Counter
        body_font_size = Counter(font_sizes).most_common(1)[0][0]

    h1_threshold = body_font_size * 1.3
    h2_threshold = body_font_size * 1.15

    markdown_pages = []

    for page_num, page in enumerate(doc, start=1):
        page_blocks = []
        if include_page_markers:
            page_blocks.append(f"<!-- Page {page_num} -->\n")

        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if "lines" not in block:
                continue

            block_text_parts = []
            max_span_size = 0.0
            is_bold = False

            for line in block["lines"]:
                line_parts = []
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    span_size = span.get("size", 10.0)
                    font_flags = span.get("font", "").lower()

                    if span_size > max_span_size:
                        max_span_size = span_size
                    if "bold" in font_flags or "black" in font_flags:
                        is_bold = True

                    line_parts.append(span_text)
                block_text_parts.append("".join(line_parts))

            raw_block_text = "\n".join(block_text_parts)
            cleaned_text = _clean_text_block(raw_block_text)

            if not cleaned_text:
                continue

            if max_span_size >= h1_threshold and len(cleaned_text) < 120:
                page_blocks.append(f"# {cleaned_text}")
            elif max_span_size >= h2_threshold and len(cleaned_text) < 150:
                page_blocks.append(f"## {cleaned_text}")
            elif is_bold and len(cleaned_text) < 80 and not cleaned_text.endswith("."):
                page_blocks.append(f"### {cleaned_text}")
            else:
                page_blocks.append(cleaned_text)

        if page_blocks:
            markdown_pages.append("\n\n".join(page_blocks))

    full_markdown = "\n\n---\n\n".join(markdown_pages)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_markdown)
        logger.info(f"Successfully saved converted Markdown to '{output_path}'.")

    return full_markdown


def extract_pdf_to_text(pdf_path: str, output_path: str = None) -> str:
    return extract_pdf_to_markdown(pdf_path, output_path=output_path, include_page_markers=False)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        base_name = os.path.splitext(os.path.basename(pdf_file))[0]
        out_file = sys.argv[2] if len(sys.argv) > 2 else os.path.join("data", f"{base_name}.md")
        print(f"Extracting '{pdf_file}' to '{out_file}'...")
        result = extract_pdf_to_markdown(pdf_file, output_path=out_file)
        print(f"Extracted {len(result)} characters and saved to '{out_file}'.")
    else:
        sample_pdf = os.path.join("data", "rich_dad_poor_dad_by_robert_t-_kiyosaki.pdf")
        if os.path.exists(sample_pdf):
            base_name = os.path.splitext(os.path.basename(sample_pdf))[0]
            output_md = os.path.join("data", f"{base_name}.md")
            print(f"Testing extraction on sample PDF: {sample_pdf}")
            result = extract_pdf_to_markdown(sample_pdf, output_path=output_md)
            print(f"Extracted {len(result)} characters and saved to '{output_md}'.")

