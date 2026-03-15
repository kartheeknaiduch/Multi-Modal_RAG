from ocr_utils import ocr_image

from unstructured.partition.pdf import partition_pdf
from bs4 import BeautifulSoup
from io import BytesIO
from pypdf import PdfReader
import tempfile
import os


def _extract_text_with_pypdf(file_bytes):
    reader = PdfReader(BytesIO(file_bytes))
    structured_output = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        structured_output.append({
            "type": "NarrativeText",
            "content": text,
            "page": page_number,
            "category": None,
            "filetype": "application/pdf",
            "image_path": None,
            "metadata": {
                "text_as_html": None,
            },
        })

    return structured_output


def _has_meaningful_text(elements):
    for el in elements:
        text = getattr(el, "text", "") or ""
        if text.strip():
            return True
    return False


def _partition_pdf_for_demo(tmp_path):
    # Use the fast parser first to reduce upload latency for demo-friendly PDFs.
    fast_elements = partition_pdf(
        filename=tmp_path,
        include_page_breaks=True,
        strategy="fast",
    )

    if _has_meaningful_text(fast_elements):
        return fast_elements

    return partition_pdf(
        filename=tmp_path,
        include_page_breaks=True,
        infer_table_structure=True,
        strategy="hi_res",
    )


# ---------------------------------------------------------------------------------------------------
# PDF → Unstructured Elements → Normalize Elements → Multi-Modal Chunks → Embeddings → FAISS Retriver
# ---------------------------------------------------------------------------------------------------

def multi_modal_ingest(file_bytes: bytes):
    """
    Ingest a PDF using Unstructured.io and return a list of structured elements.
    Each element will be converted into a plain dict for easy storage & retrieval.

    Args:
        file_bytes (bytes): The PDF file in bytes.

    Returns:
        list: List of structured elements as dictionaries.
    """

    try:
        # Use a native text extractor first because it is much faster for text PDFs.
        structured_output = _extract_text_with_pypdf(file_bytes)
        if structured_output:
            return structured_output
    except Exception:
        pass

    # Save the uploaded bytes as a temporary PDF file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # Partition PDF with a fast-first strategy to reduce upload latency.
        elements = _partition_pdf_for_demo(tmp_path)
    except Exception:
        return []
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


    structured_output = []

    for el in elements:
        structured_output.append({
            "type": el.__class__.__name__,            # e.g., NarrativeText, Table, Title, Picture
            "content": el.text if hasattr(el, "text") else "",
            "page": getattr(el.metadata, "page_number", None),
            "category": getattr(el.metadata, "category_depth", None),
            "filetype": getattr(el.metadata, "filetype", None),
            "image_path": getattr(el, "image_path", None),
            "metadata": {
                "text_as_html": getattr(el.metadata, "text_as_html", None),
            },
        })

    return structured_output


def normalize_element(el, pdf_bytes):
    """
    Convert the Unstructured element into a unified RAG chunk.
    
    Args:
        el (dict): The structured element dictionary.
        pdf_bytes (bytes): The original PDF file in bytes.

    Returns:
        dict: A normalized RAG chunk dictionary.
    """

    element_type = el['type']
    page = el.get('page')
    text = el.get("content", "").strip()
    html = el.get("metadata", {}).get("text_as_html")
    img_path = el.get("image_path")

    # -------------------------------------
    # CASE 1: NORMAL TEXT ELEMENTS
    # -------------------------------------
    if element_type not in ["Table", "Image", "Picture"]:
        return {
            "type": "text",
            "page": page,
            "content": text,
            "image_path": None,
            "embedding_text": f"Text: {text}"
        }

    # -------------------------------------
    # CASE 2: TABLE HANDLING
    # -------------------------------------
    if element_type == "Table":

        
        if html:
            soup = BeautifulSoup(html, "html.parser")
            rows = soup.find_all("tr")

            markdown = []
            for r in rows:
                cols = [c.get_text(strip=True) for c in r.find_all(["td", "th"])]

                
                if len(cols) > 1:
                    markdown.append("| " + " | ".join(cols) + " |")

            
            if markdown:
                table_md = "\n".join(markdown)

                return {
                    "type": "table",
                    "page": page,
                    "content": table_md,
                    "image_path": None,
                    "embedding_text": f"Table: {table_md}",
                }

        
        if img_path:
            ocr_text = ocr_image(img_path)

            return {
                "type": "ocr",
                "page": page,
                "content": ocr_text,
                "image_path": img_path,
                "embedding_text": f"OCR: {ocr_text}",
            }

        # 2C: fallback → treat as text
        return {
            "type": "text",
            "page": page,
            "content": text,
            "image_path": None,
            "embedding_text": f"Text: {text}",
        }

    # -------------------------------------
    # CASE 3: IMAGE / PICTURE → OCR
    # -------------------------------------
    if element_type in ["Image", "Picture"] and img_path:
        ocr_text = ocr_image(img_path)

        return {
            "type": "ocr",
            "page": page,
            "content": ocr_text,
            "image_path": img_path,
            "embedding_text": f"OCR: {ocr_text}",
        }

    # -------------------------------------
    # DEFAULT FALLBACK
    # -------------------------------------
    return None





