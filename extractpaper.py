import os
import glob
import fitz  # PyMuPDF

OUTPUT_FILE = "researchpaper.txt"

def extract_info_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            return None, None
        
        page = doc[0]
        
        # ---------------------------------------------------------
        # 1. BULLETPROOF TITLE EXTRACTION
        # ---------------------------------------------------------
        dict_text = page.get_text("dict")
        blocks = dict_text.get("blocks", [])
        
        font_blocks = []
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        size = round(span["size"], 1)
                        
                        # Rule 1: Must be longer than 4 characters (Ignores Drop Caps)
                        # Rule 2: Ignore common journal header text
                        if len(text) > 4 and "IEEE" not in text and "VOL." not in text:
                            font_blocks.append((size, text))
        
        title = "Title not found"
        if font_blocks:
            # Sort all text fragments by font size (largest first)
            font_blocks.sort(key=lambda x: x[0], reverse=True)
            largest_size = font_blocks[0][0]
            
            # Combine all text fragments that share this largest font size
            title_parts = [text for size, text in font_blocks if size == largest_size]
            title = " ".join(title_parts)

        # ---------------------------------------------------------
        # 2. BULLETPROOF ABSTRACT EXTRACTION (Reading by Blocks)
        # ---------------------------------------------------------
        # Extract physical blocks of text to avoid cross-column reading
        text_blocks = page.get_text("blocks")
        
        # Sort blocks vertically (top to bottom)
        text_blocks.sort(key=lambda b: b[1])
        
        abstract_text = ""
        recording = False
        
        for b in text_blocks:
            block_content = b[4].strip() # The actual text in the block
            lower_content = block_content.lower()
            
            # If we are currently recording the abstract, look for the stop sign
            if recording:
                if lower_content.startswith("index terms") or \
                   lower_content.startswith("keywords") or \
                   lower_content.startswith("i. intro") or \
                   lower_content.startswith("1. intro") or \
                   "introduction" in lower_content[:20]:
                    break # Stop recording!
                
                # Replace newlines with spaces to make it a clean paragraph
                abstract_text += " " + block_content.replace("\n", " ")
            
            # Look for the start of the abstract
            elif lower_content.startswith("abstract—") or lower_content.startswith("abstract"):
                recording = True
                abstract_text += " " + block_content.replace("\n", " ")

        abstract = abstract_text.strip()
        
        # Clean up the word "Abstract" from the beginning
        if abstract.lower().startswith("abstract—"):
            abstract = abstract[9:].strip()
        elif abstract.lower().startswith("abstract"):
            abstract = abstract[8:].strip(" -:.")

        if not abstract:
            abstract = "No abstract found."

        return title, abstract

    except Exception as e:
        print(f"⚠️ Error reading {os.path.basename(pdf_path)}: {e}")
        return None, None

def main():
    print(f"\n📂 PDF to Text Extractor (Block-Reading Version)")
    print(f"Safely appending to: {OUTPUT_FILE}")
    print("─" * 50)
    
    folder_path = input("Enter the full path to the folder with PDFs: ").strip()

    if not os.path.isdir(folder_path):
        print("❌ Invalid folder path. Please check and try again.")
        return

    search_pattern = os.path.join(folder_path, "*.pdf")
    pdf_files = glob.glob(search_pattern)

    if not pdf_files:
        print("No PDF files found in that directory.")
        return

    print(f"\nFound {len(pdf_files)} PDFs. Extracting data...")
    
    added_count = 0
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for pdf in pdf_files:
            print(f"Reading: {os.path.basename(pdf)}...")
            title, abstract = extract_info_from_pdf(pdf)
            
            if title and abstract != "No abstract found.":
                f.write(f"Title: {title}\n")
                f.write(f"Abstract: {abstract}\n")
                f.write("-" * 60 + "\n\n")
                added_count += 1
            else:
                print(f"   ⚠️ Could not confidently parse {os.path.basename(pdf)}. Skipping.")

    if added_count > 0:
        print(f"\n✅ Done! Successfully appended {added_count} papers to {OUTPUT_FILE}.")
    else:
        print("\n⚠️ Could not extract clean data from any of the PDFs.")

if __name__ == "__main__":
    main()