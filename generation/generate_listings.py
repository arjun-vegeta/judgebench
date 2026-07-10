"""
Module: generation.generate_listings
Pulls real catalog metadata listings and generates synthetic titles & descriptions
using LLMs (Gemini 3.1 Flash Lite) to create a dataset with varying quality, policy violations, and hallucinations.
"""

import os
import json
import gzip
import random
import urllib.request
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

ABO_METADATA_URL = "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_0.json.gz"
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
OUTPUT_FILE = os.path.join(DATA_DIR, "generated_listings.json")


def extract_localized_val(val: Any) -> str:
    """Extract string from localized list or string format."""
    if not val:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list) and len(val) > 0:
        for item in val:
            if isinstance(item, dict) and "value" in item:
                return item["value"].strip()
            elif isinstance(item, str):
                return item.strip()
    return str(val)


def fetch_abo_listings(count: int = 180) -> List[Dict[str, Any]]:
    """Fetch and parse raw catalog metadata listings from dataset S3 bucket."""
    print(f"[INFO] Fetching raw metadata from {ABO_METADATA_URL}...")
    listings = []

    req = urllib.request.Request(
        ABO_METADATA_URL, headers={"User-Agent": "Mozilla/5.0"}
    )
    try:
        with urllib.request.urlopen(req) as response:
            with gzip.GzipFile(fileobj=response) as gz:
                lines_read = 0
                while len(listings) < count and lines_read < 1000:
                    line = gz.readline()
                    lines_read += 1
                    if not line:
                        break
                    try:
                        data = json.loads(line.decode("utf-8"))
                        item_id = data.get("item_id")
                        orig_title = extract_localized_val(data.get("item_name"))
                        brand = extract_localized_val(data.get("brand"))
                        bullets_raw = data.get("bullet_point", [])

                        bullets = []
                        if isinstance(bullets_raw, list):
                            for b in bullets_raw:
                                b_val = (
                                    extract_localized_val([b])
                                    if isinstance(b, dict)
                                    else str(b)
                                )
                                if b_val:
                                    bullets.append(b_val)

                        # Only take items with title and brand
                        if item_id and orig_title:
                            listings.append(
                                {
                                    "item_id": item_id,
                                    "original_title": orig_title,
                                    "brand": brand or "Generic",
                                    "bullet_points": bullets[:3],
                                    "domain": data.get("domain_name", "store.com"),
                                    "main_product_type": extract_localized_val(
                                        data.get("main_product_type", "Product")
                                    ),
                                }
                            )
                    except Exception:
                        continue
    except Exception as e:
        print(f"[WARN] S3 fetch failed ({e}). Using offline sample generator.")
        listings = generate_fallback_abo_sample(count)

    print(f"[SUCCESS] Retrieved {len(listings)} product listings.")
    return listings


def generate_fallback_abo_sample(count: int = 180) -> List[Dict[str, Any]]:
    """Fallback sample generator if network is unreachable."""
    brands = [
        "BasicTech",
        "Stone & Beam",
        "Solimo",
        "Pinzon",
        "Rivindex",
        "Rivet",
        "Denali",
    ]
    categories = [
        "Home Kitchen",
        "Electronics",
        "Office Products",
        "Apparel",
        "Appliance",
        "Tools",
    ]
    items = []
    for i in range(count):
        brand = random.choice(brands)
        cat = random.choice(categories)
        items.append(
            {
                "item_id": f"B0{random.randint(10000000, 99999999)}",
                "original_title": f"{brand} Premium {cat} Heavy Duty Item Model {i + 100}",
                "brand": brand,
                "bullet_points": [
                    f"High-durability material for daily use in {cat.lower()}.",
                    "Ergonomic design with heat-resistant handle.",
                    "Backed by 1-year limited warranty.",
                ],
                "domain": "store.com",
                "main_product_type": cat,
            }
        )
    return items


def generate_ai_listing_prompt(
    abo_item: Dict[str, Any], inject_flaw: str = "none"
) -> str:
    """Construct prompt to generate product title & description with controlled quality/flaws."""
    prompt = f"""You are an e-commerce product copywriter.
Product Brand: {abo_item["brand"]}
Product Category: {abo_item["main_product_type"]}
Source Features: {", ".join(abo_item["bullet_points"])}

Write a title (max 150 chars) and 3 bullet points description.
"""
    if inject_flaw == "hallucination":
        prompt += "\nINSTRUCTION: Deliberately hallucinate a fake material (e.g. 100% Italian Cashmere or Aerospace Titanium) not mentioned in the source features."
    elif inject_flaw == "policy_violation":
        prompt += "\nINSTRUCTION: Deliberately use prohibited promotional superlatives such as 'WORLD'S BEST #1 GUARANTEED TO CURE ALL ILLS BESTSELLER'."
    elif inject_flaw == "length_violation":
        prompt += "\nINSTRUCTION: Make the title excessively long (>250 characters) with repetitive keyword stuffing."

    return prompt


def generate_synthetic_listings(
    abo_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate listing content (title + description) for ABO items."""
    os.makedirs(DATA_DIR, exist_ok=True)
    results = []

    gemini_key = os.environ.get("GEMINI_API_KEY")
    client = None
    if gemini_key:
        try:
            from google import genai

            client = genai.Client(api_key=gemini_key)
            print("[+] Connected to Gemini 3.1 Flash Lite API for generation.")
        except Exception as e:
            print(f"[!] Warning: Could not initialize Google GenAI SDK: {e}")

    flaw_types = [
        "none",
        "none",
        "none",
        "hallucination",
        "policy_violation",
        "length_violation",
    ]

    for idx, item in enumerate(abo_items):
        flaw = flaw_types[idx % len(flaw_types)]

        gen_title = ""
        gen_desc = ""

        if client:
            try:
                prompt = generate_ai_listing_prompt(item, inject_flaw=flaw)
                response = client.models.generate_content(
                    model="gemini-3.1-flash-lite", contents=prompt
                )
                text = response.text or ""
                lines = [
                    line_item.strip()
                    for line_item in text.split("\n")
                    if line_item.strip()
                ]
                if lines:
                    gen_title = lines[0].replace("Title:", "").strip()
                    gen_desc = " | ".join(lines[1:])
            except Exception as e:
                print(f"[!] Gemini generation error on item {idx}: {e}")

        # Fallback / rule-based realistic generation if API call wasn't available or failed
        if not gen_title:
            if flaw == "none":
                gen_title = f"{item['brand']} {item['main_product_type']} - High Quality {item['bullet_points'][0] if item['bullet_points'] else 'Utility'}"
                gen_desc = f"Features premium craftsmanship. {item['brand']} original design. Ideal for home and office."
            elif flaw == "hallucination":
                gen_title = f"{item['brand']} {item['main_product_type']} crafted from 100% Pure Siberian Cashmere & Solid Gold"
                gen_desc = "Luxury grade materials imported directly from outer space. Fits all standard sizes."
            elif flaw == "policy_violation":
                gen_title = f"WORLD'S BEST #1 GUARANTEED {item['brand']} {item['main_product_type']} AMAZING CHEAP SALE!!!"
                gen_desc = "Unbeatable top seller! Cures fatigue and makes you 100% smarter instantly!"
            elif flaw == "length_violation":
                gen_title = (
                    f"{item['brand']} {item['main_product_type']} " * 12
                    + "EXTRA LONG TITLE FOR KEYWORD STUFFING BEST BUY NOW DISCOUNT SALE CHEAP HARDWARE"
                )
                gen_desc = "Excessive text stuffed with terms for ranking manipulation."

        results.append(
            {
                "listing_id": f"LST-{idx + 1:04d}",
                "item_id": item["item_id"],
                "brand": item["brand"],
                "category": item["main_product_type"],
                "original_title": item["original_title"],
                "source_bullets": item["bullet_points"],
                "generated_title": gen_title,
                "generated_description": gen_desc,
                "intended_flaw": flaw,
            }
        )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(
        f"[SUCCESS] Successfully saved {len(results)} generated listings to {OUTPUT_FILE}"
    )
    return results


if __name__ == "__main__":
    items = fetch_abo_listings(180)
    generate_synthetic_listings(items)
