#!/usr/bin/env python
# coding: utf-8

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import json
import re
from functools import lru_cache
import time
from collections import defaultdict
import random
from ast import literal_eval

from openai import OpenAI
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Verify OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("=" * 70)
    print("⚠️  WARNING: OPENAI_API_KEY not found!")
    print("=" * 70)
    raise ValueError("OPENAI_API_KEY environment variable is required")

# --------------------------------------------------
# CATEGORY MAPPING
# --------------------------------------------------
CATEGORY_MAPPING = {
    'Home & Living': ['Home & Kitchen', 'Outdoor & Garden'],
    'Tech & Electronics': ['Electronics & Gadgets'],
    'Fashion & Jewelry': ['Fashion & Accessories', 'Jewelry', 'Travel & Luggage'],
    'Health & Beauty': ['Beauty & Personal Care', 'Grocery & Gourmet Food', 'Sports & Fitness'],
    'Toys & Games': ['Toys & Games'],
    'Kids & Baby': ['Baby & Kids'],
    'Arts & Crafts': ['Office & School', 'Arts & Crafts'],
}

SUBCATEGORY_TO_MAIN = {}
for main_cat, sub_cats in CATEGORY_MAPPING.items():
    for sub_cat in sub_cats:
        SUBCATEGORY_TO_MAIN[sub_cat] = main_cat

# --------------------------------------------------
# GENDER EXCLUSION KEYWORDS
# --------------------------------------------------
# Products containing these keywords should be EXCLUDED for opposite gender
FEMALE_KEYWORDS = [
    'for her', 'for women', 'for woman', 'for girls', 'for ladies',
    'women\'s', 'womens', 'ladies', 'feminine', 'girlfriend', 'wife',
    'mom', 'mother', 'daughter', 'sister', 'grandma', 'grandmother',
    'bridal', 'bride', 'maternity', 'pregnancy'
]

MALE_KEYWORDS = [
    'for him', 'for men', 'for man', 'for boys', 'for guys',
    'men\'s', 'mens', 'masculine', 'boyfriend', 'husband',
    'dad', 'father', 'son', 'brother', 'grandpa', 'grandfather',
    'groomsmen', 'groom'
]

# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------
app = FastAPI(
    title="GiveAura AI Gift Recommender API",
    description="Enhanced AI-powered gift recommendation engine with strict filtering",
    version="6.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# LOAD PRODUCTS
# --------------------------------------------------
df_products = pd.read_csv("gifts_data_all.csv").fillna("")


def safe_to_list(x):
    """Convert various formats to list"""
    if x is None:
        return []
    if isinstance(x, (np.ndarray, pd.Series)):
        return x.tolist() if len(x) > 0 else []
    if isinstance(x, list):
        return x
    try:
        if pd.isna(x):
            return []
    except (TypeError, ValueError):
        pass
    x_str = str(x).strip()
    if x_str == "" or x_str.lower() in ["nan", "none", "null"]:
        return []
    if x_str.startswith("[") and x_str.endswith("]"):
        try:
            result = literal_eval(x_str)
            if isinstance(result, list):
                return result
        except:
            pass
    parts = [p.strip() for p in x_str.split(",") if p.strip()]
    return parts


def normalize_product(row):
    """Return product in consistent format"""
    if isinstance(row, dict):
        data = row
    else:
        data = row.to_dict() if hasattr(row, 'to_dict') else dict(row)

    clean_cat = data.get("clean_category", "")
    main_category = SUBCATEGORY_TO_MAIN.get(clean_cat, "Other")

    return {
        "asin": data.get("asin", ""),
        "short_title": data.get("short_title", ""),
        "title": data.get("title", ""),
        "category": data.get("category", ""),
        "affiliate_url": data.get("affiliate_url", ""),
        "tags": safe_to_list(data.get("tags", [])),
        "image_urls": safe_to_list(data.get("image_urls", [])),
        "clean_category": clean_cat,
        "main_category": main_category,
        "ai_summary": data.get("ai_summary", ""),
        "brand": data.get("brand", ""),
        "predicted_occasions": safe_to_list(data.get("predicted_occasions", [])),
        "predicted_gender": safe_to_list(data.get("predicted_gender", [])),
        "predicted_relationships": safe_to_list(data.get("predicted_relationships", [])),
        "predicted_age_groups": safe_to_list(data.get("predicted_age_groups", [])),
        "predicted_personality": safe_to_list(data.get("predicted_personality", [])),
    }


# Normalize multivalued columns
multivalued_cols = [
    "predicted_occasions", "predicted_gender", "predicted_relationships",
    "predicted_age_groups", "predicted_personality", "tags", "image_urls",
]

for col in multivalued_cols:
    if col in df_products.columns:
        df_products[col] = df_products[col].apply(safe_to_list)

ALL_PRODUCTS_RAW = df_products.to_dict(orient="records")
ALL_PRODUCTS = [normalize_product(p) for p in ALL_PRODUCTS_RAW]
titles = [p["short_title"] for p in ALL_PRODUCTS]


# --------------------------------------------------
# Pydantic Schemas
# --------------------------------------------------
class RecommendRequest(BaseModel):
    query: str
    occasions: Optional[List[str]] = []
    gender: Optional[List[str]] = []
    relationships: Optional[List[str]] = []
    age_groups: Optional[List[str]] = []
    personality: Optional[List[str]] = []
    top_k: int = 25


# --------------------------------------------------
# EMBEDDING MODELS
# --------------------------------------------------
client = OpenAI()
EMBED_MODEL = "text-embedding-3-large"


def embed_texts(texts: List[str]):
    res = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in res.data])


def embed_query(text: str):
    res = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(res.data[0].embedding)


print("🔄 Computing product embeddings...")
start_time = time.time()
PRODUCT_EMB = embed_texts(titles)
print(f"✅ Embeddings computed in {time.time() - start_time:.2f}s")

# --------------------------------------------------
# SEARCH INFRASTRUCTURE
# --------------------------------------------------
print("🔄 Building search indices...")

searchable_text = []
for p in ALL_PRODUCTS:
    text = (
            p["short_title"].lower() + " " +
            p["title"].lower() + " " +
            p["clean_category"].lower() + " " +
            p["brand"].lower() + " " +
            " ".join(p["tags"]).lower()
    )
    searchable_text.append(text)

tokenized_corpus = [text.split() for text in searchable_text]
bm25 = BM25Okapi(tokenized_corpus)

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000, ngram_range=(1, 2),
    min_df=1, max_df=0.8, stop_words='english'
)
tfidf_matrix = tfidf_vectorizer.fit_transform(searchable_text)

print("✅ Search indices built")


# --------------------------------------------------
# ENHANCED FILTERING & SCORING FUNCTIONS
# --------------------------------------------------

def check_gender_exclusion(product: Dict, user_gender: List[str]) -> bool:
    """
    Check if product should be EXCLUDED based on gender mismatch.
    Returns True if product should be EXCLUDED.
    """
    if not user_gender:
        return False  # No gender preference, don't exclude

    # Get product text for keyword matching
    product_text = (
            product.get("short_title", "").lower() + " " +
            product.get("title", "").lower() + " " +
            " ".join(product.get("tags", [])).lower()
    )

    user_wants_male = any(g.lower() in ['male', 'men', 'man', 'boy'] for g in user_gender)
    user_wants_female = any(g.lower() in ['female', 'women', 'woman', 'girl'] for g in user_gender)

    # If user wants MALE, exclude products with female keywords
    if user_wants_male and not user_wants_female:
        for keyword in FEMALE_KEYWORDS:
            if keyword in product_text:
                return True  # EXCLUDE this product

    # If user wants FEMALE, exclude products with male keywords
    if user_wants_female and not user_wants_male:
        for keyword in MALE_KEYWORDS:
            if keyword in product_text:
                return True  # EXCLUDE this product

    return False  # Don't exclude


def check_gender_match(product: Dict, user_gender: List[str]) -> float:
    """
    Check if product's predicted gender matches user preference.
    Returns: 1.0 (match), 0.5 (unisex/neutral), 0.0 (mismatch)
    """
    if not user_gender:
        return 0.5  # No preference

    product_gender = [g.lower() for g in product.get("predicted_gender", [])]
    user_gender_lower = [g.lower() for g in user_gender]

    if not product_gender:
        return 0.3  # Product has no gender info

    # Check for unisex
    if 'unisex' in product_gender:
        return 0.8  # Unisex products are good for everyone

    # Check for direct match
    for ug in user_gender_lower:
        if ug in product_gender:
            return 1.0
        # Handle variations
        if ug == 'male' and any(g in ['male', 'men', 'man', 'boy'] for g in product_gender):
            return 1.0
        if ug == 'female' and any(g in ['female', 'women', 'woman', 'girl'] for g in product_gender):
            return 1.0

    return 0.0  # Mismatch


def overlap_score_strict(user_list: List[str], product_list: List[str], use_aliases: bool = False) -> float:
    """
    Calculate overlap score with stricter penalties.
    Returns: 1.0 (perfect match), 0.0 (no match or empty)
    """
    if not user_list:
        return 0.3  # No preference = slight penalty (not neutral)
    if not product_list:
        return 0.0  # Product has no data = bad

    # Normalize: lowercase, strip whitespace, and handle common variations
    def normalize(s):
        s = s.lower().strip()
        # Handle common variations
        s = s.replace('-', ' ').replace('_', ' ')  # tech-savvy -> tech savvy
        s = s.replace('savy', 'savvy')  # Fix common typo
        return s

    user_set = set([normalize(u) for u in user_list])
    product_set = set([normalize(p) for p in product_list])

    # If using aliases, expand the matching
    if use_aliases:
        expanded_product_set = set(product_set)
        for product_item in product_set:
            # Check if this item has aliases
            for key, aliases in RELATIONSHIP_ALIASES.items():
                if product_item == key or product_item in aliases:
                    expanded_product_set.update([a for a in aliases])
        product_set = expanded_product_set

    overlap = user_set.intersection(product_set)

    if len(overlap) == 0:
        return 0.0

    # Return ratio of matched criteria
    return len(overlap) / len(user_set)

    overlap = user_set.intersection(product_set)

    if len(overlap) == 0:
        return 0.0

    # Return ratio of matched criteria
    return len(overlap) / len(user_set)


def compute_semantic_similarity(query: str, product_idx: int) -> float:
    """Compute semantic similarity"""
    try:
        q_emb = embed_query(query)
        p_emb = PRODUCT_EMB[product_idx]
        similarity = cosine_similarity([q_emb], [p_emb])[0][0]
        return max(0.0, min(1.0, similarity))
    except:
        return 0.5


@lru_cache(maxsize=1000)
def get_query_embedding(query: str):
    return embed_query(query)


# --------------------------------------------------
# RELATIONSHIP SYNONYMS & EXPANSION
# --------------------------------------------------

RELATIONSHIP_SYNONYMS = {
    'son': ['son', 'boy', 'brother', 'kids', 'child'],
    'daughter': ['daughter', 'girl', 'sister', 'kids', 'child'],
    'mom': ['mom', 'mother', 'wife', 'grandma'],
    'dad': ['dad', 'father', 'husband', 'grandpa'],
    'husband': ['husband', 'boyfriend', 'him'],
    'wife': ['wife', 'girlfriend', 'her'],
    'brother': ['brother', 'son', 'boy', 'him'],
    'sister': ['sister', 'daughter', 'girl', 'her'],
    'grandma': ['grandma', 'grandmother', 'mom', 'mother'],
    'grandpa': ['grandpa', 'grandfather', 'dad', 'father'],
    'boyfriend': ['boyfriend', 'husband', 'him', 'best friend'],
    'girlfriend': ['girlfriend', 'wife', 'her', 'best friend'],
    'best friend': ['best friend', 'friend'],
    'couples': ['couples', 'husband', 'wife', 'boyfriend', 'girlfriend'],
}


def expand_relationships(relationships: List[str]) -> List[str]:
    """Expand relationship terms to include synonyms"""
    expanded = set()
    for rel in relationships:
        rel_lower = rel.lower().strip()
        expanded.add(rel_lower)
        # Add synonyms
        if rel_lower in RELATIONSHIP_SYNONYMS:
            for syn in RELATIONSHIP_SYNONYMS[rel_lower]:
                expanded.add(syn)
    return list(expanded)


# --------------------------------------------------
# QUERY KEYWORD DETECTION
# --------------------------------------------------

INTEREST_KEYWORDS = {
    'gaming': ['game', 'games', 'gaming', 'gamer', 'playstation', 'xbox', 'nintendo', 'video game', 'board game'],
    'tech': ['tech', 'technology', 'gadget', 'gadgets', 'electronic', 'electronics', 'smart', 'digital'],
    'sports': ['sport', 'sports', 'fitness', 'athletic', 'workout', 'gym', 'exercise'],
    'cooking': ['cook', 'cooking', 'kitchen', 'chef', 'baking', 'culinary'],
    'reading': ['book', 'books', 'reading', 'reader', 'bookworm', 'literature', 'literary'],
    'outdoor': ['outdoor', 'outdoors', 'camping', 'hiking', 'adventure'],
    'art': ['art', 'artistic', 'creative', 'craft', 'crafts', 'drawing', 'painting', 'decor', 'decorative'],
    'music': ['music', 'musical', 'instrument', 'audio', 'sound', 'headphones'],
    'luxury': ['luxury', 'premium', 'high-end', 'elegant', 'sophisticated', 'spa'],
    'toys': ['toy', 'toys', 'play', 'playful', 'lego', 'building', 'puzzle'],
    'nature': ['nature', 'plant', 'plants', 'garden', 'gardening', 'flower', 'flowers', 'botanical', 'tree', 'green',
               'crystal'],
    'food': ['food', 'snack', 'snacks', 'gourmet', 'chocolate', 'candy', 'tea', 'coffee', 'nut', 'nuts', 'basket',
             'foodie'],
    'wellness': ['wellness', 'spa', 'relaxation', 'candle', 'aromatherapy', 'self-care', 'bath', 'massage'],
}

# Map interests to personality traits for boosting
INTEREST_TO_PERSONALITY = {
    'gaming': ['Gamer', 'Tech-Savvy'],
    'tech': ['Tech-Savvy'],
    'toys': ['Creative', 'Artistic', 'Gamer'],
    'art': ['Artistic', 'Creative'],
    'outdoor': ['Adventurous', 'Nature Lover'],
    'sports': ['Fitness Enthusiast', 'Adventurous'],
    'cooking': ['Foodie', 'Creative'],
    'food': ['Foodie', 'Luxury-Loving'],
    'nature': ['Nature Lover', 'Creative', 'Artistic'],
    'wellness': ['Luxury-Loving', 'Romantic'],
    'reading': ['Bookworm', 'Creative'],
    'luxury': ['Luxury-Loving'],
    'music': ['Creative', 'Artistic'],
}

# Relationship aliases for better matching
RELATIONSHIP_ALIASES = {
    'son': ['son', 'boy', 'kid', 'child', 'children'],
    'daughter': ['daughter', 'girl', 'kid', 'child', 'children'],
    'brother': ['brother', 'bro', 'sibling'],
    'sister': ['sister', 'sis', 'sibling'],
    'mom': ['mom', 'mother', 'mum', 'mama'],
    'dad': ['dad', 'father', 'papa'],
    'wife': ['wife', 'spouse'],
    'husband': ['husband', 'spouse'],
    'girlfriend': ['girlfriend', 'gf'],
    'boyfriend': ['boyfriend', 'bf'],
    'grandma': ['grandma', 'grandmother', 'granny', 'nana'],
    'grandpa': ['grandpa', 'grandfather', 'gramps'],
    'best friend': ['best friend', 'friend', 'bestie', 'bff'],
    'couples': ['couples', 'couple', 'partner'],
}


def extract_interests_from_query(query: str) -> List[str]:
    """Extract interest categories from query"""
    query_lower = query.lower()
    detected = []
    for category, keywords in INTEREST_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            detected.append(category)
    return detected


def product_matches_interests(product: Dict, interests: List[str]) -> float:
    """Check if product matches detected interests from query"""
    if not interests:
        return 0.5  # Neutral

    product_text = (
            product.get("short_title", "").lower() + " " +
            product.get("title", "").lower() + " " +
            product.get("category", "").lower() + " " +
            product.get("clean_category", "").lower() + " " +
            " ".join(product.get("tags", [])).lower()
    )

    # Also check product personality for interest matching
    product_personality = [p.lower() for p in product.get("predicted_personality", [])]

    matches = 0
    for interest in interests:
        keywords = INTEREST_KEYWORDS.get(interest, [])

        # Check keywords in product text
        if any(kw in product_text for kw in keywords):
            matches += 1
            continue

        # Check if product personality matches interest-related traits
        related_personalities = INTEREST_TO_PERSONALITY.get(interest, [])
        if any(rp.lower() in product_personality for rp in related_personalities):
            matches += 0.7  # Partial match for personality alignment

    if matches == 0:
        return 0.0
    return min(1.0, matches / len(interests))


# --------------------------------------------------
# ENHANCED RANKING ENGINE
# --------------------------------------------------

def rank_products_enhanced(
        query: str,
        user_occasions: List[str] = [],
        user_gender: List[str] = [],
        user_relationships: List[str] = [],
        user_age_groups: List[str] = [],
        user_personality: List[str] = [],
        top_k: int = 50
) -> List[Dict]:
    """
    Enhanced recommendation engine with:
    1. Hard gender filtering (exclude mismatches)
    2. Query-based interest detection
    3. Weighted multi-factor scoring
    4. Bonus for multiple criteria matches
    """
    print(f"\n🎯 Enhanced Recommendation Engine v6.0:")
    print(f"   Query: '{query}'")
    print(f"   Occasions: {user_occasions}")
    print(f"   Gender: {user_gender}")
    print(f"   Relationships: {user_relationships}")
    print(f"   Age Groups: {user_age_groups}")
    print(f"   Personality: {user_personality}")

    # Extract interests from query
    detected_interests = extract_interests_from_query(query)
    print(f"   🔍 Detected interests from query: {detected_interests}")

    candidates = []
    excluded_count = 0

    # Check if user has any filter preferences
    has_filters = bool(user_occasions or user_gender or user_relationships or user_age_groups or user_personality)

    # Normalize age groups to handle "adult" vs "adults"
    user_age_normalized = []
    for age in user_age_groups:
        user_age_normalized.append(age)
        if age.lower() == 'adult':
            user_age_normalized.append('Adults')
        elif age.lower() == 'adults':
            user_age_normalized.append('Adult')

    for idx, product in enumerate(ALL_PRODUCTS):
        # STEP 1: Hard gender exclusion
        if check_gender_exclusion(product, user_gender):
            excluded_count += 1
            continue

        # STEP 2: Calculate individual scores
        occasion_score = overlap_score_strict(user_occasions, product.get("predicted_occasions", []))
        relationship_score = overlap_score_strict(user_relationships, product.get("predicted_relationships", []),
                                                  use_aliases=True)
        gender_score = check_gender_match(product, user_gender)
        age_score = overlap_score_strict(user_age_normalized, product.get("predicted_age_groups", []))
        personality_score = overlap_score_strict(user_personality, product.get("predicted_personality", []))

        # STEP 3: Query interest matching (NEW!)
        interest_score = product_matches_interests(product, detected_interests)

        # STEP 4: Semantic similarity (only if query is meaningful)
        if query and query.lower() not in ['find me the perfect gift', 'gift', 'gifts', '']:
            semantic_score = compute_semantic_similarity(query, idx)
        else:
            semantic_score = 0.5  # Neutral for generic queries

        # STEP 5: Calculate weighted final score
        if has_filters or detected_interests:
            # When filters or interests detected, prioritize matches
            final_score = (
                    0.15 * semantic_score +  # Reduced
                    0.15 * occasion_score +
                    0.15 * relationship_score +
                    0.15 * gender_score +
                    0.10 * age_score +
                    0.10 * personality_score +
                    0.20 * interest_score  # NEW: Query interest matching
            )

            # BONUS: Extra points for matching multiple criteria
            match_count = sum([
                1 if occasion_score > 0.5 else 0,
                1 if relationship_score > 0.5 else 0,
                1 if gender_score > 0.5 else 0,
                1 if age_score > 0.5 else 0,
                1 if personality_score > 0.5 else 0,
                1 if interest_score > 0.5 else 0,
            ])

            # Add bonus for multiple matches (up to 0.30 extra)
            final_score += match_count * 0.05

            # PENALTY: Heavily penalize gender mismatches
            if user_gender and gender_score == 0.0:
                final_score *= 0.1  # 90% penalty for gender mismatch

            # PENALTY: Penalize if no relationship match when relationship is specified
            if user_relationships and relationship_score == 0.0:
                final_score *= 0.5  # 50% penalty

            # PENALTY: Penalize if query mentions specific interests but product doesn't match
            if detected_interests and interest_score == 0.0:
                final_score *= 0.3  # 70% penalty for not matching query interests

        else:
            # No filters, use semantic similarity primarily
            final_score = semantic_score

        candidates.append({
            'product': product,
            'score': final_score,
            'scores': {
                'semantic': semantic_score,
                'occasion': occasion_score,
                'relationship': relationship_score,
                'gender': gender_score,
                'age': age_score,
                'personality': personality_score,
                'interest': interest_score
            }
        })

    # Sort by score descending
    candidates.sort(key=lambda x: x['score'], reverse=True)

    print(f"   ❌ Excluded {excluded_count} products (gender mismatch)")
    print(f"   ✅ Returning top {top_k} from {len(candidates)} candidates")

    # Debug: Show top 5 scores
    for i, c in enumerate(candidates[:5]):
        print(f"   #{i + 1}: {c['product']['short_title'][:40]}... Score: {c['score']:.3f}")
        print(f"        Scores: {c['scores']}")

    return [c['product'] for c in candidates[:top_k]]


# --------------------------------------------------
# RECOMMENDATION REASONING
# --------------------------------------------------

def generate_recommendation_reason(
        product: Dict[str, Any],
        user_occasions: List[str],
        user_relationships: List[str],
        user_personality: List[str],
        user_gender: List[str],
        user_age_groups: List[str]
) -> str:
    """Generate personalized reason for recommendation"""
    reasons = []

    def normalize_field(field):
        if isinstance(field, list):
            return [str(x).strip().lower() for x in field if x]
        elif isinstance(field, str):
            return [x.strip().lower() for x in field.split(',') if x.strip()]
        return []

    # Check each criterion
    product_occasions = normalize_field(product.get('predicted_occasions', []))
    matching_occasions = [occ for occ in user_occasions if occ.lower() in product_occasions]
    if matching_occasions:
        reasons.append(f"Perfect for {matching_occasions[0]}")

    product_relationships = normalize_field(product.get('predicted_relationships', []))
    matching_relationships = [rel for rel in user_relationships if rel.lower() in product_relationships]
    if matching_relationships:
        reasons.append(f"Ideal for your {matching_relationships[0].lower()}")

    product_gender = normalize_field(product.get('predicted_gender', []))
    matching_gender = [g for g in user_gender if g.lower() in product_gender]
    if matching_gender:
        reasons.append(f"Great for {matching_gender[0].lower()}s")

    product_age = normalize_field(product.get('predicted_age_groups', []))
    matching_age = [age for age in user_age_groups if age.lower() in product_age]
    if matching_age:
        reasons.append(f"Suited for {matching_age[0].lower()}")

    product_personality = normalize_field(product.get('predicted_personality', []))
    matching_personality = [pers for pers in user_personality if pers.lower() in product_personality]
    if matching_personality:
        if len(matching_personality) > 1:
            reasons.append(f"Matches {', '.join(matching_personality[:2])} personalities")
        else:
            reasons.append(f"Matches {matching_personality[0]} personality")

    if reasons:
        return " • ".join(reasons[:5])

    ai_summary = product.get('ai_summary', '')
    if ai_summary and len(ai_summary) > 50:
        first_sentence = ai_summary.split('.')[0][:100]
        return f"Recommended: {first_sentence}..."

    return "Highly rated gift choice"


def calculate_match_score(
        product: Dict[str, Any],
        user_occasions: List[str],
        user_relationships: List[str],
        user_personality: List[str],
        user_gender: List[str],
        user_age_groups: List[str]
) -> Dict[str, Any]:
    """Calculate detailed match score"""
    scores = {
        'occasion': 0,
        'relationship': 0,
        'personality': 0,
        'gender': 0,
        'age_group': 0,
        'total_score': 0
    }

    def normalize(s):
        """Normalize string for comparison"""
        s = str(s).lower().strip()
        s = s.replace('-', ' ').replace('_', ' ')
        s = s.replace('savy', 'savvy')
        return s

    def normalize_field(field):
        if isinstance(field, list):
            return [normalize(x) for x in field if x]
        elif isinstance(field, str):
            return [normalize(x) for x in field.split(',') if x.strip()]
        return []

    # Normalize user inputs
    user_occasions_norm = [normalize(x) for x in user_occasions]
    user_relationships_norm = [normalize(x) for x in user_relationships]
    user_personality_norm = [normalize(x) for x in user_personality]
    user_gender_norm = [normalize(x) for x in user_gender]
    user_age_norm = [normalize(x) for x in user_age_groups]

    product_occasions = normalize_field(product.get('predicted_occasions', []))
    if any(occ in product_occasions for occ in user_occasions_norm):
        scores['occasion'] = 1

    product_relationships = normalize_field(product.get('predicted_relationships', []))
    # Expand with aliases
    expanded_relationships = set(product_relationships)
    for rel in product_relationships:
        for key, aliases in RELATIONSHIP_ALIASES.items():
            if rel == key or rel in aliases:
                expanded_relationships.update(aliases)
    if any(rel in expanded_relationships for rel in user_relationships_norm):
        scores['relationship'] = 1

    product_personality = normalize_field(product.get('predicted_personality', []))
    if any(pers in product_personality for pers in user_personality_norm):
        scores['personality'] = 1

    product_gender = normalize_field(product.get('predicted_gender', []))
    if any(gen in product_gender for gen in user_gender_norm) or 'unisex' in product_gender:
        scores['gender'] = 1

    product_age = normalize_field(product.get('predicted_age_groups', []))
    # Handle "adult" vs "adults" variation
    user_age_expanded = user_age_norm.copy()
    if 'adult' in user_age_norm:
        user_age_expanded.append('adults')
    if 'adults' in user_age_norm:
        user_age_expanded.append('adult')

    if any(age in product_age for age in user_age_expanded) or 'all ages' in product_age:
        scores['age_group'] = 1

    scores['total_score'] = sum([
        scores['occasion'],
        scores['relationship'],
        scores['personality'],
        scores['gender'],
        scores['age_group']
    ])

    return scores


# --------------------------------------------------
# ENDPOINTS
# --------------------------------------------------

@app.get("/products/all")
def get_all_products():
    return ALL_PRODUCTS


@app.get("/products/categories")
def get_categories():
    category_counts = {}
    for product in ALL_PRODUCTS:
        main_cat = product.get("main_category", "Other")
        category_counts[main_cat] = category_counts.get(main_cat, 0) + 1

    return sorted([
        {"name": name, "count": count}
        for name, count in category_counts.items()
    ], key=lambda x: x["name"])


@app.get("/products/by_category")
def get_products_by_category(category: str):
    if category == "All":
        return ALL_PRODUCTS
    return [p for p in ALL_PRODUCTS if p.get("main_category") == category]


@app.post("/recommend")
def recommend(req: RecommendRequest):
    """
    Enhanced AI recommendation endpoint with strict filtering
    """
    print(f"\n{'=' * 60}")
    print(f"🎯 RECOMMENDATION REQUEST")
    print(f"{'=' * 60}")
    print(f"   Query: {req.query}")
    print(f"   Occasions: {req.occasions}")
    print(f"   Gender: {req.gender}")
    print(f"   Relationships: {req.relationships}")
    print(f"   Age Groups: {req.age_groups}")
    print(f"   Personality: {req.personality}")

    # Get recommendations using enhanced engine
    ranked_results = rank_products_enhanced(
        query=req.query,
        user_occasions=req.occasions or [],
        user_gender=req.gender or [],
        user_relationships=req.relationships or [],
        user_age_groups=req.age_groups or [],
        user_personality=req.personality or [],
        top_k=50
    )

    # Add personalized reasoning
    enhanced_results = []
    for product in ranked_results:
        product['recommendation_reason'] = generate_recommendation_reason(
            product,
            req.occasions or [],
            req.relationships or [],
            req.personality or [],
            req.gender or [],
            req.age_groups or []
        )

        match_details = calculate_match_score(
            product,
            req.occasions or [],
            req.relationships or [],
            req.personality or [],
            req.gender or [],
            req.age_groups or []
        )

        product['match_score'] = match_details['total_score']
        product['match_details'] = match_details
        enhanced_results.append(product)

    # Sort by match_score first, then by original ranking
    enhanced_results.sort(key=lambda x: x['match_score'], reverse=True)

    final_results = enhanced_results[:req.top_k]

    print(f"\n   ✅ Returning {len(final_results)} final results")
    print(f"{'=' * 60}\n")

    return final_results


@app.get("/filter_options")
def get_filter_options():
    """Return all unique filter options"""
    occasions, genders, relationships, age_groups, personalities = set(), set(), set(), set(), set()

    for product in ALL_PRODUCTS:
        for occ in product.get('predicted_occasions', []):
            if occ: occasions.add(occ)
        for gen in product.get('predicted_gender', []):
            if gen: genders.add(gen)
        for rel in product.get('predicted_relationships', []):
            if rel: relationships.add(rel)
        for age in product.get('predicted_age_groups', []):
            if age: age_groups.add(age)
        for pers in product.get('predicted_personality', []):
            if pers: personalities.add(pers)

    return {
        "occasions": sorted(list(occasions)),
        "genders": sorted(list(genders)),
        "relationships": sorted(list(relationships)),
        "age_groups": sorted(list(age_groups)),
        "personalities": sorted(list(personalities))
    }


@app.get("/api/search")
def search(query: str):
    """Search endpoint"""
    results = []
    q_emb = get_query_embedding(query)
    scores = cosine_similarity([q_emb], PRODUCT_EMB)[0]

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = [ALL_PRODUCTS[idx] for idx, _ in ranked[:12]]

    return {"suggestions": [], "results": results}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "products_loaded": len(ALL_PRODUCTS),
        "engine": "Enhanced Multi-Factor Ranking v6.0 (Strict Filtering)"
    }


@app.get("/")
def root():
    return {
        "message": "🎁 GiveAura - Enhanced AI Gift Recommendation Engine v6.0",
        "version": "6.0.0",
        "improvements": [
            "Hard gender exclusion filtering",
            "Keyword-based gender detection in product titles",
            "Rebalanced weights (filters > semantic similarity)",
            "Multi-match bonus scoring",
            "Strict penalties for mismatches"
        ],
        "ranking_weights_with_filters": {
            "semantic_similarity": "20%",
            "occasion": "20%",
            "relationship": "20%",
            "gender": "20%",
            "age_group": "10%",
            "personality": "10%",
            "multi_match_bonus": "up to +25%"
        }
    }


# ============================================================
# ADD THIS CODE TO YOUR FASTAPI SERVER (main.py)
# ============================================================
#
# Insert these additions in the appropriate sections:
#
# 1. Add the Pydantic model near other models (around line 120)
# 2. Add the endpoint near other endpoints (around line 500)
# ============================================================


# --------------------------------------------------
# SECTION 1: Add this Pydantic model with other models
# --------------------------------------------------

class SimilarProductsRequest(BaseModel):
    """Request model for similar products based on wishlist"""
    wishlist_asins: List[str]
    exclude_asins: Optional[List[str]] = []
    top_k: int = 8


# --------------------------------------------------
# SECTION 2: Add this endpoint with other endpoints
# --------------------------------------------------

@app.post("/recommend/similar")
def recommend_similar(req: SimilarProductsRequest):
    """
    Recommend products similar to user's wishlist items using semantic similarity.

    This endpoint:
    1. Gets embeddings of all wishlist products
    2. Computes average embedding to represent user preferences
    3. Finds most similar products not in wishlist
    4. Returns personalized recommendations with similarity scores
    """
    print(f"\n{'=' * 60}")
    print(f"🎯 SIMILAR PRODUCTS REQUEST")
    print(f"{'=' * 60}")
    print(f"   Wishlist items: {len(req.wishlist_asins)}")
    print(f"   Exclude items: {len(req.exclude_asins)}")
    print(f"   Top K: {req.top_k}")

    if not req.wishlist_asins:
        return {"recommendations": [], "message": "No wishlist items provided"}

    # Find indices of wishlist products
    wishlist_indices = []
    wishlist_products = []

    for idx, product in enumerate(ALL_PRODUCTS):
        if product['asin'] in req.wishlist_asins:
            wishlist_indices.append(idx)
            wishlist_products.append(product)

    if not wishlist_indices:
        print("   ⚠️ No matching products found for wishlist ASINs")
        return {"recommendations": [], "message": "No matching products found"}

    print(f"   Found {len(wishlist_indices)} wishlist products")

    # Get embeddings for wishlist products
    wishlist_embeddings = PRODUCT_EMB[wishlist_indices]

    # Compute centroid (average) embedding representing user preferences
    user_preference_embedding = np.mean(wishlist_embeddings, axis=0)

    # Also collect metadata from wishlist for better recommendations
    wishlist_categories = set()
    wishlist_personalities = set()
    wishlist_occasions = set()
    wishlist_genders = set()
    wishlist_age_groups = set()

    for product in wishlist_products:
        if product.get('clean_category'):
            wishlist_categories.add(product['clean_category'])
        for p in product.get('predicted_personality', []):
            wishlist_personalities.add(p.lower())
        for o in product.get('predicted_occasions', []):
            wishlist_occasions.add(o.lower())
        for g in product.get('predicted_gender', []):
            wishlist_genders.add(g.lower())
        for a in product.get('predicted_age_groups', []):
            wishlist_age_groups.add(a.lower())

    print(f"   Wishlist categories: {wishlist_categories}")
    print(f"   Wishlist personalities: {wishlist_personalities}")

    # Compute similarity scores for all products
    similarities = cosine_similarity([user_preference_embedding], PRODUCT_EMB)[0]

    # Create exclude set
    exclude_set = set(req.exclude_asins) if req.exclude_asins else set()
    exclude_set.update(req.wishlist_asins)  # Always exclude wishlist items

    # Rank products by similarity with metadata boosting
    candidates = []
    for idx, (product, sim_score) in enumerate(zip(ALL_PRODUCTS, similarities)):
        if product['asin'] in exclude_set:
            continue

        # Calculate metadata match bonus
        metadata_bonus = 0.0

        # Category match bonus
        if product.get('clean_category') in wishlist_categories:
            metadata_bonus += 0.05

        # Personality match bonus
        product_personalities = set(p.lower() for p in product.get('predicted_personality', []))
        if product_personalities.intersection(wishlist_personalities):
            metadata_bonus += 0.03

        # Occasion match bonus
        product_occasions = set(o.lower() for o in product.get('predicted_occasions', []))
        if product_occasions.intersection(wishlist_occasions):
            metadata_bonus += 0.02

        # Gender match bonus
        product_genders = set(g.lower() for g in product.get('predicted_gender', []))
        if product_genders.intersection(wishlist_genders) or 'unisex' in product_genders:
            metadata_bonus += 0.02

        # Age group match bonus
        product_age = set(a.lower() for a in product.get('predicted_age_groups', []))
        if product_age.intersection(wishlist_age_groups) or 'all ages' in product_age:
            metadata_bonus += 0.01

        # Combined score
        final_score = sim_score + metadata_bonus

        candidates.append({
            'product': product,
            'similarity_score': float(sim_score),
            'metadata_bonus': metadata_bonus,
            'final_score': final_score
        })

    # Sort by final score
    candidates.sort(key=lambda x: x['final_score'], reverse=True)

    # Take top K and add recommendation reasons
    recommendations = []
    for candidate in candidates[:req.top_k]:
        product = candidate['product'].copy()
        product['similarity_score'] = candidate['similarity_score']

        # Generate recommendation reason
        reasons = []

        # Check what matched
        if product.get('clean_category') in wishlist_categories:
            reasons.append(f"Similar to items in your wishlist")

        product_personalities = set(p.lower() for p in product.get('predicted_personality', []))
        matching_personalities = product_personalities.intersection(wishlist_personalities)
        if matching_personalities:
            pers_list = list(matching_personalities)[:2]
            reasons.append(f"Matches your {', '.join(pers_list)} style")

        product_occasions = set(o.lower() for o in product.get('predicted_occasions', []))
        matching_occasions = product_occasions.intersection(wishlist_occasions)
        if matching_occasions:
            occ_list = list(matching_occasions)[:1]
            reasons.append(f"Great for {occ_list[0]}")

        if not reasons:
            reasons.append("Recommended based on your taste")

        product['recommendation_reason'] = " • ".join(reasons[:2])
        recommendations.append(product)

    print(f"   ✅ Returning {len(recommendations)} recommendations")

    # Debug: Show top 3
    for i, rec in enumerate(recommendations[:3]):
        print(f"   #{i + 1}: {rec['short_title'][:40]}... (score: {rec['similarity_score']:.3f})")

    print(f"{'=' * 60}\n")

    return {
        "recommendations": recommendations,
        "based_on_count": len(wishlist_indices),
        "wishlist_categories": list(wishlist_categories),
        "wishlist_personalities": list(wishlist_personalities)
    }


# --------------------------------------------------
# OPTIONAL: Add endpoint to get similar products for a single item
# --------------------------------------------------

@app.get("/products/{asin}/similar")
def get_similar_to_product(asin: str, top_k: int = 6):
    """
    Get products similar to a specific product.
    Useful for "You might also like" on product detail pages.
    """
    # Find the product
    product_idx = None
    target_product = None

    for idx, product in enumerate(ALL_PRODUCTS):
        if product['asin'] == asin:
            product_idx = idx
            target_product = product
            break

    if product_idx is None:
        return {"error": "Product not found", "recommendations": []}

    # Get embedding for this product
    product_embedding = PRODUCT_EMB[product_idx]

    # Compute similarities
    similarities = cosine_similarity([product_embedding], PRODUCT_EMB)[0]

    # Rank and exclude the product itself
    candidates = []
    for idx, (product, sim_score) in enumerate(zip(ALL_PRODUCTS, similarities)):
        if product['asin'] == asin:
            continue
        candidates.append({
            'product': product,
            'similarity_score': float(sim_score)
        })

    # Sort by similarity
    candidates.sort(key=lambda x: x['similarity_score'], reverse=True)

    # Return top K
    recommendations = []
    for candidate in candidates[:top_k]:
        product = candidate['product'].copy()
        product['similarity_score'] = candidate['similarity_score']
        recommendations.append(product)

    return {
        "source_product": target_product['short_title'],
        "recommendations": recommendations
    }


@app.on_event("startup")
async def startup_event():
    print("=" * 70)
    print("🎁 GiveAura AI - Enhanced Recommendation Engine v6.0")
    print("=" * 70)
    print(f"📦 Products Loaded: {len(ALL_PRODUCTS)}")
    print(f"🎯 Improvements:")
    print(f"   ✓ Hard gender exclusion (keyword-based)")
    print(f"   ✓ Rebalanced weights (filters prioritized)")
    print(f"   ✓ Multi-match bonus scoring")
    print(f"   ✓ Strict mismatch penalties")
    print(f"🚀 API Ready on http://localhost:8000")
    print("=" * 70)