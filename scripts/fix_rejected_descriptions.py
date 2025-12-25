"""
Script để tìm và sửa các descriptions bị LLM reject (trả về "I can't help")
"""
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def find_rejected_descriptions(descriptions_path):
    """
    Tìm các descriptions bị reject
    
    Args:
        descriptions_path: Path to descriptions JSON file
        
    Returns:
        List of rejected news_ids
    """
    print("=" * 60)
    print("FINDING REJECTED DESCRIPTIONS")
    print("=" * 60)
    
    with open(descriptions_path, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)
    
    # Patterns indicating rejection
    reject_patterns = [
        "i can't help",
        "i cannot help",
        "i can't assist",
        "i cannot assist",
        "i can't",
        "i cannot",
        "i can’t"
        "i'm unable to",
        "i am unable to",
        "i apologize",
        "sorry, i can't",
        "sorry, i cannot"
    ]
    
    rejected = []
    for news_id, desc in descriptions.items():
        desc_lower = desc.lower()
        if any(pattern in desc_lower for pattern in reject_patterns):
            rejected.append(news_id)
    
    print(f"Total descriptions: {len(descriptions)}")
    print(f"Rejected descriptions: {len(rejected)}")
    print(f"Rejection rate: {len(rejected)/len(descriptions)*100:.2f}%")
    
    return rejected


def fix_rejected_descriptions(descriptions_path, news_meta_path, output_path=None):
    """
    Sửa các descriptions bị reject bằng cách concat title + abstract
    
    Args:
        descriptions_path: Path to descriptions JSON file
        news_meta_path: Path to news_meta JSON file
        output_path: Path to save fixed descriptions (optional)
    """
    print("\n" + "=" * 60)
    print("FIXING REJECTED DESCRIPTIONS")
    print("=" * 60)
    
    # Load descriptions
    with open(descriptions_path, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)
    
    # Load news_meta
    with open(news_meta_path, 'r', encoding='utf-8') as f:
        news_meta = json.load(f)
    
    # Find rejected
    reject_patterns = [
        "i can't help",
        "i cannot help",
        "i can't assist",
        "i cannot assist",
        "i can't",
        "i cannot",
        "i can’t"
        "i'm unable to",
        "i am unable to",
        "i apologize",
        "sorry, i can't",
        "sorry, i cannot"
    ]
    
    fixed_count = 0
    not_found = []
    
    for news_id, desc in descriptions.items():
        desc_lower = desc.lower()
        
        # Check if rejected
        if any(pattern in desc_lower for pattern in reject_patterns):
            # Try to fix
            if news_id in news_meta:
                title = news_meta[news_id].get('title', '')
                abstract = news_meta[news_id].get('abstract', '')
                category = news_meta[news_id].get('category', '')
                
                # Create simple description
                if title and abstract:
                    # Concat title + abstract, limit to 200 chars
                    category = news_meta[news_id].get('category', 'unknown')
                    new_desc = (
                        f"This article is about {category}. "
                        f"{title}. {abstract}"
                    )[:200]
                    
                    descriptions[news_id] = new_desc
                    fixed_count += 1
                    print(f"Fixed {news_id}: {new_desc[:80]}...")
                else:
                    print(f"Warning: {news_id} has empty title/abstract")
                    not_found.append(news_id)
            else:
                print(f"Warning: {news_id} not found in news_meta")
                not_found.append(news_id)
    
    print(f"\nFixed {fixed_count} descriptions")
    if not_found:
        print(f"Could not fix {len(not_found)} descriptions: {not_found[:10]}...")
    
    # Save
    if output_path is None:
        output_path = descriptions_path.replace('.json', '_fixed.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(descriptions, f, ensure_ascii=False, indent=2)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSaved fixed descriptions to: {output_path}")
    print(f"File size: {file_size:.2f} MB")
    
    return descriptions


def analyze_rejected_content(descriptions_path, news_meta_path):
    """
    Phân tích nội dung của các bài bị reject để hiểu nguyên nhân
    
    Args:
        descriptions_path: Path to descriptions JSON file
        news_meta_path: Path to news_meta JSON file
    """
    print("\n" + "=" * 60)
    print("ANALYZING REJECTED CONTENT")
    print("=" * 60)
    
    # Load data
    with open(descriptions_path, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)
    
    with open(news_meta_path, 'r', encoding='utf-8') as f:
        news_meta = json.load(f)
    
    # Find rejected
    reject_patterns = [
        "i can't help",
        "i cannot help",
        "i can't assist",
        "i cannot assist",
        "i'm unable to",
        "i am unable to",
        "i can't",
        "i cannot",
        "i can’t"
        "i apologize",
        "sorry, i can't",
        "sorry, i cannot"
    ]
    
    rejected_by_category = {}
    
    for news_id, desc in descriptions.items():
        desc_lower = desc.lower()
        
        if any(pattern in desc_lower for pattern in reject_patterns):
            if news_id in news_meta:
                category = news_meta[news_id].get('category', 'unknown')
                
                if category not in rejected_by_category:
                    rejected_by_category[category] = []
                
                rejected_by_category[category].append({
                    'news_id': news_id,
                    'title': news_meta[news_id].get('title', ''),
                    'category': category
                })
    
    # Print analysis
    print("\nRejected by category:")
    for category, items in sorted(rejected_by_category.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n{category}: {len(items)} articles")
        # Show first 3 examples
        for item in items[:3]:
            print(f"  - {item['news_id']}: {item['title'][:60]}...")
    
    return rejected_by_category


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Find and fix rejected LLM descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find rejected descriptions
  python scripts/fix_rejected_descriptions.py \\
    --descriptions Data/generated/news_descriptions.json \\
    --news_meta Data/processed/train/news_meta.json \\
    --action find
  
  # Fix rejected descriptions
  python scripts/fix_rejected_descriptions.py \\
    --descriptions Data/generated/news_descriptions.json \\
    --news_meta Data/processed/train/news_meta.json \\
    --action fix
  
  # Analyze rejected content
  python scripts/fix_rejected_descriptions.py \\
    --descriptions Data/generated/news_descriptions.json \\
    --news_meta Data/processed/train/news_meta.json \\
    --action analyze
        """
    )
    
    parser.add_argument(
        "--descriptions",
        required=True,
        help="Path to descriptions JSON file"
    )
    
    parser.add_argument(
        "--news_meta",
        required=True,
        help="Path to news_meta JSON file"
    )
    
    parser.add_argument(
        "--action",
        choices=["find", "fix", "analyze"],
        default="fix",
        help="Action to perform (default: fix)"
    )
    
    parser.add_argument(
        "--output",
        help="Output path for fixed descriptions (default: <input>_fixed.json)"
    )
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.descriptions):
        print(f"Error: Descriptions file not found: {args.descriptions}")
        return
    
    if not os.path.exists(args.news_meta):
        print(f"Error: News meta file not found: {args.news_meta}")
        return
    
    # Perform action
    if args.action == "find":
        rejected = find_rejected_descriptions(args.descriptions)
        if rejected:
            print(f"\nRejected news IDs: {rejected[:20]}...")
            if len(rejected) > 20:
                print(f"... and {len(rejected) - 20} more")
    
    elif args.action == "fix":
        find_rejected_descriptions(args.descriptions)
        fix_rejected_descriptions(args.descriptions, args.news_meta, args.output)
        print("\n✅ Done! Use the fixed file for training.")
    
    elif args.action == "analyze":
        find_rejected_descriptions(args.descriptions)
        analyze_rejected_content(args.descriptions, args.news_meta)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
