from typing import Dict, Optional


class PromptTemplate:
    """Base class for prompt templates."""
    
    def __init__(self):
        self.few_shot_examples = self._get_few_shot_examples()
        self.category_instructions = self._get_category_instructions()
    
    def _get_few_shot_examples(self) -> Dict[str, list]:
        """Get few-shot examples for different categories."""
        return {
            "sports": [
                {
                    "title": "Lakers Beat Warriors in Overtime Thriller",
                    "abstract": "LeBron James scored 35 points as the Los Angeles Lakers defeated the Golden State Warriors 128-125 in overtime.",
                    "description": "In an electrifying overtime showdown, LeBron James delivered a masterclass performance with 35 points, leading the Lakers to a narrow 128-125 victory over the Warriors. The game showcased intense competition and clutch plays from both teams."
                },
                {
                    "title": "Serena Williams Announces Retirement",
                    "abstract": "Tennis legend Serena Williams announced her retirement from professional tennis after 27 years.",
                    "description": "Tennis icon Serena Williams has announced her retirement after an illustrious 27-year career that redefined women's tennis. Her legacy includes 23 Grand Slam singles titles and countless contributions to the sport."
                }
            ],
            "news": [
                {
                    "title": "New Climate Agreement Reached at Summit",
                    "abstract": "World leaders agreed on new carbon emission targets at the international climate summit.",
                    "description": "In a landmark achievement, world leaders have reached a comprehensive climate agreement establishing ambitious carbon emission reduction targets. The summit marks a significant step forward in global environmental cooperation."
                },
                {
                    "title": "Economic Growth Exceeds Expectations",
                    "abstract": "The economy grew by 3.2% in the last quarter, surpassing analyst predictions.",
                    "description": "The economy demonstrated robust performance with 3.2% growth in the latest quarter, exceeding market expectations and signaling strong economic momentum. Analysts attribute the growth to increased consumer spending and business investment."
                }
            ],
            "entertainment": [
                {
                    "title": "New Marvel Movie Breaks Box Office Records",
                    "abstract": "The latest Marvel superhero film earned $250 million in its opening weekend.",
                    "description": "Marvel's newest blockbuster has shattered box office records with an impressive $250 million opening weekend, demonstrating the franchise's continued dominance in the entertainment industry and strong audience appeal."
                },
                {
                    "title": "Grammy Awards Celebrate Music Excellence",
                    "abstract": "The 65th Grammy Awards honored outstanding achievements in music across various genres.",
                    "description": "The 65th Grammy Awards ceremony celebrated musical excellence across diverse genres, recognizing both established artists and emerging talents. The event showcased memorable performances and emotional acceptance speeches."
                }
            ],
            "finance": [
                {
                    "title": "Federal Reserve Raises Interest Rates",
                    "abstract": "The Fed increased rates by 0.25% to combat inflation concerns.",
                    "description": "The Federal Reserve has implemented a 0.25% interest rate increase as part of its ongoing strategy to address inflationary pressures. The decision reflects the central bank's commitment to maintaining economic stability."
                },
                {
                    "title": "Tech Stocks Rally on Earnings Reports",
                    "abstract": "Major technology companies reported strong quarterly earnings, boosting market confidence.",
                    "description": "Technology stocks experienced significant gains following impressive quarterly earnings reports from major companies. The strong financial performance has renewed investor confidence in the tech sector's growth prospects."
                }
            ],
            "health": [
                {
                    "title": "New Vaccine Shows Promise in Clinical Trials",
                    "abstract": "A novel vaccine demonstrated 95% efficacy in phase 3 trials.",
                    "description": "Breakthrough clinical trial results show a new vaccine achieving 95% efficacy in phase 3 testing, offering hope for improved disease prevention. The findings represent a significant advancement in medical research."
                },
                {
                    "title": "Study Links Exercise to Better Mental Health",
                    "abstract": "Research shows regular physical activity reduces depression and anxiety symptoms.",
                    "description": "A comprehensive study has established a strong connection between regular exercise and improved mental health outcomes, demonstrating significant reductions in depression and anxiety symptoms among active individuals."
                }
            ],
            "technology": [
                {
                    "title": "AI Breakthrough in Natural Language Processing",
                    "abstract": "Researchers developed a new AI model that achieves human-level language understanding.",
                    "description": "Scientists have achieved a major breakthrough in artificial intelligence with a new model demonstrating human-level natural language understanding capabilities. This advancement could revolutionize how machines interact with human communication."
                },
                {
                    "title": "Apple Unveils Next-Generation iPhone",
                    "abstract": "Apple announced its latest iPhone featuring advanced camera technology and improved battery life.",
                    "description": "Apple has revealed its next-generation iPhone, showcasing cutting-edge camera innovations and enhanced battery performance. The new device represents the company's continued commitment to technological excellence and user experience."
                }
            ]
        }
    
    def _get_category_instructions(self) -> Dict[str, str]:
        """Get category-specific instructions for generation."""
        return {
            "sports": "Focus on the competitive aspects, key players, scores, and the excitement of the event. Highlight athletic achievements and game-changing moments.",
            "news": "Provide objective, factual information about current events. Focus on the who, what, when, where, and why. Maintain a neutral, journalistic tone.",
            "entertainment": "Emphasize the cultural impact, audience reception, and creative aspects. Capture the excitement and appeal of entertainment content.",
            "finance": "Focus on economic implications, market trends, and financial data. Use precise terminology and explain the significance for investors and the economy.",
            "health": "Highlight medical significance, research findings, and health implications. Use clear language to explain complex medical concepts.",
            "technology": "Emphasize innovation, technical capabilities, and potential impact. Explain how the technology works and its practical applications.",
            "lifestyle": "Focus on practical advice, trends, and personal relevance. Make the content relatable and actionable for readers.",
            "travel": "Highlight destinations, experiences, and practical travel information. Capture the appeal and unique aspects of locations.",
            "foodanddrink": "Emphasize flavors, culinary techniques, and dining experiences. Make the content appetizing and engaging.",
            "autos": "Focus on vehicle features, performance, and automotive innovation. Highlight technical specifications and driving experience.",
            "video": "Describe visual content, key moments, and viewer appeal. Capture what makes the video compelling.",
            "weather": "Provide clear, actionable weather information. Focus on conditions, forecasts, and potential impacts.",
            "music": "Highlight musical style, artist achievements, and cultural significance. Capture the emotional and artistic aspects.",
            "movies": "Focus on plot, performances, and cinematic quality. Provide insight without spoilers.",
            "tv": "Highlight show premise, character development, and viewer appeal. Capture what makes the series engaging.",
            "default": "Create a clear, engaging summary that captures the essence of the article. Focus on key information and maintain reader interest."
        }
    
    def format_prompt(self, title: str, abstract: str, category: Optional[str] = None,
                     use_few_shot: bool = True) -> str:
        """
        Format a prompt for news description generation.
        
        Args:
            title: News article title
            abstract: News article abstract
            category: News category (optional)
            use_few_shot: Whether to include few-shot examples
            
        Returns:
            Formatted prompt string
        """
        category = (category or "default").lower()
        
        # Get category-specific instruction
        instruction = self.category_instructions.get(
            category,
            self.category_instructions["default"]
        )
        
        # Build prompt
        prompt_parts = [
            "You are a professional news editor creating engaging article descriptions.",
            f"\nTask: Generate a concise, informative description (150-200 characters) for a news article.",
            f"\nCategory: {category.title()}",
            f"\nGuidelines: {instruction}",
        ]
        
        # Add few-shot examples if requested
        if use_few_shot and category in self.few_shot_examples:
            prompt_parts.append("\nExamples:")
            for i, example in enumerate(self.few_shot_examples[category][:2], 1):
                prompt_parts.append(
                    f"\nExample {i}:"
                    f"\nTitle: {example['title']}"
                    f"\nAbstract: {example['abstract']}"
                    f"\nDescription: {example['description']}"
                )
        
        # Add the actual article to process
        prompt_parts.extend([
            "\n\nNow generate a description for this article:",
            f"\nTitle: {title}",
            f"\nAbstract: {abstract}",
            "\nDescription:"
        ])
        
        return "\n".join(prompt_parts)
    
    def format_batch_prompt(self, articles: list, category: Optional[str] = None) -> str:
        """
        Format a prompt for batch processing multiple articles.
        
        Args:
            articles: List of dicts with 'title' and 'abstract' keys
            category: News category (optional)
            
        Returns:
            Formatted prompt string
        """
        category = (category or "default").lower()
        instruction = self.category_instructions.get(
            category,
            self.category_instructions["default"]
        )
        
        prompt_parts = [
            "You are a professional news editor creating engaging article descriptions.",
            f"\nTask: Generate concise descriptions (150-200 characters) for multiple news articles.",
            f"\nCategory: {category.title()}",
            f"\nGuidelines: {instruction}",
            "\n\nGenerate descriptions for these articles (respond with JSON format):",
        ]
        
        for i, article in enumerate(articles, 1):
            prompt_parts.append(
                f"\n{i}. Title: {article['title']}"
                f"\n   Abstract: {article['abstract']}"
            )
        
        prompt_parts.append(
            '\n\nRespond in JSON format: {"1": "description1", "2": "description2", ...}'
        )
        
        return "\n".join(prompt_parts)


class SimplePromptTemplate(PromptTemplate):
    """Simplified prompt template without few-shot examples."""
    
    def format_prompt(self, title: str, abstract: str, category: Optional[str] = None,
                     use_few_shot: bool = False) -> str:
        """Format a simple prompt without examples."""
        category = (category or "default").lower()
        instruction = self.category_instructions.get(
            category,
            self.category_instructions["default"]
        )
        
        return (
            f"Create a concise, engaging news description (150-200 characters) for this article.\n"
            f"Category: {category.title()}\n"
            f"Guidelines: {instruction}\n\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}\n\n"
            f"Description:"
        )


class MinimalPromptTemplate(PromptTemplate):
    """Minimal prompt template for cost-sensitive applications."""
    
    def format_prompt(self, title: str, abstract: str, category: Optional[str] = None,
                     use_few_shot: bool = False) -> str:
        """Format a minimal prompt."""
        if category:
            category_text = f" [{category.upper()}]"
        else:
            category_text = ""
        
        return (
            f"Summarize this{category_text} news article in 150-200 characters:\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}\n"
            f"Summary:"
        )


def get_prompt_template(template_type: str = "default") -> PromptTemplate:
    """
    Get a prompt template by type.
    
    Args:
        template_type: Type of template ("default", "simple", "minimal")
        
    Returns:
        PromptTemplate instance
    """
    templates = {
        "default": PromptTemplate,
        "simple": SimplePromptTemplate,
        "minimal": MinimalPromptTemplate,
    }
    
    template_class = templates.get(template_type.lower(), PromptTemplate)
    return template_class()
