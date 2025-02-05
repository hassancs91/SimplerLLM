from typing import List
from pydantic import BaseModel
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm.reliable import ReliableLLM
from SimplerLLM.language.llm_addons import generate_pydantic_json_model_reliable

# Using the same model from test_complex_json.py for consistency
class EffectivenessScore(BaseModel):
    Score: int
    Explanation: str

class ScannabilityScore(BaseModel):
    Score: int
    Explanation: str

class SentimentAnalysis(BaseModel):
    Tone: str
    Explanation: str

class LengthAnalysis(BaseModel):
    number_of_chars: str
    Explanation: str

class SpamTriggers(BaseModel):
    Triggers: List[str]
    Explanation: str

class AllCapsWords(BaseModel):
    Words: List[str]
    Impact: str

class Emojis(BaseModel):
    Recommendation: str
    Explanation: str

class EmailSubjectLineAnalysis(BaseModel):
    Effectiveness_Score: EffectivenessScore
    Scannability_Score: ScannabilityScore
    Sentiment_Analysis: SentimentAnalysis
    Length_Analysis: LengthAnalysis
    Spam_Triggers: SpamTriggers
    All_Caps_Words: AllCapsWords
    Emojis: Emojis
    Suggested_Preview_Text: List[str]
    Alternative_Subject_Lines: List[str]

# Create primary and secondary LLM instances
primary_llm = LLM.create(
    provider=LLMProvider.ANTHROPIC,
    model_name="claude-3-opus-latest",
)

secondary_llm = LLM.create(
    provider=LLMProvider.OPENAI,
    model_name="gpt-4o",
)

# Create ReliableLLM instance
reliable_llm = ReliableLLM(primary_llm=primary_llm, secondary_llm=secondary_llm)

# Test prompt
subject_line_grader_prompt = """
You are a professional email subject line copywriter tasked with analyzing and providing feedback on a given subject line. Your analysis should be thorough, considering multiple factors to ensure the best possible subject line optimization.

Here is the information you need to analyze:

<subject_line>
{subject_line}
</subject_line>

<email_type>
{email_type}
</email_type>

<is_time_sensitive>
{is_time_sensitive}
</is_time_sensitive>

<is_followup>
{is_followup}
</is_followup>

<is_series>
{is_series}
</is_series>

Please analyze the subject line considering the following aspects:
1. Effectiveness
2. Scannability
3. Sentiment
4. Length
5. Spam Triggers
6. All Caps Words
7. Emoji Usage
"""

# Format the prompt with test data
generated_prompt = subject_line_grader_prompt.format(
    subject_line="3 seo tips",
    email_type="Newsletter",
    is_time_sensitive="No",
    is_followup="No",
    is_series="No"
)

# Generate model using ReliableLLM
model_response, used_provider = generate_pydantic_json_model_reliable(
    model_class=EmailSubjectLineAnalysis,
    prompt=generated_prompt,
    reliable_llm=reliable_llm,
    max_retries=2,
    max_tokens=4096,
    system_prompt="Generate in English Language"
)

print("Response from ReliableLLM using provider:", used_provider)
print(model_response)
