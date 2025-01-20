


from typing import List
from pydantic import BaseModel
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.llm_addons import generate_pydantic_json_model


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

subject_line_grader_prompt_2 = """

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

For each aspect, wrap your thought process in <subject_line_analysis> tags before providing your final evaluation. In your analysis:
a) List relevant observations from the subject line
b) Consider how these observations relate to the email type and other provided information
c) Evaluate the potential impact on email performance

Consider how each aspect interacts with the others and how they collectively impact the subject line's performance.

After your analysis, provide a detailed report in the following structure:

1. Effectiveness Score (50-100):
   [Score]
   [Brief explanation of the score]

2. Scannability Score (1-10):
   [Score]
   [Brief explanation of the score]

3. Sentiment Analysis:
   [Positive/Negative/Neutral]
   [Brief explanation of the sentiment]

4. Length Analysis:
   [Optimal/Too Long/Too Short]
   [Number of characters]
   [Number of words]
   [Brief explanation of the length analysis]

5. Spam Triggers:
   [List each potential spam trigger with a number prefix, if any]
   [Brief explanation of their impact]

6. All Caps Words:
   [List each word in all caps with a number prefix, if any]
   [Brief explanation of their impact]

7. Emojis:
   [List any emojis present]
   [Recommendation on emoji usage]
   [Suggested emoji, if applicable]

8. Preview Texts:
   Provide 2-3 preview text examples that complement the subject line (max 2-3 words)
   - [Preview text 1]
   - [Preview text 2]
   - [Preview text 3]

9. Alternative Subject Lines:
   Suggest 3-5 improved alternatives based on your analysis:
   - [Alternative 1]
   - [Alternative 2]
   - [Alternative 3]
   - [Alternative 4]
   - [Alternative 5]

Ensure that your analysis is comprehensive and that your suggestions take into account all the provided information, including the email type and whether it's time-sensitive, a follow-up, or part of a series.


"""
openai_instance  = LLM.create(provider=LLMProvider.OPENAI,model_name="gpt-4o")

generated_prompt = subject_line_grader_prompt_2.format(subject_line="3 seo tips",
                                                               email_type = "Newsletter",
                                                               is_time_sensitive="No",
                                                               is_followup = "No",
                                                               is_series = "No")


ai_response = generate_pydantic_json_model(model_class=EmailSubjectLineAnalysis,
                                                            prompt=generated_prompt,
                                                            llm_instance=openai_instance,
                                                            max_retries=2,
                                                            max_tokens=4096,
                                                            system_prompt="Generate in Arabic Language")


print(ai_response)