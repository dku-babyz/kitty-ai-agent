# agents/planner_llm.py
import openai
import os
import json
import re

class PlannerLLM:
    def __init__(
        self,
        model="gpt-4o",
        rule_path="rules/planner_rule.txt",
        prompt_path="prompts/planner_prompt.txt"
    ):
        self.model = model
        self.rule = self._load_file(rule_path)
        self.template = self._load_file(prompt_path)

    def _load_file(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _strip_json_block(self, text):
        if text.startswith("```json"):
            return re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.DOTALL)
        elif text.startswith("```"):
            return re.sub(r"^```\s*|\s*```$", "", text.strip(), flags=re.DOTALL)
        return text

    def _determine_tone_label(self, risk_score: int) -> str:
        if risk_score <= 10:
            return "POSITIVE"
        elif risk_score <= 30:
            return "NEUTRAL"
        else:
            return "CHALLENGE"

    def build_prompt(self, risk_score: int, story_prompt: str, image_prompt: str) -> str:
        tone_label = self._determine_tone_label(risk_score)
        return (
            self.template
                .replace("{RULE}", self.rule)
                .replace("{risk_score}", str(risk_score))
                .replace("{tone_label}", tone_label)
                .replace("{story_prompt}", story_prompt.strip())
                .replace("{image_prompt}", image_prompt.strip())
        )

    def generate(self, risk_score: int, story_prompt: str, image_prompt: str) -> dict:
        prompt = self.build_prompt(risk_score, story_prompt, image_prompt)

        print("\n📄 === 최종 전달된 GPT Prompt ===")
        print(prompt)

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict prompt reviewer. Your ONLY job is to analyze the input "
                        "and return a JSON response. You must NEVER ask for any further input. "
                        "Just return valid JSON as instructed."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        content = response.choices[0].message.content.strip()
        content_cleaned = self._strip_json_block(content)

        try:
            return json.loads(content_cleaned)
        except json.JSONDecodeError as e:
            return {"error": "Invalid JSON", "raw": content}

# ✅ 외부 호출용
def call(
    risk_score: int,
    story_prompt: str,
    image_prompt: str,
    model: str = "gpt-4o",
    rule_path: str = "rules/planner_rule.txt",
    prompt_path: str = "prompts/planner_prompt.txt"
) -> dict:
    planner = PlannerLLM(model=model, rule_path=rule_path, prompt_path=prompt_path)
    return planner.generate(risk_score, story_prompt, image_prompt)

# 🧪 단독 실행
if __name__ == "__main__":
    print("🧠 Planner LLM 실행 중...")
    risk_score = int(input("👉 오늘의 Risk Score를 입력하세요 (0~100): ").strip())

    with open("prompts/story_prompt.txt", encoding="utf-8") as f:
        story_prompt = f.read()

    with open("prompts/image_prompt.txt", encoding="utf-8") as f:
        image_prompt = f.read()

    result = call(risk_score, story_prompt, image_prompt)
    print("\n📤 결과 출력 (JSON):")
    print(json.dumps(result, indent=2, ensure_ascii=False))
