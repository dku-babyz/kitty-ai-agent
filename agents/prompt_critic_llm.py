# agents/prompt_critic_llm.py
import openai
import os
import json
import re

class PromptCriticLLM:
    def __init__(
        self,
        model="gpt-4o",
        rule_path="rules/prompt_critic_rule.txt",
        story_prompt_path="prompts/story_prompt.txt",
        image_prompt_path="prompts/image_prompt.txt",
        story_feedback_path="memory/story_feedback.txt",
        image_feedback_path="memory/image_feedback.txt",
        output_path="results/revised_story_prompt.txt"
    ):
        self.model = model
        self.rule_path = rule_path
        self.story_prompt_path = story_prompt_path
        self.image_prompt_path = image_prompt_path
        self.story_feedback_path = story_feedback_path
        self.image_feedback_path = image_feedback_path
        self.output_path = output_path

        self.rule = self._load_file(rule_path)
        self.story_prompt = self._load_file(story_prompt_path)
        self.image_prompt = self._load_file(image_prompt_path)
        self.story_feedback = self._load_file(story_feedback_path)
        self.image_feedback = self._load_file(image_feedback_path)

    def _load_file(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _extract_json_from_text(self, text):
        match = re.search(r"\{(?:.|\n)*\}", text)
        return match.group(0) if match else text

    def build_prompt(self):
        return self.rule \
            .replace("{STORY_PROMPT}", self.story_prompt.strip()) \
            .replace("{IMAGE_PROMPT}", self.image_prompt.strip()) \
            .replace("{STORY_FEEDBACK}", self.story_feedback.strip()) \
            .replace("{IMAGE_FEEDBACK}", self.image_feedback.strip())

    def generate(self):
        full_prompt = self.build_prompt()
        print("\n📄 === 전달된 평가 프롬프트 ===")
        print(full_prompt)

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "너는 프롬프트 평가자이자 편집자야. RULE을 따르고, JSON 형식으로 평가와 리비전을 반환해."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.5,
            max_tokens=1200
        )

        content = response.choices[0].message.content.strip()
        content_cleaned = self._extract_json_from_text(content)

        try:
            result = json.loads(content_cleaned)
        except json.JSONDecodeError:
            result = {"error": "Invalid JSON format", "raw": content}

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 프롬프트 리비전 완료 및 저장됨: {self.output_path}")
        return result


# ✅ 외부 모듈에서 쉽게 쓸 수 있게 call() 함수 제공
def call(
    model="gpt-4o",
    rule_path="rules/prompt_critic_rule.txt",
    story_prompt_path="prompts/story_prompt.txt",
    image_prompt_path="prompts/image_prompt.txt",
    story_feedback_path="memory/story_feedback.txt",
    image_feedback_path="memory/image_feedback.txt",
    output_path="results/revised_story_prompt.txt"
):
    critic = PromptCriticLLM(
        model=model,
        rule_path=rule_path,
        story_prompt_path=story_prompt_path,
        image_prompt_path=image_prompt_path,
        story_feedback_path=story_feedback_path,
        image_feedback_path=image_feedback_path,
        output_path=output_path
    )
    return critic.generate()


# 🧪 단독 실행 시
if __name__ == "__main__":
    print("🧠 Prompt Critic LLM 실행 중...")
    result = call()
    print("\n📤 평가 결과:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
