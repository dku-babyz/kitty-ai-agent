# story_llm.py
import openai
import os
import json

class StoryLLM:
    def __init__(
        self,
        model="gpt-4o",
        rule_path="rules/story_rule.txt",
        prompt_path="prompts/story_prompt.txt",
        past_story_path="memory/previous_story.txt"
    ):
        self.model = model
        self.rule_template = self._load_file(rule_path)
        self.prompt = self._load_file(prompt_path)
        self.past_story = self._load_file(past_story_path)

    def _load_file(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def build_prompt(self):
        prompt_with_past = self.rule_template \
            .replace("{PROMPT}", self.prompt.strip()) \
            .replace("{PAST_STORY}", self.past_story.strip())
        return prompt_with_past

    def generate(self):
        full_prompt = self.build_prompt()

        print("\n📄 === 최종 전달된 GPT Prompt ===")
        print(full_prompt)

        client = openai.OpenAI()

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 어린이 시리즈 스토리를 쓰는 작가입니다. 반드시 RULE을 따르고, \n"
                        "JSON 형식으로 출력하세요. 다른 말은 절대 하지 마세요."
                    )
                },
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )

        content = response.choices[0].message.content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format", "raw": content}

if __name__ == "__main__":
    print("🧚‍♀️ Story Creator LLM 실행 중...")
    llm = StoryLLM()
    result = llm.generate()

    print("\n📤 생성된 결과 (JSON):")
    print(json.dumps(result, indent=2, ensure_ascii=False))
