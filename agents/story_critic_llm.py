# story_critic_llm.py
import openai
import os
import json

class StoryCriticLLM:
    def __init__(
        self,
        model="gpt-4o",
        rule_path="rules/story_critic_rule.txt",
        prompt_template_path="prompts/story_critic_prompt.txt",
        story_prompt_path="prompts/story_prompt.txt",
        previous_story_path="memory/previous_story.txt",
        present_story_path="memory/present_story.txt"
    ):
        self.model = model
        self.rule = self._load_file(rule_path)
        self.prompt_template = self._load_file(prompt_template_path)
        self.story_prompt = self._load_file(story_prompt_path)
        self.previous_story = self._load_file(previous_story_path)
        self.present_story = self._load_file(present_story_path)

    def _load_file(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def build_prompt(self):
        filled_prompt = self.prompt_template \
            .replace("{PAST_STORY}", self.previous_story.strip()) \
            .replace("{TODAY_STORY}", self.present_story.strip()) \
            .replace("{STORY_PROMPT}", self.story_prompt.strip())
        return f"{self.rule}\n\n### INPUT\n{filled_prompt}"

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
                        "너는 story 평가자입니다. 반드시 RULE을 따르고, JSON으로만 출력하세요."
                    )
                },
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.4,
            max_tokens=1000
        )

        content = response.choices[0].message.content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format", "raw": content}

if __name__ == "__main__":
    print("🧠 Story Critic LLM 실행 중...")
    critic = StoryCriticLLM()
    result = critic.generate()

    print("\n📤 평가 결과 (JSON):")
    print(json.dumps(result, indent=2, ensure_ascii=False))