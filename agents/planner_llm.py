import openai
import os
import json


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

        try:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            json_text = content[json_start:json_end]
            return json.loads(json_text)
        except Exception as e:
            print("[!] Failed to parse response as JSON.")
            print("[Raw Output]:\n", content)
            return {"error": str(e), "raw": content}


if __name__ == "__main__":
    print("🧠 Planner LLM 실행 중...")

    risk_score = int(input("👉 오늘의 Risk Score를 입력하세요 (0~100): ").strip())

    story_prompt = """이 콘텐츠는 하루에 하나 생성되는 어린이용 이야기이며, 짧은 동화 또는 일기처럼 구성되어야 합니다.

1. 이야기의 주제는 “오늘 하루의 특정 경험이나 인상적인 장면”에서 출발합니다.  
   - 예: 친구와 놀던 순간, 길을 잃었다가 다시 만난 경험, 가족과 함께한 특별한 날

2. 구조는 다음과 같은 순서를 따릅니다:
   - (1) 상황 제시: 장소, 시간, 등장인물 소개
   - (2) 감정 표현: 주인공이 어떤 감정을 느꼈는지 구체적으로 묘사
   - (3) 사건 전개: 갈등이나 특별한 상황이 있었는가?
   - (4) 마무리: 오늘의 교훈이나 느낀 점으로 정리

3. 문체는 다음과 같은 기준을 따릅니다:
   - 문장은 짧고 단순해야 하며, 1문장 1개념 원칙을 지킵니다.
   - 추상적이거나 무거운 단어보다는 구체적이고 밝은 어휘를 사용합니다.
   - 폭력, 죽음, 공포, 불안, 혐오 등의 표현은 절대 사용하지 않습니다.
   - 의도적인 감정 강요(“반드시 기뻐해야 해”)는 피하고, 자연스럽게 느끼게 만듭니다.

4. 전체 글의 분위기는 **오늘 하루를 차분히 회고하는 일기**처럼 구성되어야 하며,  
   마지막에는 ‘그래서 나는 이렇게 느꼈다’는 식의 교훈 또는 정리 문장으로 마무리하세요."""
    


    image_prompt = """This prompt is for generating one single image that complements a short daily story.  
The image must follow the mood and simplicity appropriate for children.

1. Structure of the prompt:
   - (1) Define the time of day and weather: “a sunny afternoon”, “a cloudy morning”
   - (2) Set a recognizable location: “a playground”, “a classroom”, “a quiet forest”
   - (3) Add 1–2 main visual elements (subject + action): “a child hugging a cat”, “a rabbit sitting under a tree”
   - (4) Optional background activity or atmosphere: “soft sunlight through leaves”, “toys scattered in the background”

2. Style guidelines:
   - Language must be simple, clear, and visual-first.
   - Avoid complex emotions or metaphorical phrases.
   - Use gentle adjectives: soft, warm, peaceful, friendly, bright.
   - Avoid: dark, scary, surreal, chaotic, hyper-realistic.

3. Framing:
   - The image should feel like a snapshot from the story – a single moment in time.
   - Focus on 1 main scene or emotion, not multiple story beats.
   - Keep the scene open and friendly for interpretation by a child.

4. Format recommendations:
   - Use commas to separate scene elements clearly.
   - Avoid chaining too many concepts in one prompt.
   - Avoid negations like “not scary” – instead, specify what *is* visible."""
    

    planner = PlannerLLM()
    result = planner.generate(
        risk_score=risk_score,
        story_prompt=story_prompt,
        image_prompt=image_prompt
    )

    print("\n📤 결과 출력 (JSON):")
    print(json.dumps(result, indent=2, ensure_ascii=False))
