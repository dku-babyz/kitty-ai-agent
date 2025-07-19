# agents/image_llm.py
import os
import requests

class ImageLLM:
    def __init__(self,
                 rule_path="rules/image_rule.txt",
                 prompt_template_path="prompts/image_prompt.txt",
                 story_path="memory/present_story.txt",
                 image_prompt_path="prompts/image_prompt.txt",
                 output_path="memory/generated_image.png",
                 style="fantasy-art",
                 seed=123456):

        self.api_key = os.getenv("STABILITY_API_KEY")
        assert self.api_key, "❌ STABILITY_API_KEY 환경변수가 설정되지 않았습니다."

        self.rule = self._load_file(rule_path)
        self.prompt_template = self._load_file(prompt_template_path)
        self.story = self._load_file(story_path)
        self.image_prompt = self._load_file(image_prompt_path)
        self.output_path = output_path
        self.style = style
        self.seed = seed

        self.negative_prompt = (
            "photo, realistic, dark, scary, messy, blue background, distorted face, extra limbs, watermark, text"
        )

    def _load_file(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def build_prompt(self):
        prompt_text = self.rule \
            .replace("{IMAGE_PROMPT}", self.image_prompt.strip()) \
            .replace("{STORY}", self.story.strip())
        return prompt_text.strip()

    def generate(self):
        full_prompt = self.build_prompt()
        print("\n📄 === Stability에 전달할 최종 Prompt ===")
        print(full_prompt)

        response = requests.post(
            "https://api.stability.ai/v2beta/stable-image/generate/core",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "accept": "image/*"
            },
            files={"none": ''},
            data={
                "prompt": full_prompt,
                "negative_prompt": self.negative_prompt,
                "style_preset": self.style,
                "output_format": "png",
                "seed": self.seed
            },
            timeout=120
        )

        if response.status_code == 200:
            with open(self.output_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ 이미지 생성 완료: {self.output_path}")
        else:
            print(f"❌ Stability API 오류 발생: {response.status_code}")
            try:
                print(response.json())
            except Exception:
                print(response.text)
            raise RuntimeError("Stable Diffusion 이미지 생성 실패")


# ✅ 외부에서 호출 가능한 함수
def call(
    rule_path="rules/image_rule.txt",
    prompt_template_path="prompts/image_prompt.txt",
    story_path="memory/present_story.txt",
    image_prompt_path="prompts/image_prompt.txt",
    output_path="memory/generated_image.png",
    style="fantasy-art",
    seed=123456
):
    generator = ImageLLM(
        rule_path=rule_path,
        prompt_template_path=prompt_template_path,
        story_path=story_path,
        image_prompt_path=image_prompt_path,
        output_path=output_path,
        style=style,
        seed=seed
    )
    generator.generate()


# 🧪 단독 실행
if __name__ == "__main__":
    call()
