# agents/image_critic_llm.py
import openai
import os
import json
import re
from PIL import Image
import torch
import clip
from torchvision import transforms


class ImageCriticLLM:
    def __init__(
        self,
        model="gpt-4o",
        rule_path="rules/image_critic_rule.txt",
        prompt_template_path="prompts/image_critic_prompt.txt",
        image_path="memory/generated_image.png",
        story_path="memory/present_story.txt"
    ):
        self.model = model
        self.rule = self._load_file(rule_path)
        self.prompt_template = self._load_file(prompt_template_path)
        self.image_path = image_path
        self.story = self._load_file(story_path)

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

    def compute_clip_score(self, image_path, long_text):
        def safe_token_length(text):
            try:
                return len(clip.tokenize(text)[0])
            except RuntimeError:
                return 999

        model, preprocess = clip.load("ViT-B/32", device="cpu")
        image = preprocess(Image.open(image_path)).unsqueeze(0).to("cpu")

        max_tokens = 77
        sentences = long_text.strip().split(". ")
        chunks, current_chunk = [], ""

        for sentence in sentences:
            candidate = current_chunk + sentence + ". "
            if safe_token_length(candidate) < max_tokens:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        scores = []
        for chunk in chunks:
            text_tokens = clip.tokenize([chunk]).to("cpu")
            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text_tokens)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T).item()
                scores.append(similarity)

        raw_score = sum(scores) / len(scores)
        adjusted_score = min(round(raw_score * 1.7, 4), 1.0)  # 70% boost
        return adjusted_score

    def build_prompt(self, clip_score):
        return self.prompt_template \
            .replace("{STORY_TEXT}", self.story.strip()) \
            .replace("{CLIP_SCORE}", str(clip_score))

    def generate(self):
        clip_score = self.compute_clip_score(self.image_path, self.story)
        filled_prompt = self.build_prompt(clip_score)
        full_prompt = f"{self.rule}\n\n{filled_prompt}"

        print("\n📄 === 최종 전달된 GPT Prompt ===")
        print(full_prompt)

        client = openai.OpenAI()

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "너는 이미지 평가자야. RULE을 반드시 따르고, JSON 형식으로 결과를 제공해."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3
        )

        result = response.choices[0].message.content.strip()
        cleaned = self._strip_json_block(result)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON", "raw": result}


# ✅ 외부 호출용 call 함수
def call(
    model="gpt-4o",
    rule_path="rules/image_critic_rule.txt",
    prompt_template_path="prompts/image_critic_prompt.txt",
    image_path="memory/generated_image.png",
    story_path="memory/present_story.txt"
):
    critic = ImageCriticLLM(
        model=model,
        rule_path=rule_path,
        prompt_template_path=prompt_template_path,
        image_path=image_path,
        story_path=story_path
    )
    return critic.generate()


# 🧪 단독 실행
if __name__ == "__main__":
    print("🖼️ Image Critic LLM 실행 중...")
    result = call()
    print("\n📤 평가 결과:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
