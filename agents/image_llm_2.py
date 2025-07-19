# image_llm.py
import os
import json
import requests


def call_stability(prompt: str, output_path: str = "kitty_story_text2img.png") -> None:
    """Stable Diffusion API 호출 및 이미지 저장"""
    STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
    assert STABILITY_API_KEY, "❌ STABILITY_API_KEY 환경변수가 설정되지 않았습니다."

    NEGATIVE_PROMPT = (
        "photo, realistic, dark, scary, messy, distorted face, extra limbs, watermark, text"
    )

    response = requests.post(
        "https://api.stability.ai/v2beta/stable-image/generate/core",
        headers={
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "accept": "image/*",
        },
        files={"none": ''},
        data={
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "style_preset": "fantasy-art",
            "output_format": "png",
            "seed": 123456,
        },
        timeout=120,
    )

    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"✅ 이미지 생성 완료 → {output_path}")
    else:
        print(f"❌ Stability API 오류 발생: {response.status_code}")
        try:
            print(response.json())
        except Exception:
            print(response.text)
        raise RuntimeError("Stable Diffusion 이미지 생성 실패")


def main():
    # ① Planner LLM 결과 JSON 파일에서 프롬프트 불러오기
    planner_result_path = "memory/planner_result.json"  # 반드시 해당 경로에 있어야 함
    if not os.path.exists(planner_result_path):
        raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {planner_result_path}")

    with open(planner_result_path, encoding="utf-8") as f:
        data = json.load(f)

    if "image_generation_prompt" not in data:
        raise KeyError("planner_result.json에 'image_generation_prompt' 키가 없습니다.")

    sd_prompt = data["image_generation_prompt"]

    print("\n🖼️ Stable Diffusion Prompt (from planner):\n")
    print(sd_prompt, "\n")

    # ② 이미지 생성
    output_path = "memory/generated_image.png"
    call_stability(prompt=sd_prompt, output_path=output_path)


if __name__ == "__main__":
    main()
