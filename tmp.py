import os
import requests

STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
assert STABILITY_API_KEY, "❌ STABILITY_API_KEY 환경변수가 설정되지 않았습니다."

OUTPUT_IMAGE_PATH = "kitty_story_text2img.png"



PROMPT = (
    "A warm, whimsical storybook illustration of a blue cartoon cat named Kitty entering a magical forest for the first time. "
    "Kitty has soft blue fur, a cream-colored belly, a round head, and large black and white eyes. "
    "In this scene, Kitty is crouching near a small pond under a tree, staring curiously at the water where shiny rocks and little fish swim. "
    "She looks slightly startled but begins to smile as sunlight sparkles off the water. "
    "Butterflies flutter around her. Some trees arch above like a gentle tunnel, and leaves glow with the morning sun. "
    "The perspective is angled slightly from the side, with Kitty partially in profile, adding depth and variation to the scene. "
    "The style is soft, detailed, and painted like a cozy fantasy picture book."
)

# PROMPT = (
#     "A detailed storybook illustration of a blue cartoon cat named Kitty exploring a magical forest for the first time. "
#     "Kitty has soft blue fur, a round head, a cream-colored belly, and large black and white eyes filled with curiosity. "
#     "Sunlight gently shines through tall green leaves above, casting dappled light on the forest floor. "
#     "Colorful butterflies flutter through the air around Kitty. "
#     "At first, Kitty looks slightly nervous, standing near the trees. "
#     "Soon she discovers a small, calm pond hidden under a tree. "
#     "In the water, sparkling rocks and tiny cheerful fish can be seen as if saying hello. "
#     "Kitty smiles with relief and wonder, feeling warmth in the forest. "
#     "The scene is vibrant, magical, and heartwarming, capturing the feeling of discovery and friendship with nature. "
#     "The art style is soft, whimsical, warm, and illustrated like a children’s storybook."
# )

NEGATIVE_PROMPT = (
    "photo, realistic, dark, scary, messy, blue background, distorted face, extra limbs, watermark, text"
)

response = requests.post(
    "https://api.stability.ai/v2beta/stable-image/generate/core",
    headers={
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "accept": "image/*"
    },
    files={"none": ''},
    data={
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "style_preset": "fantasy-art",  # 또는 "digital-art"
        "output_format": "png",
        "seed": 123456
    }
)

# 결과 저장
if response.status_code == 200:
    with open(OUTPUT_IMAGE_PATH, 'wb') as f:
        f.write(response.content)
    print(f"✅ 이미지 생성 완료: {OUTPUT_IMAGE_PATH}")
else:
    print(f"❌ 오류 발생: {response.status_code}")
    try:
        print(response.json())
    except:
        print(response.text)
