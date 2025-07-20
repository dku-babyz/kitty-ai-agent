# core/agent_manager.py
import json
import os

from agents import planner_llm, story_llm

class AgentManager:
    def __init__(
        self,
        story_prompt_path="prompts/story_prompt.txt",
        image_prompt_path="prompts/image_prompt.txt",
        story_direction_path="prompts/story_direction.txt",
        planner_result_path="memory/planner_result.json",
        present_story_path_en="memory/present_story.txt",
        present_story_path_kr="memory/present_story_kr.txt",
        previous_story_path="memory/previous_story.txt"
    ):
        self.story_prompt_path = story_prompt_path
        self.image_prompt_path = image_prompt_path
        self.story_direction_path = story_direction_path
        self.planner_result_path = planner_result_path
        self.present_story_path_en = present_story_path_en
        self.present_story_path_kr = present_story_path_kr
        self.previous_story_path = previous_story_path

        # 캐시 변수
        self.tone_label = ""
        self.story_direction = ""

    def _load_file(self, path):
        with open(path, encoding="utf-8") as f:
            return f.read()

    def _save_file(self, path, content):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _save_json(self, path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def run_planner(self, risk_score: int):
        # 1. 프롬프트 불러오기
        story_prompt = self._load_file(self.story_prompt_path)
        image_prompt = self._load_file(self.image_prompt_path)

        # 2. Planner 실행
        result = planner_llm.call(risk_score, story_prompt, image_prompt)

        # 3. Revision 적용 여부 확인 및 저장
        story_info = result["prompt_for_story_creator"]
        image_info = result["prompt_for_image_creator"]

        if story_info["how_to_generate"] == 1:
            print("🔧 story_prompt 리비전 적용됨")
            self._save_file(self.story_prompt_path, story_info["Revision"])

        if image_info["how_to_generate"] == 1:
            print("🔧 image_prompt 리비전 적용됨")
            self._save_file(self.image_prompt_path, image_info["Revision"])

        # 4. story_direction 저장
        self.tone_label = result["tone_label"]
        self.story_direction = story_info["story_direction"]
        story_direction_text = f"{self.tone_label}\n{self.story_direction}"
        self._save_file(self.story_direction_path, story_direction_text)

        # 5. planner_result 전체 저장
        self._save_json(self.planner_result_path, result)

        print("✅ Planner 단계 완료")

    def run_story_llm(self):
        print("📚 Story LLM 실행 중...")

        result = story_llm.call(
            rule_path="rules/story_rule.txt",
            prompt_path=self.story_prompt_path,
            past_story_path=self.previous_story_path,
            direction_path=self.story_direction_path
        )

        if "korean_story" in result and "english_story" in result:
            self._save_file(self.present_story_path_kr, result["korean_story"])
            self._save_file(self.present_story_path_en, result["english_story"])
            print("✅ Story 생성 및 저장 완료")
        else:
            print("❌ Story LLM 결과에 문제가 있습니다.")
            print(result)


# 🧪 단독 실행 예시
if __name__ == "__main__":
    print("🧠 AgentManager 실행 중...")
    manager = AgentManager()
    risk = int(input("👉 오늘의 Risk Score 입력 (0~100): "))
    manager.run_planner(risk)
    manager.run_story_llm()
