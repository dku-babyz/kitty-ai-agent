# core/agent_manager.py

import json
import os
import shutil
import pandas as pd
from agents import planner_llm, story_llm, story_critic_llm, image_llm, image_critic_llm, prompt_critic_llm


class AgentManager:
    def __init__(
        self,
        story_prompt_path="prompts/story_prompt.txt",
        image_prompt_path="prompts/image_prompt.txt",
        story_direction_path="prompts/story_direction.txt",
        planner_result_path="memory/planner_result.json",
        present_story_path_en="memory/present_story.txt",
        present_story_path_kr="memory/present_story_kr.txt",
        previous_story_path="memory/previous_story.txt",
        story_feedback_path="memory/story_feedback.txt",
        image_feedback_path="memory/image_feedback.txt",
        memory_csv_path="memory/memory.csv"
    ):
        self.story_prompt_path = story_prompt_path
        self.image_prompt_path = image_prompt_path
        self.story_direction_path = story_direction_path
        self.planner_result_path = planner_result_path
        self.present_story_path_en = present_story_path_en
        self.present_story_path_kr = present_story_path_kr
        self.previous_story_path = previous_story_path
        self.story_feedback_path = story_feedback_path
        self.image_feedback_path = image_feedback_path
        self.memory_csv_path = memory_csv_path

    def _load_file(self, path):
        with open(path, encoding="utf-8") as f:
            return f.read()

    def _save_file(self, path, content):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _append_file(self, path, content):
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)

    def _save_json(self, path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def run_agent(self, risk_score: int):
        print("\n🚀 AgentManager 시작")

        # ────── 저장용 원본 프롬프트 백업 (리비전 전)
        original_story_prompt = self._load_file(self.story_prompt_path)
        original_image_prompt = self._load_file(self.image_prompt_path)

        # ────── 1. Run Planner
        print("\n📋 플래너 실행 중...")
        planner_result = planner_llm.call(risk_score, original_story_prompt, original_image_prompt)

        if planner_result["prompt_for_story_creator"]["how_to_generate"] == 1:
            story_prompt = planner_result["prompt_for_story_creator"]["Revision"]
            self._save_file(self.story_prompt_path, story_prompt)

        if planner_result["prompt_for_image_creator"]["how_to_generate"] == 1:
            image_prompt = planner_result["prompt_for_image_creator"]["Revision"]
            self._save_file(self.image_prompt_path, image_prompt)

        tone_label = planner_result["tone_label"]
        story_direction = planner_result["prompt_for_story_creator"]["story_direction"]
        self._save_file(self.story_direction_path, f"{tone_label}\n{story_direction}")
        self._save_json(self.planner_result_path, planner_result)
        print("✅ Planner 완료")

        # ────── 2. Story + Critic 루프
        best_score = -1
        best_story = {"korean": "", "english": ""}
        feedback_log = ""

        for i in range(1, 4):
            print(f"\n🔁 [LOOP{i}] Story 생성 중...")
            result = story_llm.call(
                rule_path="rules/story_rule.txt",
                prompt_path=self.story_prompt_path,
                past_story_path=self.previous_story_path,
                direction_path=self.story_direction_path,
                feedback_path=self.story_feedback_path
            )
            if "korean_story" not in result or "english_story" not in result:
                print("❌ 스토리 생성 실패")
                continue

            self._save_file(self.present_story_path_kr, result["korean_story"])
            self._save_file(self.present_story_path_en, result["english_story"])

            print(f"🧐 [LOOP{i}] Critic 평가 중...")
            critic_result = story_critic_llm.call(
                rule_path="rules/story_critic_rule.txt",
                prompt_template_path="prompts/story_critic_prompt.txt",
                story_prompt_path=self.story_prompt_path,
                previous_story_path=self.previous_story_path,
                present_story_path=self.present_story_path_en
            )

            if "scores" not in critic_result:
                print("❌ 평가 실패")
                continue

            total = sum([v for v in critic_result["scores"].values() if isinstance(v, (int, float))])
            if total > best_score:
                best_score = total
                best_story["korean"] = result["korean_story"]
                best_story["english"] = result["english_story"]

            feedback_log += f"[LOOP{i}]\n{json.dumps(critic_result, ensure_ascii=False, indent=2)}\n\n"
            self._save_file(self.story_feedback_path, feedback_log)

        self._save_file(self.present_story_path_kr, best_story["korean"])
        self._save_file(self.present_story_path_en, best_story["english"])
        print("\n✅ 최종 스토리 저장 완료")

        # ────── 3. Image + Critic 루프
        best_score = -1
        best_image_path = ""

        for i in range(1, 4):
            print(f"🖼️ [LOOP{i}] 이미지 생성 중...")
            output_path = f"memory/generated_image_{i}.png"
            self._save_file(self.image_feedback_path, "")

            image_llm.call(
                rule_path="rules/image_rule.txt",
                prompt_template_path="prompts/image_prompt.txt",
                story_path=self.present_story_path_en,
                image_prompt_path=self.image_prompt_path,
                feedback_path=self.image_feedback_path,
                output_path=output_path
            )

            print(f"🔎 [LOOP{i}] 이미지 평가 중...")
            critic_result = image_critic_llm.call(
                rule_path="rules/image_critic_rule.txt",
                prompt_template_path="prompts/image_critic_prompt.txt",
                image_path=output_path,
                story_path=self.present_story_path_en
            )

            if "scores" not in critic_result:
                print("❌ 이미지 평가 실패")
                continue

            score_sum = sum([v for v in critic_result["scores"].values() if isinstance(v, (int, float))])
            if score_sum > best_score:
                best_score = score_sum
                best_image_path = output_path

            self._save_file(self.image_feedback_path, critic_result.get("feedback", ""))

        if best_image_path:
            final_image_path = "memory/generated_image.png"
            shutil.copy(best_image_path, final_image_path)
            for i in range(1, 4):
                temp_path = f"memory/generated_image_{i}.png"
                if os.path.exists(temp_path) and temp_path != best_image_path:
                    os.remove(temp_path)
            print(f"🏁 최종 이미지 저장 완료: {final_image_path}")
        else:
            print("❌ 최종 이미지 선택 실패")

        # ────── 4. Prompt Critic 실행
        print("\n📝 프롬프트 평가 및 리비전 수행 중...")
        critic_result = prompt_critic_llm.call(
            rule_path="rules/prompt_critic_rule.txt",
            story_prompt_path=self.story_prompt_path,
            image_prompt_path=self.image_prompt_path,
            story_feedback_path=self.story_feedback_path,
            image_feedback_path=self.image_feedback_path,
            output_path="results/revised_story_prompt.txt"
        )

        if "revisions" in critic_result:
            if critic_result["revisions"].get("revised_story_prompt"):
                self._save_file(self.story_prompt_path, critic_result["revisions"]["revised_story_prompt"])
            if critic_result["revisions"].get("revised_image_prompt"):
                self._save_file(self.image_prompt_path, critic_result["revisions"]["revised_image_prompt"])

        # ────── 5. Memory 정리 및 기록
        print("\n📦 Memory 정리 중...")
        if os.path.exists(self.memory_csv_path):
            df = pd.read_csv(self.memory_csv_path)
        else:
            df = pd.DataFrame(columns=[
                "cycle", "story_prompt", "image_prompt", "story_direction",
                "story_feedback", "image_feedback", "previous_story",
                "present_story", "image_name"])

        if df.empty:
            cycle = 1
        else:
            existing = pd.to_numeric(df["cycle"], errors="coerce").dropna()
            cycle = int(existing.max()) + 1 if not existing.empty else 1

        entry = {
            "cycle": cycle,
            "story_prompt": original_story_prompt,
            "image_prompt": original_image_prompt,
            "story_direction": self._load_file(self.story_direction_path),
            "story_feedback": self._load_file(self.story_feedback_path),
            "image_feedback": self._load_file(self.image_feedback_path),
            "previous_story": self._load_file(self.previous_story_path),
            "present_story": self._load_file(self.present_story_path_en),
            "image_name": f"image_{cycle}.png"
        }

        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        df.to_csv(self.memory_csv_path, index=False)

        # 파일 리셋/이동 처리
        shutil.copy(self.present_story_path_en, self.previous_story_path)
        os.remove(self.story_direction_path)
        os.remove(self.story_feedback_path)
        os.remove(self.image_feedback_path)
        os.remove(self.present_story_path_en)
        os.rename("memory/generated_image.png", f"memory/image_{cycle}.png")
        print(f"🧾 Memory 저장 완료 (cycle {cycle})")

        # ────── 6. 최종 결과 반환
        print("\n📤 최종 결과 반환 중...")
        return {
            "final_story": best_story["english"],
            "final_image_path": f"memory/image_{cycle}.png"
        }


# 🧪 단독 실행
if __name__ == "__main__":
    print("🧠 AgentManager: Full Agent 파이프라인 실행")
    manager = AgentManager()
    risk = int(input("👉 오늘의 Risk Score 입력 (0~100): "))
    result = manager.run_agent(risk)
    print("\n🎉 완료된 결과:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
