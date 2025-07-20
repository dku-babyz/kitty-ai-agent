from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
import uvicorn
import os
import sys
from dotenv import load_dotenv

# .env 파일 로드 (환경 변수 로드)
load_dotenv()

# 현재 스크립트의 디렉토리를 sys.path에 추가하여 agents 모듈을 임포트할 수 있도록 합니다.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.story_llm import call as story_llm_call

app = FastAPI()

class StoryGenerationRequest(BaseModel):
    model: str = "gpt-4o"
    rule_path: str = "rules/story_rule.txt"
    prompt_path: str = "prompts/story_prompt.txt"
    past_story_path: str = "memory/previous_story.txt"
    direction_path: str = "prompts/story_direction.txt"
    feedback_path: str = "memory/story_feedback.txt"

# API 키 검증 함수
def verify_api_key(x_api_key: str = Header(...)):
    expected_api_key = os.getenv("KITTY_API_KEY")
    if not expected_api_key:
        raise HTTPException(status_code=500, detail="Server configuration error: API key not set.")
    if x_api_key != expected_api_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key

# 파일 경로 유효성 검증 함수
def validate_file_path(base_dir: str, relative_path: str, allowed_subdirs: list[str]):
    # 디렉토리 탐색 공격 방지 (../)
    if ".." in relative_path or relative_path.startswith("/"):
        raise HTTPException(status_code=400, detail=f"Invalid path: {relative_path}. Path cannot contain '..' or start with '/'.")

    full_path = os.path.join(base_dir, relative_path)
    normalized_full_path = os.path.normpath(full_path)

    # 허용된 서브디렉토리 내에 있는지 확인
    is_allowed = False
    for subdir in allowed_subdirs:
        allowed_path_prefix = os.path.normpath(os.path.join(base_dir, subdir))
        if normalized_full_path.startswith(allowed_path_prefix):
            is_allowed = True
            break
    
    if not is_allowed:
        raise HTTPException(status_code=400, detail=f"Access to path {relative_path} is not allowed.")

    return normalized_full_path

@app.get("/")
async def read_root():
    return {"message": "Welcome to Kitty AI Agent API!"}

@app.post("/generate_story/")
async def generate_story(request: StoryGenerationRequest, api_key: str = Depends(verify_api_key)):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 허용된 디렉토리 목록
        allowed_dirs = ["rules", "prompts", "memory"]

        # 각 경로에 대해 유효성 검증 및 절대 경로 생성
        rule_abs_path = validate_file_path(base_dir, request.rule_path, allowed_dirs)
        prompt_abs_path = validate_file_path(base_dir, request.prompt_path, allowed_dirs)
        past_story_abs_path = validate_file_path(base_dir, request.past_story_path, allowed_dirs)
        direction_abs_path = validate_file_path(base_dir, request.direction_path, allowed_dirs)
        feedback_abs_path = validate_file_path(base_dir, request.feedback_path, allowed_dirs)

        story_data = story_llm_call(
            model=request.model,
            rule_path=rule_abs_path,
            prompt_path=prompt_abs_path,
            past_story_path=past_story_abs_path,
            direction_path=direction_abs_path,
            feedback_path=feedback_abs_path
        )
        return story_data
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {e}")
    except HTTPException as e: # validate_file_path에서 발생한 HTTPException을 다시 발생시킴
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during story generation: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)