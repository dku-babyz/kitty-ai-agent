import os
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn
from pathlib import Path
from dotenv import load_dotenv

from agent_manager import AgentManager

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Kitty AI Agent API",
    description="AI 에이전트를 통해 사용자의 Risk Score에 따라 동화와 이미지를 생성하는 API",
    version="1.0.0"
)

# --- Security (API Key Verification) ---
def verify_api_key(x_api_key: str = Header(..., description="API Key for authentication")):
    """API 키를 검증하는 의존성 함수"""
    expected_api_key = os.getenv("KITTY_API_KEY")
    if not expected_api_key or x_api_key != expected_api_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return True

# --- AgentManager Initialization ---
agent_manager = AgentManager()

# --- Pydantic Models ---
class StoryRequest(BaseModel):
    risk_score: int = Field(..., ge=0, le=100, description="사용자의 위험 점수 (0-100 사이)")

class StoryResponse(BaseModel):
    final_story: str
    final_image_path: str

# --- API Endpoints ---
@app.post(
    "/generate-story", 
    response_model=StoryResponse, 
    summary="스토리 및 이미지 생성", 
    description="사용자의 risk_score를 기반으로 AI 에이전트가 동화 텍스트와 관련 이미지를 생성합니다.",
    dependencies=[Depends(verify_api_key)] # 엔드포인트에 API 키 인증 적용
)
async def generate_story_endpoint(request: StoryRequest):
    """
    `risk_score` (0-100)를 입력받아 `AgentManager`를 실행하고, 생성된 스토리와 이미지 경로를 반환합니다.

    - **request**: `risk_score`를 포함하는 JSON 객체.
    - **return**: 생성된 `final_story`와 접근 가능한 `final_image_path`.
    """
    try:
        print(f"Received risk_score: {request.risk_score}")
        result = agent_manager.run_agent(risk_score=request.risk_score)
        
        image_filename = os.path.basename(result["final_image_path"])
        api_image_path = f"/images/{image_filename}"

        print(f"Story and image generated successfully. Image path: {api_image_path}")
        
        return StoryResponse(
            final_story=result["final_story"],
            final_image_path=api_image_path
        )
    except Exception as e:
        # 운영 환경에서는 상세 오류를 로그에만 기록하고, 클라이언트에게는 일반적인 메시지를 반환합니다.
        print(f"An error occurred during story generation: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get(
    "/images/{filename}", 
    summary="생성된 이미지 조회", 
    description="`generate-story` API 호출 결과로 받은 이미지 경로를 통해 실제 이미지 파일을 조회합니다."
)
async def get_image(filename: str):
    """
    파일 이름을 받아 `memory` 디렉토리에서 해당 이미지를 찾아 반환합니다.
    경로 조작 공격을 방지하기 위해 파일 경로를 검증합니다.

    - **filename**: 조회할 이미지 파일 이름 (e.g., `image_1.png`).
    - **return**: 이미지 파일.
    """
    try:
        image_directory = Path("memory").resolve()
        # Path.joinpath()를 사용하여 안전하게 경로 결합
        file_path = (image_directory / Path(filename).name).resolve()

        # 요청된 파일이 의도된 디렉토리 내에 있는지 확인하여 디렉토리 탈출(Directory Traversal) 방지
        if not str(file_path).startswith(str(image_directory)):
            raise HTTPException(status_code=400, detail="Invalid file path.")

        if file_path.is_file():
            return FileResponse(file_path)
        else:
            raise HTTPException(status_code=404, detail="Image not found.")
    except Exception as e:
        print(f"An error occurred while serving image: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# --- Server Execution ---
if __name__ == "__main__":
    print("Starting Kitty AI Agent API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
