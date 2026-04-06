# KCC 2025 Multi-hop QA Experiment Setup

이 문서는 HotpotQA 기반 실험(Vanilla RAG, IRCoT, Ours)을 시작하기 위한 설치, 폴더 구조, 실행 순서를 정리합니다.

## 1) 현재 상태 점검 결과 (2026-04-06)

- 완료: Python 3.11 설치됨
- 완료: Python 3.14 설치됨
- 완료: 프로젝트 가상환경(.venv) 있음 (현재 Python 3.14로 생성됨)
- 미완료: 필수 패키지 설치 안 됨
- 미완료: OPENAI_API_KEY 설정 안 됨
- 완료: 실험 기본 폴더 구조 생성됨
- 완료: requirements.txt 생성됨
- 완료: .env.example 생성됨

## 2) 먼저 할 일 (권장 순서)

1. Python 3.11 가상환경 생성
2. 가상환경 활성화
3. 패키지 설치
4. .env 파일 생성 및 API 키 입력
5. 설치 검증
6. 실험 실행

## 3) 폴더 구조

- src: 실험 코드
- scripts: 실행 스크립트
- configs: 설정 파일
- outputs: 결과(json/csv/log)
- data: 캐시/임시 데이터

## 4) Windows PowerShell 설치 명령

아래를 프로젝트 루트에서 순서대로 실행하세요.

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

Copy-Item .env.example .env

그 다음 .env 파일을 열어서 OPENAI_API_KEY 값을 실제 키로 수정하세요.

## 5) 설치 검증 명령

python -c "import datasets, faiss, openai, tiktoken, numpy, pandas, sklearn, tqdm, dotenv; print('ok')"
python -c "from datasets import load_dataset; ds = load_dataset('hotpot_qa','distractor'); print(len(ds['validation']))"

## 5-1) 자주 발생하는 설치 오류 해결

1. .venv가 Python 3.14로 만들어졌다면 삭제 후 3.11로 재생성

Remove-Item -Recurse -Force .venv
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

2. hotpot-evaluate 패키지 오류

- hotpot-evaluate는 pip 배포가 불안정해서 설치 실패할 수 있음
- 대신 HotpotQA 공식 평가 스크립트(hotpot_evaluate_v1.py)를 저장소에서 받아 사용 권장

## 6) 실험 고정값 (공정 비교)

- LLM: gpt-4o-mini
- Embedding: text-embedding-3-small
- top-k: 5
- chunk size: 200
- overlap: 20
- seed: 42

## 7) 실행 순서 (코드 구현 후)

1. Vanilla RAG
2. IRCoT
3. Ours (decomposition + ordering)
4. Ablation A (ordering 제거)
5. Ablation B (decomposition 제거)

각 실행 결과는 outputs 폴더에 다음 필드를 저장 권장:

- sample_id
- method
- prediction
- answer
- em
- f1
- latency_ms
- prompt_tokens
- completion_tokens

## 8) 참고

- faiss-cpu는 Python 3.14에서 호환 이슈가 날 수 있으므로 3.11 가상환경을 권장합니다.
- API 호출 에러 대비 재시도(backoff) 로직을 코드에 넣는 것을 권장합니다.
