# router

3개 후보 모델(`ax31-light`, `ax31`, `axk1-think`) 중 쿼리마다 하나를 선택해
정확도-비용 트레이드오프를 최적화하는 라우터. `기획안 수정 및 실행 계획` 문서의
설계를 구현한 스캐폴드다.

## ossp-2026-llm-router-challenge에 결합됨

이 스캐폴드는 sibling 저장소 `ossp-2026-llm-router-challenge/`(SKT OSSP 2026
대회 하네스)의 라우터 구현체로 결합되어 있다. 결합 지점:

- `router/ossp_adapter.py` — 대회의 중첩 JSON(InputBatch/OutcomeBatch)을
  이 패키지의 wide-DataFrame 포맷으로 변환하는 오프라인 1회성 어댑터.
- `router/scripts/calibrate_against_official_scorer.py` — 아래 "실데이터로
  전환" 참고.
- `ossp-2026-llm-router-challenge/src/ossp_router/learned_router.py` — 대회의
  `router-run` 인터페이스를 구현하는 추론 어댑터. 이 패키지를
  `ossp-2026-llm-router-challenge/src/router/`에 그대로 vendoring한 사본을
  가져다 쓴다 (원본을 고치면 `robocopy router ossp-2026-llm-router-challenge/src/router /E /XD .venv artifacts __pycache__ .pytest_cache /XF *.pyc`로
  재동기화해야 한다).
- 학습 완료 아티팩트는 `ossp-2026-llm-router-challenge/artifacts/router-model/`에
  커밋되어 있다 (`router/scripts/train.py`의 산출물).

## 실데이터로 전환 완료

실제 대회 데이터(Train 1,760 / Dev 880, `ossp-2026-llm-router-challenge/data/`)로
이미 학습·보정을 마쳤다. `synthetic_data.py`는 이제 `router/tests/`에서만
쓰이는 단위 테스트용 fixture다 (§10.3 당시엔 실데이터가 없어 전체 파이프라인
스모크 테스트 용도로 먼저 만들어졌다).

`config.TIERS.lambda_star_total/per_call`(기획안의 사전 추정치)은 더 이상
쓰지 않는다. 대신 `router/scripts/calibrate_against_official_scorer.py`가
ossp의 실제 Decimal 스코어러(`ossp_router.scoring`)를 실제 Dev에 직접 돌려 λ를
등급별로 재보정하고, 그 결과를 `calibrated_lambda.json`으로 저장한다 —
`learned_router.py`는 이 파일이 있으면 무조건 우선 사용한다. 안전계수(기본
0.90, 등급 예산의 90%를 목표)를 쓰는 이유는 공개 Dev에서 예산을 통과해도
비공개 평가에서 초과할 수 있다는 대회 baseline(`hash-regex`, 공개 Dev
3.985/4.0 → 비공개 평가 약 4.2/4.0으로 Premium 0점)의 전례 때문이다.

## 오늘 바로 실행

레포 루트(이 폴더의 부모 디렉터리)에서:

```bash
python3 -m venv router/.venv
source router/.venv/bin/activate
pip install -r router/requirements.txt

python -m router.scripts.generate_fixture --out-dir router/artifacts/fixture --n-queries 200
python -m router.scripts.train --train-path router/artifacts/fixture/train.parquet \
    --prompts-path router/artifacts/fixture/prompts.parquet --out-dir router/artifacts/model
python -m router.scripts.make_submission --dev-path router/artifacts/fixture/dev.parquet \
    --prompts-path router/artifacts/fixture/prompts.parquet \
    --pipeline-dir router/artifacts/model --tier Balanced \
    --out-path router/artifacts/submission.json
pytest router/tests -v
```

`USE_EMBEDDING_BRANCH=True`(기본값)면 `train`/`make_submission` 실행 시
`sentence-transformers`가 `intfloat/multilingual-e5-small`을 최초 1회 다운로드한다
(네트워크 필요). 오프라인이거나 §10.1 ARM 게이트가 실패하면
`router/config.py`의 `USE_EMBEDDING_BRANCH = False`로 바꿔 TF-IDF 전용 모드로
축소할 수 있다 (`pytest`는 이 모드로 실행되므로 네트워크 없이도 통과한다).

## 문서 섹션 ↔ 파일 매핑

| 문서 섹션 | 파일 |
| --- | --- |
| §1, §2 문제 구조 / rho 스코어 | `decision.py::compute_rho` |
| §3.1 티어/λ* 테이블 | `config.py::TIERS` |
| §4 시스템 구성 | `pipeline.py::RouterPipeline` |
| §5.1 피처 추출 | `features/` |
| §5.2 승률 분류기 + regret 가중 | `schema.py`, `models/win_probability.py` |
| §5.3 출력토큰 분위회귀 (주력 컴포넌트) | `models/output_tokens.py` |
| §5.4 입력토큰 추정 | `models/input_tokens.py` |
| §5.5 결정규칙 | `decision.py::decide` |
| §6 λ 보정 | `calibration.py::calibrate_tier` |
| §8.1 검증기 (A/B1-B4/C) | `validate.py` |
| §10.1 ARM 임베딩 게이트 | `config.py::USE_EMBEDDING_BRANCH`, `features/embeddings.py` |
| §10.2 토큰 집계 방식 진단 | `calibration.py::diagnose_token_convention`, `scripts/diagnose_token_convention.py` |

## 미결 항목

- `config.TOKEN_ACCOUNTING`: 더 이상 쓰지 않음 — ossp의 실제 채점(`scoring.py`)이
  outcome의 `input_tokens`/`output_tokens` 필드를 그대로 쓰고, λ도 사전 추정
  테이블 대신 실제 Dev + 공식 스코어러로 다시 보정하므로 "total vs per_call"
  구분 자체가 이 파이프라인에는 더 이상 영향을 주지 않는다.
  `diagnose_token_convention.py`는 남아있지만 참고용 진단일 뿐이다.
- `config.USE_EMBEDDING_BRANCH` / `EMBEDDING_BACKEND` (§10.1 ARM 레이턴시 게이트) —
  현재 `True`/`sentence_transformers`로 진행 중이나, **실제 linux/arm64
  컨테이너에서 이미지 크기(1GiB 압축 한도)와 등급당 90초 실행 시간을 아직
  검증하지 못했다** (Docker가 없는 환경에서 작업). 실패 시
  `USE_EMBEDDING_BRANCH = False`(TF-IDF만)로 즉시 전환 가능하도록 구조는
  이미 되어 있다.
- `features/handcrafted.py`의 20개 피처 vs 문서 개요의 "~34개" 표기 불일치 — 문서
  재확인 필요 (자세한 내용은 `features/handcrafted.py` 모듈 docstring)
