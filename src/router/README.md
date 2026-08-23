# router

3개 후보 모델(`ax31-light`, `ax31`, `axk1-think`) 중 쿼리마다 하나를 선택해
정확도-비용 트레이드오프를 최적화하는 라우터. `기획안 수정 및 실행 계획` 문서의
설계를 구현한 스캐폴드다.

## 실데이터 없음 — 지금은 synthetic fixture로 동작

실제 대회 데이터(prompt/outcomes/episode 메타)는 아직 없다(문서 §10.3). 대신
`synthetic_data.py`가 문서의 정성적 특징(§1.4 light↔think 상관 ≈0.44, §1.5 think
출력토큰 20배 편차 등)을 재현한 fixture를 만들어, 실데이터가 도착하기 전에도
전체 파이프라인을 오늘 바로 돌려볼 수 있게 한다.

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

## 미결 항목 (실데이터 도착 후 확정 필요)

- `config.TOKEN_ACCOUNTING` (`"total"` vs `"per_call"`, §10.2) — 실데이터로
  `python -m router.scripts.diagnose_token_convention --outcomes-path <실제 outcomes>`
  실행 후 수동으로 확정
- `config.USE_EMBEDDING_BRANCH` / `EMBEDDING_BACKEND` (§10.1 ARM 레이턴시 게이트) —
  ARM 컨테이너에서 벤치마크 후 확정
- `features/handcrafted.py`의 20개 피처 vs 문서 개요의 "~34개" 표기 불일치 — 문서
  재확인 필요 (자세한 내용은 `features/handcrafted.py` 모듈 docstring)
