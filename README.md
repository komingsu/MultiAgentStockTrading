# stockRL

강화학습 기반 한국 주식 트레이딩 연구용 저장소입니다. KIS/KRX 데이터를 `build_kis_dataset.py`로 준비하고, 단일 DRL 에이전트를 원하는 만큼 반복 학습하는 구조입니다.

## 환경 요약
- Python 3.10 이상 (권장 3.11)
- 주요 의존성: stable-baselines3, gymnasium, numpy, pandas, matplotlib
- 데이터 파일: `data/train_data.csv`, `data/stock_test_data.csv`, `data/trade_data.csv`

## 학습 파이프라인
### 1. 데이터 재구성 (선택)
```bash
python build_kis_dataset.py \
  --start 2023-01-01 --end 2025-10-21 \
  --train-start 2023-01-02 --train-end 2024-12-30 \
  --test-start 2025-01-02 --test-end 2025-04-29 \
  --trade-start 2025-05-01 --trade-end 2025-08-31 \
  --backtest-end 2025-10-21
```
(인자를 생략하면 `config.py`의 기본 분할 사용)

### 2. 단일 에이전트 학습
```bash
python scripts/run_train_agent.py \
  --algo ppo \
  --experiment-name exp_ppo
```
- 지원 알고리즘: `a2c`, `ddpg`, `ppo`, `sac`, `td3`
- 옵션:
  - `--timesteps-scale`, `--total-timesteps`: 학습 스텝 조정
  - `--turbulence-threshold`: trade 평가 시 임계치 지정
- 결과: `experiments/exp_ppo/` 아래에 모델(`models/agent_ppo.zip`), test/trade 자산 CSV·PNG, 로그(`logs/experiment.log`) 저장

### 랜덤 초기 포트폴리오
학습 에피소드마다 `hyperparams.py`에 정의된 규칙으로 초기 상태를 샘플링합니다.
- 총자본 1억: 현금 5천만 + 주식 목표 5천만
- 초기 보유 종목 3~8개, 종목당 최대 25% 가치 캡
- 남는 금액은 현금으로 유지

## 액션 추론 CLI
이미 학습된 모델로 현재 보유 상태에서 다음 액션을 추론할 수 있습니다.
```bash
python scripts/infer_action.py \
  --experiment experiments/exp_ppo \
  --algo ppo \
  --cash 25000000 \
  --holdings holdings.json \
  --date 2025-08-29
```
- `holdings`는 파일 경로 또는 JSON 문자열 (`{"005930": 120, ...}`)
- 미지정 시 `data/trade_data.csv` 마지막 날짜 사용
- 출력: 원 액션, 실행 액션, 체결 후 수량, 현금/총자산 변화

## 하이퍼파라미터 관리
`hyperparams.py`
- `DEFAULT_ALGO_CONFIG`: 각 알고리즘 timesteps 및 SB3 인자
- `ENV_PARAMS`: `hmax`, 거래비용, 보상 스케일
- `PORTFOLIO_INIT`: 초기 포트폴리오 총액·현금 비중·종목수 범위·티커별 캡
수정 시 학습/추론/환경이 자동으로 반영됩니다.

## 소스 구조
- `config.py` : 데이터 기간, 지표 목록, 공통 경로
- `hyperparams.py` : 모든 하이퍼파라미터
- `env.py` : Gymnasium 환경 + 랜덤 초기 포트폴리오 로직
- `train.py` : 실험 디렉터리/로그 관리, 단일 에이전트 학습/평가 유틸
- `models.py` : SB3 에이전트 래퍼(get/train/predict)
- `scripts/`
  - `run_train_agent.py` : 학습/평가 CLI
  - `infer_action.py` : 보유 상태 → 액션 추론 CLI
  - `plot_training_results.py` : SB3 모니터 로그 기반 학습 곡선 PNG 생성
- `vis_util.py` : 자산 곡선 시각화
- `data/` : 전처리된 CSV
- `experiments/` : 실험 산출물

## 로그 및 산출물
- 학습/평가지표는 `logs/experiment.log`에 `metric|value|step` 형식으로 누적
- SB3 모니터 & 로거: `logs/sb3/train_monitor/monitor.csv`, `logs/sb3/eval/`(평가 콜백), `logs/sb3/progress.csv`, TensorBoard 이벤트(`logs/sb3/events.*`, `tensorboard` 설치 시)
- Test/Trade 평가 결과는 `results/` CSV + `plots/` PNG

## 기타 스크립트
- `helper_function.py` : 지표 계산, 데이터 스플릿 유틸
- `kis_auth.py`, `refresh_kis_token.py` : KIS 인증/토큰 관리
- `build_kis_dataset.py` : KIS API 기반 데이터 빌더

필요 시 `PORTFOLIO_INIT` 값을 조정하거나 `run_train_agent.py`를 반복 실행해 다양한 모델을 얻으세요.
