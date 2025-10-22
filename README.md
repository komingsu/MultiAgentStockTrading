# stockRL

한국 주식 데이터를 기반으로 강화학습형 트레이딩 실험을 구성한 저장소입니다. `build_kis_dataset.py`로 최신 데이터를 수집/전처리하고, `scripts/run_experiment.py`로 5개의 DRL 에이전트(A2C, DDPG, PPO, TD3, SAC)와 앙상블 전략을 일괄 학습/평가할 수 있습니다.

## 데이터 재구성
```bash
python build_kis_dataset.py \
  --start 2023-01-01 --end 2025-10-21 \
  --train-start 2023-01-02 --train-end 2024-12-30 \
  --test-start 2025-01-02 --test-end 2025-04-29 \
  --trade-start 2025-05-01 --trade-end 2025-08-31 \
  --backtest-end 2025-10-21
```
> `config.py`에 정의된 분할 구간을 사용하면 위 인자 없이 실행해도 동일한 결과를 얻습니다.

## 학습/실험 실행
5개의 에이전트를 충분히 학습하고 연속 백테스트 및 앙상블 평가까지 수행하려면 다음과 같이 실행합니다.
```bash
python scripts/run_experiment.py --experiment-name <실험이름>
```
- `--experiment-name`은 결과가 저장될 폴더 이름입니다. 지정하지 않으면 타임스탬프가 자동 할당됩니다.
- `--output-root`(기본값 `experiments`) 아래에 `<실험이름>/` 구조로 **모델 체크포인트**, **SB3 로그**, **결과 CSV/PNG**, **단일 `.log` 파일**이 정리됩니다.
- `--timesteps-scale`을 사용하면 학습 스텝을 일괄적으로 스케일링할 수 있습니다(예: `0.1`은 빠른 리허설, 기본값 `1.0`).
- 옵션으로 `--turbulence-threshold` 등도 조정 가능합니다.

TensorBoard 또는 pyfolio를 설치하면(선택 사항) 로그 시각화와 추가 성능 지표를 자동으로 저장합니다.

## 폴더 구조
- `build_kis_dataset.py` : KIS API에서 일별 시세를 수집하고 기술지표, VIX, 터뷸런스를 계산해 `data/`에 저장.
- `config.py` : 데이터 분할 구간, 사용 지표, 대상 종목 목록 및 모델 하이퍼파라미터 정의.
- `data/` : 최신 전처리 결과 (train/test/trade/backtest, `kis_full_data.csv`, `vix_daily.csv`, `turbulence_daily.csv`).
- `env.py` : 강화학습 환경 정의. 모든 중간 산출물을 `config.RESULTS_DIR` 경로 아래로 저장.
- `helper_function.py` : 기술지표 계산, 데이터 스플릿, 베이스라인 지표 계산 유틸리티.
- `models.py` : Stable-Baselines3 에이전트 래퍼, 학습/예측 함수, 앙상블 전략 구현.
- `scripts/`
  - `run_experiment.py` : 5개 에이전트 학습 → 트레이드 백테스트 → 앙상블 전략 → 시각화/로그를 자동 수행하는 CLI.
- `experiments/` : 각 실험의 산출물이 생성되는 루트 디렉터리 (커밋에는 `.gitkeep`만 포함).
- `kis_auth.py`, `refresh_kis_token.py` : KIS API 토큰 관리.
- `p3.ipynb`, `make_dataset.ipynb` : 기존 실험 노트북(참고용).

필요 없는 임시 산출물과 디버그 디렉터리는 정리되었으며, 새 실험을 실행하면 `experiments/<실험이름>/` 아래에 결과가 깔끔하게 정리됩니다.
