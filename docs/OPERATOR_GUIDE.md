# StockRL Live Trading MVP — 운영 가이드(KIS)

본 문서는 운영자가 이 레포를 이용해 데이터 준비 → 학습 → 인퍼런스(AOC) → 주문 계획 생성(Plan JSON) → 프리플라이트/제출(드라이런/모의/실전) 순서를 안전하게 수행하도록 절차 중심으로 정리되었습니다. 문서의 모든 명령과 값은 “예시”이며, 실제 실행은 사용자의 환경과 검토에 따라 진행하십시오.

- 지원 범위: 실행 모드 2가지만 지원
  - MOO(개장 시장가)
  - LIMIT_OHLC(일봉 지정가)
- 비범위: 전략/모델 변경, 데이터 파이프라인 확장, 정정/취소/추격 체결 같은 복잡 로직
- 최소 요구환경: Python 3.11, (선택) 가상환경, 한국투자증권(KIS) App Key/Secret/계좌(모의 가능)

디렉토리 스냅샷(요지)
- `env.py` — 환경(ExecutionMode, 체결 시뮬)
- `execution/` — AOC(Planner), SimBroker, OrderSpec
- `live/` — 주문 변환(to_orders), KIS Router
- `scripts/` — `infer_action.py`, `submit_orders.py`
- `tests/` — 유닛·패리티·틱 테스트
- `config/` — 환경 설정(예: 휴장일 JSON)

> 주의: 본 가이드는 기능 제안/코드 변경을 포함하지 않습니다. 모든 실행은 운영자의 판단하에 진행하십시오.

---

## 0. 빠른 검증(첫 실행 전)

### 환경 점검
- [ ] Python 3.11, (선택) 가상환경 사용
- [ ] KIS 자격 정보(App Key/Secret/계좌)는 환경변수로 주입하고, 파일/저장소에 커밋하지 않음
- [ ] 운영 서버의 시스템 타임존이 KST 기준으로 처리되도록 확인(또는 코드의 KST 기준 가드를 인지)

### 레포 자체 테스트 실행(권장)
```bash
pytest -q tests
# 기대: 24 passed
```
→ 실패 시 운영 전 중단하고 원인을 해소합니다.

## 1. 데이터 준비(맨바닥부터)

### 1.1 입력 데이터 스키마(필수)
- 반드시 포함할 컬럼
  - `date`(str, YYYY-MM-DD), `open`(float), `high`(float), `low`(float), `close`(float), `volume`(float/int), `tic`(str)
  - (선택) `turbulence` 등 기술지표는 없더라도 동작합니다.
- 인덱스 규칙
  - `date` 단위로 factorize되어 `df.index`가 0,1,…(거래일 순번)이 되도록 준비
- 멀티심볼
  - 동일 `date`에 여러 `tic` 행 존재(멀티 심볼 지원)

### 1.2 수집/적재 방법(개요)
- 실거래 대비 권장: KIS 시세 API(일자별/기간별 시세)로 CSV/Parquet 적재
- 임시: `data/` 하위 CSV 사용 가능(스키마만 충족하면 동작)
- ADV 참조: Planner는 `prev_volume` 또는 `sma5_volume` 방식으로 t-1 거래량을 ADV 상한으로 사용

### 1.3 데이터 검증 체크리스트
- [ ] 각 `date`의 모든 심볼에 `open/high/low/close/volume` NaN 없음
- [ ] 심볼들의 `tic` 문자열이 일관
- [ ] 거래일 인덱스(0..N-1) 연속성 확보

---

## 2. 실행 모드/핵심 개념(요약)

- ExecutionMode
  - `MOO`: T 관측 → T+1 개장가 시장가 체결(초기 MVP는 부분체결 없음 가정)
  - `LIMIT_OHLC`: T 관측, 전일 종가±offset으로 지정가 설정 → T+1 OHLC로 체결/미체결 판정(ADV/현금/보유 캡 반영)
- AOC(Planner): 목표 비중/델타 → 현금/보유/ADV/턴오버/최소금액/노트레이드밴드/최대종목수까지 고려하여 정수 수량 벡터 산출
- 주문 변환(to_orders): 정수 수량 → OrderSpec(MOO=MARKET, LIMIT_OHLC=LIMIT) 변환, 국내 주식 **KRX 틱 라운딩 + 정수 호가 적용**
- Preflight: 보유/현금 게이트 + 시간/휴장일 게이트 → **Fail-Closed**
- LiveBroker(KIS): 주문/잔고/현금 조회, 1회 재시도/토큰갱신/백오프 내장

> 도해(개략)
> 상태 → [AOC] → desired_qtys → [to_orders] → OrderSpec(시장가/지정가) → [Preflight] → [LiveBroker 제출]

---

## 3. 구성/파라미터(요약 표)

### 3.1 실행/체결 관련

| 키 | 값/예시 | 설명 |
|---|---|---|
| `exec_mode` | `MOO` / `LIMIT_OHLC` | 실행 모드 |
| `limit_offset_bps` | 0, 50, 100 | LIMIT 지정가 오프셋(bps) |
| `adv_fraction` | 0.1 | ADV 상한 비율 |
| `slippage_bps` | 0~ | 슬리피지(bps) |
| `fee_buy[]`/`fee_sell[]` | 예: 0.001 | 종목별 수수료(소수) |
| `day_only` | True | DAY 유효 |

### 3.2 AOC(Planner)

| 키 | 값/예시 | 설명 |
|---|---|---|
| `w_max` | 0.10 | 종목별 최대 비중 |
| `turnover_max` | 0.20 | 일일 턴오버 한도 |
| `min_notional` | 50000 | 최소 주문 금액(원) |
| `epsilon_w` | 0.005 | 노트레이드 밴드 |
| `cash_buffer_pct` | 0.001 | 현금 버퍼 비율 |
| `max_orders` | 20 | 일 최대 종목 수 |
| `lot_size` | 1 | 주식 lot 크기 |
| `adv_ref` | `prev_volume`/`sma5_volume` | ADV 기준 |

### 3.3 인퍼런스/로깅

| 옵션 | 예시 | 설명 |
|---|---|---|
| `--report-dir` | `logs/aoc_reports` | AOC 리포트 저장 경로 |
| `--write-plan-json` | `logs/plans/plan_YYYYMMDD.json` | Plan JSON 저장 경로 |

### 3.4 제출/가드

| 항목 | 값/예시 | 설명 |
|---|---|---|
| MOO 시간창 | 08:30–09:00 KST | 이외 시간 차단(옵션으로 `--force-submit` 우회 가능) |
| LIMIT 시간창 | 09:00–15:20 KST | 동일 |
| `--holiday-file` | `config/holiday_kr.json` | `["YYYY-MM-DD", ...]` |
| `--force-submit` | 플래그 | 시간/휴일 가드 우회(테스트용) |

Holiday 파일 샘플(예시)
```json
// config/holiday_kr.json (예시)
[
  "2025-01-01",
  "2025-02-28",
  "2025-03-01"
]
```
> 파일이 없으면 빈 집합으로 처리되어 운영은 계속됩니다. 휴장일 차단이 필요하면 위와 같이 날짜를 추가합니다.

### 3.5 KIS 크리덴셜(환경변수 권장)

| 키 | 설명 |
|---|---|
| `KIS_APP_KEY` / `KIS_API_KEY` | App Key |
| `KIS_APP_SECRET` / `KIS_API_SECRET` | App Secret |
| `KIS_ACCOUNT_NO` | 계좌번호(CANO) |
| `KIS_PRODUCT_CODE` | `01`(일반) |

---

## 4. 학습(Training) — 권장 루틴
> 전략/모델 변경은 비범위입니다. 아래는 기존 학습 스크립트 사용법 “예시”입니다(실행 환경에 맞게 조정하세요).

- 데이터 범위: (예시) 시작~T-5일 학습, T-4~T 검증(5일)
- 게이트(예시): 검증에서 `수익률 > 0`, `턴오버 ≤ 20%`, `비용 반영 샤프 ≥ 0.5` → 실패 시 NO_TRADE
- 산출물: 모델 체크포인트(`experiments/<name>/models/...`), 로그/리포트(`experiments/<name>/logs`, `results`)

예시 명령(실행 예시이며 강제 아님)
```bash
# (예시) PPO 학습
python scripts/run_train_agent.py \
  --experiment experiments/overnight_ppo \
  --algo ppo --timesteps 1000000
```

---

## 5. 인퍼런스 → 플랜 생성(AOC 포함)

### 5.1 인퍼런스 스텝
- 모델 로드 → 최신 상태 구성 → AOC 계획 산출 → Plan JSON 저장
- 중요: 인퍼런스 경로에서 `env.hmax = 1`(코드 내부 강제), 환경 추적/플롯 비활성

예시 명령(실제 실행은 환경에 맞게)
```bash
python scripts/infer_action.py \
  --experiment experiments/overnight_ppo \
  --cash 10000000 \
  --holdings '{"005930": 100, "000660": 0}' \
  --action-schema WEIGHT \
  --w-max 0.10 --turnover-max 0.20 --min-notional 50000 --epsilon-w 0.005 \
  --adv-ref prev_volume --adv-frac 0.2 \
  --exec-mode MOO --limit-offset-bps 0 \
  --report-dir logs/aoc_reports \
  --write-plan-json logs/plans/plan_YYYYMMDD.json \
  --data data/stock_test_data.csv
```

### 5.2 Plan JSON 스키마
```json
[
  {"symbol": "005930", "desired_qty": 3, "close": 70000.0},
  {"symbol": "000660", "desired_qty": 0, "close": 123000.0}
]
```
- 최소 1개는 `desired_qty ≠ 0`

BUY/SELL(음수 수량) 포함 예시
```json
[
  {"symbol": "005930", "desired_qty":  3,  "close": 70000.0},  // BUY
  {"symbol": "000660", "desired_qty": -2,  "close": 123000.0}  // SELL
]
```
*SELL 게이트*: `desired_qty<0`인 항목은 보유 수량 이내에서만 프리플라이트 통과(초과 시 실패).

### 5.3 AOC 리포트 읽는 법(요지)
- `planning.per_symbol[].{w_curr,w_tgt,delta_shares,adv_cap,limit_px}`
- `clips.{epsilon,adv,cash,turnover}`, `ok.{cash_nonnegative,no_short,turnover_respected,...}`

---

## 6. 프리플라이트/제출(드라이런 → 모의 → 실전)

### 중요: Plan 파일명 날짜 = 오늘(KST)
제출 스크립트는 `plan_YYYYMMDD.json`의 **YYYYMMDD가 오늘(KST)과 같지 않으면 제출을 차단**합니다.

- 오류 예: `Plan date 2025-04-29 does not match today 2025-04-30 (KST)`
- 해결: 당일 플랜으로 다시 생성하거나 `--force-submit`(테스트 전용) 사용

### 6.1 프리플라이트(드라이런) — 추천 절차
- 현금/보유 게이트
  - SELL 합계 ≤ 보유 수량, BUY 합계 ≤ `cash_available × (1−1e−3)`
  - MARKET은 기준가격(실시간 시세 가능 시 사용, 없으면 plan close)으로 보수 추정
- 시간/휴장일 가드: KST 창 내에서만 제출(옵션으로 `--force-submit`)
- 산출물: `logs/submit_orders_dryrun.json`
  - `orders[]`(symbol/side/qty/order_type/limit_price/client_order_id)
  - `responses[]`(accepted/message/returned id 등)

예시 명령(실제 실행은 환경에 맞게)
```bash
python scripts/submit_orders.py \
  --plan-json logs/plans/plan_YYYYMMDD.json \
  --exec-mode MOO --limit-offset-bps 0 \
  --dry-run --use-paper \
  --holiday-file config/holiday_kr.json \
  --output logs/submit_orders_dryrun.json
```

### 6.2 모의 제출(Paper)
- 환경변수로 KIS 자격 세팅 후 `--use-paper`, `--dry-run` 제거
- 응답에서 **주문번호(ODNO 등)** 기록 확인

예시(모의)
```bash
python scripts/submit_orders.py \
  --plan-json logs/plans/plan_YYYYMMDD.json \
  --exec-mode MOO --limit-offset-bps 0 \
  --use-paper \
  --output logs/submit_orders_paper.json
```

### 6.3 실전 제출
- `--use-paper` 제거(실전 base URL 이용), 소액/저빈도(일 1회)부터 시작

#### 드라이런 → 모의 → 실전 전환 순서

| 단계        | 목적                | 필수 플래그                  | 기대 산출                            |
| --------- | ----------------- | ----------------------- | -------------------------------- |
| 드라이런      | 프리플라이트·라운딩 검증     | `--dry-run --use-paper` | `logs/submit_orders_dryrun.json` |
| 모의(Paper) | KIS 모의 응답·주문번호 확인 | `--use-paper`           | `logs/submit_orders_paper.json`  |
| 실전        | 실제 제출(소액/저빈도)     | (없음)                    | `logs/submit_orders.json`        |

---

## 7. 틱·정수 호가 규칙(실수 방지 요약)
- LIMIT 가격은 **KRW 정수**이며 **KRX 틱 규칙** 적용
- 매수는 **내림**, 매도는 **올림**(불리/유리 방향)
- 간단 예시
  - 10,003원: 매수 → 10,000 / 매도 → 10,010
  - 50,510원: 매수 → 50,500 / 매도 → 50,600

> 구현 상태: `to_orders()`에서 틱 규칙과 정수 라운딩을 적용하여 LIMIT 가격 생성

빠른 참조(가격대별 틱)

| 가격대(원)            | 틱(원) |
| --------------------- | ------ |
| < 2,000               | 1      |
| 2,000 ~ 4,999         | 5      |
| 5,000 ~ 9,999         | 10     |
| 10,000 ~ 19,999       | 10     |
| 20,000 ~ 49,999       | 50     |
| 50,000 ~ 99,999       | 100    |
| 100,000 ~ 199,999     | 100    |
| 200,000 ~ 499,999     | 500    |
| ≥ 500,000             | 1,000  |

---

## 8. 체크리스트(런 전 마지막 10개)
1. [ ] Plan JSON에 `desired_qty>0` 항목 1개 이상 있음
2. [ ] LIMIT 가격 정수/틱 규칙 충족
3. [ ] MARKET 비용 추정 정상(시세/plan close)
4. [ ] SELL ≤ 보유, BUY ≤ 가용현금×(1−1e−3)
5. [ ] 시간/휴장일 가드 통과(또는 테스트 시 `--force-submit` 사용)
6. [ ] 응답 로그의 client_order_id ↔ returned id 매핑 확인
7. [ ] 401/429/5xx 1회 재시도 후 Fail-Closed (로그 확인)
8. [ ] TR-ID/헤더/hashkey 세팅 확인
9. [ ] 인퍼런스 모드에서 추적/플롯 비활성(운영 경로 최소 부하)
10. [ ] `config/holiday_kr.json` 없으면 빈 집합 처리(있으면 휴장일 차단)

---

## 9. 오류 메시지 ↔ 조치(Troubleshooting)

| 메시지 | 원인 | 조치 |
|---|---|---|
| `MOO orders permitted only 08:30-09:00 KST` | 시간 가드 | KST 창 내 재실행 또는 `--force-submit` (테스트만) |
| `sell_qty_exceeds_position:SYM:A>B` | 보유 초과 SELL | 보유수량 확인 후 Plan 조정 |
| `insufficient_cash:X>Y` | 현금 부족 BUY | 현금 충족되게 Plan 조정 / LIMIT offset 상향 |
| `invalid limit price` | 틱/정수 불일치 | 틱 라운딩 규칙 확인(매수=내림, 매도=올림) |
| `retry_exhausted` / `error:...` | 네트워크/인증 실패 | 네트워크/자격 확인, 재시도(실패 시 Fail-Closed) |
| `duplicate_client_order_id` | 중복 제출 | 파일명/날짜 포함한 멱등 키 재생성 |
| `Plan date ... does not match today (KST)` | Plan 파일명 날짜 ≠ 오늘(KST) | 당일 플랜으로 재생성 또는 `--force-submit`(테스트 전용) |

---

## 10. 부록

### 10.1 CLI 옵션 표(요지)

`scripts/infer_action.py`
- 실행 모드: `--exec-mode {MOO,LIMIT_OHLC}`
- Planner: `--w-max --turnover-max --min-notional --epsilon-w --cash-buffer-pct --max-orders`
- ADV 기준: `--adv-ref {prev_volume,sma5_volume}`, `--adv-frac`
- 로깅/플랜: `--report-dir`, `--write-plan-json`
- 데이터/보유/현금: `--data`, `--holdings`, `--cash`

`scripts/submit_orders.py`
- 제출 모드: `--exec-mode`, `--limit-offset-bps`, `--use-paper`, `--dry-run`
- 가드: `--holiday-file`, `--force-submit`
- 출력: `--output`

### 10.2 JSON 예시 모음

Plan JSON
```json
[
  {"symbol": "005930", "desired_qty": 3, "close": 70000.0}
]
```

Submit 결과(JSON)
```json
{
  "plan_path": "logs/plans/plan_YYYYMMDD.json",
  "plan_date": "YYYY-MM-DD",
  "orders": [
    {"symbol":"005930","side":"BUY","qty":3,"order_type":"MARKET","limit_price":null,"client_order_id":"YYYYMMDD-005930-BUY-3"}
  ],
  "responses": [
    {"order_id":"YYYYMMDD-005930-BUY-3","accepted":true,"message":"mock","symbol":"005930","qty":3,"side":"BUY"}
  ]
}
```

### 10.3 로그/산출물 위치
- `logs/plans/` — Plan JSON
- `logs/aoc_reports/` — AOC 리포트(JSON)
- `logs/submit_orders*.json` — 제출/응답 로그

### 10.4 보안 유의
- 자격정보(KIS App Key/Secret/계좌)는 **환경변수로 주입**하고, `.env`나 키 파일은 절대 커밋하지 않습니다.
- 제출 로그(`logs/submit_orders*.json`)에는 **종목·수량·클라이언트 주문ID**가 포함되므로 **사내 저장소 또는 암호화 디스크**에 보관하십시오.

### 10.5 사전 자가 테스트(runbook)
```bash
# (1) 자체 테스트
pytest -q tests

# (2) 플랜 생성 (desired_qty>0 확인)
python scripts/infer_action.py ... \
  --write-plan-json logs/plans/plan_$(date +%Y%m%d).json

# (3) 드라이런 제출
python scripts/submit_orders.py \
  --plan-json logs/plans/plan_$(date +%Y%m%d).json \
  --exec-mode MOO --limit-offset-bps 0 --dry-run --use-paper \
  --holiday-file config/holiday_kr.json \
  --output logs/submit_orders_dryrun.json
```

---

## 11. 수용 기준(Definition of Done)
- 본 문서만으로 **데이터 스키마 → 학습 → 인퍼런스(AOC) → 플랜 JSON → 드라이런 프리플라이트 → (선택) 모의 제출**까지 절차/옵션/산출물 이해 및 수행 가능
- “런 전 마지막 10개” 체크리스트 포함
- 틱/정수 호가 규칙과 예시 명시
- 오류→대응 표 제공으로 운영 중단 없이 원인 파악 가능
- 문서 내 명령은 모두 예시이며, 실제 실행은 운영자 환경에 맞게 조정함을 명시
