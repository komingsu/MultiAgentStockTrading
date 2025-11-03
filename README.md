# StockRL Live Trading MVP — 빠른 시작(운영자용)

이 레포는 한국투자증권(KIS) Open API를 이용한 한국 주식 실거래 직전(MVP) 파이프라인을 제공합니다.
핵심 경로(Env → AOC → to_orders → Preflight → LiveRouter)가 동일 규칙으로 동작하도록 설계되어,
백테스트와 라이브의 동등성(Parity)을 유지합니다.

자세한 운영 매뉴얼은 `docs/OPERATOR_GUIDE.md`를 참고하세요.

## 1) 한 줄 요약 & 범위
- 지원 실행 모드: MOO(개장가 시장가), LIMIT_OHLC(일봉 지정가)
- 비범위: 전략/모델 변경, 데이터 파이프라인 확장, 정정/취소/추격 체결

## 2) 설치/요구사항
- Python 3.11 권장, 가상환경 권장
- KIS 자격/계좌(모의 가능): `KIS_APP_KEY`, `KIS_APP_SECRET`, `KIS_ACCOUNT_NO`, `KIS_PRODUCT_CODE=01`
- (선택) `.env`에 KIS 키/토큰 관리. 민감정보는 반드시 비커밋

## 3) 디렉터리 맵(요약)
- `env.py`                : 실행 모드(MOO/LIMIT_OHLC) 규칙 포함 Gym 환경
- `execution/*.py`        : AOC(plan_from_weights), OrderSpec, SimBroker(Parity)
- `live/route.py`         : 수량→OrderSpec 변환(틱 라운딩/정수 호가)
- `live/kis_router.py`    : KIS REST 주문 래퍼(모의/실전 공용)
- `scripts/infer_action.py`: 추론→AOC→Plan JSON 생성
- `scripts/submit_orders.py`: 플랜 제출(드라이런/모의/실전) + 프리플라이트
- `tests/`                : Parity/틱/플래너 유닛 테스트(24개)

## 4) 3줄 런북(자가 테스트 → 플랜 생성 → 드라이런)
```bash
pytest -q tests                               # 기대: 24 passed
python scripts/infer_action.py \
  --experiment experiments/exp_ppo --algo ppo \
  --cash 25000000 --holdings '{"005930": 10}' \
  --write-plan-json logs/plans/plan_$(date +%Y%m%d).json
python scripts/submit_orders.py \
  --plan-json logs/plans/plan_$(date +%Y%m%d).json \
  --exec-mode MOO --limit-offset-bps 0 --dry-run --use-paper \
  --holiday-file config/holiday_kr.json --output logs/submit_orders_dryrun.json
```

## 5) 일일 운영 루틴
1) 전일 데이터 동기화 → 2) 인퍼런스(AOC)로 플랜 생성 → 3) 드라이런 프리플라이트
→ 4) (선택) 모의 제출 → 5) 실전 제출(소액·저빈도) → 6) 응답/주문번호 보관

## 6) 시간/휴일 가드 & 플랜 날짜 주의
- 플랜 파일명 날짜(YYYYMMDD)가 오늘(KST)와 다르면 제출 차단
- 시간창: MOO 08:30–09:00 / LIMIT 09:00–15:20 (KST)
- `--force-submit`은 테스트 전용

## 7) KRX 틱·정수 호가 요약
| 가격대(원)            | 틱(원) |
|---------------------|-------|
| < 2,000             | 1     |
| 2,000 ~ 4,999       | 5     |
| 5,000 ~ 9,999       | 10    |
| 10,000 ~ 19,999     | 10    |
| 20,000 ~ 49,999     | 50    |
| 50,000 ~ 99,999     | 100   |
| 100,000 ~ 199,999   | 100   |
| 200,000 ~ 499,999   | 500   |
| ≥ 500,000           | 1,000 |

매수=내림, 매도=올림으로 라운딩하며, LIMIT 가격은 정수 문자열로 전송합니다.

## 8) 안전장치(요약)
- Fail-Closed: 프리플라이트 실패/네트워크/토큰 오류 시 즉시 중단
- 401/429/5xx 1회 재시도(백오프)
- 멱등성: `client_order_id` 중복 제출 차단

## 9) Troubleshooting(대표 에러 ↔ 조치)
- `Plan date ... does not match today (KST)` → 당일 플랜으로 재생성 또는 `--force-submit`
- `sell_qty_exceeds_position:SYM:A>B` → 보유수량 확인, 플랜 조정
- `insufficient_cash:X>Y` → 가용 현금 대비 비용 축소(플랜 조정/오프셋 상향)
- `invalid limit price` → 틱/정수 호가 규칙 점검
- `retry_exhausted` → 네트워크/인증 실패. 로그 확인 후 재시도

## 10) 법적/리스크 면책
본 저장소는 연구용 자료입니다. 실제 투자/운용에 따른 손실 책임은 전적으로 사용자에게 있습니다.

— 운영 매뉴얼 전체판: `docs/OPERATOR_GUIDE.md`
