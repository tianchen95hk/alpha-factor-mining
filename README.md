# Alpha Factor Mining (Binance USD-M)

一个面向 Binance USDT 永续的横截面因子研究与回测框架。

核心流程：

`数据拉取/缓存 -> 因子生成 -> 多周期评估 -> 去冗余选因子 -> 组合信号 -> 多策略回测 -> 报告输出`

## What It Does

- 自动构建加密因子库（基础因子 + 参数扩展 + 可选组合因子）
- 评估因子有效性（IC、IC-IR、分层 spread 等）
- 支持多周期评估（默认 `6,12,24` bars）
- 支持按因子相关性去冗余（降低“高度相似因子”重复入选）
- 输出单因子、组合因子、多因子风控策略的统一回测结果

## Project Structure

```text
config/                  参数与配置
scripts/run_research.py  运行入口
src/alpha_mining/        核心实现（data/factors/evaluation/backtest/pipeline）
data/cache/              行情缓存
results/                 研究与回测输出
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

如需私有接口，配置 `.env`：

```bash
BINANCE_API_KEY=
BINANCE_SECRET_KEY=
```

说明：仅公共行情时可留空 API key（可能受限频影响）。

## Run

1. 刷新交易所数据并跑完整研究

```bash
python3 scripts/run_research.py --refresh-cache
```

2. 使用本地缓存复现（推荐日常迭代）

```bash
python3 scripts/run_research.py --use-cache-only
```

3. 控制规模（更快）

```bash
python3 scripts/run_research.py --use-cache-only --max-factors 40 --disable-combos --no-charts
```

4. 仅策略回测（不重新挖掘全量因子）

```bash
python3 scripts/run_research.py --use-cache-only --strategy-only --selected-top-n 12 --no-charts
```

## Key Outputs (results/)

- `factor_summary.csv`：主周期因子评估
- `factor_summary_multi_horizon.csv`：多周期聚合评估
- `factor_summary_by_horizon.csv`：分周期明细评估
- `top_strategy_factors.csv`：入选策略因子
- `factor_diversification.csv`：去冗余筛选诊断
- `factor_effectiveness_report.csv`：因子有效性对比报告
- `strategy_summary.csv`：策略指标对比（Sharpe/回撤/年化等）
- `strategy_comparison_report.csv`：策略对比汇总
- `auto_comparison_report.md`：自动生成的总览报告

## Important Configs

配置位于 `config/settings.py`，常用项：

- `TIMEFRAME`：K 线周期（默认 `4h`）
- `LOOKBACK_DAYS`：回溯天数
- `HORIZON_BARS`：主评估预测周期
- `FACTOR_MAX_COUNT`：最大因子数
- `FACTOR_ENABLE_COMBOS`：是否生成组合因子
- `FACTOR_EVAL_HORIZONS`：多周期评估列表（如 `6,12,24`）
- `FACTOR_CORR_THRESHOLD`：因子去冗余相关性阈值
- `TOP_K_FACTORS`：组合信号选取因子数

## Notes

- 回测结果依赖数据区间、交易成本参数和因子规模。
- 建议先用小规模参数做迭代验证，再放大 `FACTOR_MAX_COUNT`。
- 提交仓库前建议确认 `.env` 不被追踪。
