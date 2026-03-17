# 4Alpha Factor Mining（Binance USD-M）

这是一个面向加密永续合约的横截面因子挖掘与评估框架，目标是把“数据拉取 -> 因子生成 -> 统计检验 -> 组合信号 -> 策略回测 -> 可视化”做成可重复、可扩展、可对比的研究流水线。

项目当前聚焦 Binance USDT 本位线性永续（`binanceusdm`），并内置过去 3 年窗口的默认研究配置。

## 1. 研究目标

- 在多币种横截面上构建因子库，评估因子对未来收益的解释能力。
- 通过统一指标（IC、IC-IR、命中率、分层价差）筛选稳定因子。
- 基于有效因子生成组合信号，作为后续策略/回测输入。

## 2. 流程总览

```text
交易所数据(OHLCV+Funding)
        |
        v
数据清洗与对齐（UTC、bar 对齐、缺失处理）
        |
        v
因子库生成（核心因子 + 参数扩展 + 组合因子）
        |
        v
横截面标准化（每个时间截面 rank(pct=True)）
        |
        v
标签构建（未来 horizon 收益）
        |
        v
统计评估（IC/IC-IR/HitRate/T-Stat/分层 Spread）
        |
        v
输出报告 + Top-K 因子加权组合信号
        |
        v
多策略回测对比（收益/回撤/Sharpe/换手/成本）
        |
        v
回测图表输出（净值/回撤/滚动Sharpe/指标柱状图）
```

## 3. 环境与安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

编辑已有 `.env`（若本地不存在再创建，按需填写）：

```bash
BINANCE_API_KEY=
BINANCE_SECRET_KEY=
```

说明：

- 仅公共行情研究时，留空 API 也可运行（受交易所限频影响）。
- 默认使用 `config/settings.py`，`--config` 参数已兼容保留但不再生效。

## 4. 数据层设计

数据加载逻辑位于 `src/alpha_mining/data_loader.py`。

- 标的池：
  - 若 `SYMBOLS` 非空，使用显式列表。
  - 否则按 `quoteVolume` 选取前 `TOP_N_SYMBOLS` 的 USDT 永续。
- 行情数据：分批拉取 OHLCV（每批最多 1500 条）。
- 资金费率：分批拉取 funding history（每批最多 1000 条）。
- 时间对齐：统一为 UTC，并按 timeframe 向下取整到 bar。
- 合并与补全：funding 按币种前向填充，空值补 0。
- 缓存：保存到 `data/cache/market_data_4h.csv`，支持 cache-only 复现实验。

## 5. 因子工程（当前实现）

实现文件：`src/alpha_mining/factors.py`。

### 5.1 核心因子（13 个）

> 说明：以下“方向”均指该因子原始值变大时的经济含义。进入研究流程后，所有因子会做横截面 `rank(pct=True)`，因此 `factor_value` 越高代表该币种在该因子上相对越强。

1. `mom_24`（24 bar 动量）
   `close.pct_change(24)`  
   含义：中短期趋势延续，涨幅更高的币倾向继续强势（趋势类）。

2. `mom_72`（72 bar 动量）
   `close.pct_change(72)`  
   含义：更慢节奏的趋势因子，用来捕捉中期强弱分化。

3. `short_reversal_6`（6 bar 短期反转）
   `-close.pct_change(6)`  
   含义：短期涨太多的币更可能回落，短期跌多的币更可能反弹（均值回归）。

4. `volatility_adjusted_trend`（波动率调整趋势）
   `(close - rolling_mean_48) / rolling_std_48`  
   含义：价格偏离中枢的强度按波动率标准化，避免“高波动币天然更极端”的偏差。

5. `intrabar_reversal`（K线内反转）
   `-mean_12((close-open)/(high-low))`  
   含义：若经常收在K线高位，可能存在短期过热；负号使其偏向反转逻辑。

6. `range_expansion`（振幅扩张）
   `mean_12((high-low)/close)`  
   含义：衡量近期波动区间是否放大，可用于识别风险偏好切换或拥挤交易阶段。

7. `volume_shock`（成交量冲击）
   `volume / mean_48(volume) - 1`  
   含义：当前成交量相对历史基线的放大量，反映资金活跃度突变。

8. `illiquidity_reversal`（非流动性反转）
   `-mean_24(abs(return)/(close*volume))`  
   含义：Amihud 非流动性越高，冲击成本越大，后续更容易均值回归（负号反转化）。

9. `funding_crowding_unwind`（资金费率拥挤反转）
   `-mean_9(funding_rate)`  
   含义：长期高正 funding 常对应多头拥挤，负号表达“拥挤后出清”的反转假设。

10. `funding_zscore_revert`（资金费率Z分数回归）
    `-(funding - mean_72(funding)) / std_72(funding)`  
    含义：衡量 funding 偏离常态程度，偏离越极端越倾向回归。

11. `vol_regime_reversion`（波动率状态回归）
    `-(std_24(return)/std_168(return) - 1)`  
    含义：短期波动率相对长期波动率异常升高时，通常伴随风险溢价变化与后续回归。

12. `downside_pressure_revert`（下行波动压力回归）
    `-(std_48(return.clip(upper=0))/std_48(return) - 1)`  
    含义：下行波动占比越高，说明恐慌/抛压更集中，负号表达“超跌后修复”。

13. `trend_volume_confirmation`（趋势-成交量确认）
    `pct_change(24) * sign(volume/mean_48(volume)-1)`  
    含义：趋势若被成交量放大同向确认，信号更可靠；若价量背离，信号会被削弱或反转。

这些核心因子共同覆盖：趋势、反转、波动率结构、流动性冲击、资金费率拥挤度、价量一致性。

### 5.2 参数扩展因子

在核心模板上做窗口遍历，自动生成更多候选因子（例如不同 `mom_window`、`vol_window`、`funding_window`）。

- 目的：增加搜索空间，不改变研究框架。
- 控制：通过 `FACTOR_MAX_COUNT` 约束最终因子总量，避免无限膨胀。

### 5.3 组合因子（可开关）

对入选基础因子做横截面 zscore 后，生成以下组合：

- `add`: `z(f1) + z(f2)`
- `sub`: `z(f1) - z(f2)`
- `mul`: `z(f1) * z(f2)`

相关参数：

- `FACTOR_ENABLE_COMBOS`
- `FACTOR_COMBO_SOURCE_TOP_N`
- `FACTOR_COMBO_MAX_COUNT`

### 5.4 横截面标准化

每个时点、跨币种做 `rank(pct=True)`，输出 `[0,1]` 范围的因子值，降低极值影响并增强可比性。

## 6. 标签与评估方法

实现文件：`src/alpha_mining/evaluation.py`。

### 6.1 标签构建

- 标签定义：未来 `HORIZON_BARS` 收益率。
- 计算：`future_return = pct_change(horizon).shift(-horizon)`。
- 保障：标签与因子严格时间错位，避免前视偏差。

### 6.2 评估指标

对每个因子计算：

- `mean_ic`：截面 Spearman IC 均值。
- `std_ic`：IC 波动。
- `ic_ir`：`mean_ic / std_ic`。
- `hit_rate`：IC>0 的时间占比。
- `t_stat`：IC 均值显著性。
- `spread_mean/std/ir`：按因子分 5 桶后 `Q5-Q1` 的统计。

注：单时点可交易资产数不足 `MIN_ASSETS_PER_TIMESTAMP` 时，该时点不参与统计。

## 7. 组合信号生成（策略输入）

实现：`build_composite_signal`。

- 从 `factor_summary` 里选 `mean_ic > 0` 的前 `TOP_K_FACTORS`。
- 以 `mean_ic` 归一化作为权重。
- 同时点同币种对加权因子求和，得到 `composite_signal`。

这一步是“研究信号聚合”，会作为后续交易策略输入之一。

## 8. 交易策略与回测（精简）

实现文件：`src/alpha_mining/backtest.py`、`src/alpha_mining/visualization.py`。

三类策略（都基于已挖掘因子）：

- `single_factor_long_short[...]`：从 `factor_summary.csv` 选最优单因子（结合 `spread_ir + ic_ir`），按方向做横截面多空。
- `composite_long_short`：对优选因子加权合成信号，再做横截面多空。
- `risk_controlled_multi_factor`：多因子加权后叠加风险约束（逆波动缩放、单资产权重上限）。

统一执行逻辑：

- 交易收益按下一根 bar 结算（`t` 信号用于 `t->t+1`）。
- 默认按 `HORIZON_BARS` 调仓，且支持信号平滑与分步调仓（降换手）。
- 手续费按换手计入：`cost = taker_fee * turnover`。

核心输出：

- `strategy_summary.csv`：`annual_return / sharpe / max_drawdown / avg_turnover / avg_cost` 等。
- `strategy_returns.csv`、`strategy_nav.csv`：逐期收益与净值回撤。
- 图表：`backtest_equity_curve.png`、`backtest_drawdown.png`、`backtest_rolling_sharpe.png`、`backtest_strategy_metrics.png`。

## 9. 默认配置（重点）

`config/settings.py` 当前关键默认值：

- `TIMEFRAME = "4h"`
- `LOOKBACK_DAYS = 1095`（过去 3 年）
- `HORIZON_BARS = 12`
- `TOP_N_SYMBOLS = 40`
- `MIN_ASSETS_PER_TIMESTAMP = 8`
- `TOP_K_FACTORS = 5`
- `FACTOR_MAX_COUNT = 80`
- `FACTOR_ENABLE_COMBOS = True`
- `FACTOR_COMBO_SOURCE_TOP_N = 16`
- `FACTOR_COMBO_MAX_COUNT = 40`
- `INITIAL_CAPITAL = 100000`
- `TAKER_FEE = 0.001`
- `BACKTEST_TOP_QUANTILE = 0.2`
- `BACKTEST_MAX_ABS_WEIGHT = 0.2`
- `BACKTEST_VOL_LOOKBACK = 24`
- `BACKTEST_MULTI_FACTOR_TOP_K = 8`
- `BACKTEST_SIGNAL_SMOOTHING_SPAN = 6`
- `BACKTEST_EXECUTION_ALPHA = 0.35`
- `BACKTEST_REBALANCE_INTERVAL = 0`（0 表示自动使用 `HORIZON_BARS`）
- `BACKTEST_TARGET_GROSS_EXPOSURE = 1.0`
- `BACKTEST_SAVE_POSITION_EVENTS = True`
- `BACKTEST_POSITION_EVENT_THRESHOLD = 0.0001`

说明：默认参数兼顾覆盖面与运行速度。若机器性能有限，建议先降 `FACTOR_MAX_COUNT` 或关闭组合因子。

## 10. 运行方式

### 10.1 刷新数据并研究（含回测+图表）

```bash
python3 scripts/run_research.py --refresh-cache
```

### 10.2 仅用本地缓存复现（推荐日常迭代）

```bash
python3 scripts/run_research.py --use-cache-only
```

指定回测时间段（命令行覆盖配置）：

```bash
python3 scripts/run_research.py --use-cache-only --start-date 2024-01-01 --end-date 2025-12-31
```

### 10.3 临时扩大因子数量

```bash
python3 scripts/run_research.py --use-cache-only --max-factors 200
```

### 10.4 禁用组合因子

```bash
python3 scripts/run_research.py --use-cache-only --disable-combos
```

### 10.5 快速小规模验证（更快）

```bash
python3 scripts/run_research.py --use-cache-only --max-factors 20 --disable-combos
```

### 10.6 仅用已确定因子直接跑策略（避免全量 factor_values）

当你已经在 `results/factor_summary.csv` 中确定了因子，可以跳过全量因子挖掘：

```bash
# 自动从 factor_summary.csv 选前 8 个因子，直接跑策略
python3 scripts/run_research.py --use-cache-only --strategy-only --selected-top-n 8 

# 或者手动指定因子名（逗号分隔）
python3 scripts/run_research.py --use-cache-only --strategy-only \
  --selected-factors "volatility_adjusted_trend_96,illiquidity_reversal,mom_72" 
```

可选：如果需要保存本次选中因子的明细，再加 `--save-selected-factors`。

## 11. 输出文件说明

结果默认保存到 `results/`：

- `factor_values.csv`：每时点-币种-因子 的标准化因子值。
- `factor_summary.csv`：每个因子的核心统计摘要。
- `factor_ic_timeseries.csv`：每个因子的 IC 时间序列。
- `factor_spread_timeseries.csv`：每个因子分层价差（Q5-Q1）时间序列。
- `composite_signal.csv`：Top-K 因子加权后的组合信号。
- `top_strategy_factors.csv`：用于策略的 Top N 因子清单（含 score/方向）。
- `strategy_returns.csv`：逐期收益、换手、交易成本。
- `strategy_nav.csv`：净值、权益、回撤时间序列。
- `strategy_summary.csv`：多策略回测指标对比。
- `strategy_position_events.csv`：逐笔仓位事件（建仓/加仓/减仓/平仓/反手）及触发信号信息（`signal_value/signal_rank/trigger_reason`）。
- `backtest_equity_curve.png`：净值曲线图。
- `backtest_equity_vs_btc.png`：策略净值与同期 BTC 基准对比图。
- `backtest_drawdown.png`：回撤曲线图。
- `backtest_rolling_sharpe.png`：滚动 Sharpe 曲线图。
- `backtest_strategy_metrics.png`：策略指标柱状对比图。
- `selected_factor_values.csv`：仅在 `--strategy-only --save-selected-factors` 时输出。

## 12. 推荐研究实践

1. 先用 `--use-cache-only` 小规模试验（例如 40~80 因子）验证流程稳定。
2. 观察 `factor_summary.csv` 的 `mean_ic / ic_ir / hit_rate / t_stat` 是否一致改善。
3. 观察 `strategy_summary.csv` 的 Sharpe、最大回撤、换手和成本，避免只看收益率。
4. 对优选因子与策略做时间分段稳定性检查（牛熊/震荡分段）。
5. 再扩大到上百因子，并控制组合因子上限，防止过拟合和计算爆炸。
6. 最终建议做样本外回测与滚动重估，再考虑实盘迁移。

## 13. 已知边界与风险

- 本项目是“因子研究 + 回测”框架，不直接保证实盘收益。
- 因子挖掘存在数据窥探与过拟合风险，必须做样本外验证。
- 仅使用交易所行情与 funding，未纳入链上、订单簿、宏观、情绪等外部信息。
- 回测对成交冲击、滑点、资金容量的刻画仍较简化，实盘需更保守。
- 多因子组合的稳定性高度依赖市场状态，建议滚动重估。
