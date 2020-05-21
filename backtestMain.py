from indexing import carry_backtest
import yaml



m = carry_backtest("configs.yml")
# m.mainBacktester()
m.mainPerformanceAnalysis()


