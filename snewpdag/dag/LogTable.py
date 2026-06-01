"""
LogTable - table of logs of integers and factorials
"""
import numpy as np

class LogTable:

  maxn = 1
  logs = np.array([0])
  logfacts = np.array([0])

  def __init__(self, maxn=1000):
    self.ensure(maxn)

  def ensure(self, n):
    nm = np.max(n) # since n could be an array of indices
    if nm >= LogTable.maxn:
      n0 = LogTable.maxn
      n1 = nm + 1 # 因為python永遠都唔計最尾嗰個index 所以我地要加一
      nvals = np.log(np.arange(n0, n1)) # [ln(n0), ln(n0+1), ... , ln(nm)] # need learn this kind of clean writing style!
      LogTable.logs = np.append(LogTable.logs, nvals)
      LogTable.logfacts = np.append(LogTable.logfacts, nvals) # 注意看：佢係專登擺啲普通野落去個factorial array 度
      np.cumsum(LogTable.logfacts[n0-1:], out=LogTable.logfacts[n0-1:]) # 喺呢一步就係滾雪球 
      LogTable.maxn = n1

  def log(self, n):
    self.ensure(n)
    return LogTable.logs[n]

  def logfact(self, n):
    self.ensure(n)
    return LogTable.logfacts[n]

