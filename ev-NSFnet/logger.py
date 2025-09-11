"""Minimal logger for restored baseline.
集中格式化輸出 (rank0)。
"""
import sys, time, os
from datetime import datetime
import torch

class SimpleLogger:
    def __init__(self, name: str = "PINN", rank: int = 0, enable_file: bool = True):
        self.name = name
        self.rank = rank
        self.start_time = time.time()
        self.file = None
        if enable_file and rank == 0:
            os.makedirs('logs', exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.file = open(f'logs/{name}_{ts}.log','w',encoding='utf-8')

    def _emit(self, level: str, msg: str):
        if self.rank != 0:
            return
        line = f"{level} | {msg}"
        print(line)
        if self.file:
            self.file.write(line+"\n"); self.file.flush()

    def info(self, msg: str):
        self._emit('INFO', msg)

    def warning(self, msg: str):
        self._emit('WARN', msg)

    def error(self, msg: str):
        self._emit('ERROR', msg)

    def header(self, title: str):
        self.info('='*60)
        self.info(title)
        self.info('='*60)

    def stage(self, name: str, alpha: float, epochs: int, lr: float):
        self.info(f"🎯 {name}: alpha={alpha}, epochs={epochs:,}, lr={lr:.2e}")

    def close(self):
        if self.file:
            self.file.close()

_logger = None

def get_logger(name: str = 'PINN', rank: int = 0):
    global _logger
    if _logger is None:
        _logger = SimpleLogger(name=name, rank=rank)
    return _logger
