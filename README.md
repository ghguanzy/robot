# 灵巧操作学习实验室

目标：跳过 Python、C++、Linux 和 ROS 2 入门，用 24 周（每周 8～10 小时）完成一条可验收的灵巧操作学习路线，并最终做出一个“视觉/状态输入 → 操作策略 → 仿真评测 → 真机准备”的作品集项目。

## 现在已经落地的内容

第一阶段不是阅读清单，而是一个可以直接运行的机器人学实验：

- 平面多关节机械臂正运动学；
- 末端雅可比矩阵；
- 阻尼最小二乘逆运动学（DLS IK）；
- 笛卡尔直线轨迹生成与逐点跟踪；
- 数值雅可比、可达目标、不可达目标和轨迹误差自动测试；
- 明确的阶段验收门槛。

完整安排见 [ROADMAP.md](ROADMAP.md)，学习记录模板见 [LEARNING_LOG.md](LEARNING_LOG.md)。

## 运行第一阶段

在本目录执行：

```powershell
python scripts/run_phase1.py
python scripts/benchmark_phase1.py
python -m unittest discover -s tests -v
```

如果 Windows 的 `python` 是商店占位程序，可显式使用可用的 Python 解释器。项目只依赖 NumPy。

## 第一阶段的学习方式

1. 阅读 `src/dexlab/planar_arm.py`，手推二连杆正运动学和雅可比。
2. 运行演示，观察目标误差、迭代次数和奇异位形附近的行为。
3. 完成 `exercises/phase1_tasks.md` 中的任务。
4. 修改代码后运行全部测试，记录失败原因和实验数据。
5. 达到验收条件后才进入三维机械臂、MoveIt 2 和接触控制。

## 目录

```text
dexterous-manipulation-lab/
├── ROADMAP.md
├── LEARNING_LOG.md
├── exercises/
├── scripts/
├── src/dexlab/
└── tests/
```

## 学习原则

- 每一阶段必须产出代码、实验记录和可复现指标。
- 先二指夹爪、后多指灵巧手；先可靠闭环、后端到端模型。
- 先在仿真中证明算法，再考虑真机。
- 不用“看完课程”作为完成标准，只用可测量的任务成功率和误差作为标准。


