# sciai

[English](README.md) | 简体中文

`sciai` 是一个早期的、框架中立的统一 AI for Science（科学智能）开发者体验基础框架。它的目标是让科学数据语义、模型、训练、任务管线与模拟器耦合在各个学科之间可组合复用，而不强迫所有领域绑定到同一个机器学习后端。

> **状态：`0.1.0a1` 基础版本。** 稳定契约与 NumPy 参考实现路径已完成。生产级模型权重、基准数据集以及原生商业求解器集成尚未打包发布。能力状态均被如实标注，而不是把路线图中的条目当作已完成功能来宣传。

## 为什么选择这种架构

- **统一的科学本体：** 物理量纲、单位、坐标系与参考系在所有领域之间共享。
- **八种通用数据结构：** 标量场、矢量场、张量场、图、粒子、网格、时间序列与光谱。
- **统一的模型契约：** `ScienceModel` 标准化了 `forward`、`predict`、`train`、`save` 与 `load`，而将张量执行交给后端插件。
- **领域插件而非单体框架：** 分子科学与流体力学是内置的参考插件；第三方包通过 `sciai.domains` 入口点接入。
- **显式耦合：** 模拟器适配器通过有向耦合图交换具名科学变量。
- **如实的能力发现：** 七大领域族与 30 余个子领域被如实归类为 `available`（可用）、`plugin-ready`（插件就绪）或 `roadmap`（规划中）。

## 安装

需要 Python 3.10 或更高版本。

```bash
python -m pip install -e ".[dev]"
pytest
```

仓库名为 `sciai`，导入包名为 `sciai`。PyPI 发行版暂时命名为 `sciai-core`，因为 `sciai` 这个发行名已被 MindSpore SciAI 项目占用。

## 统一 API

```python
from sciai import AutoDataset, AutoModel, ScienceTrainer, TrainingConfig, pipeline
from sciai.training import Adam

model = AutoModel.from_pretrained("sciai/molecule-baseline")
dataset = AutoDataset.load("sciai/toy-molecules")

trainer = ScienceTrainer(
    config=TrainingConfig(epochs=100, batch_size=len(dataset)),
    optimizer=Adam(learning_rate=0.01),
)
trainer.fit(model, dataset)

predictor = pipeline("property-prediction", model=model)
print(predictor("CCO"))
```

内置的分子模型是一个透明、可训练的描述符基线，用于验证模型、训练器、工件（artifact）、Auto 与管线契约。它不会被宣传成 MACE、SchNet 或其他预训练 SOTA 模型。

## 科学数据

```python
import numpy as np
from sciai import Quantity, ScalarField
from sciai.operators import gradient

length = Quantity(2.5, "km").to("m")
x = np.linspace(0.0, 1.0, 101)
field = ScalarField(values=x**2, coordinates=(x,), unit="K")
derivative = gradient(field)
```

## 领域 API

```python
from sciai.molecule import Molecule
from sciai.fluid import FlowField, IncompressibilityModel

molecule = Molecule.from_smiles("CCO")
flow = FlowField.from_arrays(velocity, pressure=pressure, coordinates=(x, y))
residual = IncompressibilityModel()(flow)
```

无依赖的 SMILES 解析器有意只支持一个保守的子集。完整的立体化学、带电原子与规范化化学语义应由基于 RDKit 的插件提供。

## 多物理场耦合

```python
from sciai.multiphysics import CoupledSimulator, InMemoryAdapter

fluid = InMemoryAdapter("fluid", "fluid", {"pressure": 101325.0})
solid = InMemoryAdapter("solid", "solid", {"displacement": 0.0})

simulation = CoupledSimulator([fluid, solid])
simulation.couple("fluid", "solid", variables=["pressure"], interface="wall")
result = simulation.run(steps=10)
```

OpenFOAM、ABAQUS、COMSOL、VASP、GROMACS 等外部程序都接入同一套 `SimulationAdapter` 契约。它们是集成目标，而不是核心包的依赖。

## 社区领域包

```bash
sciai scaffold-domain plasma-physics --destination ./plugins
```

生成的包会声明一个 `sciai.domains` 入口点，并可独立拥有领域数据、算子、模型、数据集、管线与模拟器适配器。

## 命令

```bash
sciai catalog
sciai catalog --json
sciai doctor
sciai scaffold-domain <name>
```

## 架构

```text
应用层
  AutoModel / AutoDataset / AutoTrainer
  任务管线 / 工作流 DAG / 多物理场耦合
                         |
领域插件层              | Python 入口点 + 类型化注册表
  molecule / fluid / 社区包
                         |
通用基础层
  本体 / 8 种数据类型 / 算子 / ScienceModel
  ScienceTrainer / 指标 / 工件格式 / 仿真协议
```

契约与交付阶段详见 [docs/architecture.md](docs/architecture.md)、[docs/capability-matrix.md](docs/capability-matrix.md) 与 [docs/roadmap.md](docs/roadmap.md)。

## 项目原则

1. 科学语义在边界处被校验。
2. 核心接口保持独立于 PyTorch、JAX、MindSpore 或 TensorFlow。
3. 工件默认使用不可执行的 JSON + NPZ 格式。
4. 可选集成在后端缺失时给出清晰的失败提示。
5. 可复现性元数据、来源追溯、单位与领域身份是一等公民。

## 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。
