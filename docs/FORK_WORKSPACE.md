# 本 fork 的工作区

本仓库的 `origin` 是 `ggkiller-air/OpenViTac`，`upstream` 是
`FVL-Repo/OpenViTac`。日常改动直接提交并推送至 `main`。

| 仓库 | 职责 |
| --- | --- |
| [AdaViP](https://github.com/ggkiller-air/AdaViP) | 我们的方法，含 `adavip/openvitac` 适配和阶段训练 |
| [UniVTAC](https://github.com/ggkiller-air/UniVTAC) | 独立 UniVTAC 实验 |
| [OpenViTac](https://github.com/ggkiller-air/OpenViTac) | 官方环境、资产和本 fork 的 ACT 基线 |

## ACT Pull Drawer

现有独立 ACT 项目已作为 subtree 导入
[`experiments/ACTPullDrawer`](../experiments/ACTPullDrawer/README.md)。导入版本是
原本地项目的 `d6967a5`，其源代码、配置、49 个官方文件的校验清单和历史工作日志
全部保留；使用这里作为后续开发入口。

原项目完整仓库归档在
`/public/home/wangzihao/project-archive/2026-11-02/legacy/OpenViTac-ACT-pull-drawer`。
训练数据和产物仍分别使用 `/data/wangzihao/datasets/openvitac/` 与
`/data/wangzihao/outputs/openvitac_act/`，已有冻结源码快照保留原位。

```bash
cd /public/home/wangzihao/projects/OpenViTac/experiments/ACTPullDrawer
/public/home/wangzihao/.venvs/openvitac-act/bin/python -m pytest -q tests
```

测试需要本机已经转换好的 100 个 Pull Drawer 演示。提交训练的入口仍是
`submit_training.py`，具体的数据约定、环境和部署流程见项目 README。

## 本机大型资源

2026-11-02 整理时，大型 Git 资源包连续发生连接中断。本机采用 partial clone
和 sparse checkout：全部源代码、notebook 和配置已经检出，暂缓 103 个至少
1 MiB 的模型、贴图、网格和传感器资源。暂缓文件清单位于归档根目录的
`openvitac-deferred-resources.json`；这些资源仍由 Git 跟踪。

ACT 数据处理与训练测试可直接使用当前 checkout。在本机运行完整仿真前补齐
资源，在 OpenViTac 仓库根目录执行：

```bash
git sparse-checkout disable
```

Git 会从 `origin` 按需取回缺失对象。其他机器普通 clone 不受本机 sparse
配置影响。单独 clone 的训练依赖和各项目的数据、checkpoint 仍独立管理。
