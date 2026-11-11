# towing_dwa

由 [ref_dwa](../ref_dwa/README.md) 的最小经典 DWA 初始化的独立副本。
修改本目录 `optimizer.py`、`controller.py`、`config.py` 即可开发，不影响 ref_dwa。
`rollout.py` 私有实现候选预测/制动评估，不依赖通用适配器。

支持差速与阿克曼，通过 `model=` 注入；车体形状通过 `footprint=` 独立传入。
未指定车体时使用通用 `Footprint()`，Web 切换算法保留当前车型和几何参数。
这里尚未增加挂车或牵引约束。
接口与测试方式见[开发说明](../README.md)。
