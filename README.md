# 玉米生育期识别 · 论文复现/审稿用发布包

本目录为独立于主仓库路径的**可打包上传**版本：含测试脚本、与训练一致的模型代码、**不含**大体积权重文件（请自行放入 `weights/` 或审稿通过后单独托管）。

## 目录结构

```
github/
├── README.md                 # 本说明
├── requirements-test.txt     # 测试与推理依赖
├── scripts/
│   └── test_improved_corn_growth_model.py
├── src/
│   ├── improved_corn_growth_model.py
│   └── model_compatibility.py
├── weights/                  # 将 improved_corn_model_best.pth 放在此处（不上传 Git 时可仅用 Zenodo）
├── testdata/sample/          # 按生育期建子文件夹的示例图像（可自行添加）
└── results/                  # 运行后生成（可忽略或加入 .gitignore）
```

## 权重文件（本仓库不附带）

1. 将训练得到的 **`improved_corn_model_best.pth`**（或你论文中的同名权重）复制到：

   `weights/improved_corn_model_best.pth`

2. 若期刊要求**开放存档**：建议将权重上传 **Zenodo / Figshare** 等获取 DOI，在论文 **Data Availability** 中写 DOI；GitHub 仅保留代码与本说明。

默认测试参数已指向上述相对路径，**不再使用**任何 `D:\...` 绝对路径。

## 环境

- Python 3.10+（与训练环境一致更佳）
- CUDA（可选，CPU 可跑但较慢）

```bash
cd github
pip install -r requirements-test.txt
```

## 测试数据格式

- `--data_dir` 下为若干**子文件夹**，文件夹名为生育期代码：`11`, `21`, `31`, `41`, `61`, `71`, `81`, `91`, `99`（与 `stage_mapping` 一致）。
- 每类文件夹内为图像：`jpg/jpeg/png/bmp/tiff`。
- 若模型启用时间特征，脚本会从**文件名**解析拍摄日期（与 `test_improved_corn_growth_model.py` 内逻辑一致）。

可将少量脱敏样本放在 `testdata/sample/` 下供审稿验证流程。

## 运行测试

在 **`github/`** 目录下执行：

```bash
python scripts/test_improved_corn_growth_model.py
```

显式指定路径示例：

```bash
python scripts/test_improved_corn_growth_model.py ^
  --model_path weights/improved_corn_model_best.pth ^
  --data_dir testdata/sample ^
  --output_dir results ^
  --device cuda
```

（Linux/macOS 将 `^` 换为 `\`。）

成功运行后，`results/` 下会生成报告与图表（以脚本实际输出为准）。

## 与主仓库的关系

- `src/` 中文件从 `D:\code\cornmx\src\` **同步拷贝**；若主仓库模型代码有更新，发版前请重新复制并提交。
- 主仓库完整训练代码未全部包含在此包内；期刊若只要求 **权重 + 测试说明**，当前内容通常可满足。

## 许可与数据

请根据期刊要求在正文中补充 **Acknowledgments / Funding / Data Availability**；若原始影像不可公开，在 Data Availability 中说明**仅提供示例子集与推理脚本**。
