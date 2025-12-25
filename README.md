# 2025KDD-SGD-DYG-

## 批处理 2、3、4、6、7（dblp、digg、last_fm、wiki_eo、wiki_gl）
已在 `scripts/process_datasets.py` 添加批处理脚本，默认一次性处理 dblp、digg、last_fm、wiki_eo、wiki_gl 五个数据集。

### 使用方式
```bash
# 处理默认 5 个数据集
python scripts/process_datasets.py

# 指定数据路径/输出路径
python scripts/process_datasets.py --data-dir data --output-dir data/processed

# 仅处理部分数据集
python scripts/process_datasets.py --datasets dblp digg

# 如果不想删除空行或重复行
python scripts/process_datasets.py --keep-empty-rows --keep-duplicates
```

### 输入/输出约定
- 输入查找顺序（按名称 `X`）：`data/X.csv` 或 `data/X/X.csv`
- 输出路径：`data/processed/X/X_processed.csv`
