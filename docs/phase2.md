# CineMagic Phase 2 — 验收报告

## 环境

| 项目 | 版本 |
|------|------|
| conda env | `cinemagic` (Python 3.12) |
| PyTorch | 2.14.0.dev20260625+cu130 (nightly) |
| CUDA | 13.0 |
| PyG | 2.8.0 |
| GPU | NVIDIA RTX 5070 Laptop (sm_120 ✅) |

## 新增文件

```
src/models/
  lightgcn.py          ★ LightGCN 召回模型 (PyG, BPR Loss, GPU)
  deepfm.py             ★ DeepFM 精排模型 (FM+DNN, FeatureProcessor, CTRDataset)

scripts/
  train_lightgcn.py     ★ LightGCN 训练脚本
  train_deepfm.py       ★ DeepFM 训练脚本
  recommend.py           ★ 已更新: --model lightgcn / lightgcn+deepfm

config/config.yaml       ★ 已更新: lightgcn + deepfm 超参数
```

## 模型效果对比

| 模型 | RMSE↓ | Precision@10 | Recall@10 | NDCG@10 | Coverage |
|------|-------|-------------|-----------|---------|----------|
| **LightGCN** | 1.33 | **0.0493** | **0.2470** | **0.1464** | 0.96 |
| SVD (Phase 1 最佳) | **0.80** | 0.0227 | 0.0895 | 0.0579 | 0.32 |
| DeepFM | 3.46 | 0.0148 | 0.0655 | 0.0383 | 0.22 |

### 解读

1. **LightGCN 在排序指标上碾压 SVD**: NDCG@10 = 0.146 vs 0.058，提升 **2.5 倍**。Recall@10 = 0.25 vs 0.09，提升 **2.8 倍**。图卷积的高阶信号传播真正见效了。

2. **DeepFM 效果不如预期**: 主要因为 Top200 数据量太小（仅 1280 用户 × 192 电影），CTR 模型需要更多样本来学习有效的特征交叉。另外评分右偏严重（pos/neg = 6930/1052 = 6.6:1），正负样本极度不均衡。另外 DeepFM 在 Top200 上作为独立推荐器使用时效果差，因为它的设计目标是精排（在已有候选集上排序），而不是全量召回。

3. **LightGCN 召回 + DeepFM 精排**: 两阶段架构的预期效果是 LightGCN 取 top 50 候选 → DeepFM 精排 top 10。但由于 DeepFM 在 Top200 小数据上表现差，建议当前阶段直接用 LightGCN 作为独立推荐器，DeepFM 留待 Phase 3 全量数据时再启用两阶段模式。

## 使用方式

```bash
# 训练
conda activate cinemagic
python scripts/train_lightgcn.py --device cuda --epochs 100
python scripts/train_deepfm.py --device cuda --epochs 50

# 推荐（LightGCN 作为独立推荐器）
python scripts/recommend.py --user <user_id> --model lightgcn --topk 10
python scripts/recommend.py --similar-to 1292052 --model lightgcn
```

## 已知问题

1. **DeepFM 在 Top200 上效果差**: 数据量不足 + 正负比失衡。解决方案：Phase 3 扩展到全量 14 万电影 × 60 万用户，或调整正负采样策略。
2. **LightGCN Embedding 映射到 1-5 评分有偏**: RMSE 差是因为内积×sigmoid 映射不等于真实评分尺度。但这不影响排序质量（NDCG 很好）。
3. **推荐脚本 DeepFM 加载路径**: recommend.py 中的 `--model lightgcn+deepfm` 模式还需要完善（目前 deepfm.load 需要传入 movies_df）。
