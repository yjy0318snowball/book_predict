import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import scipy.sparse as sp

# — 1. 配置参数 (与之前hybrid版本基本一致) —
class Config:
    data_path = './'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 🔥 针对小数据集优化
    epochs = 50
    batch_size = 64 # 减小批次以提高更新频率
    learning_rate = 0.003 # 提高学习率
    weight_decay = 1e-6 # 降低正则化
    lgcn_embedding_dim = 64
    lgcn_n_layers = 2 # 减少层数防止过拟合

    sasrec_embedding_dim = 64
    sasrec_maxlen = 30 # 减小序列长度
    sasrec_transformer_layers = 2 # 简化模型
    sasrec_transformer_heads = 4
    sasrec_dropout_rate = 0.1
    recall_k = 150 # 大幅增加召回数量

    # ... 其他参数 ...
    submission_top_k = 1  # 为每个用户推荐5本书

# — 评估函数 (修正版) —

def evaluate_model(recommendations_topk, ground_truth, model_name=""):
    """
    根据比赛要求计算 Precision, Recall 和 F1，并输出结果。
    Args:
        recommendations_topk (dict): {user_id: book_id} 或 {user_id: [book_ids]}
        ground_truth (pd.DataFrame): 包含 'user_id' 和 'book_id' 的真实借阅数据
        model_name (str): 模型名称

    Returns:
        float: F1 Score
    """

    # 规范化 ground_truth
    gt = ground_truth.copy()
    gt['user_id'] = gt['user_id'].astype(str)
    gt['book_id'] = gt['book_id'].astype(str)

    # 构建真实借阅字典：{user_id: set of book_ids}
    truth_dict = gt.groupby('user_id')['book_id'].apply(set).to_dict()

    TP = 0  # True Positives
    FP = 0  # False Positives
    FN = 0  # False Negatives

    # 统计所有验证集用户
    all_validation_users = set(truth_dict.keys())

    # 规范化推荐结果的 user_id
    normalized_recs = {}
    for uid, books in recommendations_topk.items():
        uid_str = str(uid)
        # 处理单个 book 或列表
        if isinstance(books, (list, tuple)):
            book_list = [str(b) for b in books]
        else:
            book_list = [str(books)]
        normalized_recs[uid_str] = book_list

    # 对每个验证集用户进行评估
    for user_id_str in all_validation_users:
        true_books = truth_dict[user_id_str]  # 该用户真实借阅的图书集合
        
        # 获取推荐列表
        if user_id_str in normalized_recs:
            recommended_books = set(normalized_recs[user_id_str])
        else:
            recommended_books = set()
        
        # 计算 TP, FP, FN
        tp = len(recommended_books & true_books)  # 推荐且真实借阅的
        fp = len(recommended_books - true_books)  # 推荐但未借阅的
        fn = len(true_books - recommended_books)  # 未推荐但真实借阅的
        
        TP += tp
        FP += fp
        FN += fn

    # 计算指标
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # 输出结果
    print(f"\n--- {model_name} 评估结果 ---")
    print(f"验证集用户数: {len(all_validation_users)}")
    print(f"有推荐的用户数: {len(normalized_recs)}")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    print(f"精确率 (P): {precision:.6f}")
    print(f"召回率 (R): {recall:.6f}")
    print(f"F1 值: {f1_score:.6f}")

    # 判断是否达到阈值
    threshold = 0.0055
    status = "✓ 有效成绩" if f1_score >= threshold else "✗ 无效成绩"
    print(f"阈值判定 (>= {threshold}): {status}")
    print("--------------------------------\n")

    return f1_score
    

# =====================================================================================
# — LightGCN (关键改动: BPRDataset_LGCN 和 训练循环) —
# =====================================================================================
class BPRDataset_LGCN(Dataset):
    def __init__(self, inter_df, n_items):
        self.users = torch.LongTensor(inter_df['user_idx'].values)
        self.pos_items = torch.LongTensor(inter_df['book_idx'].values)
        # — 关键改动: 加入兴趣强度 —
        self.interest_scores = torch.FloatTensor(inter_df['兴趣强度'].values)
        self.n_items = n_items
    def __len__(self): return len(self.users)
    def __getitem__(self, idx):
        user, pos_item = self.users[idx], self.pos_items[idx]
        interest_score = self.interest_scores[idx]
        while True:
            neg_item = np.random.randint(0, self.n_items)
            if neg_item != pos_item: break
        # --- 关键改动: 返回兴趣强度 ---
        return user, pos_item, torch.LongTensor([neg_item]).squeeze(), interest_score
        
class LightGCN(nn.Module):
    # (模型定义无需改动)
    def __init__(self, n_users, n_items, config):
        super(LightGCN, self).__init__()
        self.n_users, self.n_items = n_users, n_items
        self.graph = None
        self.user_embedding = nn.Embedding(n_users, config.lgcn_embedding_dim)
        self.item_embedding = nn.Embedding(n_items, config.lgcn_embedding_dim)
        self.n_layers = config.lgcn_n_layers
        nn.init.normal_(self.user_embedding.weight, std=0.005)
        nn.init.normal_(self.item_embedding.weight, std=0.005)
        
    def get_embeddings(self):
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        return torch.split(torch.mean(embs, dim=1), [self.n_users, self.n_items])
        
    def predict(self, user_indices, user_history, top_k):
        self.eval()
        with torch.no_grad():
            # 确保在正确的设备上运行
            all_users_emb, all_items_emb = self.get_embeddings()
            
            recommendations = {}
            for user_idx in user_indices:
                user_emb = all_users_emb[user_idx]
                scores = torch.matmul(user_emb, all_items_emb.t())
                
                # 过滤历史记录
                if user_idx in user_history:
                    # 使用 torch.tensor 索引，并确保在正确的设备上
                    history_items = torch.LongTensor(user_history[user_idx]).to(scores.device)
                    scores[history_items] = -np.inf
                    
                _, top_indices = torch.topk(scores, k=top_k)
                recommendations[user_idx] = top_indices.cpu().numpy().tolist()
            return recommendations
# =====================================================================================
# — SASRec (无需任何改动) —
# =====================================================================================
class SASRecDataset(Dataset):
    # … (代码与之前完全相同) …
    def __init__(self, user_sequences, maxlen, n_items):
        self.user_sequences, self.maxlen, self.n_items = user_sequences, maxlen, n_items
        self.pad_token = n_items; self.instances = []
        for user_id, seq in self.user_sequences.items():
            for i in range(1, len(seq)):
                target = seq[i]; start = max(0, i - self.maxlen); source = seq[start:i]
                padded_source = [self.pad_token] * (self.maxlen - len(source)) + source
                self.instances.append((padded_source, target))
    def __len__(self): return len(self.instances)
    def __getitem__(self, idx):
        source, target = self.instances[idx]
        while True:
            neg_target = np.random.randint(0, self.n_items)
            if neg_target != target: break
        return torch.LongTensor(source), torch.LongTensor([target]), torch.LongTensor([neg_target])
# =====================================================================================
# — SASRec (关键修改：容量、forward索引、predict截断) —
# =====================================================================================
class SASRec(nn.Module):
    def __init__(self, n_items, config):
        super(SASRec, self).__init__()
        self.n_items, self.pad_token = n_items, n_items
        self.embedding_dim, self.maxlen = config.sasrec_embedding_dim, config.sasrec_maxlen
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_dim, padding_idx=self.pad_token)
        
        # 🌟 修正 1: 扩大位置嵌入容量 (maxlen + 2)，以彻底避免 Index Error
        self.positional_embedding = nn.Embedding(self.maxlen + 2, self.embedding_dim) 
        
        self.emb_dropout = nn.Dropout(config.sasrec_dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=config.sasrec_transformer_heads, dim_feedforward=self.embedding_dim * 4, dropout=config.sasrec_dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.sasrec_transformer_layers)
        self.output_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(self, input_seq):
        item_embs = self.item_embedding(input_seq)
        seq_len = input_seq.size(1)
        
        position_indices = torch.arange(1, seq_len + 1, dtype=torch.long, device=input_seq.device).unsqueeze(0)
        pos_embs = self.positional_embedding(position_indices)
        
        seq_embedding = self.emb_dropout(item_embs + pos_embs)
        padding_mask = (input_seq == self.pad_token)
        
        # 🔥 重新引入因果注意力掩码
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_seq.device), diagonal=1).bool()
        
        output = self.transformer_encoder(
            seq_embedding,
            mask=attn_mask,  # 恢复因果掩码
            src_key_padding_mask=padding_mask
        )
        
        return self.output_layer_norm(output)


    # bpr_forward 方法无需改动，因为它调用的是修正后的 forward
    def bpr_forward(self, input_seq, pos_target, neg_target):
        output_seq = self.forward(input_seq)
        last_item_rep = output_seq[:, -1, :]
        pos_emb = self.item_embedding(pos_target).squeeze(1)
        neg_emb = self.item_embedding(neg_target).squeeze(1)
        pos_scores = torch.sum(last_item_rep * pos_emb, dim=-1)
        neg_scores = torch.sum(last_item_rep * neg_emb, dim=-1)
        return pos_scores, neg_scores

    # predict 方法无需改动，因为它也调用的是修正后的 forward
    def predict(self, user_idx, user_sequences, top_k):
        self.eval()
        with torch.no_grad():
            if user_idx not in user_sequences: return []
            
            history_seq = user_sequences[user_idx]
            
            # 确保 history_seq 是一个列表
            if not isinstance(history_seq, list):
                history_seq = list(history_seq)
                
            if len(history_seq) > self.maxlen:
                history_seq = history_seq[-self.maxlen:]
            
            input_seq = [self.pad_token] * (self.maxlen - len(history_seq)) + history_seq
            # 再次确保只取 maxlen 长度，防止意外
            input_seq = input_seq[-self.maxlen:] 
            input_seq = torch.LongTensor([input_seq]).to(self.item_embedding.weight.device)
            
            last_item_rep = self.forward(input_seq)[0, -1, :]
            
            scores = torch.matmul(last_item_rep, self.item_embedding.weight.t())
            
            # 过滤历史记录
            history_indices = torch.LongTensor(history_seq).to(scores.device)
            scores[history_indices] = -np.inf
            scores[self.pad_token] = -np.inf 
            
            _, top_indices = torch.topk(scores, k=top_k)
            return top_indices.cpu().numpy().tolist()
# =====================================================================================
# — 主函数 main (整合了加权训练) —
# =====================================================================================
def main():
    config = Config()
    print(f"Using device: {config.device}")

    # --- 1. 全局数据加载与ID映射 (修正版) ---
    print("--- [Step 1] 加载数据并建立ID映射 ---")

    # 🔥 修正：直接从 user.csv 和 book.csv 读取
    user_df = pd.read_csv(os.path.join(config.data_path, 'user.csv'))
    book_df = pd.read_csv(os.path.join(config.data_path, 'book.csv'))
    train_df = pd.read_csv(os.path.join(config.data_path, 'local_train.csv'))
    validation_df = pd.read_csv(os.path.join(config.data_path, 'local_validation.csv'))

    # 🔥 修正：使用正确的列名 'user_id' 和 'book_id'
    user_to_idx = {user_id: i for i, user_id in enumerate(user_df['user_id'].unique())}
    book_to_idx = {book_id: i for i, book_id in enumerate(book_df['book_id'].unique())}
    idx_to_user = {i: user_id for user_id, i in user_to_idx.items()}
    idx_to_book = {i: book_id for book_id, i in book_to_idx.items()}

    n_users, n_items = len(user_to_idx), len(book_to_idx)

    print(f"数据统计: {n_users} 用户, {n_items} 物品")
    print(f"训练集: {len(train_df)} 条, 验证集: {len(validation_df)} 条")

    # 映射训练数据的索引
    train_df['user_idx'] = train_df['user_id'].map(user_to_idx)
    train_df['book_idx'] = train_df['book_id'].map(book_to_idx)
    # 原始代码中缺少了 dropna 和 astype，这里保持原意，假定数据是完整的
    train_df.dropna(subset=['user_idx', 'book_idx'], inplace=True)
    train_df['user_idx'] = train_df['user_idx'].astype(int)
    train_df['book_idx'] = train_df['book_idx'].astype(int)

    print(f"映射后训练集: {len(train_df)} 条")

    user_history_for_filter = train_df.groupby('user_idx')['book_idx'].apply(list).to_dict()

    # --- 2. 训练 LightGCN 模型 (使用加权图和加权损失) ---
    print("\n--- [Stage 1] Training Weighted LightGCN Model ---")
    lgcn_model = LightGCN(n_users, n_items, config).to(config.device)

    # --- 关键改动: 使用'兴趣强度'作为图的权重 ---
    train_interactions = sp.csr_matrix((train_df['兴趣强度'].values, (train_df['user_idx'], train_df['book_idx'])), shape=(n_users, n_items))

    adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj_mat = adj_mat.tolil(); R = train_interactions.tolil()
    adj_mat[:n_users, n_users:] = R; adj_mat[n_users:, :n_users] = R.T
    adj_mat = adj_mat.todok(); rowsum = np.array(adj_mat.sum(axis=1)); d_inv = np.power(rowsum, -0.5).flatten(); d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv); norm_adj = d_mat.dot(adj_mat).dot(d_mat).tocoo()
    lgcn_model.graph = torch.sparse.FloatTensor(torch.LongTensor(np.vstack((norm_adj.row, norm_adj.col))), torch.FloatTensor(norm_adj.data), torch.Size(norm_adj.shape)).to(config.device)

    lgcn_loader = DataLoader(BPRDataset_LGCN(train_df, n_items), batch_size=config.batch_size, shuffle=True)
    optimizer_lgcn = torch.optim.Adam(lgcn_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    for epoch in range(config.epochs):
        lgcn_model.train()
        for users, pos_items, neg_items, interest_scores in tqdm(lgcn_loader, desc=f"LGCN Epoch {epoch+1}/{config.epochs}"):
            users, pos_items, neg_items, interest_scores = users.to(config.device), pos_items.to(config.device), neg_items.to(config.device), interest_scores.to(config.device)
            optimizer_lgcn.zero_grad()
            all_users_emb, all_items_emb = lgcn_model.get_embeddings()
            
            pos_scores = torch.sum(all_users_emb[users] * all_items_emb[pos_items], dim=1)
            neg_scores = torch.sum(all_users_emb[users] * all_items_emb[neg_items], dim=1)
            
            # 1. 获取当前批次用户和物品的原始嵌入
            users_emb_ori = lgcn_model.user_embedding.weight[users]
            pos_emb_ori = lgcn_model.item_embedding.weight[pos_items]
            neg_emb_ori = lgcn_model.item_embedding.weight[neg_items]
        
            # 2. BPR Loss (加权)
            bpr_loss = -torch.mean(interest_scores * torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
            
            # 3. L2 正则化项 (直接使用原始嵌入)
            # ❗ 修正：添加 L2 正则化项，并乘以 weight_decay ❗
            l2_reg = config.weight_decay * (
                torch.norm(users_emb_ori, p=2).pow(2) + 
                torch.norm(pos_emb_ori, p=2).pow(2) + 
                torch.norm(neg_emb_ori, p=2).pow(2)
            ) / users.size(0) # 通常是除以 batch_size
            
            # 4. 总损失
            loss = bpr_loss + l2_reg
            
            loss.backward()
            optimizer_lgcn.step()

    # --- 3. 训练 SASRec 模型 (无需改动) —
    print("\n--- [Stage 2] Training SASRec Model ---")
    sasrec_model = SASRec(n_items, config).to(config.device)
    train_df.sort_values(by=['user_idx', '借阅时间'], inplace=True)
    user_sequences = train_df.groupby('user_idx')['book_idx'].apply(list).to_dict()
    sasrec_loader = DataLoader(SASRecDataset(user_sequences, config.sasrec_maxlen, n_items), batch_size=config.batch_size, shuffle=True)
    optimizer_sasrec = torch.optim.Adam(sasrec_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    bpr_loss_sasrec = lambda pos, neg: -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()
    
    for epoch in range(config.epochs):
        sasrec_model.train()
        # ... Stage 2 训练循环 (修正)
        for input_seq, pos_target, neg_target in tqdm(sasrec_loader, desc=f"SASRec Epoch {epoch+1}/{config.epochs}"):
            input_seq, pos_target, neg_target = input_seq.to(config.device), pos_target.to(config.device), neg_target.to(config.device)
            optimizer_sasrec.zero_grad()
            # 使用新的 bpr_forward 方法来计算训练得分
            pos_scores, neg_scores = sasrec_model.bpr_forward(input_seq, pos_target, neg_target) 
            # SASRec 模型中的 L2 正则化由 Adam optimizer 的 weight_decay 自动处理
            loss = bpr_loss_sasrec(pos_scores, neg_scores) 
            loss.backward()
            optimizer_sasrec.step()
   # 修改评估和提交部分（替换原来的 Stage 3 和 Stage 4）
    # --- 4. 召回、融合与评估 ---
    print("\n--- [Stage 3] Recall, Fusion and Evaluation ---")
    validation_user_ids = validation_df['user_id'].unique()
    validation_user_indices = [user_to_idx[uid] for uid in validation_user_ids if uid in user_to_idx]

    lgcn_recs_by_idx = lgcn_model.predict(validation_user_indices, user_history_for_filter, config.recall_k)
    sasrec_recs_by_idx = {uidx: sasrec_model.predict(uidx, user_sequences, config.recall_k) 
                        for uidx in tqdm(validation_user_indices, desc="SASRec Recalling")}

    # 🔥 改为 Top-K 推荐
    K = config.submission_top_k

    # 单模型 Top-K 评估
    lgcn_topk = {idx_to_user[uidx]: [idx_to_book[i] for i in recs[:K]] 
                for uidx, recs in lgcn_recs_by_idx.items() if recs}
    sasrec_topk = {idx_to_user[uidx]: [idx_to_book[i] for i in recs[:K]] 
                for uidx, recs in sasrec_recs_by_idx.items() if recs}

    evaluate_model(lgcn_topk, validation_df, f"LightGCN Top-{K}")
    evaluate_model(sasrec_topk, validation_df, f"SASRec Top-{K}")

    # 混合模型 Top-K
    final_hybrid_recs = {}
    for uidx in tqdm(validation_user_indices, desc="Fusing and Re-ranking"):
        lgcn_list = lgcn_recs_by_idx.get(uidx, [])
        sasrec_list = sasrec_recs_by_idx.get(uidx, [])
        
        if not lgcn_list and not sasrec_list: 
            continue
        
        candidate_scores = {}
        lgcn_weight = 0.6
        sasrec_weight = 0.4
        
        for rank, book_idx in enumerate(lgcn_list[:50]):  # 只取前50
            candidate_scores[book_idx] = candidate_scores.get(book_idx, 0) + \
                                        lgcn_weight / (rank + 1)
        
        for rank, book_idx in enumerate(sasrec_list[:50]):
            candidate_scores[book_idx] = candidate_scores.get(book_idx, 0) + \
                                        sasrec_weight / (rank + 1)
        
        if not candidate_scores: 
            continue
        
        # 推荐 Top-K
        sorted_candidates = sorted(candidate_scores.items(), 
                                key=lambda item: item[1], 
                                reverse=True)
        top_k_indices = [book_idx for book_idx, _ in sorted_candidates[:K]]
        final_hybrid_recs[idx_to_user[uidx]] = [idx_to_book[i] for i in top_k_indices]

    evaluate_model(final_hybrid_recs, validation_df, f"Hybrid Top-{K}")

    # --- 5. 生成提交文件 ---
    print("\n--- [Stage 4] Generating Submission File ---")

    submission_records = []

    for uidx in tqdm(validation_user_indices, desc="Generating Submission"):
        lgcn_list = lgcn_recs_by_idx.get(uidx, [])
        sasrec_list = sasrec_recs_by_idx.get(uidx, [])
        
        if not lgcn_list and not sasrec_list:
            user_hist = set(user_history_for_filter.get(uidx, []))
            all_items = set(range(n_items))
            available_items = list(all_items - user_hist)
            
            if available_items:
                top_k_indices = np.random.choice(available_items, 
                                            size=min(K, len(available_items)), 
                                            replace=False)
            else:
                top_k_indices = list(range(K))
        else:
            candidate_scores = {}
            
            for rank, book_idx in enumerate(lgcn_list[:50]):
                candidate_scores[book_idx] = candidate_scores.get(book_idx, 0) + 0.6 / (rank + 1)
            
            for rank, book_idx in enumerate(sasrec_list[:50]):
                candidate_scores[book_idx] = candidate_scores.get(book_idx, 0) + 0.4 / (rank + 1)
            
            if not candidate_scores:
                top_k_indices = [0] * K
            else:
                sorted_candidates = sorted(candidate_scores.items(), 
                                        key=lambda item: item[1], 
                                        reverse=True)
                top_k_indices = [book_idx for book_idx, _ in sorted_candidates[:K]]
        
        user_id = idx_to_user[uidx]
        for book_idx in top_k_indices:
            book_id = idx_to_book[book_idx]
            submission_records.append({'user_id': user_id, 'book_id': book_id})

    submission_df = pd.DataFrame(submission_records)
    submission_df = submission_df.sort_values(['user_id', 'book_id']).reset_index(drop=True)

    submission_path = os.path.join(config.data_path, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)

    print(f"\n✓ 提交文件已生成: {submission_path}")
    print(f"提交文件包含 {len(submission_df)} 条推荐记录")
    print(submission_df.head(15))



if __name__ == '__main__':
    main()