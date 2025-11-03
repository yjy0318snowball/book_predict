import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import zipfile
from tqdm import tqdm
from collections import defaultdict


class Config:
    # 基础参数（适配稀疏数据）
    random_seed = 42
    embedding_dim = 64 # 更小维度，避免过拟合
    epochs = 20  # 增加训练轮数，缓慢学习
    batch_size = 512
    lr = 0.0006  # 更低学习率，适合稀疏数据
    weight_decay = 1e-6  # 轻微正则
    patience = 3
    top_k = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 兜底核心参数（保持不变）
    fallback_ratio = 0.98  #对难推荐的用户推荐结果的最后一本书作为兜底
    interact_weight = 0.5
    acc_weight = 0.5

    # 优化特征与负样本（适配稀疏数据）
    content_weight = 0.8  # 提高内容特征权重，依赖图书属性
    user_freq_weight = 0.2
    neg_sample_num = 3
    pop_threshold = 90
    hard_neg_ratio = 0.2
    pos_loss_weight = 1.5  # 大幅提高正样本权重
    neg_loss_weight = 0.5  # 降低负样本权重
    margin = 0.5  # 增大margin，强化正样本得分

    # 数据路径
    book_path = 'item.csv'
    inter_path = 'inter.csv'
    user_path = 'user.csv'
    model_path = 'best_model.pth'
    submission_csv = 'submission.csv'
    submission_zip = 'submission.zip'
    num_workers = 0 if torch.cuda.is_available() else 2


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None


def calculate_metrics(pred_items, true_items, user_train_items):
    pred_filtered = [item for item in pred_items if item not in user_train_items]
    if not pred_filtered or not true_items:
        return 0.0, 0.0, 0.0
    tp = len(set(pred_filtered) & true_items)
    precision = tp / len(pred_filtered)
    recall = tp / len(true_items)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def load_and_preprocess():
    # 加载数据
    book_df = pd.read_csv(Config.book_path)
    inter_df = pd.read_csv(Config.inter_path)
    user_df = pd.read_csv(Config.user_path)

    # 处理用户ID和时间特征
    user_df = user_df.rename(columns={'借阅人': 'user_id'})
    inter_df['借阅时间'] = pd.to_datetime(inter_df['借阅时间'], errors='coerce')
    inter_df = inter_df.dropna(subset=['user_id', 'book_id', '借阅时间']).reset_index(drop=True)
    inter_df['借阅月份'] = inter_df['借阅时间'].dt.month

    # 使用所有用户
    interacted_user_ids = inter_df['user_id'].unique()
    print(f"确认有交互记录的用户数：{len(interacted_user_ids)}人")

    # 保留所有图书
    print(f"原始图书总数：{len(book_df)}")

    # 编码ID
    user_encoder = LabelEncoder()
    book_encoder = LabelEncoder()
    all_book_ids = book_df['book_id'].astype(str).unique()
    book_encoder.fit(all_book_ids)
    inter_df['user_idx'] = user_encoder.fit_transform(inter_df['user_id'].astype(str))
    inter_df['book_idx'] = book_encoder.transform(inter_df['book_id'].astype(str))
    n_users = len(user_encoder.classes_)
    n_books = len(book_encoder.classes_)
    print(f"编码后用户数：{n_users}，图书总数（含无交互）：{n_books}")

    # 统计用户交互次数
    user_interact_count = inter_df.groupby('user_idx')['book_idx'].count().to_dict()
    for user in range(n_users):
        if user not in user_interact_count:
            user_interact_count[user] = 0
    interact_counts = list(user_interact_count.values())
    print(f"用户交互次数分布：最小值={min(interact_counts)}, 最大值={max(interact_counts)}, 平均值={np.mean(interact_counts):.1f}")

    # 计算交互频率特征
    inter_df['user_item_freq'] = inter_df.groupby(['user_idx', 'book_idx']).cumcount() + 1
    inter_df['user_freq'] = inter_df.groupby('user_idx')['book_idx'].transform('count')
    inter_df['item_freq'] = inter_df.groupby('book_idx')['user_idx'].transform('count').fillna(0)
    all_book_idx = set(range(n_books))
    interacted_book_idx = set(inter_df['book_idx'].unique())
    non_interacted_book_idx = all_book_idx - interacted_book_idx
    item_freq_dict = dict(inter_df.groupby('book_idx')['user_idx'].count())
    for idx in non_interacted_book_idx:
        item_freq_dict[idx] = 0
    inter_df['item_freq'] = inter_df['book_idx'].map(item_freq_dict)

    # 【优化数据划分：适配稀疏数据】
    train_data = []
    val_data = []
    test_data = []
    for user in inter_df['user_idx'].unique():
        user_inter = inter_df[inter_df['user_idx'] == user].sort_values('借阅时间')
        n = len(user_inter)
        if n >= 3:  # 至少3条交互才拆分，保证验证/测试有数据
            train_idx = int(n * 0.8)  # 80%训练
            val_idx = train_idx + 1  # 1条验证
            train_data.append(user_inter.iloc[:train_idx])
            val_data.append(user_inter.iloc[train_idx:val_idx])
            test_data.append(user_inter.iloc[val_idx:])  # 剩余测试
        else:
            # 交互太少的用户，全部放训练集
            train_data.append(user_inter)

    # 处理空验证集/测试集（用训练集随机采样补充）
    train_df = pd.concat(train_data).reset_index(drop=True)
    val_df = pd.concat(val_data).reset_index(drop=True) if val_data else train_df.sample(frac=0.05, random_state=Config.random_seed)
    test_df = pd.concat(test_data).reset_index(drop=True) if test_data else val_df.sample(frac=0.5, random_state=Config.random_seed)
    print(f"训练集交互数：{len(train_df)}，验证集交互数：{len(val_df)}，测试集交互数：{len(test_df)}")

    # 记录用户最后借阅图书（兜底用）
    user_last_book = {}
    for user in inter_df['user_idx'].unique():
        user_inter = inter_df[inter_df['user_idx'] == user].sort_values('借阅时间')
        last_book = user_inter.iloc[-1]['book_idx']
        user_last_book[user] = last_book

    # 构建用户历史类别
    user_category_history = defaultdict(set)
    cat_map = None
    if '一级分类' in book_df.columns:
        book_df['book_idx'] = book_encoder.transform(book_df['book_id'].astype(str))
        cat_map = dict(zip(book_df['book_idx'], book_df['一级分类']))
        for _, row in train_df.iterrows():
            user_category_history[row['user_idx']].add(cat_map[row['book_idx']])

    # 【优化内容特征：增加热门图书标识】
    book_features = None
    if '一级分类' in book_df.columns:
        # 一级分类独热编码
        one_hot_cat = pd.get_dummies(book_df['一级分类'], drop_first=True)
        # 图书流行度归一化
        item_freq_norm = (book_df['book_idx'].map(item_freq_dict) /
                          (max(item_freq_dict.values()) if item_freq_dict else 1)).fillna(0).values.reshape(-1, 1)
        # 新增：是否为热门图书（前20%）
        pop_threshold = np.percentile(list(item_freq_dict.values()), 80) if item_freq_dict else 0
        book_df['is_popular'] = book_df['book_idx'].map(item_freq_dict) >= pop_threshold
        is_popular = book_df['is_popular'].astype(int).values.reshape(-1, 1)
        # 合并所有特征
        book_features = np.hstack([one_hot_cat, item_freq_norm, is_popular])
        book_features = torch.tensor(book_features.astype(np.float32))
        print(f"内容特征维度：{book_features.shape[1]}（含分类+流行度+热门标识）")

    # 构建用户特征
    user_features = np.zeros((n_users, 13))  # 1个频率+12个月份
    for user in range(n_users):
        user_inter = train_df[train_df['user_idx'] == user]
        if len(user_inter) == 0:
            continue
        user_freq = len(user_inter) / (train_df['user_idx'].value_counts().max() if not train_df.empty else 1)
        user_features[user, 0] = user_freq
        months = user_inter['借阅月份'].value_counts().index
        for m in months:
            user_features[user, m] = 1.0
    user_features = torch.tensor(user_features.astype(np.float32))

    # 物品流行度
    item_popularity = item_freq_dict

    # 正负样本集合
    user_pos_train = {u: set(train_df[train_df['user_idx'] == u]['book_idx']) for u in train_df['user_idx'].unique()}
    user_pos_val = {u: set(val_df[val_df['user_idx'] == u]['book_idx']) for u in val_df['user_idx'].unique()}
    user_pos_test = {u: set(test_df[test_df['user_idx'] == u]['book_idx']) for u in test_df['user_idx'].unique()}
    all_interacted_users = inter_df['user_idx'].unique().tolist()

    return {
        'train_df': train_df, 'val_df': val_df, 'test_df': test_df,
        'book_encoder': book_encoder, 'user_encoder': user_encoder,
        'n_users': n_users, 'n_books': n_books,
        'user_pos_train': user_pos_train, 'user_pos_val': user_pos_val, 'user_pos_test': user_pos_test,
        'interacted_users': all_interacted_users,
        'book_features': book_features, 'user_features': user_features,
        'item_popularity': item_popularity, 'user_category_history': user_category_history,
        'cat_map': cat_map, 'user_last_book': user_last_book, 'user_interact_count': user_interact_count
    }


class EnhancedMatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, book_features=None, user_features=None):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(0.0))

        self.user_proj = None
        if user_features is not None and user_features.shape[1] > 0:
            self.user_proj = nn.Linear(user_features.shape[1], embedding_dim)
            self.user_features = user_features

        self.content_proj = None
        if book_features is not None and book_features.shape[1] > 0:
            self.content_proj = nn.Linear(book_features.shape[1], embedding_dim)
            self.book_features = book_features
            self.content_weight = nn.Parameter(torch.tensor(Config.content_weight))

        # 初始化参数（更保守保守）
        nn.init.xavier_normal_(self.user_emb.weight, gain=0.03)
        nn.init.xavier_normal_(self.item_emb.weight, gain=0.03)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users, items):
        user_emb = self.user_emb(users)
        if self.user_proj is not None:
            user_feat_emb = self.user_proj(self.user_features[users])
            user_emb = user_emb + user_feat_emb

        item_emb = self.item_emb(items)
        if self.content_proj is not None:
            content_emb = self.content_proj(self.book_features[items])
            item_emb = item_emb + torch.sigmoid(self.content_weight) * content_emb  # 强化内容特征影响

        dot_product = (user_emb * item_emb).sum(dim=1)
        return dot_product + self.user_bias(users).squeeze() + self.item_bias(items).squeeze() + self.global_bias

    def get_embeddings(self):
        user_emb = self.user_emb.weight
        if self.user_proj is not None:
            user_feat_emb = self.user_proj(self.user_features)
            user_emb = user_emb + user_feat_emb

        item_emb = self.item_emb.weight
        if self.content_proj is not None:
            content_emb = self.content_proj(self.book_features)
            item_emb = item_emb + torch.sigmoid(self.content_weight) * content_emb
        return user_emb, item_emb


class BalancedNegDataset(Dataset):
    def __init__(self, df, n_items, user_pos_train, item_popularity,
                 user_category_history, cat_map, pop_threshold=85, hard_neg_ratio=0.2):
        self.users = df['user_idx'].values
        self.items = df['book_idx'].values
        self.n_items = n_items
        self.user_pos_train = user_pos_train
        self.item_popularity = item_popularity
        self.user_category_history = user_category_history
        self.cat_map = cat_map
        pop_values = list(item_popularity.values()) if item_popularity else [0]
        self.pop_threshold = np.percentile(pop_values, pop_threshold) if pop_values else 0
        self.hard_neg_ratio = hard_neg_ratio
        self.neg_items = self._sample_neg()

    def _sample_neg(self):
        neg_items = []
        all_items = set(range(self.n_items))
        user_hard_candidates = defaultdict(list)
        # 构建难负样本候选集（仅少量）
        for user in self.user_category_history:
            user_pos = self.user_pos_train.get(user, set())
            user_cats = self.user_category_history[user]
            for item in all_items - user_pos:
                if self.cat_map and self.cat_map.get(item) in user_cats:
                    user_hard_candidates[user].append(item)

        # 采样负样本（减少难负样本比例）
        for i, user in enumerate(self.users):
            user_pos = self.user_pos_train.get(user, set())
            if np.random.random() < self.hard_neg_ratio and len(user_hard_candidates.get(user, [])) > 0:
                neg = np.random.choice(user_hard_candidates[user])
            else:
                # 普通负样本：非已借阅+低流行度
                while True:
                    neg = np.random.randint(0, self.n_items)
                    if neg not in user_pos and self.item_popularity.get(neg, 0) <= self.pop_threshold:
                        break
            neg_items.append(neg)
        return np.array(neg_items)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.neg_items[idx]


def train(model, train_df, val_df, test_df, n_books, user_pos_train, user_pos_val, user_pos_test,
          item_popularity, user_category_history, cat_map, user_last_book, user_interact_count):
    dataset = BalancedNegDataset(
        train_df, n_books, user_pos_train, item_popularity,
        user_category_history, cat_map,
        Config.pop_threshold, Config.hard_neg_ratio
    )
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)
    optimizer = optim.Adam(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-7)  # 适配更多epoch

    model.to(Config.device)
    best_val_f1 = -1.0
    best_test_metrics = None
    best_user_acc = {}
    no_improve_epochs = 0

    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0.0
        for users, pos_items, neg_items in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{Config.epochs}"):
            users = users.to(Config.device)
            pos_items = pos_items.to(Config.device)
            neg_items = neg_items.to(Config.device)

            optimizer.zero_grad()
            pos_score = model(users, pos_items)
            neg_score = model(users, neg_items)

            # 【核心优化：强化正样本学习】
            pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-8).mean() * Config.pos_loss_weight  # 正样本权重翻倍
            neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-8).mean() * Config.neg_loss_weight  # 负样本权重降低
            margin_loss = torch.mean(
                torch.max(torch.tensor(0.0, device=Config.device), Config.margin - (pos_score - neg_score))
            )  # 增大margin，强制正样本得分更高
            loss = pos_loss + neg_loss + 0.3 * margin_loss  # 增加margin损失占比

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        print(f"Epoch {epoch + 1} | 损失: {avg_loss:.4f} | 学习率: {scheduler.get_last_lr()[0]:.7f}")

        # 验证与测试
        model.eval()
        with torch.no_grad():
            val_metrics = evaluate(model, val_df, user_pos_val, user_pos_train, user_last_book)
            test_metrics = evaluate(model, test_df, user_pos_test, user_pos_train, user_last_book)
        print(f"验证集 P: {val_metrics['precision']:.4f}, R: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"测试集 P: {test_metrics['precision']:.4f}, R: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")

        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_test_metrics = test_metrics
            best_user_acc = val_metrics['user_acc']
            no_improve_epochs = 0
            torch.save(model.state_dict(), Config.model_path)
            print(f"  保存最佳模型（验证集F1: {best_val_f1:.4f}）")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= Config.patience:
                print(f"  早停触发！最佳验证集F1: {best_val_f1:.4f}")
                break

    model.load_state_dict(torch.load(Config.model_path))
    return model, best_test_metrics, best_user_acc


def evaluate(model, eval_df, user_pos_true, user_pos_train, user_last_book):
    user_emb, item_emb = model.get_embeddings()
    user_bias = model.user_bias.weight.squeeze()
    item_bias = model.item_bias.weight.squeeze()
    global_bias = model.global_bias

    eval_users = eval_df['user_idx'].unique()
    total_p, total_r, total_f1 = 0.0, 0.0, 0.0
    count = 0
    user_acc = {}

    for user in eval_users:
        true_items = user_pos_true.get(user, set())
        if not true_items:
            continue
        user_train_items = user_pos_train.get(user, set())

        # 模型推荐（过滤已借图书）
        scores = (user_emb[user] @ item_emb.T) + user_bias[user] + item_bias + global_bias
        scores[list(user_train_items)] = -1e9
        _, top_idx = torch.topk(scores, k=Config.top_k)
        model_pred = top_idx.cpu().numpy().tolist()

        # 计算用户级准确率
        model_p, _, _ = calculate_metrics(model_pred, true_items, user_train_items)
        user_acc[user] = model_p

        # 累计指标
        p, r, f1 = calculate_metrics(model_pred, true_items, user_train_items)
        total_p += p
        total_r += r
        total_f1 += f1
        count += 1

    return {
        'precision': total_p / count if count > 0 else 0.0,
        'recall': total_r / count if count > 0 else 0.0,
        'f1': total_f1 / count if count > 0 else 0.0,
        'user_acc': user_acc
    }


def generate_submission(model, user_encoder, book_encoder, interacted_users, user_pos_train,
                        user_last_book, user_interact_count, user_acc):
    model.eval()
    user_emb, item_emb = model.get_embeddings()
    user_bias = model.user_bias.weight.squeeze()
    item_bias = model.item_bias.weight.squeeze()
    global_bias = model.global_bias

    submissions = []
    total_users = len(interacted_users)
    fallback_num = int(Config.fallback_ratio * total_users)

    # 计算用户推荐难度
    user_difficulty = {}
    max_interact = max(user_interact_count.values()) if user_interact_count else 1
    for user in interacted_users:
        interact = user_interact_count.get(user, 0)
        interact_norm = 1 - (interact / max_interact)
        acc = user_acc.get(user, 0.0)
        acc_norm = 1 - acc
        difficulty = (interact_norm * Config.interact_weight) + (acc_norm * Config.acc_weight)
        user_difficulty[user] = difficulty

    # 筛选兜底用户
    sorted_users = sorted(interacted_users, key=lambda x: user_difficulty[x], reverse=True)
    fallback_users = set(sorted_users[:fallback_num])

    # 生成推荐
    for user in interacted_users:
        if user in fallback_users:
            book_idx = user_last_book.get(user, 0)
        else:
            scores = (user_emb[user] @ item_emb.T) + user_bias[user] + item_bias + global_bias
            user_train_items = user_pos_train.get(user, set())
            scores[list(user_train_items)] = -1e9
            _, top_idx = torch.topk(scores, k=Config.top_k)
            book_idx = top_idx.item()

        book_id = book_encoder.inverse_transform([book_idx])[0]
        user_id = user_encoder.inverse_transform([user])[0]
        submissions.append({'user_id': str(user_id), 'book_id': str(book_id)})

    # 输出兜底比例
    print(f"\n兜底推荐使用比例：{len(fallback_users)}/{total_users} ({len(fallback_users) / total_users:.1%})")
    print(f"（按推荐难度排序，取前{Config.fallback_ratio * 100}%最难推荐的用户）")

    # 生成提交文件
    pd.DataFrame(submissions).to_csv(Config.submission_csv, index=False)
    with zipfile.ZipFile(Config.submission_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(Config.submission_csv)
    print(f"提交文件生成：{Config.submission_zip}")


def main():
    set_random_seed(Config.random_seed)
    print("=== 启动推荐系统（适配稀疏数据优化版） ===")
    print(f"兜底规则：按推荐难度自动筛选前{Config.fallback_ratio * 100}%的用户")
    print(f"难度计算：交互次数少（权重{Config.interact_weight}）+ 模型准确率低（权重{Config.acc_weight}）")

    # 数据预处理
    print("\n=== 数据预处理 ===")
    data = load_and_preprocess()
    train_df = data['train_df']
    val_df = data['val_df']
    test_df = data['test_df']
    n_users = data['n_users']
    n_books = data['n_books']
    user_encoder = data['user_encoder']
    book_encoder = data['book_encoder']
    user_pos_train = data['user_pos_train']
    user_pos_val = data['user_pos_val']
    user_pos_test = data['user_pos_test']
    interacted_users = data['interacted_users']
    book_features = data['book_features']
    user_features = data['user_features']
    item_popularity = data['item_popularity']
    user_category_history = data['user_category_history']
    cat_map = data['cat_map']
    user_last_book = data['user_last_book']
    user_interact_count = data['user_interact_count']

    # 模型训练
    print("\n=== 模型训练 ===")
    model = EnhancedMatrixFactorization(n_users, n_books, Config.embedding_dim, book_features, user_features)
    model, best_test_metrics, best_user_acc = train(
        model, train_df, val_df, test_df, n_books, user_pos_train, user_pos_val,
        user_pos_test, item_popularity, user_category_history, cat_map, user_last_book, user_interact_count
    )

    # 最终评估
    print("\n=== 测试集最终评估结果 ===")
    print(f"精确率（Precision）: {best_test_metrics['precision']:.4f}")
    print(f"召回率（Recall）: {best_test_metrics['recall']:.4f}")
    print(f"F1值（F1-Score）: {best_test_metrics['f1']:.4f}")

    # 生成提交
    print("\n=== 生成提交文件 ===")
    generate_submission(model, user_encoder, book_encoder, interacted_users, user_pos_train,
                        user_last_book, user_interact_count, best_user_acc)

    print("\n=== 完成 ===")


if __name__ == '__main__':
    main()