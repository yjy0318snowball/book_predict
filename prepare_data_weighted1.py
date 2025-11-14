import pandas as pd
import numpy as np

def preprocess_and_split_data_v2():
    """
    改进版预处理：增加数据统计分析，使用更严格的用户筛选
    """
    print("="*60)
    print("开始数据预处理 v2.0")
    print("="*60)
    
    # --- Step 1: 加载数据 ---
    try:
        inter_df = pd.read_csv("inter.csv")
    except FileNotFoundError:
        print("错误: 找不到 inter.csv")
        return

    print(f"\n原始数据: {len(inter_df)} 条记录")
    
    # --- Step 2: 时间处理 ---
    inter_df["借阅时间"] = pd.to_datetime(inter_df["借阅时间"], errors="coerce")
    inter_df["还书时间"] = pd.to_datetime(inter_df["还书时间"], errors="coerce")
    inter_df["续借时间"] = pd.to_datetime(inter_df["续借时间"], errors="coerce")
    
    # 删除缺失借阅时间的记录
    inter_df = inter_df.dropna(subset=['借阅时间']).copy()
    print(f"删除无借阅时间后: {len(inter_df)} 条")
    
    # 修正续借次数
    inter_df.loc[(inter_df["续借时间"].notna()) & (inter_df["续借次数"] == 0), "续借次数"] = 1
    
    # --- Step 3: 填充还书时间 ---
    has_return = inter_df.dropna(subset=['还书时间'])
    
    # 计算平均借阅时长
    non_renewal = has_return[has_return['续借次数'] == 0]
    avg_days_no_renew = (non_renewal['还书时间'] - non_renewal['借阅时间']).dt.days.mean()
    avg_days_no_renew = 30.0 if pd.isna(avg_days_no_renew) else max(avg_days_no_renew, 1.0)
    
    renewal = has_return[has_return['续借次数'] > 0].dropna(subset=['续借时间'])
    avg_days_renew = (renewal['还书时间'] - renewal['续借时间']).dt.days.mean()
    avg_days_renew = 30.0 if pd.isna(avg_days_renew) else max(avg_days_renew, 1.0)
    
    print(f"平均借阅时长: 不续借={avg_days_no_renew:.1f}天, 续借={avg_days_renew:.1f}天")
    
    # 填充缺失的还书时间
    mask_no_renew = (inter_df['续借次数'] == 0) & inter_df['还书时间'].isna()
    mask_renew = (inter_df['续借次数'] > 0) & inter_df['还书时间'].isna()
    
    inter_df.loc[mask_no_renew, '还书时间'] = \
        inter_df.loc[mask_no_renew, '借阅时间'] + pd.Timedelta(days=avg_days_no_renew)
    inter_df.loc[mask_renew, '还书时间'] = \
        inter_df.loc[mask_renew, '续借时间'].fillna(inter_df.loc[mask_renew, '借阅时间']) + \
        pd.Timedelta(days=avg_days_renew)
    
    inter_df = inter_df.dropna(subset=['还书时间'])
    
    # --- Step 4: 计算兴趣强度 ---
    inter_df["借阅时长"] = (inter_df["还书时间"] - inter_df["借阅时间"]).dt.days
    inter_df["借阅时长比"] = inter_df["借阅时长"] / avg_days_no_renew
    inter_df["兴趣强度"] = 1.0 + inter_df["续借次数"] * 0.5 + inter_df["借阅时长比"] * 0.3
    
    # 过滤异常值
    inter_df = inter_df[
        (inter_df["兴趣强度"] > 0) & 
        (inter_df["兴趣强度"] < 10) &
        (inter_df["借阅时长"] >= 0) &
        (inter_df["借阅时长"] <= 365)
    ].copy()
    
    print(f"过滤异常值后: {len(inter_df)} 条")
    
    # --- Step 5: 数据质量分析 ---
    print("\n" + "="*60)
    print("数据质量分析")
    print("="*60)
    
    user_stats = inter_df.groupby('user_id').agg({
        'book_id': 'count',
        '借阅时间': ['min', 'max']
    }).reset_index()
    user_stats.columns = ['user_id', 'record_count', 'first_date', 'last_date']
    
    book_stats = inter_df.groupby('book_id').size().reset_index(name='borrow_count')
    
    print(f"总用户数: {inter_df['user_id'].nunique()}")
    print(f"总图书数: {inter_df['book_id'].nunique()}")
    print(f"用户借阅记录分布:")
    print(f"  - 平均: {user_stats['record_count'].mean():.1f}")
    print(f"  - 中位数: {user_stats['record_count'].median():.0f}")
    print(f"  - 最小: {user_stats['record_count'].min()}")
    print(f"  - 最大: {user_stats['record_count'].max()}")
    
    # --- Step 6: 智能划分训练集和验证集 ---
    print("\n" + "="*60)
    print("数据划分策略")
    print("="*60)
    
    # 🔥 关键改进：只选择活跃用户和热门图书
    MIN_USER_RECORDS = 5  # 至少5条记录
    MIN_BOOK_BORROWS = 3  # 图书至少被借3次
    
    active_users = user_stats[user_stats['record_count'] >= MIN_USER_RECORDS]['user_id']
    popular_books = book_stats[book_stats['borrow_count'] >= MIN_BOOK_BORROWS]['book_id']
    
    # 筛选数据
    filtered_df = inter_df[
        inter_df['user_id'].isin(active_users) &
        inter_df['book_id'].isin(popular_books)
    ].copy()
    
    print(f"筛选后: {len(filtered_df)} 条 (活跃用户+热门图书)")
    print(f"  - 用户数: {filtered_df['user_id'].nunique()}")
    print(f"  - 图书数: {filtered_df['book_id'].nunique()}")
    
    # 按时间排序
    filtered_df = filtered_df.sort_values(['user_id', '借阅时间'])
    
    # 每个用户的最后一条作为验证集
    validation_df = filtered_df.groupby('user_id').tail(1).copy()
    train_df = filtered_df.drop(validation_df.index).copy()
    
    print(f"\n最终划分:")
    print(f"  - 训练集: {len(train_df)} 条")
    print(f"  - 验证集: {len(validation_df)} 条")
    print(f"  - 训练集用户数: {train_df['user_id'].nunique()}")
    print(f"  - 验证集用户数: {validation_df['user_id'].nunique()}")
    print(f"  - 平均每用户训练记录: {train_df.groupby('user_id').size().mean():.1f}")
    
    # --- Step 7: 保存文件 ---
    train_df.to_csv('local_train.csv', index=False)
    validation_df.to_csv('local_validation.csv', index=False)
    
    # 保存ID列表
    unique_users = pd.DataFrame({'user_id': filtered_df['user_id'].unique()})
    unique_books = pd.DataFrame({'book_id': filtered_df['book_id'].unique()})
    unique_users.to_csv('user.csv', index=False)
    unique_books.to_csv('book.csv', index=False)
    
    print("\n✓ 预处理完成！文件已保存。")
    print("="*60)

if __name__ == '__main__':
    preprocess_and_split_data_v2()
