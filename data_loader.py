# data_loader.py
import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# ========== 关键修改1：使用正确的标签列 ==========
TARGET_COLUMN = "Label"  # 改为使用label列
# ============================================

NON_FEATURE_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Dataset", "Attack"]  # 新增Attack列
CATEGORICAL_COLS = ["PROTOCOL", "L7_PROTO", "ICMP_IPV4_TYPE"]
MAX_FEATURE_VALUE = 1e4
LOG_FILE = "data_processing.log"

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode='w')
    ]
)

class IntrusionDataLoader:
    """客户端数据加载器（最终修复版）"""
    def __init__(self, client_id: int, data_root: str, batch_size: int = 32):
        self.client_id = client_id
        self.data_root = os.path.join(data_root, f"client_{client_id}")
        self.batch_size = batch_size
        self._load_data()
        self._feature_dim = self.train_df.shape[1] - 1  # 特征维度
    
    @property
    def feature_dim(self):
        return self._feature_dim
        
    def _sanitize_data(self, df):
        """彻底清洗数据，确保所有特征为数值类型"""
        sanitized_df = df.copy()
    
        # 1. 仅删除明确指定的非特征列
        cols_to_drop = [col for col in NON_FEATURE_COLS if col in sanitized_df.columns]
        sanitized_df = sanitized_df.drop(columns=cols_to_drop, errors="ignore")
    
        # 2. 确保标签列存在
        if TARGET_COLUMN not in sanitized_df.columns:
            raise ValueError(f"清洗后数据缺失标签列 {TARGET_COLUMN}")
    
        # 3. 分离特征和标签
        features = sanitized_df.drop(columns=[TARGET_COLUMN])
        labels = sanitized_df[TARGET_COLUMN]
    
        # 4. 仅对特征列进行处理
        # 4.1 转换非数值列
        for col in features.columns:
            if features[col].dtype not in [np.float32, np.float64, np.int32, np.int64]:
                try:
                    features[col] = pd.to_numeric(features[col], errors='coerce')
                except:
                    logging.warning(f"列 {col} 无法转换为数值类型，保留原样")
    
        # 4.2 处理缺失值
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    
        # 4.3 合并回数据框
        sanitized_df = pd.concat([features, labels], axis=1)
    
        # 5. 确保标签列为整数
        sanitized_df[TARGET_COLUMN] = sanitized_df[TARGET_COLUMN].astype(int)
    
        return sanitized_df

    def _load_data(self):
        """加载预处理后的客户端数据"""
        try:
            # 读取原始数据
            train_path = os.path.join(self.data_root, "train.csv")
            test_path = os.path.join(self.data_root, "test.csv")
            
            raw_train_df = pd.read_csv(train_path)
            raw_test_df = pd.read_csv(test_path)
            
            # 彻底清洗数据
            self.train_df = self._sanitize_data(raw_train_df)
            self.test_df = self._sanitize_data(raw_test_df)
            
            # 验证标签列
            if TARGET_COLUMN not in self.train_df.columns:
                raise ValueError(f"训练数据缺少标签列 {TARGET_COLUMN}")
                
            # 检查标签值是否合法（0/1）
            label_values = self.train_df[TARGET_COLUMN].unique()
            invalid_values = [v for v in label_values if v not in (0, 1)]
            if invalid_values:
                raise ValueError(f"非法标签值: {invalid_values}")
            
            self._feature_dim = self.train_df.shape[1] - 1
            logging.info(f"客户端{self.client_id} 特征维度: {self._feature_dim}")
        except FileNotFoundError as e:
            logging.error(f"客户端{self.client_id} 数据加载失败: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"客户端{self.client_id} 数据处理异常: {str(e)}")
            raise

    def sample_batch(self) -> tuple:
        """采样批量数据，确保返回特征为float32，标签为int64"""
        if len(self.train_df) == 0:
            raise RuntimeError(f"客户端{self.client_id} 训练数据为空")
            
        # 随机选择索引
        indices = np.random.choice(len(self.train_df), self.batch_size, replace=False)
        
        # 获取特征和标签
        features = self.train_df.drop(TARGET_COLUMN, axis=1).iloc[indices].values
        labels = self.train_df[TARGET_COLUMN].iloc[indices].values
        
        # 确保特征为二维数组，形状为 (batch_size, feature_dim)
        if features.ndim == 1:
            features = features.reshape(-1, 1)  # 如果是一维，则转换为二维
        elif features.ndim != 2:
            features = features.reshape(features.shape[0], -1)  # 展平为二维
        
        # 检查特征维度
        if features.shape[1] != self.feature_dim:
            raise ValueError(f"特征维度错误: 期望 {self.feature_dim}, 实际 {features.shape[1]}")
        
        # 转换数据类型
        features = features.astype(np.float32)
        labels = labels.astype(np.int64)
        
        return features, labels

class FederatedDataProcessor:
    """联邦数据处理器（最终修复版）"""
    def __init__(self, raw_dir: str, num_clients: int = 4):
        self.raw_dir = raw_dir
        self.num_clients = num_clients
        self.datasets = ["cicids2018.csv", "bot-iot.csv", "ton-iot.csv", "nb15.csv"]
        self.global_features = set()

    def process(self):
        """主处理流程（增强健壮性）"""
        client_root = os.path.join("data", "clients")
        os.makedirs(client_root, exist_ok=True)

        # 第一阶段：处理所有客户端并收集全局特征
        all_features = set()
        success_clients = []
        
        for client_id in range(self.num_clients):
            try:
                self._process_client(client_id, client_root)
                
                # 读取处理后的数据以收集特征
                train_path = os.path.join(client_root, f"client_{client_id}/train.csv")
                if os.path.exists(train_path):
                    df = pd.read_csv(train_path)
                    # 收集所有特征（排除标签列）
                    all_features.update(df.columns.difference([TARGET_COLUMN]))
                
                success_clients.append(client_id)
            except Exception as e:
                logging.error(f"客户端{client_id} 处理失败: {str(e)}", exc_info=True)

        # 更新全局特征集
        self.global_features = all_features
        logging.info(f"全局特征集合: {self.global_features}")
        
        # 第二阶段：对成功处理的客户端进行特征对齐
        self._align_features(client_root, success_clients)
        logging.info("联邦数据处理完成")

    def _process_client(self, client_id: int, client_root: str):
        """处理单个客户端数据（关键修复：使用label列）"""
        # 加载原始数据
        raw_path = os.path.join(self.raw_dir, self.datasets[client_id])
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"原始文件 {raw_path} 不存在")

        df = pd.read_csv(raw_path)
        
        # 基础清洗
        df = df.drop(columns=NON_FEATURE_COLS, errors="ignore")
        df = df.dropna(axis=1, how="all").dropna()

        # 分类特征编码
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                # 使用更健壮的编码方式（处理未知值）
                unique_vals = df[col].astype(str).unique()
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                df[col] = df[col].astype(str).map(mapping).fillna(len(unique_vals))

        # ========== 关键修复：使用label列作为标签 ==========
        # 确保标签列存在
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"数据集缺少目标列 {TARGET_COLUMN}")
        
        # 分离标签列（防止被标准化）
        labels = df[TARGET_COLUMN].copy()
        df = df.drop(columns=[TARGET_COLUMN])
        
        # 转换为整数类型
        labels = labels.astype(int)
        
        # 验证标签值
        unique_labels = labels.unique()
        if not set(unique_labels).issubset({0, 1}):
            invalid = [x for x in unique_labels if x not in (0, 1)]
            raise ValueError(f"非法标签值: {invalid}")
        # ============================================

        # 数值处理（仅特征列）
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            # 修复 FutureWarning：先复制数据，然后进行裁剪
            numeric_data = df[numeric_cols].copy()
            numeric_data = numeric_data.clip(-MAX_FEATURE_VALUE, MAX_FEATURE_VALUE)
            df[numeric_cols] = numeric_data
            
            # 标准化
            df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])

        # 添加回标签列
        df[TARGET_COLUMN] = labels

        # 保存数据
        client_dir = os.path.join(client_root, f"client_{client_id}")
        os.makedirs(client_dir, exist_ok=True)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df.to_csv(os.path.join(client_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(client_dir, "test.csv"), index=False)
        logging.info(f"客户端{client_id} 数据已保存至 {client_dir}")  # 修复：使用 client_id 而非 self.client_id

        # 注册特征
        self.global_features.update(train_df.columns[:-1])

    def _align_features(self, client_root: str, success_clients: list):
        """仅对齐成功处理的客户端"""
        logging.info(f"全局特征集合: {self.global_features}")
        for client_id in success_clients:
            client_dir = os.path.join(client_root, f"client_{client_id}")
            for file in ["train.csv", "test.csv"]:
                path = os.path.join(client_dir, file)
                df = pd.read_csv(path)
                # 添加缺失特征
                # 记录对齐前的特征
                logging.info(f"客户端{client_id} 对齐前特征数: {len(df.columns)}")
                logging.info(f"对齐前列名: {df.columns.tolist()}")
                # 添加缺失特征
                for feat in self.global_features:
                    if feat not in df.columns:
                        df[feat] = 0
                # 确保列顺序一致
                df = df[list(self.global_features) + [TARGET_COLUMN]]
                # 记录对齐后的特征
                logging.info(f"客户端{client_id} 对齐后特征数: {len(df.columns)}")
                logging.info(f"对齐后列名: {df.columns.tolist()}")
                df.to_csv(path, index=False)
        logging.info(f"全局特征对齐完成（有效客户端: {len(success_clients)}）")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="./data/raw", help="原始数据目录路径")
    parser.add_argument("--num_clients", type=int, default=4, help="客户端数量")
    args = parser.parse_args()

    processor = FederatedDataProcessor(raw_dir=args.raw_dir, num_clients=args.num_clients)
    processor.process()