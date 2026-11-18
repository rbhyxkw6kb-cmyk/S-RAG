from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # 服务配置
    app_name: str = "知源 · 产线作业助手"
    host: str = "127.0.0.1"
    port: int = 8000
    dataset_dir: str = "./dataset"
    knowledge_db: str = "./data/instructions.sqlite3"
    max_pdf_mb: int = Field(default=30, ge=1)
    max_excel_mb: int = Field(default=128, ge=1)
    staff_password: SecretStr | None = None
    admin_password: SecretStr | None = None
    query_max_inflight: int = Field(default=32, ge=1)
    query_max_per_station: int = Field(default=4, ge=1)
    query_worker_threads: int = Field(default=4, ge=1)

    # 嵌入模型
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    embedding_dim: int = 1024
    embedding_batch_size: int = 32
    embedding_device: str = "cuda"
    # BGE 查询指令前缀
    query_instruction: str = "为这个句子生成表示以用于检索相关文章："
    rag_embedding_model: str = ""
    rag_embedding_cache: str = "./data/embedding-cache"
    rag_embedding_path: str | None = None
    rag_embedding_revision: str = "1"
    rag_embedding_threads: int = Field(default=4, ge=1)
    rag_vector_min_score: float = Field(default=0.65, ge=0, le=1)
    rag_candidate_limit: int = Field(default=20, ge=5, le=100)
    rag_reranker_model: str = ""
    rag_reranker_path: str | None = None

    # 重排序模型
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_device: str = "cuda"
    reranker_top_k: int = 5
    reranker_threshold: float = 0.3

    # LLM (vLLM OpenAI 兼容接口)
    llm_base_url: str = "http://localhost:8001/v1"
    llm_model_name: str = ""
    llm_api_key: str = "EMPTY"  # vLLM 默认
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_top_p: float = 0.9
    llm_evidence_only: bool = False
    llm_max_inflight: int = Field(default=8, ge=1)
    llm_requests_per_minute: int = Field(default=0, ge=0)
    llm_queue_seconds: float = Field(default=0.2, ge=0, le=10)
    llm_budget_seconds: float = Field(default=8.0, gt=0)
    llm_connect_seconds: float = Field(default=2.0, gt=0)
    llm_breaker_failures: int = Field(default=3, ge=1)
    llm_breaker_cooldown: float = Field(default=15.0, gt=0)
    deepseek_api_key: SecretStr | None = Field(default=None, validation_alias="DEEPSEEK_API_KEY")

    @property
    def uses_deepseek(self) -> bool:
        from urllib.parse import urlsplit

        return urlsplit(self.llm_base_url).hostname == "api.deepseek.com"

    @property
    def llm_token(self) -> str:
        if self.uses_deepseek:
            return self.deepseek_api_key.get_secret_value().strip() if self.deepseek_api_key else ""
        return self.llm_api_key

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_model_name and self.llm_token)

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "zhiyuan_chunks"
    milvus_index_type: str = "HNSW"
    milvus_metric_type: str = "IP"  # Inner Product, BGE 推荐

    # BM25 (Whoosh)
    whoosh_index_dir: str = "./data/whoosh_index"
    bm25_top_k: int = 20

    # 向量检索
    vector_top_k: int = 20

    # 分块参数
    chunk_size: int = 512
    chunk_overlap: int = 64

    # 对话记忆
    memory_window_size: int = 10
    session_ttl: int = 86400  # 24小时

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # PostgreSQL
    database_url: str = "postgresql://zhiyuan:zhiyuan@localhost:5432/zhiyuan"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", env_prefix="ZHIYUAN_")

settings = Settings()
