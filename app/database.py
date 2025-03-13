# 定义了数据库连接和会话管理

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

## 设置 URL 来指定数据库连接
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:your_password@localhost/database_name"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)