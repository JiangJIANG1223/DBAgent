# 定义了数据库模型HumanSingleCellTrackingTable，该模型与MySQL中的Human_SingleCell_TrackingTable表对应

from sqlalchemy import Column, String, Integer, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
    
class HumanSingleCellTrackingTable(Base):
    __tablename__ = "Your_table_name"
    id = Column(Integer, primary_key=True, autoincrement=True)
    cell_id = Column("cell_id", String(50))
    brain_region = Column("brain_region", String(50))
    dye_name = Column("dye_name", String(50))    ## 新增列
    perfusion_date = Column("perfusion_date", String(50))
    immunohistochemistry = Column("immunohistochemistry", int)
    shooting_date = Column("shooting_date", String(50))
    image_file = Column("image_file", Text)
    v3dpbd_file = Column("v3dpbd_file", Text)
