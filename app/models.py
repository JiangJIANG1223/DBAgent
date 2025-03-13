# 定义了数据库模型HumanSingleCellTrackingTable，该模型与MySQL中的Human_SingleCell_TrackingTable表对应

from sqlalchemy import Column, String, Integer, Text, Date, TIMESTAMP, func, DateTime, ForeignKey, Enum, UniqueConstraint,Float
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()
    
class HumanSingleCellTrackingTable(Base):
    __tablename__ = "human_singlecell_trackingtable_20240712"
    id = Column(Integer, primary_key=True, autoincrement=True)
    PTRSB = Column("PTRS(B)", String(50))
    cell_id = Column("Cell ID", String(50))
    brain_region = Column("脑区", String(255))
    dye_name = Column("染料名称", String(255))    ## 新增列
    perfusion_date = Column("灌注日期", String(255))
    ihc_category = Column("类别", String(255))  ## 新增
    immunohistochemistry = Column("免疫染色(0:否;1:是)", String(255))
    xy_resolution = Column("xy拍摄分辨率(*10e-3μm/px)", String(255))
    z_resolution = Column("z拍摄分辨率(*10e-3μm/px)", String(255))
    shooting_date = Column("拍摄日期", String(255))
    image_file = Column("image_file", Text)
    v3dpbd_file = Column("v3dpbd_file", Text)
