# # 定义了Pydantic模式，用于数据验证和序列化
# from pydantic import BaseModel
# from typing import Optional

# # 定义一个基础模型 HumanSingleCellTrackingTableBase，用于存储公共字段
# class HumanSingleCellTrackingTableBase(BaseModel):
#     PTRSB: Optional[str] = None
#     cell_id: str
#     brain_region: Optional[str] = None
#     dye_name: Optional[str] = None    ## 新增列
#     perfusion_date: str
#     ihc_category: Optional[str] = None
#     immunohistochemistry: Optional[str] = None
#     xy_resolution: Optional[str] = None
#     z_resolution: Optional[str] = None
#     shooting_date: Optional[str] = None
#     image_file: Optional[str] = None
#     v3dpbd_file: Optional[str] = None

# # 定义一个创建模型 HumanSingleCellTrackingTableCreate，继承 HumanSingleCellTrackingTableBase
# # 这个模型用于创建新的单细胞数据
# class HumanSingleCellTrackingTableCreate(HumanSingleCellTrackingTableBase):
#     pass

# # 定义一个响应模型 HumanSingleCellTrackingTable，继承 HumanSingleCellTrackingTableBase
# # 这个模型用于返回数据库中的单细胞数据
# class HumanSingleCellTrackingTable(HumanSingleCellTrackingTableBase):
#     id: int

#     class Config:
#         from_attributes = True
#         # orm_mode = True    # 启用 ORM 模式，允许 Pydantic 模型与 SQLAlchemy 模型进行交互
