import os
import subprocess
import cv2
import base64
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from . import models, schemas
from typing import List, Union, Optional
import csv
import shutil

### Database operations(CRUD)
def insert_data_from_csv(csv_path: str, db: Session) -> str:
    """
    Reads a local CSV file and inserts its rows into the HumanSingleCellTrackingTable.
    Returns a success message or error message.
    
    CSV 内容示例（需与你的表结构对应）:
    cell_id, brain_region, dye_name, immunohistochemistry, image_file
    12345, "MFG", "Alexa488", "Synaptophysin", "/path/to/12345.tif"
    ...
    """
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                # 对 row 做一些预处理，比如 cell_id zfill(5)
                row["cell_id"] = row["cell_id"].zfill(5) 

                # 1) 使用 row 创建一个 Pydantic 模型的实例
                create_data = schemas.HumanSingleCellTrackingTableCreate(**row)

                # 2) 将这个 Pydantic 实例转为 dict，传给 SQLAlchemy 模型
                new_record = models.HumanSingleCellTrackingTable(**create_data.dict())

                db.add(new_record)
                count += 1
            db.commit()
        return f"Inserted {count} rows into the database from {csv_path}"
    except FileNotFoundError:
        return f"Error: CSV file not found at {csv_path}"
    except SQLAlchemyError as e:
        db.rollback()
        return f"Error inserting data into database: {str(e)}"
    except Exception as e:
        db.rollback()
        return f"Unexpected error during CSV insertion: {str(e)}"

def retrieve_data(filter_field: str, filter_value: str, db: Session) -> Union[List[models.HumanSingleCellTrackingTable], str]:
    """
    查询 HumanSingleCellTrackingTable 中符合条件的记录。
    
    参数:
      filter_field: 用于过滤的字段名称，比如 "cell_id", "brain_region", "dye_name", 等
      filter_value: 用于匹配的值
      db: SQLAlchemy 的 Session 对象
      
    返回:
      成功时返回符合条件的记录列表，如果没有找到则返回提示信息；
      发生错误时返回错误提示信息。
    """
    try:
        field_mapping = {
            "cell_id": models.HumanSingleCellTrackingTable.cell_id,
            "brain_region": models.HumanSingleCellTrackingTable.brain_region,
            "dye_name": models.HumanSingleCellTrackingTable.dye_name,
            "immunohistochemistry": models.HumanSingleCellTrackingTable.immunohistochemistry,
        }
        if filter_field not in field_mapping:
            return f"Error: Unrecognized filter field '{filter_field}'"

        filter_column = field_mapping[filter_field]
        rows = db.query(models.HumanSingleCellTrackingTable).filter(filter_column == filter_value).all()

        if not rows:
            return f"No records found where {filter_field} = '{filter_value}'."

        return rows
    except SQLAlchemyError as e:
        db.rollback()
        return f"Error querying data: {str(e)}"

def update_data(filter_field: str, filter_value: str, target_field: str, new_value: str, db: Session) -> str:
    """
    Updates the 'target_field' to 'new_value' for all rows where (filter_field == filter_value).
    e.g., change dye_name => 'Cascade Blue' for cell_id = '12345'
    
    filter_field: 'cell_id', 'brain_region', 'dye_name', etc.
    filter_value: the old value to match
    target_field: which field we want to update
    new_value: the new value we want to assign
    """
    try:
        field_mapping = {
            "cell_id": models.HumanSingleCellTrackingTable.cell_id,
            "brain_region": models.HumanSingleCellTrackingTable.brain_region,
            "dye_name": models.HumanSingleCellTrackingTable.dye_name,
            "immunohistochemistry": models.HumanSingleCellTrackingTable.immunohistochemistry,
            "image_file": models.HumanSingleCellTrackingTable.image_file
        }
        if filter_field not in field_mapping or target_field not in field_mapping:
            return "Error: Unrecognized filter_field or target_field."

        # Get the SQLAlchemy column objects
        filter_column = field_mapping[filter_field]
        target_column = field_mapping[target_field]
        print("filter_column: ", filter_column)
        print("target_column: ", target_column)

        rows = db.query(models.HumanSingleCellTrackingTable).filter(filter_column == filter_value).all()
        if not rows:
            return f"No records found where {filter_field} = '{filter_value}'."

        for row in rows:
            setattr(row, target_field, new_value)  # row.target_field = new_value
        db.commit()

        print("DEBUG - db connection:", db.bind.engine)

        # 立即再次查询确认
        rows_after = db.query(models.HumanSingleCellTrackingTable).filter(filter_column == filter_value).all()
        for r in rows_after:
            print("DEBUG after update => cell_id:", r.cell_id, "dye_name:", r.dye_name)

        return f"Successfully updated {len(rows)} record(s), set {target_field}='{new_value}' where {filter_field}='{filter_value}'."
    except SQLAlchemyError as e:
        db.rollback()

        return f"Error updating data: {str(e)}"

def delete_data(field: str, value: str, db: Session) -> str:
    """
    Deletes records from the table matching (field == value).
    e.g., field="cell_id", value="12345"  OR  field="brain_region", value="MFG"
    Returns a success or failure message.
    """
    try:
        field_mapping = {
            "cell_id": models.HumanSingleCellTrackingTable.cell_id,
            "brain_region": models.HumanSingleCellTrackingTable.brain_region,
            "dye": models.HumanSingleCellTrackingTable.dye_name,
            "immunohistochemistry": models.HumanSingleCellTrackingTable.immunohistochemistry,
        }
        if field not in field_mapping:
            return f"Error: Unrecognized field '{field}'"
        
        query_column = field_mapping[field]
        rows = db.query(models.HumanSingleCellTrackingTable).filter(query_column == value).all()
        
        if not rows:
            return f"No records found where {field} = '{value}'."

        for row in rows:
            db.delete(row)
        db.commit()

        return f"Successfully deleted {len(rows)} record(s) where {field} = '{value}'."
    except SQLAlchemyError as e:
        db.rollback()
        return f"Error deleting data: {str(e)}"


### image process and analysis
vaa3d_exe = "Path\\to\\your\\Vaa3D-x.exe"

# 定义输出目录并确保其存在
output_dir = '.\\results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def _make_subdir(pipeline_dir: Optional[str], stage_name: str) -> str:
    """
    根据 pipeline_dir + stage_name 生成并返回子目录，如:
      results/pipeline_20250101_2026/preprocessing
    如果 pipeline_dir=None，则默认使用 ./results
    """
    if pipeline_dir:
        sub_dir = os.path.join(pipeline_dir, stage_name)
    else:
        sub_dir = os.path.join('.', 'results', stage_name)

    os.makedirs(sub_dir, exist_ok=True)
    return sub_dir

def search_by_id(cell_id: str, db: Session) -> str:
    ''' 
    This function searches MIP path by cell_id.       
    Returns a string of file path or error message.
    '''
    cell_id = cell_id.zfill(5)
    try:
        ## 查询数据库
        image_path = db.query(models.HumanSingleCellTrackingTable.v3dpbd_file).filter(
            models.HumanSingleCellTrackingTable.cell_id == cell_id
        ).first()
        
        if image_path:
            # .first() 返回的是一个元组或单列对象，需要加 [0] 获取真正的字符串
            return image_path[0]
        else:
            return f"No image found for cell_id: {cell_id}"

    except SQLAlchemyError as e:
        # 捕获数据库查询异常并返回错误信息
        print(f"Database error while searching for cell_id {cell_id}: {e}")
        return "An error occurred while searching for the image. Please try again."

def search_by_criteria(field: str, value: str, db: Session) -> List[str]:
    """
    Searches multiple MIP paths by a given field and value.
    E.g. field='brain_region', value='MFG' or field='dye', value='Alexa488'.
    Returns a list of file paths or an empty list if none found.
    """
    try:
        # 1) 根据 field 动态选择要查询的列
        field_mapping = {
        "brain_region": models.HumanSingleCellTrackingTable.brain_region,
        "dye": models.HumanSingleCellTrackingTable.dye_name,
        "immunohistochemistry": models.HumanSingleCellTrackingTable.immunohistochemistry,
        }
        if field not in field_mapping:
            return [f"Error: Unrecognized field '{field}'"]
        
        query_column = field_mapping[field]

        # 2) 查询数据库，可能返回多条
        rows = db.query(models.HumanSingleCellTrackingTable)\
         .with_entities(models.HumanSingleCellTrackingTable.v3dpbd_file)\
         .filter(query_column == value)\
         .all()
        
        if not rows:
            return [f"No images found for {field}='{value}'"]
        
        file_paths = [row.image_file for row in rows]
        
        return file_paths
    
    except SQLAlchemyError as e:
        print(f"Database error while searching for {field}={value}: {e}")
        return [f"An error occurred while searching for the images. Please try again."]

def display_images(image_path_or_paths: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    读取一个或多个 .tif 图像文件，将其缩略后转换为 PNG 格式编码为 Base64，
    并返回 Markdown 格式的图像标签，使其可在对话框中直接显示缩略图。
    
    参数:
      image_path_or_paths: 单个图像路径 (str) 或图像路径列表 (List[str])
    
    返回:
      如果传入单个路径，则返回一个 Markdown 格式的图像标签字符串；
      如果传入列表，则返回多个图像标签组成的列表。
    """
    def convert_tif_to_markdown(image_path: str) -> str:
        # 读取 .tif 图像
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return f"Error: Cannot load image at {image_path}"
        
        # 获取原图尺寸
        height, width = img.shape[:2]
        max_dim = 90 # 设置最大边长为200像素
        # 计算缩放比例，保持长宽比
        scale = max_dim / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # 缩放图像
        thumbnail = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # 将缩略图编码为 PNG 格式（在内存中）
        success, buffer = cv2.imencode('.png', thumbnail)
        if not success:
            return f"Error: Failed to encode image at {image_path}"
        
        # 将编码后的数据转换为 Base64 字符串
        b64_str = base64.b64encode(buffer).decode('utf-8')
        
        # 返回 Markdown 格式的图像标签
        return f"![Image](data:image/png;base64,{b64_str})"
    
    if isinstance(image_path_or_paths, str):
        return convert_tif_to_markdown(image_path_or_paths)
    elif isinstance(image_path_or_paths, list):
        return [convert_tif_to_markdown(path) for path in image_path_or_paths]
    else:
        return "Error: Invalid input type for display_image. Must be str or list."

def preprocess_image(image_path_or_paths: Union[str, List[str]],
                     pipeline_dir: Optional[str] = None) -> Union[str, List[str]]:
    """
    Preprocess one or multiple images using the Vaa3D imPreProcess plugin.
    If pipeline_dir is provided, outputs will be saved to pipeline_dir/preprocessing/,
    otherwise, outputs will be saved to ./results/preprocessing/.
    """
    # 1) Create the output subdirectory
    sub_dir = _make_subdir(pipeline_dir, "preprocessing")

    def _preprocess_single_image(image_path: str) -> str:
        base_name = os.path.basename(image_path)
        name_no_ext, ext = os.path.splitext(base_name)
        output_file_name = f"{name_no_ext}_preprocessed{ext}"
        output_path = os.path.join(sub_dir, output_file_name)
        # Build the command to call the Vaa3D plugin
        cmd = [
            "Vaa3D-x.exe", "/x", "imPreProcess", "/f", "im_enhancement",
            "/i", image_path, "/o", output_path,
            "/p", "3", "1", "35", "3", "25", "1", "1"
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            return f"Error: Vaa3D plugin pre-processing failed for {image_path}: {str(e)}"
        return output_path

    # 判断输入是单个字符串还是列表
    if isinstance(image_path_or_paths, str):
        return _preprocess_single_image(image_path_or_paths)
    elif isinstance(image_path_or_paths, list):
        results = []
        for p in image_path_or_paths:
            results.append(_preprocess_single_image(p))
        return results
    else:
        return "Error: Invalid input type for preprocess_image (must be str or list)."

def auto_tracing(image_path_or_paths: Union[str, List[str]],
                 pipeline_dir: Optional[str] = None
                 ) -> Union[str, List[str]]:
    """
    Performs automatic tracing on one or multiple images using Vaa3D, generating SWC files.
    Returns the path(s) of the generated SWC file(s).
    Auto tracing, results go to pipeline_dir/auto_tracing/ if pipeline_dir is specified,
    else ./results/auto_tracing/.
    """
    sub_dir = _make_subdir(pipeline_dir, "auto_tracing")

    def _trace_single_image(image_path: str) -> str:
        tif_filename = os.path.splitext(os.path.basename(image_path))[0]
        swc_filename = tif_filename + "_app2.swc"
        swc_path = os.path.join(sub_dir, swc_filename)
        
        cmd = [vaa3d_exe, "/x", "vn2", "/f", "app2", "/i", image_path, "/o", swc_path]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running Vaa3D: {e}")
            return f"Error: Vaa3D auto-tracing failed for {image_path}"

        return swc_path

    if isinstance(image_path_or_paths, str):
        return _trace_single_image(image_path_or_paths)
    elif isinstance(image_path_or_paths, list):
        return [ _trace_single_image(path) for path in image_path_or_paths ]
    else:
        return "Error: Invalid input type for auto_tracing."

def postprocess_results(swc_path_or_paths: Union[str, List[str]],
                        pipeline_dir: Optional[str] = None
                        ) -> Union[str, List[str]]:
    """
    Performs resampling on one or multiple SWC files using the Vaa3D resample_swc plugin.
    Returns the path(s) of the resampled SWC file(s).
    """
    sub_dir = _make_subdir(pipeline_dir, "postprocess")

    def _postprocess_single_swc(swc_path: str) -> str:
        swc_filename = os.path.splitext(os.path.basename(swc_path))[0]
        resampled_swc_filename = swc_filename + "_resample.swc"
        resampled_swc_path = os.path.join(sub_dir, resampled_swc_filename)

        cmd = [vaa3d_exe, "/x", "resample_swc", "/f", "resample_swc", "/i", swc_path, "/o", resampled_swc_path, "/p", "3"]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running Vaa3D: {e}")
            return f"Error: Vaa3D postprocessing failed for {swc_path}"

        return resampled_swc_path

    if isinstance(swc_path_or_paths, str):
        return _postprocess_single_swc(swc_path_or_paths)
    elif isinstance(swc_path_or_paths, list):
        results = []
        for path in swc_path_or_paths:
            result = _postprocess_single_swc(path)
            results.append(result)
        return results
    else:
        return "Error: Invalid input type for postprocess_results (must be str or list)."

def _parse_single_swc_stdout(raw_stdout: str) -> str:
    """
    从 compute_feature 命令的标准输出 (raw_stdout) 中，提取或整理想要展示的特征内容。
    仅截取从 '--------------Neuron #1----------------' 到 'Hausdorff Dimension:' 所在行。
    """
    lines = raw_stdout.splitlines()
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if "--------------Neuron #1----------------" in line:
            start_idx = i
        if "Hausdorff Dimension:" in line:
            end_idx = i
    
    if start_idx is not None and end_idx is not None and end_idx >= start_idx:
        relevant_lines = lines[start_idx:end_idx+1]
        return "\n".join(relevant_lines)
    else:
        # 如果找不到对应范围，就返回全部 stdout 或提示
        return raw_stdout

def extract_features(swc_path_or_paths: Union[str, List[str]],
                     pipeline_dir: Optional[str] = None
                     ) -> Union[str, List[str]]:
    """
    单文件场景: 
      - 执行 'compute_feature' 命令, 不生成CSV, 捕获stdout并解析其中的特征信息返回对话.
    多文件场景: 
      - 执行 'compute_feature_in_folder' 命令, 生成一个合并CSV, 返回CSV路径.
    """
    sub_dir = _make_subdir(pipeline_dir, "extract_features")

    # 1) 单个SWC
    if isinstance(swc_path_or_paths, str):
        # 命令: compute_feature, 不加 /o 参数, 我们只捕获stdout
        cmd = [
            vaa3d_exe, "/x", "global_neuron_feature", "/f", "compute_feature",
            "/i", swc_path_or_paths
        ]
        try:
            # capture_output=True, text=True 用于捕获命令行输出并以字符串形式获取
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            raw_output = result.stdout
        except subprocess.CalledProcessError as e:
            return f"Error: feature extraction failed for {swc_path_or_paths}, {str(e)}"
        
        # 解析输出
        parsed_info = _parse_single_swc_stdout(raw_output)
        return parsed_info  # 直接把解析结果返回给对话框

    # 2) 多个SWC
    elif isinstance(swc_path_or_paths, list):
        # 将所有swc复制(或移动/硬链接)到sub_dir，再用compute_feature_in_folder
        for path in swc_path_or_paths:
            if not os.path.isfile(path):
                return f"Error: file not found => {path}"
            shutil.copy(path, sub_dir)  # 拷贝到子目录
        
        # 统一生成一个CSV
        batch_csv = os.path.join(sub_dir, "batch_features.csv")
        cmd = [
            vaa3d_exe, "/x", "global_neuron_feature", "/f", "compute_feature_in_folder",
            "/i", sub_dir, "/o", batch_csv
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            return f"Error: batch feature extraction failed in folder {sub_dir}, {str(e)}"
        
        return batch_csv  # 返回合并CSV的路径

    else:
        return "Error: Invalid input type for extract_features (must be str or list)."


# 可以用一个字典映射名称 -> 函数
TOOLS_REGISTRY = {
    "search_by_id": search_by_id,
    "search_by_criteria": search_by_criteria, 
    "display_images": display_images,
    "preprocess_image": preprocess_image,
    "auto_tracing": auto_tracing,
    "postprocess_results": postprocess_results,
    "extract_features": extract_features,
    "insert_data_from_csv": insert_data_from_csv,
    "retrieve_data": retrieve_data,
    "update_data": update_data,
    "delete_data": delete_data,
}
