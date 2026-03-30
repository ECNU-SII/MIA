import os
import json
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict, Union
from qwen_agent.tools.base import BaseTool, register_tool


test_cache_dir=os.environ.get('TEST_CACHE_DIR')

@register_tool("web_image_to_image_search", allow_overwrite=True)
class ImageSearch(BaseTool):
    name = "web_image_to_image_search"
    description = "Searches for relevant images based on the image using web search."
    parameters = {
        'type': 'object',
        'properties': { 
            'img_idx': {
                'type': 'number',
                'description': 'The index of the image (starting from 0)'
            }
        },
        'required': ['img_idx']
    }


    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict]:
        try:
            if isinstance(params, str):
                params = json.loads(params)
            if 'cache_id' not in kwargs:
                raise KeyError("cache_id is missing in kwargs")
            cache_id = kwargs["cache_id"]
            
            # 获取max_images参数，默认为-1（表示返回所有图片）
            max_images = 3
        except Exception as e:
            return "[Image Search] Invalid request format: Input must be provided with 'cache_id' field in kwargs"
        
        tool_returned_str, tool_returned_images, tool_stat = self.call_image_search(cache_id, max_images)
        # 返回一个字典，包含文本和图像结果，符合BaseTool的接口规范
        return {
            "text": tool_returned_str,
            "images": tool_returned_images
        }
        
    def call_image_search(self,cache_id: str, max_images: int = -1):
        """
        基于cache数据的图像搜索工具。
        
        根据输入的id从fvqa_train_cache或fvqa_test_cache文件夹中读取相应的搜索结果数据，
        包括图片和标题信息。

        Args:
            image_url (str): 查询图像的URL或内部标识符（当前版本中未使用）
            cache_id (str): 查询ID，对应cache文件夹中的子文件夹名称

        Returns:
            tool_returned_str (str): 格式化的图像搜索结果字符串
            tool_returned_images (List[PIL.Image.Image]): 搜索结果图片列表
            tool_stat (dict): 工具执行状态和元数据
        """
        
        # 初始化返回值
        tool_returned_images = []
        tool_returned_str = ""
        tool_success = False
        
        try:
            # 从环境变量读取cache路径，如果没有设置则使用默认路径
            test_cache_dirs_raw = test_cache_dir or "/fvqa_test_cache"
            
            # 确定cache类型和路径
            test_cache_dirs = [p.strip() for p in test_cache_dirs_raw.split(',') if p.strip()]
            cache_path = None
            for base_dir in test_cache_dirs:
                candidate_path = os.path.join(base_dir, cache_id)
                if os.path.exists(candidate_path):
                    cache_path = candidate_path
                    break
            
            # 检查cache文件夹是否存在
            if cache_path is None:
                raise FileNotFoundError(f"Cache文件夹 {cache_path} 不存在")
            
            # 读取meta.json文件
            meta_file = os.path.join(cache_path, "meta.json")
            if not os.path.exists(meta_file):
                raise FileNotFoundError(f"Meta文件 {meta_file} 不存在")
            
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            
            # 获取标题列表和图片URL列表
            # 首先尝试新格式，如果失败则使用旧格式
            if "search_results" in meta_data:
                # 新格式：从search_results中提取标题和图片URL
                title_list = [item.get("title", "") for item in meta_data.get("search_results", [])]
                image_urls = meta_data.get("image_urls", [])
            else:
                # 旧格式：直接获取title_list和image_urls
                title_list = meta_data.get("title_list", [])
                image_urls = meta_data.get("image_urls", [])
            
            # 构建返回字符串
            tool_returned_str = ""
            
            # 限制读取的图片数量
            if max_images > 0:
                title_list = title_list[:max_images]
                image_urls = image_urls[:max_images]
            
            # 读取图片文件并添加到返回列表
            for i, (title, img_url) in enumerate(zip(title_list, image_urls)):
                # 构建图片文件名
                img_filename = f"img_{i:03d}.jpg"
                img_path = os.path.join(cache_path, img_filename)
                
                # 如果jpg不存在，尝试png
                if not os.path.exists(img_path):
                    img_filename = f"img_{i:03d}.png"
                    img_path = os.path.join(cache_path, img_filename)
                    
                    # 如果png也不存在，尝试webp
                    if not os.path.exists(img_path):
                        img_filename = f"img_{i:03d}.webp"
                        img_path = os.path.join(cache_path, img_filename)
                
                # 读取图片
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                        tool_returned_images.append(img)
                        # 添加到返回字符串
                        tool_returned_str += f"{i+1}. title: {title}\n <image>"
                    except Exception as e:
                        print(f"读取图片 {img_path} 时出错: {e}")
                        # 如果图片读取失败，创建占位符图片
                        dummy_img = Image.fromarray(np.full((64, 64, 3), fill_value=100 + i * 30, dtype=np.uint8))
                        tool_returned_images.append(dummy_img)
                        tool_returned_str += f"{i+1}.title: {title}\n"
                else:
                    # 如果图片文件不存在，创建占位符图片
                    dummy_img = Image.fromarray(np.full((64, 64, 3), fill_value=100 + i * 30, dtype=np.uint8))
                    tool_returned_images.append(dummy_img)
                    tool_returned_str += f"{i+1}. title: {title}\n"
            
            tool_success = True
            
        except Exception as e:
            print(f"图像搜索工具执行出错: {e}")
            tool_returned_str = "[Image Search Results] There is an error encountered in performing search. Please reason with your own capaibilities."
            tool_returned_images = []
            tool_success = False
        
        # 构建工具状态信息
        tool_stat = {
            "success": tool_success,
            "num_images": len(tool_returned_images),
            "requested_max_images": max_images,
            "cache_id": cache_id,
            "cache_path": cache_path if 'cache_path' in locals() else None,
        }
        
        return tool_returned_str, tool_returned_images, tool_stat
    
    
