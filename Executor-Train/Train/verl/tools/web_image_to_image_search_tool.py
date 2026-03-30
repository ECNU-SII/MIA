import os
import json
from typing import Any, Optional, Tuple
from PIL import Image
import numpy as np
from uuid import uuid4
import re
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse


class WebImageToImageSearchTool(BaseTool):
    """Web image to image search tool.
    
    This tool performs image search based on cache data, reading search result data
    including images and title information from cache folders.
    """

    def __init__(self, config: dict = None, tool_schema: OpenAIFunctionToolSchema = None):
        # Define the tool schema if not provided
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema(
                type="function",
                function={
                    "name": "web_image_to_image_search",
                    "description": "Searches for relevant images and title based on the original image using web search.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            )
        
        super().__init__(config or {}, tool_schema)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the tool schema in OpenAI format."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> Tuple[str, ToolResponse]:
        """Create a tool instance.
        
        Args:
            instance_id: The instance id of the tool.
            **kwargs: May contain 'create_kwargs' with tool-specific parameters
            
        Returns:
            The instance id of the tool and tool creation response.
        """
        if instance_id is None:
            instance_id = str(uuid4())
            
        # Handle create_kwargs parameter if passed
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)
        
        cache_id = create_kwargs.get("ids")
        if cache_id is None:
            raise ValueError("Missing required 'ids' parameter in create_kwargs")
            
        # Store any provided parameters for later use in execute
        if not hasattr(self, '_instance_dict'):
            self._instance_dict = {}
        self._instance_dict[instance_id] = {
            "cache_id": cache_id,
            "response": "",
            "reward": 0.0,
        }
        return instance_id, ToolResponse()
    
    

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[ToolResponse, float, dict]:
        """Execute the image search tool.
        
        Args:
            instance_id: The instance id of the tool.
            parameters: The parameters of the tool (not used in this implementation).
            
        Returns:
            ToolResponse containing images and titles, reward score, and metrics.
        """
        # Get parameters from instance data
        instance_data = {}
        if hasattr(self, '_instance_dict') and instance_id in self._instance_dict:
            instance_data = self._instance_dict[instance_id]
            
        # Get cache_id from instance data, or generate a default one
        cache_id = instance_data.get("cache_id")
        try:
            # Call the image search function
            tool_returned_str, tool_returned_images, tool_stat = self.call_image_search(cache_id)
            
            # Create ToolResponse with images and text
            response = ToolResponse(
                text=tool_returned_str,
                image=tool_returned_images if tool_returned_images else None
            )
            
            reward = 0.0
            
            return response, reward, tool_stat
            
        except Exception as e:
            error_msg = f"[Image Search Results] Error executing image search: {str(e)}"
            return (
                ToolResponse(text=error_msg),
                -0.1,
                {"success": False, "error": str(e)}
            )
            
        
    def _open_image_safe(self, img_path: str) -> Image.Image:
        """Safely open an image file supporting jpg, png, webp."""
        try:
            img = Image.open(img_path)
            img.load()
            return img
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            return Image.fromarray(np.full((64, 64, 3), fill_value=128, dtype=np.uint8))


    def _normalize_cache_id(self, cache_id: str) -> str:
        """
        将形如 '0_20250913190519239546_level2_2' 的 cache_id 规范化为
        '0_20250913190519239546'。若是旧格式(以 fvqa_train_/fvqa_test_ 开头)或匹配不到则原样返回。
        """
        if cache_id.startswith("fvqa_train_") or cache_id.startswith("fvqa_test_") or cache_id[-5] == '_' or cache_id.startswith("0_2025112"):
            return cache_id
        m = re.match(r'^(\d+_\d+)', cache_id)  # 提取开头连续两段数字
        return m.group(1) if m else cache_id
    
    
    def call_image_search(self, cache_id: str, max_images: int = -1):
        """
        基于cache数据的图像搜索工具（跳过HTML文件，并Resize到448x448）。
        
        根据输入的id从fvqa_train_cache或fvqa_test_cache文件夹中读取相应的搜索结果数据。
        如果条目是HTML文件，则跳过，继续查找下一个条目，直到满足max_images。
        返回的图片均被resize到 448x448。

        Args:
            cache_id (str): 查询ID，对应cache文件夹中的子文件夹名称
            max_images (int): 限制返回的最大图片数量，-1表示不限制

        Returns:
            tool_returned_str (str): 格式化的图像搜索结果字符串
            tool_returned_images (List[PIL.Image.Image]): 搜索结果图片列表 (448x448)
            tool_stat (dict): 工具执行状态和元数据
        """
        # 初始化返回值
        tool_returned_images = []
        tool_returned_str = ""
        tool_success = False
        cache_path = None
        test_cache_base = None
        
        # --- vvvv 新增：定义目标尺寸 vvvv ---
        target_size = (448, 448)
        # --- ^^^^ 新增 ^^^^ ---

        try:
            # (前面的配置、路径检查、meta.json读取代码保持不变...)
            # ... [省略前面的代码] ...

            # 确定cache类型和路径
            cache_id = self._normalize_cache_id(cache_id)
            fvqa_train_cache_base = self.config.get("fvqa_train_cache_path", "fvqa_train_cache")
            all_train_cache_base = self.config.get("all_train_cache_path", "fvqa_train_cache")
            test_cache_base = self.config.get("test_cache_path", "fvqa_test_cache")
            new_cache_path = self.config.get("new_cache_path", "fvqa_train_cache")
            sim_cache_path = self.config.get("sim_cache_path", "fvqa_train_cache")
            default_max_images = self.config.get("max_images", 3)

            if max_images <= 0:
                max_images = default_max_images

            if cache_id.startswith("fvqa_train_"):
                cache_path = os.path.join(fvqa_train_cache_base, cache_id)
            elif cache_id.startswith("fvqa_test_"):
                cache_path = os.path.join(test_cache_base, cache_id)
            elif cache_id.startswith("0_2025112"):#新增的2000sim
                cache_path = os.path.join(sim_cache_path, cache_id)
            elif cache_id[-5] == '_':#新增
                cache_path = os.path.join(new_cache_path, cache_id)
            else:#新增的0_2025109
                cache_path = os.path.join(all_train_cache_base, cache_id)

            if not os.path.exists(cache_path):
                raise FileNotFoundError(f"Cache文件夹 {cache_path} 不存在")

            meta_file = os.path.join(cache_path, "meta.json")
            if not os.path.exists(meta_file):
                raise FileNotFoundError(f"Meta文件 {meta_file} 不存在")

            with open(meta_file, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)

            if "search_results" in meta_data:
                title_list = [item.get("title", "") for item in meta_data.get("search_results", [])]
                image_urls = [item.get("image_url", "") for item in meta_data.get("search_results", [])]
            else:
                title_list = meta_data.get("title_list", [])
                image_urls = meta_data.get("image_urls", [])

            
            tool_returned_str = "[Image Search Results]: \n"
            collected_images_count = 0 

            # 遍历读取图片
            for i, (title, img_url) in enumerate(zip(title_list, image_urls)):
                title = title.replace("<image>", "")
                # 1. 检查是否为HTML文件
                if cache_id.startswith("0_2025112"):#因为图像是从001开始
                    html_filename = f"img_{i+1:03d}.html"
                else:
                    html_filename = f"img_{i:03d}.html"
                html_path = os.path.join(cache_path, html_filename)
                
                if os.path.exists(html_path):
                    continue # 跳过此条目

                # 2. (如果不是HTML) 正常尝试查找图片
                img_found = False
                for ext in [".jpg", ".png", ".webp", ".gif"]:
                    if cache_id.startswith("0_2025112"):#因为图像是从001开始
                        img_filename = f"img_{i+1:03d}{ext}"
                    else:
                        img_filename = f"img_{i:03d}{ext}"
                    img_path = os.path.join(cache_path, img_filename)
                    if os.path.exists(img_path):
                        try:
                            img = self._open_image_safe(img_path)
                            
                            # --- vvvv 修改：调整尺寸 vvvv ---
                            # 使用高质量的LANCZOS/ANTIALIAS滤镜进行缩放
                            img = img.resize(target_size, Image.LANCZOS) 
                            # --- ^^^^ 修改 ^^^^ ---
                            
                            tool_returned_images.append(img)
                            tool_returned_str += f"{collected_images_count + 1}. title: {title}\n<image>\n"
                            img_found = True
                            break
                        except Exception as e:
                            print(f"读取图片 {img_path} 时出错: {e}")
                            break

                # 3. 如果没找到对应文件（但也不是HTML），则创建占位图
                if not img_found:
                    
                    # --- vvvv 修改：调整占位图尺寸 vvvv ---
                    dummy_array = np.full((target_size[1], target_size[0], 3), fill_value=100 + i * 30, dtype=np.uint8)
                    dummy_img = Image.fromarray(dummy_array)
                    # --- ^^^^ 修改 ^^^^ ---
                    
                    tool_returned_images.append(dummy_img)
                    tool_returned_str += f"{collected_images_count + 1}. title: {title}\n<image>\n"
                
                # 4. 增加计数器
                collected_images_count += 1

                # 5. 检查是否达到数量限制
                if max_images != -1 and collected_images_count >= max_images:
                    break 

            tool_success = True

        except Exception as e:
            print(f"图像搜索工具执行出错: {e}")
            tool_returned_str = "[Image Search Results] There is an error encountered in performing search. Please reason with your own capabilities."
            tool_returned_images = []
            tool_success = False

        # 构建工具状态信息
        tool_stat = {
            "success": tool_success,
            "num_images": len(tool_returned_images),
            "requested_max_images": max_images,
            "cache_id": cache_id,
            "cache_path": cache_path if cache_path else None,
            "test_cache_base": test_cache_base if test_cache_base else None,
        }

        return tool_returned_str, tool_returned_images, tool_stat
    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance.
        
        Args:
            instance_id: The instance id of the tool.
        """
        if hasattr(self, '_instance_dict') and instance_id in self._instance_dict:
            del self._instance_dict[instance_id]