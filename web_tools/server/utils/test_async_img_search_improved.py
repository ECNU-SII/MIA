import asyncio
import time
from img_search import call_ddgs_image_search

async def test_async_image_search():
    """测试异步图片搜索功能"""
    print("开始测试异步图片搜索...")
    start_time = time.time()
    
    try:
        # 测试搜索猫咪图片
        result_text, images, stats = await call_ddgs_image_search(
            query="cats",
            max_results=3,
            region="cn-en",
            safesearch="off"
        )
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000  # 转换为毫秒
        
        print(f"搜索完成，总耗时: {total_time:.2f}ms")
        print(f"返回结果统计: {stats}")
        print(f"获取到 {len(images)} 张图片")
        print(f"结果预览:\n{result_text[:500]}...")
        
        # 验证图片是否正确加载
        for i, img in enumerate(images):
            if img:
                print(f"图片 {i+1}: 尺寸 {img.size}, 模式 {img.mode}")
            else:
                print(f"图片 {i+1}: 加载失败")
                
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_async_image_search())