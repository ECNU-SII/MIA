import asyncio
import time
from server.utils.img_search import _fetch_image_robust
import httpx

async def test_concurrent_downloads():
    """测试并发下载功能"""
    print("开始测试并发下载功能...")
    
    # 使用一些公开的图片URL进行测试
    test_urls = [
        {"image": "https://httpbin.org/image/jpeg", "thumbnail": None, "source": None},
        {"image": "https://httpbin.org/image/png", "thumbnail": None, "source": None},
        {"image": "https://httpbin.org/image/svg", "thumbnail": None, "source": None},
    ]
    
    start_time = time.time()
    
    async with httpx.AsyncClient() as client:
        # 创建下载任务
        download_tasks = []
        for url_data in test_urls:
            task = _fetch_image_robust(
                client,
                image_url=url_data["image"],
                thumb_url=url_data["thumbnail"],
                source_page=url_data["source"],
                timeout=15.0,
            )
            download_tasks.append(task)
        
        # 并发执行所有下载任务
        print(f"开始并发下载 {len(download_tasks)} 个图片...")
        image_results = await asyncio.gather(*download_tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000  # 转换为毫秒
        
        # 处理结果
        successful_downloads = 0
        for i, result in enumerate(image_results):
            if isinstance(result, Exception):
                print(f"图片 {i+1} 下载失败: {result}")
            elif result is None:
                print(f"图片 {i+1} 下载失败: 返回空结果")
            else:
                print(f"图片 {i+1} 下载成功: 尺寸 {result.size}, 模式 {result.mode}")
                successful_downloads += 1
        
        print(f"\n并发下载测试完成:")
        print(f"总耗时: {total_time:.2f}ms")
        print(f"成功下载: {successful_downloads}/{len(test_urls)}")
        print(f"平均每个图片下载时间: {total_time/max(len(test_urls), 1):.2f}ms")

async def test_sequential_downloads():
    """测试串行下载功能作为对比"""
    print("\n开始测试串行下载功能...")
    
    # 使用相同的URL进行测试
    test_urls = [
        {"image": "https://httpbin.org/image/jpeg", "thumbnail": None, "source": None},
        {"image": "https://httpbin.org/image/png", "thumbnail": None, "source": None},
        {"image": "https://httpbin.org/image/svg", "thumbnail": None, "source": None},
    ]
    
    start_time = time.time()
    
    async with httpx.AsyncClient() as client:
        # 串行下载图片
        image_results = []
        for url_data in test_urls:
            result = await _fetch_image_robust(
                client,
                image_url=url_data["image"],
                thumb_url=url_data["thumbnail"],
                source_page=url_data["source"],
                timeout=15.0,
            )
            image_results.append(result)
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000  # 转换为毫秒
        
        # 处理结果
        successful_downloads = 0
        for i, result in enumerate(image_results):
            if isinstance(result, Exception):
                print(f"图片 {i+1} 下载失败: {result}")
            elif result is None:
                print(f"图片 {i+1} 下载失败: 返回空结果")
            else:
                print(f"图片 {i+1} 下载成功: 尺寸 {result.size}, 模式 {result.mode}")
                successful_downloads += 1
        
        print(f"\n串行下载测试完成:")
        print(f"总耗时: {total_time:.2f}ms")
        print(f"成功下载: {successful_downloads}/{len(test_urls)}")
        print(f"平均每个图片下载时间: {total_time/max(len(test_urls), 1):.2f}ms")

if __name__ == "__main__":
    print("=== 异步并发下载性能测试 ===")
    asyncio.run(test_concurrent_downloads())
    asyncio.run(test_sequential_downloads())
    print("\n=== 测试完成 ===")