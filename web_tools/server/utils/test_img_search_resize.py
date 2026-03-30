import asyncio
import base64
from PIL import Image
import io
from server.utils.img_search import call_ddgs_image_search_with_resize, resize_and_encode_image

async def test_image_search_with_resize():
    """测试图片搜索和resize功能"""
    print("开始测试图片搜索和resize功能...")
    
    try:
        # 测试搜索猫咪图片并自动resize
        text_result, image_data, stats = await call_ddgs_image_search_with_resize(
            query="Rani Lakshmibai",
            max_results=3,
            resize_size=(256, 256)
        )
        
        print(f"搜索完成，统计信息: {stats}")
        print(f"获取到 {len(image_data)} 张图片")
        print(f"文本结果预览:\n{text_result[:300]}...")
        
        # 验证图片数据
        for i, img_data in enumerate(image_data):
            if img_data:
                # 解码base64图片数据
                img_bytes = base64.b64decode(img_data)
                img = Image.open(io.BytesIO(img_bytes))
                print(f"图片 {i+1}: 尺寸 {img.size}, 模式 {img.mode}")
                
                # 验证图片尺寸是否正确
                if img.size == (256, 256):
                    print(f"✓ 图片 {i+1} 尺寸正确")
                else:
                    print(f"✗ 图片 {i+1} 尺寸不正确: 期望 (256, 256), 实际 {img.size}")
            else:
                print(f"图片 {i+1}: 数据为空")
                
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def test_resize_and_encode_function():
    """测试resize_and_encode_image函数"""
    print("\n开始测试resize_and_encode_image函数...")
    
    try:
        # 创建一个测试图片
        test_img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        print(f"原始图片尺寸: {test_img.size}")
        
        # 测试resize和编码功能
        encoded_img = resize_and_encode_image(test_img, (256, 256))
        print(f"编码后数据长度: {len(encoded_img)} 字符")
        
        # 验证解码后的图片
        img_bytes = base64.b64decode(encoded_img)
        decoded_img = Image.open(io.BytesIO(img_bytes))
        print(f"解码后图片尺寸: {decoded_img.size}")
        print(f"解码后图片模式: {decoded_img.mode}")
        
        if decoded_img.size == (256, 256):
            print("✓ Resize功能正常")
        else:
            print("✗ Resize功能异常")
            
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== 图片搜索和Resize功能测试 ===")
    asyncio.run(test_image_search_with_resize())
    test_resize_and_encode_function()
    print("\n=== 测试完成 ===")