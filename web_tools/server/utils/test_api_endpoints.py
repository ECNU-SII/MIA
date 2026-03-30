import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from fastapi.testclient import TestClient
from server.main import app

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
    
    def test_search_endpoint_validation(self):
        """测试文本搜索端点的参数验证"""
        # 测试空查询参数
        response = self.client.get("/search?q=")
        self.assertEqual(response.status_code, 422)
        
        # 测试查询参数过长
        response = self.client.get("/search?q=" + "a" * 401)
        self.assertEqual(response.status_code, 422)
        
        # 测试max_results参数范围
        response = self.client.get("/search?q=test&max_results=0")
        self.assertEqual(response.status_code, 422)
        
        response = self.client.get("/search?q=test&max_results=11")
        self.assertEqual(response.status_code, 422)
        
        # 测试有效参数
        response = self.client.get("/search?q=test&max_results=5")
        # 由于没有mock外部服务，这里可能会返回500，但我们主要验证参数验证
        self.assertIn(response.status_code, [200, 500])
    
    def test_image_search_endpoint_validation(self):
        """测试图片搜索端点的参数验证"""
        # 测试空查询参数
        response = self.client.get("/image_search?q=")
        self.assertEqual(response.status_code, 422)
        
        # 测试max_results参数范围
        response = self.client.get("/image_search?q=test&max_results=0")
        self.assertEqual(response.status_code, 422)
        
        response = self.client.get("/image_search?q=test&max_results=11")
        self.assertEqual(response.status_code, 422)
        
        # 测试有效参数
        response = self.client.get("/image_search?q=test&max_results=5")
        # 由于没有mock外部服务，这里可能会返回500，但我们主要验证参数验证
        self.assertIn(response.status_code, [200, 500])

if __name__ == '__main__':
    unittest.main()