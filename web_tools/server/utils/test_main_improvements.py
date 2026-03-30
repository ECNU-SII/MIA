import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from server.main import _handle_cache_logic, _cache_key_suffix

class TestMainImprovements(unittest.TestCase):
    
    def test_cache_key_suffix(self):
        """测试缓存键生成函数"""
        params1 = {"q": "test", "max_results": 5}
        params2 = {"max_results": 5, "q": "test"}  # 顺序不同
        params3 = {"q": "test", "max_results": 3}  # 值不同
        
        # 相同参数应该生成相同的哈希
        self.assertEqual(_cache_key_suffix(params1), _cache_key_suffix(params2))
        
        # 不同参数应该生成不同的哈希
        self.assertNotEqual(_cache_key_suffix(params1), _cache_key_suffix(params3))
    
    @patch('server.main.get_cache')
    @patch('server.main.CACHE_HITS')
    @patch('server.main.REQUESTS')
    async def test_handle_cache_logic_hit(self, mock_requests, mock_cache_hits, mock_get_cache):
        """测试缓存命中逻辑"""
        mock_get_cache.return_value = {"test": "data"}
        
        result, suffix = await _handle_cache_logic("test_tool", "1.0.0", {"q": "test"})
        
        # 验证返回值
        self.assertEqual(result, {"test": "data"})
        
        # 验证缓存命中统计
        mock_cache_hits.labels.assert_called_once_with("test_tool")
        mock_requests.labels.assert_called_once_with("200_cache", "test_tool")
    
    @patch('server.main.get_cache')
    @patch('server.main.REQUESTS')
    async def test_handle_cache_logic_miss_deterministic(self, mock_requests, mock_get_cache):
        """测试确定性模式下缓存未命中逻辑"""
        mock_get_cache.return_value = None
        
        result, suffix = await _handle_cache_logic("test_tool", "1.0.0", {"q": "test"}, deterministic=True)
        
        # 验证返回值
        self.assertEqual(result, "CACHE_MISS")
        
        # 验证请求统计
        mock_requests.labels.assert_called_once_with("404_cache_miss", "test_tool")
    
    @patch('server.main.get_cache')
    async def test_handle_cache_logic_miss(self, mock_get_cache):
        """测试缓存未命中逻辑"""
        mock_get_cache.return_value = None
        
        result, suffix = await _handle_cache_logic("test_tool", "1.0.0", {"q": "test"})
        
        # 验证返回值
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()