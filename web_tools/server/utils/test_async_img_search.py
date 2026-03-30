import asyncio
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.img_search import call_ddgs_image_search

async def test_async_image_search():
    """Test the async image search functionality"""
    print("Testing async image search...")
    
    # Test with a simple query
    query = "cats"
    try:
        result_str, images, stats = await call_ddgs_image_search(query, max_results=3)
        print(f"Search completed successfully!")
        print(f"Results: {stats}")
        print(f"Number of images: {len(images)}")
        print(f"Result string preview: {result_str[:200]}...")
        
        # Check if we got images
        if images:
            print("Images downloaded successfully!")
            for i, img in enumerate(images):
                if img:
                    print(f"Image {i+1}: {img.size}")
                else:
                    print(f"Image {i+1}: Failed to download")
        else:
            print("No images were downloaded.")
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_async_image_search())