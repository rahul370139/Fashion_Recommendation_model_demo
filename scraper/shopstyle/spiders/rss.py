import scrapy
import os
from pathlib import Path

class ShopStyleRSS(scrapy.Spider):
    name = "shopstyle_rss"
    start_urls = ["https://www.shopstyle.com/rss"]
    custom_settings = {"DOWNLOAD_DELAY": 1.0}

    def __init__(self, scrape_mode=False, local_data_path=None, *args, **kwargs):
        super(ShopStyleRSS, self).__init__(*args, **kwargs)
        self.scrape_mode = scrape_mode.lower() == 'true' if isinstance(scrape_mode, str) else scrape_mode
        self.local_data_path = local_data_path or "/Users/rahul/Downloads/deepfashion1_data/images"
        
        if not self.scrape_mode:
            # Disable scraping when using local data
            self.start_urls = []

    def start_requests(self):
        if self.scrape_mode:
            # Use original scraping logic
            for url in self.start_urls:
                yield scrapy.Request(url, self.parse)
        else:
            # For local data, we'll handle it in parse_local_data
            pass

    def parse(self, response):
        """Original scraping logic"""
        for item in response.css("item"):
            yield {
                "product_id": item.css("guid::text").get(),
                "url":        item.css("link::text").get(),
                "title":      item.css("title::text").get(),
                "image_url":  item.css("enclosure::attr(url)").get()
            }

    async def start(self):
        """Override start method to handle local data processing"""
        if self.scrape_mode:
            # Use original start_requests for scraping
            async for item in super().start():
                yield item
        else:
            # Process local data directly
            async for item in self.parse_local_data():
                yield item

    async def parse_local_data(self):
        """Parse local image files from the specified directory"""
        data_path = Path(self.local_data_path)
        
        if not data_path.exists():
            self.logger.error(f"Local data path does not exist: {self.local_data_path}")
            return
            
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        
        for img_file in data_path.rglob('*'):
            if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                # Create a product ID from the file path
                product_id = str(img_file.relative_to(data_path))
                
                yield {
                    "product_id": product_id,
                    "url": str(img_file),  # Local file path
                    "title": img_file.stem,  # Filename without extension
                    "image_url": str(img_file),  # Local file path
                    "local_path": str(img_file)  # Additional field for local processing
                } 