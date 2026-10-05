"""Google Shopping via Serper API data source."""

import httpx
from typing import List, Dict, Any, Optional
from src.services.product_data_source import ProductDataSource
from src.utils.config import settings
from src.analytics.logger import logger


class SerperDataSource(ProductDataSource):
    """Google Shopping data source via Serper API."""

    def __init__(self):
        self.api_key = settings.serper_api_key
        self.base_url = "https://google.serper.dev/shopping"

    def get_source_name(self) -> str:
        return "serper_google_shopping"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def search_products(
        self,
        query: str,
        num_results: int = 10,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search products using Google Shopping via Serper."""
        if not self.is_available():
            return []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.base_url,
                    headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
                    json={
                        "q": query,
                        "num": num_results,
                        "gl": "in",
                        "hl": "en",
                    },
                )
                response.raise_for_status()
                data = response.json()

                products = []
                for item in data.get("shopping", []):
                    normalized = self.normalize_product(
                        {
                            "productId": item.get("productId", ""),
                            "title": item.get("title", ""),
                            "description": item.get("description", ""),
                            "price": item.get("price", ""),
                            "currency": "INR",
                            "imageUrl": item.get("imageUrl", ""),
                            "link": item.get("link", ""),
                            "rating": item.get("rating", 0),
                            "reviews": item.get("reviews", 0),
                            "category": item.get("category", "general"),
                        }
                    )
                    products.append(normalized)

                return products
        except Exception as e:
            logger.error(f"Error fetching from Serper API: {e}")
            return []

    def normalize_product(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Serper product to standard format."""
        price_str = product.get("price", "")
        price = self._parse_price(price_str)

        return {
            "id": f"serper_{product.get('productId', '')}",
            "name": product.get("title", ""),
            "description": product.get("description", ""),
            "price": price,
            "currency": product.get("currency", "INR"),
            "original_price": None,
            "shipping_cost": 0.0,  # Serper doesn't provide shipping
            "image_url": product.get("imageUrl", ""),
            "product_url": product.get("link", ""),
            "rating": product.get("rating", 0),
            "reviews": product.get("reviews", 0),
            "category": product.get("category", "general"),
            "brand": None,
            "availability": True,
            "in_stock": True,
            "retailer": "Google Shopping",
            "source": "serper_google_shopping",
            "upc": None,
            "gtin": None,
            "ean": None,
            "sku": None,
            "metadata": {"productId": product.get("productId", "")},
            "product_metadata": {
                "productId": product.get("productId", "")
            },  # Alias for compatibility
        }

    def _parse_price(self, price_str: str) -> float:
        """Parse price strings from Google Shopping/Serper."""

        if price_str is None:
            return 0.0

        try:
            cleaned = str(price_str).strip()

            # Remove common currency symbols/codes.
            for symbol in ["₹", "$", "€", "£", "Rs.", "Rs", "INR", "USD"]:
                cleaned = cleaned.replace(symbol, "")

            # Remove commas and whitespace.
            cleaned = cleaned.replace(",", "").strip()

            # Keep only the numeric portion.
            import re

            match = re.search(r"\d+(?:\.\d+)?", cleaned)

            if not match:
                return 0.0

            return float(match.group(0))

        except (ValueError, TypeError, AttributeError):
            return 0.0
