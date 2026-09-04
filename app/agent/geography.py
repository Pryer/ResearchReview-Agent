"""兼容导入层；地域证据规则位于 :mod:`app.core.geography`。"""

from app.core.geography import geographic_bucket, has_reliable_geographic_comparison

__all__ = ["geographic_bucket", "has_reliable_geographic_comparison"]
