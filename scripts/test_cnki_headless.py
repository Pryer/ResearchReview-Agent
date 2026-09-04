"""测试 CNKI Headless 模式下的机构 IP 识别登录

使用方式：
    python scripts/test_cnki_headless.py
"""

import sys
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.clients.cnki_client import search_cnki
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def test_cnki_search():
    """测试 CNKI 搜索功能"""
    print("=" * 60)
    print("CNKI Headless 模式测试")
    print("=" * 60)
    print()
    
    # 显示当前配置
    print(f"配置信息：")
    print(f"  CNKI_HEADLESS: {settings.cnki_headless}")
    print(f"  CNKI_CHROMEDRIVER_PATH: {settings.cnki_chromedriver_path}")
    print(f"  CNKI_HOME_WAIT_SECONDS: {settings.cnki_home_wait_seconds}")
    print()
    
    # 测试查询
    query = "深度学习"
    start_year = 2022
    end_year = 2024
    max_results = 5
    
    print(f"测试查询：")
    print(f"  关键词: {query}")
    print(f"  年份: {start_year}-{end_year}")
    print(f"  最大结果数: {max_results}")
    print()
    
    print("开始搜索...")
    print("-" * 60)
    
    try:
        papers = search_cnki(
            query=query,
            start_year=start_year,
            end_year=end_year,
            max_results=max_results
        )
        
        print()
        print("=" * 60)
        print(f"搜索完成！找到 {len(papers)} 篇论文")
        print("=" * 60)
        print()
        
        if papers:
            print("前 3 篇论文：")
            for i, paper in enumerate(papers[:3], 1):
                print(f"\n{i}. {paper.title}")
                print(f"   作者: {', '.join(paper.authors[:3]) if paper.authors else '无'}")
                print(f"   年份: {paper.year or '未知'}")
                print(f"   来源: {paper.venue or '未知'}")
                print(f"   URL: {paper.url or '无'}")
            
            print("\n" + "=" * 60)
            print("✅ 测试成功！CNKI Headless 模式正常工作")
            print("=" * 60)
            return True
        else:
            print("⚠️ 未找到论文")
            print()
            print("可能的原因：")
            print("1. 机构 IP 识别失败（请检查是否在机构网络内）")
            print("2. CNKI 检测到自动化访问（尽管已做反检测处理）")
            print("3. 网络问题或 CNKI 服务异常")
            print()
            print("建议：")
            print("1. 检查 logs/app.log 查看详细错误信息")
            print("2. 尝试非 Headless 模式（CNKI_HEADLESS=false）")
            print("3. 确认机构 IP 在浏览器中可以访问 CNKI")
            return False
            
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ 测试失败！")
        print("=" * 60)
        print()
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print()
        print("请检查：")
        print("1. ChromeDriver 是否正确安装")
        print("2. Chrome 浏览器是否已安装")
        print("3. 网络连接是否正常")
        print("4. logs/app.log 中的详细错误信息")
        
        import traceback
        print()
        print("详细堆栈跟踪：")
        print("-" * 60)
        traceback.print_exc()
        
        return False


def main():
    """主函数"""
    success = test_cnki_search()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
