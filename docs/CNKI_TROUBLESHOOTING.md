# CNKI 搜索故障排查指南

> 针对 "CNKI search failed: Message:" 错误

---

## 🔍 问题分析

### 错误日志
```
2026-07-19 21:05:06 | INFO | CNKI search: query='课堂行为 目标检测 深度学习' pages=6 headless=True
2026-07-19 21:06:01 | WARNING | CNKI search failed: Message:
2026-07-19 21:06:04 | INFO | [cnki] Found 0 papers
```

### 可能原因

1. **Chrome/ChromeDriver 版本不匹配**
   - Headless 模式下更容易出现兼容性问题
   - ChromeDriver 版本与 Chrome 浏览器版本必须匹配

2. **CNKI 网站结构变化**
   - 知网可能更新了页面结构
   - 元素选择器失效

3. **反爬虫机制触发**
   - Headless 模式被检测
   - IP 被临时限制
   - 需要验证码

4. **网络问题**
   - 无法访问 CNKI
   - 连接超时

5. **Selenium WebDriver 异常**
   - Session 失效
   - 浏览器启动失败

---

## ✅ 已完成的修复

### 1. 改进错误日志

**修改文件**：`app/clients/cnki_client.py`

**修改内容**：
```python
# 之前
logger.warning("CNKI search failed: %s", exc)

# 修改后
logger.warning("CNKI search failed: %s: %s", type(exc).__name__, str(exc), exc_info=True)
```

**效果**：
- 显示异常类型（如 `TimeoutException`, `WebDriverException`）
- 显示完整堆栈跟踪
- 更容易定位问题

---

## 🔧 诊断步骤

### 步骤 1：验证 ChromeDriver 配置

```bash
# 查看 ChromeDriver 路径
type .env | findstr CNKI_CHROMEDRIVER_PATH

# 验证 ChromeDriver 是否存在
dir .\chromedriver-win64\chromedriver.exe

# 测试 ChromeDriver 版本
.\chromedriver-win64\chromedriver.exe --version
```

**预期输出**：
```
ChromeDriver 131.x.xxxx.xx
```

**如果不匹配**：
1. 查看 Chrome 版本：打开 Chrome → 设置 → 关于 Chrome
2. 下载匹配版本：https://googlechromelabs.github.io/chrome-for-testing/
3. 替换 `chromedriver-win64/chromedriver.exe`

---

### 步骤 2：测试 CNKI 访问

**手动访问**：
```
https://www.cnki.net/
```

**检查项**：
- [ ] 能否正常访问
- [ ] 是否需要登录
- [ ] 是否出现验证码
- [ ] 搜索功能是否正常

---

### 步骤 3：查看详细错误日志

**重新运行并查看日志**：
```bash
# 停止服务
# 重启服务
conda activate rragent
python run_api.py

# 另一个终端查看日志
tail -f logs/app.log
```

**关注以下信息**：
- 异常类型（`TimeoutException`, `WebDriverException` 等）
- 堆栈跟踪
- 最后访问的 URL

---

### 步骤 4：测试 Headless vs 非 Headless

**禁用 Headless 模式测试**：
```bash
# 编辑 .env
CNKI_HEADLESS=false
```

**重启服务测试**：
- 如果非 Headless 成功，说明是 Headless 模式被检测
- 可以观察浏览器窗口，查看具体卡在哪一步

---

### 步骤 5：简化搜索测试

**创建测试脚本** `test_cnki.py`：
```python
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.clients.cnki_client import search_cnki

# 测试简单查询
query = "深度学习"
start_year = 2024
end_year = 2026
max_results = 5

print(f"测试 CNKI 搜索: {query}")
print(f"年份: {start_year}-{end_year}")
print(f"最大结果: {max_results}")
print("-" * 50)

papers = search_cnki(query, start_year, end_year, max_results)

print(f"\n找到 {len(papers)} 篇论文")
for i, paper in enumerate(papers, 1):
    print(f"\n{i}. {paper.title}")
    print(f"   作者: {', '.join(paper.authors[:3])}")
    print(f"   年份: {paper.year}")
```

**运行测试**：
```bash
conda activate rragent
python test_cnki.py
```

---

## 🛠️ 常见问题解决方案

### 问题 1：ChromeDriver 版本不匹配

**症状**：
```
SessionNotCreatedException: Message: session not created: 
This version of ChromeDriver only supports Chrome version XX
```

**解决方案**：
```bash
# 1. 检查 Chrome 版本
chrome.exe --version

# 2. 下载匹配的 ChromeDriver
# https://googlechromelabs.github.io/chrome-for-testing/

# 3. 替换文件
copy /Y chromedriver.exe chromedriver-win64\chromedriver.exe
```

---

### 问题 2：Headless 模式被检测

**症状**：
- 非 Headless 正常，Headless 失败
- 页面加载后立即跳转或显示验证

**解决方案**：

**选项 A：禁用 Headless**
```bash
# .env
CNKI_HEADLESS=false
```

**选项 B：添加反检测参数**（需要修改代码）
```python
# app/clients/cnki_client.py - build_driver 函数

def build_driver(headless: bool = False, ...) -> webdriver.Chrome:
    options = Options()
    
    # 添加反检测参数
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    
    # 修改 user-agent
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    
    # ... 其余代码
```

---

### 问题 3：网络超时

**症状**：
```
TimeoutException: Message: 
```

**解决方案**：

**增加等待时间**：
```bash
# .env
CNKI_HOME_WAIT_SECONDS=5.0  # 从 3.0 增加到 5.0
```

**或修改代码中的超时设置**：
```python
# app/clients/cnki_client.py

# 搜索框等待时间
box = _fill_search_box(driver, selectors, keyword, timeout=60)  # 从 30 增加到 60

# 结果等待时间
anchors = WebDriverWait(driver, 30).until(...)  # 从 20 增加到 30
```

---

### 问题 4：CNKI 页面结构变化

**症状**：
- 元素无法定位
- `NoSuchElementException`

**解决方案**：

1. **手动访问 CNKI 查看页面结构**
2. **更新元素选择器**（需要技术人员）
3. **临时禁用 CNKI**：
   ```bash
   # .env
   DEFAULT_SEARCH_SOURCES=arxiv,semantic_scholar,openalex,crossref
   # 移除 cnki
   ```

---

### 问题 5：IP 被限制

**症状**：
- 连续多次搜索后失败
- 出现验证码页面

**解决方案**：

**增加请求间隔**：
```bash
# .env
CNKI_HOME_WAIT_SECONDS=10.0  # 增加等待时间
```

**或限制 CNKI 使用**：
```bash
# .env
CNKI_MAX_RESULTS=20  # 限制每次查询数量
```

**或完全禁用 CNKI**：
```bash
# .env
DEFAULT_SEARCH_SOURCES=arxiv,semantic_scholar,openalex,crossref
```

---

## 🔬 高级调试

### 启用 Selenium 详细日志

**修改代码**：
```python
# app/clients/cnki_client.py - build_driver 函数

import logging
from selenium.webdriver.remote.remote_connection import LOGGER

# 启用 Selenium 详细日志
LOGGER.setLevel(logging.DEBUG)

def build_driver(...):
    # 添加日志路径
    service = Service(
        chromedriver,
        service_args=['--verbose', '--log-path=logs/chromedriver.log']
    )
    return webdriver.Chrome(service=service, options=options)
```

**查看 ChromeDriver 日志**：
```bash
tail -f logs/chromedriver.log
```

---

### 截图调试

**修改 search 函数添加截图**：
```python
# app/clients/cnki_client.py - search 函数

def search(driver, keyword, home_wait=3.0):
    driver.get(CNKI_HOME)
    
    # 首页截图
    driver.save_screenshot("logs/cnki_home.png")
    
    if home_wait > 0:
        time.sleep(home_wait)
    
    # ... 填写搜索框
    
    # 搜索后截图
    driver.save_screenshot("logs/cnki_after_search.png")
```

**查看截图**：
```bash
dir logs\*.png
```

---

## 📋 完整诊断清单

### 环境检查
- [ ] Chrome 已安装
- [ ] ChromeDriver 版本匹配
- [ ] Python 环境正确（rragent）
- [ ] 依赖包已安装（selenium）

### 配置检查
- [ ] `.env` 中 `CNKI_CHROMEDRIVER_PATH` 正确
- [ ] `.env` 中 `CNKI_HEADLESS` 设置合适
- [ ] ChromeDriver 文件存在且可执行

### 网络检查
- [ ] 能访问 https://www.cnki.net/
- [ ] 无代理或防火墙限制
- [ ] 网络稳定

### 功能检查
- [ ] 手动在 CNKI 搜索正常
- [ ] 无验证码要求
- [ ] 非 Headless 模式测试通过

---

## 🚀 快速修复建议

### 方案 1：禁用 CNKI（最快）

```bash
# .env
DEFAULT_SEARCH_SOURCES=arxiv,semantic_scholar,openalex,crossref
```

**优点**：
- 立即解决问题
- 其他数据源通常足够

**缺点**：
- 失去中文论文来源

---

### 方案 2：使用非 Headless 模式

```bash
# .env
CNKI_HEADLESS=false
```

**优点**：
- 更稳定
- 可以观察执行过程

**缺点**：
- 速度较慢
- 需要图形界面

---

### 方案 3：等待下次请求重试

CNKI 搜索失败不会影响其他数据源，系统会继续使用其他来源的论文。

---

## 📞 获取帮助

如果以上方法都无法解决，请提供以下信息：

1. **完整错误日志**（包含堆栈跟踪）
2. **Chrome 版本**
3. **ChromeDriver 版本**
4. **是否能手动访问 CNKI**
5. **Headless 和非 Headless 的测试结果**
6. **截图（如果有）**

