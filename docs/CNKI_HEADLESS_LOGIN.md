# CNKI Headless 模式登录问题解决方案

> 问题：非 Headless 模式可以通过机构 IP 自动登录，但 Headless 模式失败

---

## 🔍 问题分析

### 为什么非 Headless 模式可以访问 CNKI？

**非 Headless 模式**：
```python
CNKI_HEADLESS=false
```

- ✅ 启动的浏览器实例**可能**使用默认 Chrome 配置
- ✅ 如果您的 Chrome 已经登录 CNKI，可能共享 Session
- ✅ 或者 CNKI 识别机构 IP 自动授权访问

### 为什么 Headless 模式失败？

**Headless 模式**：
```python
CNKI_HEADLESS=true
```

- ❌ 启动的是**全新的独立浏览器实例**
- ❌ 没有 Cookies、Session、登录状态
- ❌ 即使是机构 IP，CNKI 的某些功能可能需要浏览器完整会话

**本质区别**：
```
非 Headless：[您的 Chrome] → [共享配置？] → [CNKI 自动登录]
Headless：   [全新浏览器] → [无登录信息] → [CNKI 拒绝访问/要求登录]
```

---

## ✅ 解决方案

### 方案 1：使用非 Headless 模式（推荐）

**最简单的方案**：既然非 Headless 正常工作，就保持这个模式。

**修改配置**：
```bash
# .env
CNKI_HEADLESS=false
```

**优点**：
- ✅ 立即可用
- ✅ 稳定可靠
- ✅ 可以利用机构账号访问权限

**缺点**：
- ⚠️ 速度稍慢（约慢 20-30%）
- ⚠️ 会弹出浏览器窗口（可最小化）
- ⚠️ 需要图形界面环境

**适用场景**：
- 本地开发和测试
- 有图形界面的服务器
- 对速度要求不高的场景

---

### 方案 2：Headless 模式使用共享用户数据（高级）

让 Headless 模式使用您的 Chrome 用户配置，从而继承登录状态。

**⚠️ 警告**：
- 此方案有一定风险，可能导致数据冲突
- 不推荐在已经打开 Chrome 浏览器时使用
- 建议仅在测试环境使用

#### 步骤 1：找到 Chrome 用户数据目录

**Windows 默认路径**：
```
C:\Users\<您的用户名>\AppData\Local\Google\Chrome\User Data
```

**验证路径**：
```bash
# PowerShell
Test-Path "$env:LOCALAPPDATA\Google\Chrome\User Data"
```

#### 步骤 2：修改代码支持用户数据目录

**修改文件**：`app/clients/cnki_client.py`

**在 `build_driver` 函数中添加**：
```python
def build_driver(
    headless: bool = False,
    chrome_binary: str | None = None,
    chromedriver: str | None = None,
    user_data_dir: str | None = None,  # 新增参数
) -> webdriver.Chrome:
    options = Options()
    binary = chrome_binary or find_local_chrome()
    if binary:
        options.binary_location = binary
    
    # 使用用户数据目录（继承登录状态）
    if user_data_dir:
        options.add_argument(f"--user-data-dir={user_data_dir}")
        # 使用独立的 profile，避免与主浏览器冲突
        options.add_argument("--profile-directory=SeleniumProfile")
    
    if headless:
        options.add_argument("--headless=new")
    
    options.add_argument("--disable-gpu")
    options.add_argument("--start-maximized")
    options.add_argument("--lang=zh-CN")
    
    # 重要：避免与主浏览器进程冲突
    if user_data_dir:
        options.add_argument("--no-first-run")
        options.add_argument("--no-default-browser-check")
    
    if chromedriver:
        return webdriver.Chrome(service=Service(chromedriver), options=options)
    return webdriver.Chrome(options=options)
```

#### 步骤 3：添加配置项

**修改文件**：`app/core/config.py`

```python
class Settings(BaseSettings):
    # ... 其他配置
    
    # CNKI（知网）Selenium 客户端
    cnki_chromedriver_path: str = ""
    cnki_headless: bool = False
    cnki_home_wait_seconds: float = 3.0
    cnki_user_data_dir: str = ""  # 新增：Chrome 用户数据目录
```

#### 步骤 4：修改调用

**修改文件**：`app/clients/cnki_client.py` 中的 `search_cnki` 函数

```python
def search_cnki(...) -> List[PaperMetadata]:
    # ...
    
    chromedriver = settings.cnki_chromedriver_path or None
    headless = settings.cnki_headless
    user_data_dir = settings.cnki_user_data_dir or None  # 新增
    
    driver = build_driver(
        headless=headless,
        chromedriver=chromedriver,
        user_data_dir=user_data_dir,  # 新增
    )
    
    # ...
```

#### 步骤 5：配置环境变量

**首次登录 CNKI**：
```bash
# 1. 先用非 Headless 模式，确保 SeleniumProfile 创建
# .env
CNKI_HEADLESS=false
CNKI_USER_DATA_DIR=C:\Users\<您的用户名>\AppData\Local\Google\Chrome\User Data

# 2. 启动服务，访问一次 CNKI 确保登录
# 3. 然后可以切换到 Headless 模式
CNKI_HEADLESS=true
```

**限制**：
- ⚠️ 不能同时打开主 Chrome 浏览器和 Selenium（会锁定用户数据）
- ⚠️ 或者使用不同的 Profile（如上面的 SeleniumProfile）

---

### 方案 3：Cookie 注入（最灵活）

手动提取 CNKI Cookies 并在 Headless 模式下注入。

#### 步骤 1：导出 CNKI Cookies

**使用浏览器扩展**：
1. 安装 "EditThisCookie" 或类似扩展
2. 访问 CNKI
3. 导出 Cookies（JSON 格式）
4. 保存到 `cnki_cookies.json`

#### 步骤 2：修改代码注入 Cookies

**修改文件**：`app/clients/cnki_client.py`

```python
def load_cookies(driver: webdriver.Chrome, cookie_file: str):
    """从文件加载 Cookies 并注入到浏览器。"""
    import json
    from pathlib import Path
    
    cookie_path = Path(cookie_file)
    if not cookie_path.exists():
        return
    
    # 必须先访问域名，才能设置 Cookies
    driver.get("https://www.cnki.net/")
    time.sleep(1)
    
    with open(cookie_path, "r", encoding="utf-8") as f:
        cookies = json.load(f)
    
    for cookie in cookies:
        # Selenium 需要的 Cookie 格式
        driver.add_cookie({
            "name": cookie["name"],
            "value": cookie["value"],
            "domain": cookie.get("domain", ".cnki.net"),
            "path": cookie.get("path", "/"),
            "secure": cookie.get("secure", False),
        })
    
    logger.info("CNKI cookies loaded from %s", cookie_file)


def search_cnki(...) -> List[PaperMetadata]:
    # ...
    
    driver = build_driver(headless=headless, chromedriver=chromedriver)
    
    # 加载 Cookies
    cookie_file = settings.cnki_cookie_file or "cnki_cookies.json"
    if Path(cookie_file).exists():
        load_cookies(driver, cookie_file)
    
    papers: List[PaperMetadata] = []
    try:
        search(driver, query, home_wait=home_wait)
        # ...
```

#### 步骤 3：配置

```bash
# .env
CNKI_HEADLESS=true
CNKI_COOKIE_FILE=cnki_cookies.json
```

**优点**：
- ✅ Headless 模式可用
- ✅ 不依赖用户数据目录
- ✅ 更安全（独立的 Cookie 文件）

**缺点**：
- ⚠️ 需要定期更新 Cookies（过期后需重新导出）
- ⚠️ 需要手动维护

---

## 📋 方案对比

| 方案 | 复杂度 | 稳定性 | 速度 | 适用场景 |
|------|--------|--------|------|----------|
| 非 Headless | ⭐ | ⭐⭐⭐ | ⭐⭐ | 本地开发、有图形界面 |
| 共享用户数据 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | 无法同时使用主浏览器 |
| Cookie 注入 | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | 需要定期更新 Cookies |

---

## 🎯 推荐方案

### 对于您的情况

**强烈推荐**：**方案 1（非 Headless 模式）**

**理由**：
1. ✅ 您已经验证非 Headless 模式正常工作
2. ✅ 有机构 IP 自动登录，非常方便
3. ✅ 简单可靠，无需额外配置
4. ⚠️ 速度稍慢可以接受（从 10 分钟变回 15 分钟，但稳定）

**配置**：
```bash
# .env
CNKI_HEADLESS=false
FRONTEND_REQUEST_TIMEOUT=1800  # 保持 30 分钟超时
```

### 如果确实需要 Headless

可以考虑：
1. **短期**：使用方案 3（Cookie 注入）
2. **长期**：研究 CNKI 的机构 IP 认证机制，看是否可以通过 HTTP Headers 实现

---

## 🔧 实施步骤

### 立即实施（方案 1）

已为您修改配置：
```bash
# .env
CNKI_HEADLESS=false  # ✅ 已修改
```

**重启服务**：
```bash
conda activate rragent
python run_api.py
```

**预期效果**：
- ✅ CNKI 正常搜索
- ✅ 机构 IP 自动登录
- ⚠️ 会看到浏览器窗口（可以最小化）

---

## 💡 额外优化

### 减少视觉干扰

虽然不是 Headless，但可以让窗口不那么显眼：

```python
# app/clients/cnki_client.py - build_driver

def build_driver(...):
    options = Options()
    
    if not headless:
        # 非 Headless 模式下，最小化窗口启动
        options.add_argument("--window-position=-2000,-2000")  # 移到屏幕外
    
    # 或者
    # options.add_argument("--window-size=800,600")
    # options.add_argument("--window-position=2000,0")  # 移到副屏
```

---

## 📝 总结

您的发现非常关键：

**问题根源**：
- ✅ Headless 模式是**全新浏览器实例**，没有登录状态
- ✅ 机构 IP 认证可能需要完整的浏览器会话

**最佳方案**：
- ✅ 使用非 Headless 模式（已为您配置）
- ✅ 利用机构 IP 自动登录的便利
- ✅ 接受略慢的速度换取稳定性

**配置已更新**：
```bash
CNKI_HEADLESS=false  # ✅ 改回非 Headless
```

重启服务后 CNKI 应该就能正常工作了！🎉

