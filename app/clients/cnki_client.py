"""CNKI（知网）Selenium 客户端。

通过 Selenium 驱动本地 Chrome 检索中国知网，抓取论文元数据。
与其它 client 一样是模块级函数，统一签名：

    search_cnki(query, start_year, end_year, max_results=20) -> List[PaperMetadata]

注意：
- 依赖本机安装 Chrome，以及一个版本匹配的 chromedriver（见
  ``settings.cnki_chromedriver_path``，留空则交给 Selenium Manager 自动下载）。
- CNKI 有反爬/间歇性 JS alert，已由 ``dismiss_alert`` + 重试兜底处理。
- Selenium driver 有状态，**本函数非线程安全**；调度层
  ``app.tools.search_papers.search_papers`` 为顺序循环调用，无并发问题。
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse, parse_qs

# Force UTF-8 on stdout/stderr so Chinese titles print correctly on Windows
# terminals whose default code page is GBK.
def _force_utf8(stream):
    recon = getattr(stream, "reconfigure", None)
    if callable(recon):
        recon(encoding="utf-8")
        return stream
    return io.TextIOWrapper(stream.buffer, encoding="utf-8")


sys.stdout = _force_utf8(sys.stdout)
sys.stderr = _force_utf8(sys.stderr)

try:
    from selenium import webdriver
    from selenium.common.exceptions import (
        ElementNotInteractableException,
        NoAlertPresentException,
        StaleElementReferenceException,
        TimeoutException,
        UnexpectedAlertPresentException,
        WebDriverException,
    )
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    webdriver = None
    class WebDriverException(Exception): pass
    class TimeoutException(WebDriverException): pass
    class ElementNotInteractableException(WebDriverException): pass
    class NoAlertPresentException(WebDriverException): pass
    class StaleElementReferenceException(WebDriverException): pass
    class UnexpectedAlertPresentException(WebDriverException): pass
    Options = None
    Service = None
    By = None
    Keys = None
    EC = None
    WebDriverWait = None

from app.core.config import get_settings
from app.core.logger import get_logger
from app.schemas.paper_schema import PaperMetadata

logger = get_logger(__name__)
settings = get_settings()


def _record_search_diagnostic(outcome: str, *, error_code: str | None = None, message: str | None = None) -> None:
    try:
        from app.tools.search_papers import record_client_diagnostic
        record_client_diagnostic(outcome, error_code=error_code, message=message)
    except Exception:
        pass


CNKI_HOME = "https://www.cnki.net/"
RESULTS_PER_PAGE = 20
STALE_RETRY_ATTEMPTS = 3
STALE_RETRY_DELAY_SECONDS = 0.5

# CNKI 结果页标题链接选择器 — 按优先级排列；知网发版时 class 名可能变，语义选择器在前。
_RESULT_LINK_CSS_SELECTORS = [
    "table.result-table-list td.name a",
    "table.result-table-list a[href*='Detail']",
    "table.result-table a[href*='Detail']",
    "a.fz14",
]

# JS fallback：当 CSS 选择器全部失效时，按 URL 模式匹配知网文章详情链接。
_CNKI_ARTICLE_HREF_PATTERNS = [
    "/kcms2/article/",
    "/kcms/detail/",
    "kns.cnki.net/kcms",
    "dbcode=",
    "filename=",
]


# ============================================================
# Driver 构建
# ============================================================
def find_local_chrome() -> str:
    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path.home() / r"AppData\Local\Google\Chrome\Application\chrome.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return ""


def build_driver(
    headless: bool = False,
    chrome_binary: str | None = None,
    chromedriver: str | None = None,
) -> webdriver.Chrome:
    options = Options()
    binary = chrome_binary or find_local_chrome()
    if binary:
        options.binary_location = binary
    
    if headless:
        # Chrome's newer headless mode is less likely to break modern pages.
        options.add_argument("--headless=new")
    
    # 基础配置
    options.add_argument("--disable-gpu")
    options.add_argument("--start-maximized")
    options.add_argument("--lang=zh-CN")
    
    options.add_experimental_option("prefs", {"profile.managed_default_content_settings.images": 2})
    
    # 增强 Headless 模式的浏览器特征，避免被 CNKI 识别为爬虫
    # 这对机构 IP 自动识别登录很重要
    options.add_argument("--disable-blink-features=AutomationControlled")  # 移除 navigator.webdriver 标识
    options.add_experimental_option("excludeSwitches", ["enable-automation"])  # 移除自动化控制标识
    options.add_experimental_option("useAutomationExtension", False)  # 禁用自动化扩展
    
    # 设置真实的用户代理（避免默认的 HeadlessChrome 标识）
    if headless:
        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        options.add_argument(f"--user-agent={user_agent}")
    
    # 窗口大小设置（模拟真实桌面环境）
    options.add_argument("--window-size=1920,1080")
    
    # 禁用沙箱和共享内存（某些环境需要）
    # 注意：这会降低安全性，但对某些机构网络环境可能必要
    # options.add_argument("--no-sandbox")
    # options.add_argument("--disable-dev-shm-usage")
    
    if chromedriver:
        service = Service(chromedriver)
        driver = webdriver.Chrome(service=service, options=options)
    else:
        driver = webdriver.Chrome(options=options)

    # 限制页面导航时间。WebDriverWait 只能限制元素等待，无法限制 driver.get
    # 或新标签页导航；两者必须分别设置。
    driver.set_page_load_timeout(max(5, settings.cnki_page_load_timeout_seconds))
    driver.set_script_timeout(max(5, settings.cnki_page_load_timeout_seconds))
    
    # 进一步隐藏 WebDriver 特征
    # 执行 JavaScript 覆盖 navigator.webdriver 属性
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            
            // 覆盖 Chrome 对象
            window.chrome = {
                runtime: {}
            };
            
            // 覆盖 Permissions API
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
            
            // 覆盖 Plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
            
            // 覆盖 Languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['zh-CN', 'zh', 'en']
            });
        """
    })
    
    return driver


# ============================================================
# 通用 Selenium 辅助
# ============================================================
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def wait_first(driver: webdriver.Chrome, selectors: list[tuple[str, str]], timeout: int = 10):
    if not selectors:
        raise TimeoutException("No selectors supplied")

    def _any_selector_present(d):
        for by, selector in selectors:
            try:
                elems = d.find_elements(by, selector)
                if elems:
                    return elems[0]
            except Exception:
                continue
        return False

    return WebDriverWait(driver, timeout, poll_frequency=0.2).until(_any_selector_present)


def find_text(driver: webdriver.Chrome, selectors: list[tuple[str, str]], timeout: int = 3) -> str:
    try:
        element = wait_first(driver, selectors, timeout=timeout)
        return normalize_text(element.text)
    except TimeoutException:
        return ""


def find_own_text(
    driver: webdriver.Chrome,
    selectors: list[tuple[str, str]],
    timeout: int = 3,
) -> str:
    """只取元素自身的直接文本节点，忽略后代元素的文本。

    CNKI 详情页把「题录 / 引用 / 导出」等操作控件放在标题 ``h1`` 内部，
    ``element.text`` 会把控件文案一并带出——实测参考文献标题尾部出现
    " 题录"（《基于课堂行为分析的医学理论教学质量评估模型 题录》）。
    判据取 DOM 结构（直接文本节点 vs 后代元素）而不是控件文案词表：
    知网改文案或加按钮都不会让它再次失效。

    取不到直接文本节点时退回 ``element.text``，宁可带控件文案也不丢标题。
    """
    try:
        element = wait_first(driver, selectors, timeout=timeout)
    except TimeoutException:
        return ""
    own = ""
    try:
        own = driver.execute_script(
            "return Array.prototype.filter"
            ".call(arguments[0].childNodes, function (node) {"
            "  return node.nodeType === 3;"
            "}).map(function (node) { return node.textContent; }).join(' ');",
            element,
        )
    except Exception as exc:  # noqa: BLE001 - 脚本不可用时退回整体文本
        logger.debug("CNKI own-text extraction failed, fall back to element.text: %s", exc)
    own = normalize_text(str(own or ""))
    return own or normalize_text(element.text)


def dismiss_alert(driver: webdriver.Chrome) -> bool:
    """Dismiss a CNKI anti-scraping alert if one is open.

    CNKI intermittently pops a JS alert (e.g. "操作频繁") after a search.
    While it's open, every webdriver call throws UnexpectedAlertPresentException,
    so we must accept/dismiss it before doing anything else.
    """
    try:
        alert = driver.switch_to.alert
        alert.accept()
        time.sleep(1)
        return True
    except (NoAlertPresentException, WebDriverException):
        return False


def robust_click(driver: webdriver.Chrome, element) -> None:
    """Click an element, tolerating headless 'not interactable' failures.

    Native .click() refuses when the element is off-viewport or covered in
    headless mode; a JS click bypasses that. If CNKI pops an alert mid-click,
    dismiss it and retry.
    """
    try:
        element.click()
    except StaleElementReferenceException:
        # 失效元素无法通过 JS click 恢复，必须由调用方使用 locator 重新定位。
        raise
    except UnexpectedAlertPresentException:
        dismiss_alert(driver)
        try:
            driver.execute_script("arguments[0].click();", element)
        except UnexpectedAlertPresentException:
            dismiss_alert(driver)
    except (ElementNotInteractableException, WebDriverException):
        try:
            driver.execute_script("arguments[0].click();", element)
        except StaleElementReferenceException:
            raise
        except UnexpectedAlertPresentException:
            dismiss_alert(driver)


# ============================================================
# 检索流程
# ============================================================
def _on_result_page(driver: webdriver.Chrome) -> bool:
    try:
        url = driver.current_url
    except UnexpectedAlertPresentException:
        dismiss_alert(driver)
        return False
    return "defaultresult" in url or "kns8s" in url or "/kns/" in url


def search(driver: webdriver.Chrome, keyword: str, home_wait: float = 3.0) -> None:
    driver.get(CNKI_HOME)
    # 进入首页后停留，让页面充分加载，降低触发反爬 alert 的概率。
    if home_wait > 0:
        time.sleep(home_wait)
    search_box_selectors = [
        (By.ID, "txt_SearchText"),
        (By.CSS_SELECTOR, "input[name='txt_SearchText']"),
        (By.CSS_SELECTOR, "input.search-input"),
    ]
    box = _fill_search_box(driver, search_box_selectors, keyword, timeout=30)
    button = wait_first(
        driver,
        [
            (By.CSS_SELECTOR, "input.search-btn"),
            (By.CSS_SELECTOR, "button.search-btn"),
            (By.XPATH, "//input[contains(@class,'search') or @type='button']"),
            (By.XPATH, "//button[contains(., '检索') or contains(., '搜索')]"),
        ],
        timeout=10,
    )
    try:
        robust_click(driver, button)
    except StaleElementReferenceException:
        # 首页组件可能在输入关键词后重新渲染，重新定位按钮再点击。
        button = wait_first(
            driver,
            [
                (By.CSS_SELECTOR, "input.search-btn"),
                (By.CSS_SELECTOR, "button.search-btn"),
                (By.XPATH, "//input[contains(@class,'search') or @type='button']"),
                (By.XPATH, "//button[contains(., '检索') or contains(., '搜索')]"),
            ],
            timeout=10,
        )
        robust_click(driver, button)
    # CNKI 可能在点击后立刻弹反爬 alert。
    dismiss_alert(driver)

    # 新版 CNKI 上点击 input.search-btn 不一定触发检索；检测是否跳到结果页，
    # 没跳就在搜索框按回车（这能可靠触发检索）并重试几次。
    for _attempt in range(3):
        time.sleep(3)
        dismiss_alert(driver)
        if _on_result_page(driver):
            return
        try:
            # 不复用点击检索前保存的 WebElement；知网会动态替换整个搜索组件。
            box = wait_first(driver, search_box_selectors, timeout=10)
            box.send_keys(Keys.RETURN)
        except StaleElementReferenceException:
            box = wait_first(driver, search_box_selectors, timeout=10)
            box.send_keys(Keys.RETURN)
        except UnexpectedAlertPresentException:
            dismiss_alert(driver)
            box = wait_first(driver, search_box_selectors, timeout=10)
            box.send_keys(Keys.RETURN)
        time.sleep(3)
        dismiss_alert(driver)
        if _on_result_page(driver):
            return

    time.sleep(2)
    dismiss_alert(driver)


def _fill_search_box(
    driver: webdriver.Chrome,
    selectors: list[tuple[str, str]],
    keyword: str,
    timeout: int,
):
    """填写搜索框；页面重绘导致元素失效时重新定位。"""
    last_error: Exception | None = None
    for attempt in range(STALE_RETRY_ATTEMPTS):
        try:
            box = wait_first(driver, selectors, timeout=timeout)
            box.clear()
            box.send_keys(keyword)
            return box
        except StaleElementReferenceException as exc:
            last_error = exc
            logger.debug("CNKI search box became stale, retry=%d", attempt + 1)
            time.sleep(STALE_RETRY_DELAY_SECONDS)
    if last_error:
        raise last_error
    raise StaleElementReferenceException("CNKI search box remained stale")


def try_click_text(driver: webdriver.Chrome, text: str, timeout: int = 5) -> bool:
    xpaths = [
        f"//a[contains(normalize-space(), '{text}')]",
        f"//span[contains(normalize-space(), '{text}')]",
        f"//li[contains(normalize-space(), '{text}')]",
    ]
    for xpath in xpaths:
        try:
            element = WebDriverWait(driver, timeout).until(
                EC.element_to_be_clickable((By.XPATH, xpath))
            )
            robust_click(driver, element)
            time.sleep(2)
            return True
        except (UnexpectedAlertPresentException, StaleElementReferenceException, TimeoutException):
            continue
    return False


def _wait_for_result_links(driver: webdriver.Chrome, timeout: int = 20) -> str:
    """等待任一已知的选择器匹配到结果链接，返回匹配成功的 CSS 选择器。

    如果所有选择器均超时，自动保存 page_source + 截图到 logs/ 以便排查。
    """

    def _any_selector_has_links(d):
        for selector in _RESULT_LINK_CSS_SELECTORS:
            try:
                elems = d.find_elements(By.CSS_SELECTOR, selector)
                if elems:
                    return selector
            except Exception:
                continue
        return False

    try:
        return WebDriverWait(driver, timeout).until(_any_selector_has_links)
    except TimeoutException:
        _save_debug_artifacts(driver, "wait_result_links")
        raise


def _try_collect_by_url_pattern(driver: webdriver.Chrome) -> list[dict[str, str]]:
    """JS fallback：按知网文章 URL 模式在页面上搜索所有 <a> 并提取匹配项。"""
    rows = driver.execute_script(
        """
        const patterns = arguments[0];
        return Array.from(document.querySelectorAll('a')).filter(a => {
            const href = a.href || '';
            const text = (a.innerText || a.textContent || '').trim();
            return text.length > 0 && patterns.some(p => href.includes(p));
        }).map(a => ({
            href: a.href || '',
            title: (a.innerText || a.textContent || '').trim(),
            row_text: ((a.closest('tr') || a.closest('li') || a.parentElement)?.innerText || '').trim()
        }));
        """,
        _CNKI_ARTICLE_HREF_PATTERNS,
    ) or []
    return _dedupe_result_rows(rows)


def _dedupe_result_rows(rows: list[dict]) -> list[dict[str, str]]:
    """去重 + 过滤无效行。"""
    records: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for row in rows:
        url = str(row.get("href") or "").strip()
        title = normalize_text(str(row.get("title") or ""))
        if not url or not title or url in seen_urls:
            continue
        seen_urls.add(url)
        records.append({
            "url": url,
            "title": title,
            "row_text": normalize_text(str(row.get("row_text") or "")),
        })
    return records


def _save_debug_artifacts(driver: webdriver.Chrome, tag: str) -> None:
    """超时时自动保存页面源码和截图到 logs/ 目录，方便排查知网页面结构变更。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    screenshot_path = logs_dir / f"cnki_{tag}_{timestamp}.png"
    html_path = logs_dir / f"cnki_{tag}_{timestamp}.html"
    try:
        driver.save_screenshot(str(screenshot_path))
        logger.warning("CNKI debug screenshot saved: %s", screenshot_path)
    except Exception as exc:
        logger.warning("Failed to save CNKI screenshot: %s", exc)
    try:
        html_path.write_text(driver.page_source, encoding="utf-8")
        logger.warning("CNKI debug page source saved: %s", html_path)
    except Exception as exc:
        logger.warning("Failed to save CNKI page source: %s", exc)


def collect_result_records(driver: webdriver.Chrome) -> list[dict[str, str]]:
    """一次性读取结果页可见元数据，避免必须逐篇打开详情页才保留结果。

    策略：先用多组 CSS 选择器依次尝试（适配知网发版后 class 名变化），
    全部失败则用 JS 按知网文章 URL 模式兜底匹配。
    """
    last_error: Exception | None = None
    for attempt in range(STALE_RETRY_ATTEMPTS):
        try:
            records = _try_collect_with_selectors(driver)
            if records:
                return records
            # CSS 选择器全部未命中 → JS URL 模式兜底
            records = _try_collect_by_url_pattern(driver)
            if records:
                logger.info(
                    "[cnki] result links extracted via JS URL-pattern fallback (%d records)",
                    len(records),
                )
                return records
            raise TimeoutException(
                "CNKI result links not found with any CSS selector or URL pattern"
            )
        except UnexpectedAlertPresentException as exc:
            last_error = exc
            dismiss_alert(driver)
        except StaleElementReferenceException as exc:
            last_error = exc
            logger.debug("CNKI result list became stale, retry=%d", attempt + 1)
        except TimeoutException as exc:
            last_error = exc
            logger.warning("CNKI result links not found (attempt %d/%d)", attempt + 1, STALE_RETRY_ATTEMPTS)
        if attempt + 1 < STALE_RETRY_ATTEMPTS:
            time.sleep(STALE_RETRY_DELAY_SECONDS)
    if last_error:
        raise last_error
    return []


def _try_collect_with_selectors(driver: webdriver.Chrome) -> list[dict[str, str]]:
    """等待任一已知选择器命中，用匹配到的选择器执行 JS 批量提取。"""
    matched_selector = _wait_for_result_links(driver, timeout=20)
    if not isinstance(matched_selector, str):
        # Defensive compatibility for custom wait adapters; production waits
        # return the matched selector string.
        matched_selector = _RESULT_LINK_CSS_SELECTORS[0]
    selector_json = json.dumps(matched_selector, ensure_ascii=False)
    rows = driver.execute_script(
        f"""
        const selector = {selector_json};
        return Array.from(document.querySelectorAll(selector)).map(a => ({{
            href: a.href || '',
            title: (a.innerText || a.textContent || '').trim(),
            row_text: ((a.closest('tr') || a.parentElement)?.innerText || '').trim()
        }}));
        """
    ) or []
    logger.debug(
        "[cnki] extracted %d result rows via selector %r", len(rows), matched_selector
    )
    return _dedupe_result_rows(rows)


def collect_result_urls(driver: webdriver.Chrome) -> list[str]:
    """兼容旧调用；新检索流程使用 :func:`collect_result_records`。"""
    return [record["url"] for record in collect_result_records(driver)]


def extract_year(text: str) -> str:
    match = re.search(r"(19|20)\d{2}", text or "")
    return match.group(0) if match else ""


def extract_doi(text: str) -> str:
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text or "")
    return match.group(0).rstrip(".。;,，") if match else ""


def find_pdf_url(driver: webdriver.Chrome) -> str:
    for anchor in driver.find_elements(By.CSS_SELECTOR, "a"):
        text = normalize_text(anchor.text)
        href = anchor.get_attribute("href") or ""
        if href and ("PDF" in text.upper() or "下载" in text or "pdf" in href.lower()):
            return href
    return ""


def parse_detail(driver: webdriver.Chrome, url: str) -> dict:
    """打开详情页并抓取字段，返回原始 dict（字符串形式）。"""
    driver.execute_script("window.open(arguments[0]);", url)
    driver.switch_to.window(driver.window_handles[-1])
    try:
        title = find_own_text(
            driver,
            [
                (By.CSS_SELECTOR, "div.wx-tit h1"),
                (By.CSS_SELECTOR, "h1"),
                (By.XPATH, "//h1"),
            ],
            timeout=15,
        )
        abstract = find_text(
            driver,
            [
                (By.CLASS_NAME, "abstract-text"),
                (By.ID, "ChDivSummary"),
                (By.CSS_SELECTOR, ".abstract"),
                (By.XPATH, "//*[contains(@class,'abstract')]"),
            ],
        )
        keywords = find_text(
            driver,
            [
                (By.CLASS_NAME, "keywords"),
                (By.CSS_SELECTOR, ".keyword"),
                (By.XPATH, "//*[contains(., '关键词') and string-length(normalize-space()) < 500]"),
            ],
        )
        authors = find_text(
            driver,
            [
                (By.CSS_SELECTOR, ".author"),
                (By.CSS_SELECTOR, ".authors"),
                (By.CSS_SELECTOR, "h3.author"),
            ],
        )
        source_text = find_text(
            driver,
            [
                (By.CSS_SELECTOR, ".sourinfo"),
                (By.CSS_SELECTOR, ".top-tip"),
                (By.XPATH, "//*[contains(., '来源') and string-length(normalize-space()) < 500]"),
            ],
        )
        doi_text = find_text(
            driver,
            [
                (By.XPATH, "//*[contains(translate(., 'doi', 'DOI'), 'DOI')]"),
            ],
        )
        year = extract_year(source_text or driver.page_source)
        doi = extract_doi(doi_text or driver.page_source)
        pdf_url = find_pdf_url(driver)
        return {
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": abstract,
            "venue": source_text,
            "doi": doi,
            "url": driver.current_url,
            "pdf_url": pdf_url,
            "keywords": keywords,
        }
    finally:
        if len(driver.window_handles) > 1:
            driver.close()
            driver.switch_to.window(driver.window_handles[0])


def go_next_page(driver: webdriver.Chrome) -> bool:
    selectors = [
        (By.ID, "PageNext"),
        (By.CSS_SELECTOR, "a#PageNext"),
        (By.XPATH, "//a[contains(., '下一页')]"),
    ]
    for by, selector in selectors:
        for attempt in range(STALE_RETRY_ATTEMPTS):
            try:
                button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((by, selector))
                )
                robust_click(driver, button)
                try:
                    WebDriverWait(driver, 15).until(EC.staleness_of(button))
                except TimeoutException:
                    # 部分知网页面使用局部刷新并复用分页按钮，按钮不失效也不代表翻页失败。
                    pass
                _wait_for_result_links(driver, timeout=20)
                return True
            except StaleElementReferenceException:
                logger.debug("CNKI next-page button became stale, retry=%d", attempt + 1)
                time.sleep(STALE_RETRY_DELAY_SECONDS)
            except TimeoutException:
                break
    return False


# ============================================================
# 字段规整：原始字符串 -> PaperMetadata 字段
# ============================================================
def _split_authors(authors_text: str) -> list[str]:
    """把知网作者字符串拆成 List[str]，去掉数字角标（如 '孙洋杰1 王猛2'）。"""
    if not authors_text:
        return []
    # 去掉紧跟名字的数字角标
    cleaned = re.sub(r"\d+", " ", authors_text)
    # 按空白/逗号/分号/顿号切分
    parts = re.split(r"[\s,;；、]+", cleaned)
    from app.utils.title_cleaner import normalize_author_names

    return normalize_author_names([p.strip() for p in parts if p.strip()])


def _split_keywords(keywords_text: str) -> list[str]:
    """把知网关键词字符串拆成 List[str]（如 '动作识别;深度学习;'）。"""
    if not keywords_text:
        return []
    parts = re.split(r"[;；,，\s]+", keywords_text)
    return [p.strip() for p in parts if p.strip()]


def _stable_paper_id(detail_url: str) -> str:
    """从详情页 URL 取稳定标识作为 paper_id 的 id 部分。

    优先用 query 参数 v= 的值；取不到则用整段 URL 的短哈希兜底。
    """
    try:
        parsed = urlparse(detail_url)
        qs = parse_qs(parsed.query)
        v = qs.get("v", [None])[0]
        if v:
            return v[:64]
    except Exception:
        pass
    return hashlib.sha1(detail_url.encode("utf-8")).hexdigest()[:16]


def _to_paper_metadata(raw: dict) -> PaperMetadata:
    year_str = raw.get("year") or ""
    year: Optional[int] = None
    if year_str:
        try:
            year = int(year_str)
        except ValueError:
            year = None

    detail_url = raw.get("url") or ""
    
    # CNKI 当前不提供引用量，设置为 None
    # citation_count_by_source 也设置为 None（未来如果 CNKI 提供引用量，可在此添加）
    return PaperMetadata(
        paper_id=f"cnki:{_stable_paper_id(detail_url)}",
        title=raw.get("title") or "",
        authors=_split_authors(raw.get("authors") or ""),
        year=year,
        venue=raw.get("venue") or None,
        abstract=raw.get("abstract") or None,
        doi=raw.get("doi") or None,
        arxiv_id=None,
        url=detail_url or None,
        pdf_url=raw.get("pdf_url") or None,
        citation_count=None,
        citation_count_by_source=None,  # CNKI 当前不提供引用量
        source="cnki",
        is_open_access=False,
        keywords=_split_keywords(raw.get("keywords") or ""),
    )


def _result_record_to_raw(record: dict[str, str]) -> dict:
    """把结果页记录转换为最低可用元数据；详情失败时也不会整篇丢失。"""
    row_text = record.get("row_text") or ""
    return {
        "title": record.get("title") or "",
        "authors": "",
        "year": extract_year(row_text),
        "abstract": "",
        "venue": row_text,
        "doi": extract_doi(row_text),
        "url": record.get("url") or "",
        "pdf_url": "",
        "keywords": "",
    }


# ============================================================
# 对外入口
# ============================================================
def _record_year(record: dict[str, str]) -> Optional[int]:
    """从结果页行文本解析发表年份；解析不出返回 None。"""
    year_str = extract_year(record.get("row_text") or "")
    if not year_str:
        return None
    try:
        return int(year_str)
    except ValueError:
        return None


def _page_is_below_window(records: list[dict[str, str]], start_year: int) -> bool:
    """判断整页是否已全部早于年份窗口下界。

    知网结果页默认按发表时间倒序，因此一旦某页最新的一条都早于 start_year，
    后续页只会更早。年份全部解析失败时返回 False，避免因解析问题提前收工。
    """
    years = [year for year in (_record_year(r) for r in records) if year is not None]
    if not years:
        return False
    return max(years) < start_year


def _relevance_shingles(text: str) -> set[str]:
    """把文本切成轻量匹配单元：英文取词，中文取二字滑窗。

    结果页粗排只需要"够用"的相关度信号，不引入分词依赖与额外请求。
    中文用二字滑窗，使"课堂行为分析"与"行为分析"这类部分重叠也能得分。
    """
    lowered = (text or "").lower()
    words = set(re.findall(r"[a-z0-9]{2,}", lowered))
    cjk = re.sub(r"[^\u4e00-\u9fff]", "", lowered)
    bigrams = {cjk[i:i + 2] for i in range(len(cjk) - 1)}
    return words | bigrams


def _score_record_relevance(record: dict[str, str], query_shingles: set[str]) -> float:
    """按标题与行文本对检索式的覆盖率给结果页记录打分。

    标题命中权重远高于行文本（行文本含刊名、日期等噪声）。
    """
    if not query_shingles:
        return 0.0
    title_hits = len(query_shingles & _relevance_shingles(record.get("title") or ""))
    row_hits = len(query_shingles & _relevance_shingles(record.get("row_text") or ""))
    total = float(len(query_shingles))
    return (title_hits / total) * 2.0 + (row_hits / total) * 0.5


def select_records_for_detail(
    records: list[dict[str, str]],
    query: str,
    start_year: int,
    end_year: int,
    limit: int,
) -> list[int]:
    """挑出值得开详情页的记录下标（按原始顺序返回）。

    详情页增强是知网检索的主要时间成本，额度有限。按结果页出现顺序分配会把
    额度花在"最新但未必相关"的条目上，自适应翻页下更会花在越界页；因此改为
    先按相关度排序，并排除年份窗口外的记录——那些条目最终会被年份过滤剔除，
    为它们开详情页是纯浪费。

    Args:
        records: 结果页记录列表。
        query: 本次检索式。
        start_year: 年份窗口下界。
        end_year: 年份窗口上界。
        limit: 详情增强额度。

    Returns:
        选中记录在 ``records`` 中的下标，按升序（即原始结果顺序）返回。
    """
    if limit <= 0 or not records:
        return []

    query_shingles = _relevance_shingles(query)
    scored: list[tuple[float, int]] = []
    for index, record in enumerate(records):
        year = _record_year(record)
        # 年份解析失败时放行：与后续年份过滤保持一致（year is None 不剔除）。
        if year is not None and not (start_year <= year <= end_year):
            continue
        scored.append((_score_record_relevance(record, query_shingles), index))

    # 分数降序、同分按原始顺序（下标升序）稳定排序。
    scored.sort(key=lambda item: (-item[0], item[1]))
    return sorted(index for _score, index in scored[:limit])


def resolve_detail_enrichment_limit(max_results: int) -> int:
    """按下游实际需求推导详情页增强额度。

    固定额度在自适应翻页下会失配：结果池可达 200+ 条，而只有前 N 条拿到摘要，
    其余仅有标题与年份（``access_level=metadata_only``），既进不了写作证据池
    （synthesis 只接受 abstract/partial_full_text/full_text），也拿不到排序
    所需的摘要文本。额度改为跟随本次检索的 ``max_results``——它由
    ``retrieval_target`` 派生，代表下游真正要用的文献量。

    倍数 >1 是因为按相关度粗排选中的记录与最终排序入选集不完全重合，
    需留出重叠损耗；``cnki_detail_enrichment_max`` 兜住大额请求的时间成本。
    ``cnki_detail_enrichment_limit`` 为 0 时整体关闭增强（沿用旧语义），
    否则作为额度下限。
    """
    baseline = max(0, int(settings.cnki_detail_enrichment_limit or 0))
    if baseline <= 0:
        return 0
    factor = max(1.0, float(settings.cnki_detail_enrichment_demand_factor or 1.0))
    ceiling = max(1, int(settings.cnki_detail_enrichment_max or 1))
    demand = int(math.ceil(max(1, int(max_results)) * factor))
    return min(ceiling, max(baseline, demand))


def search_cnki(
    query: str,
    start_year: int,
    end_year: int,
    max_results: int = 20,
) -> List[PaperMetadata]:
    """检索中国知网，返回论文元数据列表。

    知网结果页按发表时间倒序。开启 ``cnki_adaptive_year_paging`` 时不再按
    ``max_results`` 换算固定页数，而是持续翻页直到连续若干整页都早于
    ``start_year``，从而覆盖用户请求年份窗口内的全部文献；页数、总条数与
    总时长三重硬上限用于兜底。

    详情页增强分两阶段：先翻页取全部结果页元数据（廉价），再按相关度挑出
    额度内的记录开详情页（昂贵）。这样有限的详情额度落在最相关的文献上，
    而不是落在结果页靠前但可能不相关、甚至年份越界的条目上。

    Args:
        query: 检索关键词。
        start_year: 起始年份（抓取后按 year 过滤；也是自适应翻页的停止边界）。
        end_year: 结束年份（抓取后按 year 过滤）。
        max_results: 固定翻页模式下的最大返回数；自适应模式下作为下界，
            实际上限取 ``cnki_max_results_ceiling``。

    Returns:
        论文元数据列表；失败或熔断时返回空列表（不抛异常，与其它 client 一致）。
    """
    from app.core.circuit_breaker import get_circuit_breaker

    cb = get_circuit_breaker("cnki", failure_threshold=2, recovery_timeout=120.0)
    if not cb.allow_request():
        logger.warning("CNKI search skipped due to active circuit breaker")
        _record_search_diagnostic("api_failed", error_code="CIRCUIT_OPEN", message="CNKI 暂时不可用，熔断器已打开")
        return []

    from app.core.rate_limiter import get_rate_limiter

    if not get_rate_limiter("cnki").acquire(1.0, timeout=10.0):
        # 知网自动化访问须显著降频（0.5 qps）：令牌等待超时说明已有
        # 并发会话在爬取，跳过本次而不是叠加请求导致封禁。
        logger.warning("CNKI search skipped: rate limit token wait timed out")
        _record_search_diagnostic("rate_limited", error_code="RATE_LIMIT_WAIT_TIMEOUT", message="CNKI 限流等待超时")
        return []

    if not SELENIUM_AVAILABLE:
        logger.warning("Selenium is not installed, CNKI search disabled")
        cb.record_failure()
        _record_search_diagnostic("api_failed", error_code="SELENIUM_UNAVAILABLE", message="CNKI 检索依赖不可用")
        return []

    max_results = max(1, max_results)
    adaptive = bool(settings.cnki_adaptive_year_paging)
    if adaptive:
        # 自适应模式下 max_results 只作下界；真正的收敛条件是年份边界。
        result_cap = max(max_results, int(settings.cnki_max_results_ceiling or max_results))
        pages = max(1, int(settings.cnki_max_pages or 1))
        boundary_pages_needed = max(1, int(settings.cnki_year_boundary_pages or 1))
        time_budget = max(0.0, float(settings.cnki_paging_time_budget_seconds or 0))
    else:
        result_cap = max_results
        pages = (max_results + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE
        boundary_pages_needed = 0
        time_budget = 0.0

    chromedriver = settings.cnki_chromedriver_path or None
    headless = settings.cnki_headless
    home_wait = settings.cnki_home_wait_seconds

    logger.info(
        "CNKI search: query=%r pages<=%d adaptive=%s result_cap=%d headless=%s",
        query, pages, adaptive, result_cap, headless,
    )
    started_at = time.monotonic()
    driver = None
    papers: List[PaperMetadata] = []
    raws: list[dict] = []
    detail_limit = resolve_detail_enrichment_limit(max_results)
    detail_time_budget = max(0.0, float(settings.cnki_detail_time_budget_seconds or 0))
    max_detail_failures = max(1, int(settings.cnki_max_consecutive_detail_failures or 1))
    rank_before_enrichment = bool(settings.cnki_detail_rank_before_enrichment)
    consecutive_detail_failures = 0
    consecutive_pages_below_window = 0
    enriched_count = 0
    stop_reason = "page_limit"
    try:
        driver = build_driver(headless=headless, chromedriver=chromedriver)
        search(driver, query, home_wait=home_wait)

        # ---------- 阶段一：翻页收集结果页元数据（廉价） ----------
        collected: list[dict[str, str]] = []
        for page in range(1, pages + 1):
            records = collect_result_records(driver)
            logger.info("[cnki page %d] %d result records", page, len(records))
            collected.extend(records[: max(0, result_cap - len(collected))])
            if len(collected) >= result_cap:
                stop_reason = "result_cap"
                break

            if adaptive:
                # 结果按发表时间倒序：连续多整页越过窗口下界即可收工。
                if _page_is_below_window(records, start_year):
                    consecutive_pages_below_window += 1
                    if consecutive_pages_below_window >= boundary_pages_needed:
                        stop_reason = "year_boundary"
                        logger.info(
                            "[cnki] stopped at page %d: %d consecutive pages older than %d",
                            page, consecutive_pages_below_window, start_year,
                        )
                        break
                else:
                    consecutive_pages_below_window = 0
                if time_budget and (time.monotonic() - started_at) >= time_budget:
                    stop_reason = "time_budget"
                    logger.warning(
                        "[cnki] paging time budget %.0fs exhausted at page %d", time_budget, page,
                    )
                    break

            if page < pages and not go_next_page(driver):
                stop_reason = "no_next_page"
                logger.info("[cnki] no next page, stop")
                break

        # 先落地结果页元数据：详情增强中途失败也不会退化为 0 篇。
        raws = [_result_record_to_raw(record) for record in collected]

        # ---------- 阶段二：按相关度选出详情增强对象 ----------
        if rank_before_enrichment:
            selected_indices = select_records_for_detail(
                collected, query, start_year, end_year, detail_limit,
            )
        else:
            # 兼容旧行为：按结果页顺序取前 detail_limit 条。
            selected_indices = list(range(min(detail_limit, len(collected))))
        if collected:
            logger.info(
                "[cnki] detail enrichment plan: %d/%d records (ranked=%s, limit=%d, max_results=%d)",
                len(selected_indices), len(collected), rank_before_enrichment,
                detail_limit, max_results,
            )

        # ---------- 阶段三：为选中记录补全详情（昂贵） ----------
        # 增强用独立时长预算：翻页预算从检索开始计时，若共用同一预算，翻页
        # 耗时长的会话会让增强几乎拿不到时间，退回"大量 metadata_only"。
        detail_started_at = time.monotonic()
        for index in selected_indices:
            if consecutive_detail_failures >= max_detail_failures:
                logger.warning(
                    "[cnki] detail enrichment stopped after %d consecutive failures",
                    consecutive_detail_failures,
                )
                break
            if detail_time_budget and (
                time.monotonic() - detail_started_at
            ) >= detail_time_budget:
                logger.warning(
                    "[cnki] detail time budget %.0fs exhausted after enriching %d records",
                    detail_time_budget, enriched_count,
                )
                break
            try:
                detail = parse_detail(driver, collected[index]["url"])
                raws[index].update(
                    {key: value for key, value in detail.items() if value}
                )
                enriched_count += 1
                consecutive_detail_failures = 0
            except (TimeoutException, WebDriverException) as exc:
                consecutive_detail_failures += 1
                logger.warning(
                    "[cnki] detail enrichment failed; keeping result-page metadata: %s %s",
                    type(exc).__name__, exc,
                )

        papers = [_to_paper_metadata(raw) for raw in raws]
        cb.record_success()
    except Exception as exc:
        cb.record_failure(exc)
        logger.warning("CNKI search failed: %s: %s", type(exc).__name__, str(exc), exc_info=True)
        if not papers and raws:
            # 阶段二/三异常时仍交付已获得的结果页元数据。
            papers = [_to_paper_metadata(raw) for raw in raws]
        if not papers:
            outcome = "timeout" if isinstance(exc, TimeoutException) else "api_failed"
            _record_search_diagnostic(
                outcome,
                error_code="TIMEOUT" if outcome == "timeout" else "API_ERROR",
                message=f"{type(exc).__name__}: {exc}",
            )
            return []
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass

    # 截断到本次抓取上限（自适应模式下为 result_cap，固定模式下等于 max_results）
    papers = papers[:result_cap]

    # 年份过滤（CNKI 检索不强制年份，沿用"抓后筛"）
    filtered = [
        p for p in papers
        if p.year is None or (start_year <= p.year <= end_year)
    ]
    logger.info(
        "CNKI parsed %d papers (%d after year filter %d-%d, enriched=%d, stop_reason=%s)",
        len(papers), len(filtered), start_year, end_year, enriched_count, stop_reason,
    )
    return filtered
