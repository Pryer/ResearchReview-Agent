"""CNKI Selenium smoke test scraper.

This script is for manual, small-scale connectivity and selector testing only.
It does not bypass login, captcha, paywall, or access controls.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# Force UTF-8 on stdout/stderr so Chinese titles print correctly on Windows
# terminals whose default code page is GBK. Reconfigure in place when possible
# (Python 3.7+); otherwise wrap the underlying buffer and reassign the stream.
def _force_utf8(stream):
    recon = getattr(stream, "reconfigure", None)
    if callable(recon):
        recon(encoding="utf-8")
        return stream
    return io.TextIOWrapper(stream.buffer, encoding="utf-8")


sys.stdout = _force_utf8(sys.stdout)
sys.stderr = _force_utf8(sys.stderr)

from selenium import webdriver
from selenium.common.exceptions import (
    ElementNotInteractableException,
    NoAlertPresentException,
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


CNKI_HOME = "https://www.cnki.net/"


@dataclass
class CnkiRecord:
    title: str = ""
    authors: str = ""
    year: str = ""
    abstract: str = ""
    venue: str = ""
    doi: str = ""
    url: str = ""
    pdf_url: str = ""
    keywords: str = ""


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
    options.add_argument("--disable-gpu")
    options.add_argument("--start-maximized")
    options.add_argument("--lang=zh-CN")
    if chromedriver:
        return webdriver.Chrome(service=Service(chromedriver), options=options)
    return webdriver.Chrome(options=options)


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


def wait_first(driver: webdriver.Chrome, selectors: list[tuple[str, str]], timeout: int = 10):
    last_error: Exception | None = None
    for by, selector in selectors:
        try:
            return WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((by, selector))
            )
        except TimeoutException as exc:
            last_error = exc
    if last_error:
        raise last_error
    raise TimeoutException("No selectors supplied")


def find_text(driver: webdriver.Chrome, selectors: list[tuple[str, str]], timeout: int = 3) -> str:
    try:
        element = wait_first(driver, selectors, timeout=timeout)
        return normalize_text(element.text)
    except TimeoutException:
        return ""


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def dismiss_alert(driver: webdriver.Chrome) -> bool:
    """Dismiss a CNKI anti-scraping alert if one is open.

    CNKI intermittently pops a JS alert (e.g. "操作频繁") after a search.
    While it's open, every webdriver call throws UnexpectedAlertPresentException,
    so we must accept/dismiss it before doing anything else. Returns True if an
    alert was present and dismissed.
    """
    try:
        alert = driver.switch_to.alert
        alert.accept()
        time.sleep(1)
        return True
    except NoAlertPresentException:
        return False
    except WebDriverException:
        return False


def robust_click(driver: webdriver.Chrome, element) -> None:
    """Click an element, tolerating headless 'not interactable' failures.

    Native .click() refuses when the element is off-viewport or covered in
    headless mode; a JS click bypasses that. If CNKI pops an alert mid-click
    (it can appear between the click and the next call), dismiss it and retry.
    """
    try:
        element.click()
    except UnexpectedAlertPresentException:
        dismiss_alert(driver)
        try:
            driver.execute_script("arguments[0].click();", element)
        except UnexpectedAlertPresentException:
            dismiss_alert(driver)
    except (ElementNotInteractableException, WebDriverException):
        try:
            driver.execute_script("arguments[0].click();", element)
        except UnexpectedAlertPresentException:
            dismiss_alert(driver)


def search(driver: webdriver.Chrome, keyword: str, home_wait: float = 3.0) -> None:
    driver.get(CNKI_HOME)
    # Pause after landing on the CNKI home page so it fully loads before we
    # touch the search box — helps avoid tripping anti-scraping alerts.
    if home_wait > 0:
        time.sleep(home_wait)
    box = wait_first(
        driver,
        [
            (By.ID, "txt_SearchText"),
            (By.CSS_SELECTOR, "input[name='txt_SearchText']"),
            (By.CSS_SELECTOR, "input.search-input"),
        ],
        timeout=30,
    )
    box.clear()
    box.send_keys(keyword)
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
    # CNKI may pop an anti-scraping alert right after the search click.
    dismiss_alert(driver)

    # On the current CNKI, clicking input.search-btn does not always submit.
    # Detect whether we reached the result page; if not, press RETURN in the
    # search box (this reliably triggers the search) and retry a couple times.
    for _attempt in range(3):
        time.sleep(3)
        dismiss_alert(driver)
        if _on_result_page(driver):
            return
        try:
            box.send_keys(Keys.RETURN)
        except UnexpectedAlertPresentException:
            dismiss_alert(driver)
            box.send_keys(Keys.RETURN)
        time.sleep(3)
        dismiss_alert(driver)
        if _on_result_page(driver):
            return

    # Last resort: still no result page, but keep going so the caller can fail
    # loudly on missing result links rather than hanging here.
    time.sleep(2)
    dismiss_alert(driver)


def _on_result_page(driver: webdriver.Chrome) -> bool:
    try:
        url = driver.current_url
    except UnexpectedAlertPresentException:
        dismiss_alert(driver)
        return False
    return "defaultresult" in url or "kns8s" in url or "/kns/" in url


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
        except UnexpectedAlertPresentException:
            dismiss_alert(driver)
            continue
        except TimeoutException:
            continue
    return False


def switch_to_chinese_literature(driver: webdriver.Chrome) -> None:
    # CNKI UI changes frequently; failing to click this tab is not fatal.
    try_click_text(driver, "中文文献", timeout=5)


def switch_doc_type(driver: webdriver.Chrome, doc_type: str) -> None:
    if doc_type == "doctor":
        try_click_text(driver, "博士", timeout=5)
    elif doc_type == "master":
        try_click_text(driver, "硕士", timeout=5)
    elif doc_type == "journal":
        try_click_text(driver, "期刊", timeout=5)


def collect_result_urls(driver: webdriver.Chrome) -> list[str]:
    try:
        anchors = WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.fz14"))
        )
    except UnexpectedAlertPresentException:
        dismiss_alert(driver)
        anchors = WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.fz14"))
        )
    urls: list[str] = []
    for anchor in anchors:
        href = anchor.get_attribute("href")
        title = normalize_text(anchor.text)
        if href and title:
            urls.append(href)
    return list(dict.fromkeys(urls))


def parse_detail(driver: webdriver.Chrome, url: str) -> CnkiRecord:
    driver.execute_script("window.open(arguments[0]);", url)
    driver.switch_to.window(driver.window_handles[-1])
    try:
        title = find_text(
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
        return CnkiRecord(
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            venue=source_text,
            doi=doi,
            url=driver.current_url,
            pdf_url=pdf_url,
            keywords=keywords,
        )
    finally:
        if len(driver.window_handles) > 1:
            driver.close()
            driver.switch_to.window(driver.window_handles[0])


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


def append_records(path: Path, records: list[CnkiRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(CnkiRecord()).keys()))
        if not exists:
            writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def go_next_page(driver: webdriver.Chrome) -> bool:
    selectors = [
        (By.ID, "PageNext"),
        (By.CSS_SELECTOR, "a#PageNext"),
        (By.XPATH, "//a[contains(., '下一页')]"),
    ]
    for by, selector in selectors:
        try:
            button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((by, selector))
            )
            robust_click(driver, button)
            time.sleep(3)
            return True
        except TimeoutException:
            continue
    return False


def run(
    keyword: str,
    pages: int,
    output: Path,
    doc_type: str,
    headless: bool,
    chrome_binary: str | None,
    chromedriver: str | None,
    home_wait: float = 3.0,
) -> None:
    driver = build_driver(
        headless=headless,
        chrome_binary=chrome_binary,
        chromedriver=chromedriver,
    )
    try:
        search(driver, keyword, home_wait=home_wait)
        switch_to_chinese_literature(driver)
        switch_doc_type(driver, doc_type)

        for page in range(1, pages + 1):
            print(f"[page {page}] collecting result urls")
            urls = collect_result_urls(driver)
            print(f"[page {page}] found {len(urls)} urls")

            records: list[CnkiRecord] = []
            for index, url in enumerate(urls, start=1):
                try:
                    record = parse_detail(driver, url)
                    records.append(record)
                    print(f"  [{index}/{len(urls)}] OK {record.title[:60]}")
                except (TimeoutException, WebDriverException) as exc:
                    print(f"  [{index}/{len(urls)}] FAILED {url} {type(exc).__name__}: {exc}")
            append_records(output, records)
            print(f"[page {page}] saved {len(records)} records to {output}")

            if page < pages and not go_next_page(driver):
                print("No next page found; stopped.")
                break
    finally:
        driver.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", default="计算机")
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--output", default="data/cnki_smoke.csv")
    parser.add_argument("--chrome-binary", default="")
    parser.add_argument("--chromedriver", default="")
    parser.add_argument(
        "--doc-type",
        choices=["all", "journal", "master", "doctor"],
        default="all",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--home-wait",
        type=float,
        default=3.0,
        help="Seconds to pause after loading the CNKI home page (default 3).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        keyword=args.keyword,
        pages=max(1, args.pages),
        output=Path(args.output),
        doc_type=args.doc_type,
        headless=args.headless,
        chrome_binary=args.chrome_binary or None,
        chromedriver=args.chromedriver or None,
        home_wait=args.home_wait,
    )
