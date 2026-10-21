#!/usr/bin/env python3
"""
regulatory_pipeline_scraper.py

Declarative data extraction pipeline built on Playwright.

The script drives a headless Chromium instance against a client-side rendered
data portal. Instead of parsing the DOM, it listens on the browser network
layer and captures the raw JSON payloads that the frontend itself consumes.
Pagination is driven by waiting for the completion of a specific network
response rather than by fixed sleep timers, which removes the race conditions
that occur when the table re-renders without a full page reload.

Output:
  - a normalized JSON dataset
  - an optional full-page screenshot of the hydrated browser session

Usage:
  python regulatory_pipeline_scraper.py --source quotes
  python regulatory_pipeline_scraper.py --source films
  python regulatory_pipeline_scraper.py --source custom --url URL --match PATTERN \
      --next-selector CSS [--record-key KEY]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Locator, Page, Response, async_playwright


def ms(seconds: float) -> float:
    """Playwright timeouts are expressed in milliseconds."""
    return seconds * 1000


def short_url(url: str) -> str:
    """Strip the scheme so captured URLs print on one line."""
    return url.split("://", 1)[-1]


def report_view(status: int, url: str, count: int) -> None:
    """Print one compact line per captured page of records."""
    print(f"  [intercept {status}] {short_url(url)} -> {count} records")


# --------------------------------------------------------------------------
# Source definitions
# --------------------------------------------------------------------------


@dataclass
class Source:
    """Describes one target portal and how to drive it."""

    name: str
    label: str
    url: str
    # Substring matched against the response URL of the backing API.
    api_match: str
    # How the portal advances: "next" (next-page control), "scroll" (infinite
    # scroll that triggers a fetch), or "sequence" (fixed list of controls).
    strategy: str = "next"
    # Locator for the control that advances to the next page of records.
    next_selector: str | None = None
    # Locator for a row element, used to confirm the table finished hydrating.
    row_selector: str | None = None
    # Key holding the record list inside the intercepted JSON payload.
    records_key: str | None = None
    # Fixed sequence of controls to click instead of a next-page button.
    click_sequence: list[tuple[str, str]] = field(default_factory=list)
    # Maps one raw record to the normalized output record.
    normalize: Callable[[Any], dict[str, Any]] | None = None


def _normalize_quote(raw: Any) -> dict[str, Any]:
    author = raw.get("author") or {}
    return {
        "text": (raw.get("text") or "").strip(),
        "author": author.get("name", ""),
        "author_slug": author.get("slug", ""),
        "tags": list(raw.get("tags") or []),
    }


def _normalize_film(raw: Any) -> dict[str, Any]:
    return {
        "title": (raw.get("title") or "").strip(),
        "year": raw.get("year"),
        "awards": raw.get("awards"),
        "nominations": raw.get("nominations"),
        "best_picture": bool(raw.get("best_picture", False)),
    }


SOURCES: dict[str, Source] = {
    "quotes": Source(
        name="quotes",
        label="scraping sandbox: quote registry (quotes.toscrape.com)",
        url="http://quotes.toscrape.com/scroll",
        api_match="/api/quotes",
        strategy="scroll",
        next_selector="li.next a",
        row_selector="div.quote",
        records_key="quotes",
        normalize=_normalize_quote,
    ),
    "films": Source(
        name="films",
        label="scraping sandbox: Oscar winners tables (scrapethissite.com)",
        url="https://www.scrapethissite.com/pages/ajax-javascript/",
        api_match="ajax=true",
        strategy="sequence",
        row_selector="tbody#table-body tr.film",
        records_key=None,
        normalize=_normalize_film,
        click_sequence=[
            ('.year-link[id="2015"]', "2015"),
            ('.year-link[id="2014"]', "2014"),
            ('.year-link[id="2013"]', "2013"),
            ('.year-link[id="2012"]', "2012"),
            ('.year-link[id="2011"]', "2011"),
            ('.year-link[id="2010"]', "2010"),
        ],
    ),
}


# --------------------------------------------------------------------------
# Network interception helpers
# --------------------------------------------------------------------------


class PayloadCollector:
    """
    Captures JSON responses served by the target API.

    Responses are collected asynchronously as the browser produces them, so a
    payload is never missed between the click and the network event.
    """

    def __init__(self, match: str) -> None:
        self._match = match
        self.captured: list[dict[str, Any]] = []

    def matches(self, response: Response) -> bool:
        return self._match in response.url

    async def handle(self, response: Response) -> None:
        if not self.matches(response):
            return
        try:
            body = await response.json()
        except (PlaywrightError, ValueError):
            return
        if not isinstance(body, (dict, list)):
            return
        self.captured.append(
            {"url": response.url, "status": response.status, "body": body}
        )


def extract_records(payload: dict[str, Any], source: Source) -> list[Any]:
    """Pull the record list out of an intercepted payload."""
    body = payload.get("body")
    if source.records_key:
        if isinstance(body, dict):
            return list(body.get(source.records_key) or [])
        return []
    if isinstance(body, list):
        return list(body)
    if isinstance(body, dict):
        for value in body.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return list(value)
    return []


def has_next_page(payload: dict[str, Any]) -> bool:
    """Read the pagination flag from the payload when the API exposes one."""
    body = payload.get("body")
    if isinstance(body, dict) and "has_next" in body:
        return bool(body["has_next"])
    return True


# --------------------------------------------------------------------------
# Scrape strategies
# --------------------------------------------------------------------------


async def advance_control_state(control: Locator) -> str:
    """
    Report whether the next-page control can be used.

    Returns "ok", "missing", "disabled", or "error". Checking the control
    before clicking avoids acting on a stale element.
    """
    try:
        if not await control.count():
            return "missing"
        if await control.first.is_disabled():
            return "disabled"
        return "ok"
    except PlaywrightError:
        return "error"


async def capture_payload(
    page: Page,
    collector: PayloadCollector,
    timeout: float,
    action: Callable[[], Any],
) -> dict[str, Any]:
    """
    Run an action and capture the API payload it triggers.

    The expectation is armed before the action runs, so the response cannot be
    missed between the trigger and the network event. This is the core of the
    race condition fix: no sleep timers, only explicit state changes.
    """
    async with page.expect_response(
        lambda r: collector.matches(r) and r.status == 200, timeout=ms(timeout)
    ) as info:
        await action()
    response = await info.value
    return {
        "url": response.url,
        "status": response.status,
        "body": await response.json(),
    }


async def scrape_with_next_button(
    page: Page,
    source: Source,
    collector: PayloadCollector,
    max_pages: int,
    timeout: float,
) -> list[Any]:
    """
    Click through a paginated table, one network response at a time.

    Every click is wrapped in an expect_response guard, so the next click only
    fires once the previous page payload has fully arrived. No sleep timers.
    """
    records: list[Any] = []

    # Page 1: capture the payload produced by the initial document load.
    async with page.expect_response(
        lambda r: collector.matches(r) and r.status == 200, timeout=ms(timeout)
    ) as first_info:
        await page.goto(source.url, wait_until="domcontentloaded")
    first_response = await first_info.value
    payload = {
        "url": first_response.url,
        "status": first_response.status,
        "body": await first_response.json(),
    }
    page_records = extract_records(payload, source)
    records.extend(page_records)
    report_view(payload["status"], payload["url"], len(page_records))

    for page_number in range(2, max_pages + 1):
        if not has_next_page(payload):
            print("  pagination flag reported no further pages, stopping")
            break

        if not source.next_selector:
            print("  no next-page control configured, stopping")
            break

        next_button = page.locator(source.next_selector)
        state = await advance_control_state(next_button)
        if state != "ok":
            print(f"  next-page control is {state}, stopping")
            break

        # Wait for the network response that the click triggers before moving on.
        async with page.expect_response(
            lambda r: collector.matches(r) and r.status == 200, timeout=ms(timeout)
        ) as next_info:
            await next_button.first.click()
        response = await next_info.value
        payload = {
            "url": response.url,
            "status": response.status,
            "body": await response.json(),
        }
        page_records = extract_records(payload, source)
        records.extend(page_records)
        report_view(payload["status"], payload["url"], len(page_records))

    return records


async def scrape_with_scroll(
    page: Page,
    source: Source,
    collector: PayloadCollector,
    max_pages: int,
    timeout: float,
) -> list[Any]:
    """
    Drive an infinite-scroll portal.

    The page requests the next batch when the viewport reaches the bottom.
    Every scroll is wrapped in an expect_response guard, so the next scroll
    only fires once the previous payload has fully arrived.
    """
    records: list[Any] = []

    payload = await capture_payload(
        page,
        collector,
        timeout,
        lambda: page.goto(source.url, wait_until="domcontentloaded"),
    )
    page_records = extract_records(payload, source)
    records.extend(page_records)
    report_view(payload["status"], payload["url"], len(page_records))

    scroll_timeout = min(timeout, 10.0)

    for page_number in range(2, max_pages + 1):
        if not has_next_page(payload):
            print("  pagination flag reported no further pages, stopping")
            break

        try:
            payload = await capture_payload(
                page,
                collector,
                scroll_timeout,
                lambda: page.evaluate(
                    "window.scrollTo(0, document.documentElement.scrollHeight)"
                ),
            )
        except PlaywrightError:
            print("  no further network activity after scrolling, stopping")
            break

        page_records = extract_records(payload, source)
        records.extend(page_records)
        report_view(payload["status"], payload["url"], len(page_records))

    return records


async def scrape_with_click_sequence(
    page: Page,
    source: Source,
    collector: PayloadCollector,
    max_pages: int,
    timeout: float,
) -> list[Any]:
    """Click a fixed sequence of controls, capturing the payload of each."""
    records: list[Any] = []

    await page.goto(source.url, wait_until="domcontentloaded")

    for selector, _ in source.click_sequence[:max_pages]:
        async with page.expect_response(
            lambda r: collector.matches(r) and r.status == 200, timeout=ms(timeout)
        ) as info:
            await page.locator(selector).click()
        response = await info.value
        payload = {
            "url": response.url,
            "status": response.status,
            "body": await response.json(),
        }
        page_records = extract_records(payload, source)
        records.extend(page_records)
        report_view(payload["status"], payload["url"], len(page_records))

        # Confirm the table finished hydrating before the next click.
        if source.row_selector:
            with contextlib.suppress(PlaywrightError):
                await page.wait_for_selector(source.row_selector, timeout=ms(timeout))

    return records


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract structured records from a dynamic data portal with Playwright."
    )
    parser.add_argument(
        "--source",
        choices=["quotes", "films", "custom"],
        default="quotes",
        help="Built-in portal to scrape, or 'custom' to supply your own.",
    )
    parser.add_argument("--url", help="Target URL when --source custom is used.")
    parser.add_argument(
        "--match",
        help="Substring matched against API response URLs when --source custom is used.",
    )
    parser.add_argument(
        "--next-selector", help="Next-page control for --source custom."
    )
    parser.add_argument("--row-selector", help="Row element used to confirm hydration.")
    parser.add_argument(
        "--records-key", help="Key holding the record list in the payload."
    )
    parser.add_argument(
        "--max-pages", type=int, default=10, help="Maximum pages to collect."
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="Network wait timeout in seconds."
    )
    parser.add_argument(
        "--output",
        default="output/regulatory_records.json",
        help="Path of the exported JSON dataset.",
    )
    parser.add_argument(
        "--screenshot",
        default="output/scraper_browser.png",
        help="Path of the browser screenshot.",
    )
    parser.add_argument(
        "--headed", action="store_true", help="Run with a visible browser window."
    )
    parser.add_argument(
        "--no-screenshot", action="store_true", help="Skip the screenshot capture."
    )
    parser.add_argument(
        "--full-page",
        action="store_true",
        help="Capture the whole scrollable page instead of the viewport.",
    )
    return parser


def resolve_source(args: argparse.Namespace) -> Source:
    if args.source != "custom":
        return SOURCES[args.source]
    if not args.url:
        raise SystemExit("--source custom requires --url")
    return Source(
        name="custom",
        label=args.url,
        url=args.url,
        api_match=args.match or "/api/",
        next_selector=args.next_selector,
        row_selector=args.row_selector,
        records_key=args.records_key,
        normalize=lambda raw: raw if isinstance(raw, dict) else {"value": raw},
    )


async def run(args: argparse.Namespace) -> int:
    source = resolve_source(args)
    collector = PayloadCollector(source.api_match)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not args.headed)
        context = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        # Attach the interceptor before any navigation so no payload is missed.
        page.on("response", collector.handle)

        print(f"Target: {source.label}")
        print(f"URL:    {source.url}")
        print("Launching headless Chromium, network interceptor armed\n")

        try:
            if source.strategy == "sequence":
                raw_records = await scrape_with_click_sequence(
                    page, source, collector, args.max_pages, args.timeout
                )
            elif source.strategy == "scroll":
                raw_records = await scrape_with_scroll(
                    page, source, collector, args.max_pages, args.timeout
                )
            else:
                raw_records = await scrape_with_next_button(
                    page, source, collector, args.max_pages, args.timeout
                )
        except PlaywrightError as exc:
            print(f"Playwright error: {exc}", file=sys.stderr)
            await browser.close()
            return 1

        if source.row_selector:
            with contextlib.suppress(PlaywrightError):
                await page.wait_for_selector(
                    source.row_selector, timeout=ms(args.timeout)
                )

        if not args.no_screenshot:
            # Return to the top first so the capture does not retrigger the
            # portal's infinite-scroll handler.
            with contextlib.suppress(PlaywrightError):
                await page.evaluate("window.scrollTo(0, 0)")
            shot_path = Path(args.screenshot)
            shot_path.parent.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=str(shot_path), full_page=args.full_page)
            print(f"\nScreenshot saved to {shot_path}")

    # Merge the records collected during the run with any payload that was
    # intercepted afterwards, then normalize and de-duplicate.
    for intercepted in collector.captured:
        raw_records.extend(extract_records(intercepted, source))

    normalizer = source.normalize or (lambda raw: raw)
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for raw in raw_records:
        normalized = normalizer(raw)
        key = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        records.append(normalized)

    dataset = {
        "source": source.label,
        "source_url": source.url,
        "generated_at": datetime.now(UTC).isoformat(),
        "api_requests_intercepted": len(collector.captured),
        "record_count": len(records),
        "records": records,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"API responses intercepted: {len(collector.captured)}")
    print(f"Unique records exported:   {len(records)}")
    print(f"Dataset written to:        {out_path}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
