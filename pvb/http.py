"""A small, polite HTTP client.

It identifies itself, waits between requests, retries temporary errors and
refuses URLs that robots.txt disallows. Every download in the project goes
through `PoliteSession`, so these rules are applied in one place.
"""

from __future__ import annotations

import logging
import time
from urllib import robotparser
from urllib.parse import urlsplit

import requests

from . import config

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}
MIN_BACKOFF = 5  # seconds before the first retry, even if the delay is set lower


class PoliteSession:
    def __init__(
        self,
        delay: float = config.REQUEST_DELAY,
        user_agent: str = config.USER_AGENT,
        retries: int = 3,
        timeout: float = 30,
    ):
        self.delay = delay
        self.retries = retries
        self.timeout = timeout
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self._robots: dict[str, robotparser.RobotFileParser] = {}
        self._last_request = 0.0

    def allowed(self, url: str) -> bool:
        """Check robots.txt for this URL. The rules are read once per site."""
        parts = urlsplit(url)
        site = f"{parts.scheme}://{parts.netloc}"
        if site not in self._robots:
            rules = self._read_robots(site)
            if rules is None:  # robots.txt could not be read: say no for now, ask again next time
                return False
            self._robots[site] = rules
        return self._robots[site].can_fetch(self.user_agent, url)

    def get(self, url: str, params: dict | None = None) -> requests.Response:
        """GET a URL politely. Raises PermissionError if robots.txt says no."""
        if not self.allowed(url):
            raise PermissionError(f"robots.txt disallows {url} (or could not be read)")
        for attempt in range(1, self.retries + 1):
            try:
                resp = self._throttled_get(url, params)
            except requests.RequestException as exc:
                if attempt == self.retries:
                    raise
                log.warning("%s failed (%s), retrying", url, exc)
                self._back_off(attempt)
                continue
            if resp.status_code in RETRY_STATUS and attempt < self.retries:
                log.warning("%s returned %s, retrying", url, resp.status_code)
                self._back_off(attempt, resp.headers.get("Retry-After"))
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError("unreachable")

    def _read_robots(self, site: str) -> robotparser.RobotFileParser | None:
        rules = robotparser.RobotFileParser()
        try:
            resp = self._throttled_get(site + "/robots.txt")
        except requests.RequestException as exc:
            log.warning("could not read %s/robots.txt: %s", site, exc)
            return None
        if resp.status_code >= 500:
            log.warning("%s/robots.txt returned %s", site, resp.status_code)
            return None
        if resp.status_code in (401, 403):  # access to the rules is refused: treat as "keep out"
            rules.parse(["User-agent: *", "Disallow: /"])
        elif resp.ok:
            rules.parse(resp.text.splitlines())
        else:  # 404 and similar: the site has no rules
            rules.parse([])
        return rules

    def _back_off(self, attempt: int, retry_after: str | None = None) -> None:
        wait = max(self.delay, MIN_BACKOFF) * 2 ** (attempt - 1)
        if retry_after and retry_after.isdigit():
            wait = max(wait, int(retry_after))
        time.sleep(wait)

    def _throttled_get(self, url: str, params: dict | None = None) -> requests.Response:
        wait = self.delay - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            return self.session.get(url, params=params, timeout=self.timeout)
        finally:
            self._last_request = time.monotonic()
