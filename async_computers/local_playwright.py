from playwright.sync_api import Browser, Page
from .base_playwright import AsyncBasePlaywrightComputer


class LocalPlaywrightComputer(AsyncBasePlaywrightComputer):
    """Launches a local Chromium instance using Playwright."""

    def __init__(self, headless: bool = False):
        super().__init__()
        self.headless = headless

    async def _get_browser_and_page(self) -> tuple[Browser, Page]:
        width, height = self.dimensions
        launch_args = [f"--window-size={width},{height}", "--disable-extensions", "--disable-file-system"]
        browser = await self._playwright.chromium.launch(
            chromium_sandbox=True,
            headless=self.headless,
            args=launch_args,
            env={"DISPLAY": ":0"}
        )
        
        context = await browser.new_context()
        
        # Add event listeners for page creation and closure
        context.on("page", self._handle_new_page)
        
        page = await context.new_page()
        await page.set_viewport_size({"width": width, "height": height})
        page.on("close", self._handle_page_close)

        await page.goto("https://bing.com")
        
        return browser, page
        
    async def _handle_new_page(self, page: Page):
        """Handle the creation of a new page."""
        print("New page created")
        self._page = page
        page.on("close", self._handle_page_close)
        
    def _handle_page_close(self, page: Page):
        """Handle the closure of a page."""
        print("Page closed")
        if self._page == page:
            if self._browser.contexts[0].pages:
                self._page = self._browser.contexts[0].pages[-1]
            else:
                print("Warning: All pages have been closed.")
                self._page = None
