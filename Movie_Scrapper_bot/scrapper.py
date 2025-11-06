import asyncio
from crawl4ai import AsyncWebCrawler, PruningContentFilter
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, DefaultMarkdownGenerator

async def main():
    browser_config = BrowserConfig()  # Default browser configuration
    run_config = CrawlerRunConfig( markdown_generator=DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.6),
        options={"ignore_links": True}))   # Default crawl run configuration

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url = "https://in.bookmyshow.com/movies/pune",
            config=run_config
        )
        print(result.cleaned_html)
    

if __name__ == "__main__":
    asyncio.run(main())
    