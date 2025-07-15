from playwright.sync_api import sync_playwright


def parse_coin_list(raw_text: str):
    lines = raw_text.strip().split("\n")
    parsed = []

    # Step through 4 lines at a time
    for i in range(0, len(lines), 4):
        try:
            coin = lines[i].strip()
            price = float(lines[i + 1].strip())
            change = float(lines[i + 2].replace('%', '').replace('+', '').strip())
            volume = lines[i + 3].strip()
            parsed.append({
                "coin": coin,
                "price": price,
                "change_percent": change,
                "volume": volume
            })
        except Exception as e:
            print(f"❌ Failed to parse entry starting at line {i}: {e}")

    return parsed


def extract_zoomex_pairs():
    with sync_playwright() as p:
        
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        page.goto("https://www.zoomex.finance/trade/usdt/BTCUSDT", timeout=60000)
        page.wait_for_timeout(5000)

        # Click the trading pair dropdown
        page.locator("text=BTCUSDT").first.click()
        page.wait_for_timeout(2000)

        # Switch to "New" tab
        #page.locator("text=New").first.click()
        #page.wait_for_timeout(2000)

        # Wait for the box to appear
        page.wait_for_selector(".book-symbol-table__box", timeout=10000)

        # Extract all the text
        content = page.locator(".book-symbol-table__box").inner_text()
        print("\n🧠 Extracted Content:\n", content)

        parsed = parse_coin_list(content)

        browser.close()

        return parsed


if __name__ == "__main__":
    parsed = extract_zoomex_pairs()
    print(json.dumps(parsed))
