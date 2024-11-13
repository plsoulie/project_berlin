from seleniumwire import webdriver  # Import from selenium-wire instead of selenium
import time
import random
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Load proxies from the text file
with open("proxies-berlin.txt", "r") as file:
    proxies = [line.strip() for line in file if line.strip()]

# Define the base URL without the page number
base_url = "https://www.immonet.de/classified-search?distributionTypes=Buy,Buy_Auction,Compulsory_Auction&estateTypes=Apartment&locations=AD08DE8634&page="

# Open the output file in append mode to save data
output_file_path = "data-appartement.txt"
with open(output_file_path, "a", encoding="utf-8") as output_file:

    for page in range(21, 22):  # Loop through the desired number of pages
        proxy = random.choice(proxies)
        host, port, username, password = proxy.split(":")
        print(f"Using proxy: {host}:{port} with username: {username} for page {page}")

        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-gpu")
        proxy_options = {
            'proxy': {
                'http': f'http://{username}:{password}@{host}:{port}',
                'https': f'https://{username}:{password}@{host}:{port}',
                'no_proxy': 'localhost,127.0.0.1'
            }
        }

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options, seleniumwire_options=proxy_options)
        url = f"{base_url}{page}"
        driver.get(url)

        # Pause and prompt manual CAPTCHA solving if detected
        if "captcha" in driver.page_source.lower():
            input(f"CAPTCHA detected on page {page}. Please solve it and press Enter to continue...")

        time.sleep(random.uniform(5, 10))  # Random delay

        output_file.write(f"\n--- Page {page} ---\n")
        wait = WebDriverWait(driver, 5)
        try:
            buttons = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "css-oobyeg")))
            print(f"Found {len(buttons)} buttons.")
            for button in buttons:
                title = button.get_attribute("title")
                print(title)
                output_file.write(f"{title}\n")
                time.sleep(random.uniform(1, 3))

        except Exception as e:
            print(f"An error occurred on page {page}: {str(e)}")
            output_file.write(f"Error on page {page}: {str(e)}\n")

        # Attempt to extract addresses
        try:
            addresses = driver.execute_script('return document.querySelectorAll(".css-ee7g92");')
            for element in addresses:
                address = element.get_attribute('innerHTML')
                output_file.write(f"{address}\n")
                print(address)
        except Exception as e:
            print(f"Error extracting addresses on page {page}: {str(e)}")

        driver.quit()

print(f"Data saved to {output_file_path}")
