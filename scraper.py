import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Set up Chrome options
chrome_options = Options()
# chrome_options.add_argument("--headless")  # Run Chrome in headless mode
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-gpu")

# Use WebDriver Manager to automatically manage the ChromeDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# Open the target URL
url = "https://www.immonet.de/classified-search?distributionTypes=Buy,Buy_Auction,Compulsory_Auction&estateTypes=House,Apartment&locations=AD08DE8634&order=Default&m=homepage_new_search_classified_search_result"
driver.get(url)

# Pause execution to allow you to manually solve the CAPTCHA
input("Please complete the CAPTCHA and press Enter to continue...")

# Optional: Wait for a few seconds to allow the page to load fully
time.sleep(1)

print(driver.page_source)  # Check if the expected elements are present

# Wait until the buttons load
wait = WebDriverWait(driver, 10)  # Increase wait time
try:
    # Wait for the button elements to be present
    buttons = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "css-oobyeg")))
    print(f"Found {len(buttons)} buttons.")

    # Iterate through the buttons and print their titles
    for button in buttons:
        title = button.get_attribute("title")
        print(title)

except Exception as e:
    print(f"An error occurred: {e}")

# Don't close the browser automatically to keep it open for debugging
# Close the WebDriver manually after confirming data collection
# driver.quit()

