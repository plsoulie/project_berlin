import time
import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Set up Chrome options
chrome_options = Options()
# chrome_options.add_argument("--headless")  # Comment this line out to see the browser
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-gpu")

# Specify the path to your Chrome user data directory
user_data_dir = "/Users/user/Library/Application Support/Google/Chrome"  # Update this path
chrome_options.add_argument(f"user-data-dir={user_data_dir}")
chrome_options.add_argument("profile-directory=Default")  

# Use WebDriver Manager to automatically manage the ChromeDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# Open the target URL
url = "https://www.immonet.de/classified-search?distributionTypes=Buy,Buy_Auction,Compulsory_Auction&estateTypes=House,Apartment&locations=AD08DE8634&order=Default&m=homepage_new_search_classified_search_result"
driver.get(url)

# Optional: Wait for a few seconds to allow the page to load
time.sleep(random.uniform(2, 5))  # Random sleep

# Wait until the buttons load
wait = WebDriverWait(driver, 20)  # Increase wait time
try:
    buttons = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "css-oobyeg")))
    print(f"Found {len(buttons)} buttons.")

    for button in buttons:
        title = button.get_attribute("title")
        print(title)
        
        # Introduce a random delay between processing each button
        time.sleep(random.uniform(1, 3))  # Sleep for a random time between 1 and 3 seconds

except Exception as e:
    print(f"An error occurred: {str(e)}")
    input("Please solve the CAPTCHA and press Enter to continue...")  # Wait for user to solve CAPTCHA

finally:
    driver.quit()
