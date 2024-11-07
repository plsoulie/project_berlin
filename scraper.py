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

    # Loop through the desired number of pages (e.g., 1 to 5)
    for page in range(20, 200):  # Change 200 to the desired number of pages
        # Choose a random proxy from the list for each page
        proxy = random.choice(proxies)
        print(f"Using proxy: {proxy}")  # Debugging: Log the entire proxy string
        
        # Split the proxy string into components
        host, port, username, password = proxy.split(":")
        print(f"Using proxy: {host}:{port} with username: {username} for page {page}")

        # Set up Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-gpu")
        
        # Set up selenium-wire options with proxy details
        proxy_options = {
            'proxy': {
                'http': f'http://{username}:{password}@{host}:{port}',
                'https': f'https://{username}:{password}@{host}:{port}',
                'no_proxy': 'localhost,127.0.0.1'  # Exclude local addresses
            }
        }

        # Initialize WebDriver with proxy settings
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options, seleniumwire_options=proxy_options)
        
        url = f"{base_url}{page}"  # Increment the page number in the URL
        driver.get(url)

        # Wait for a random delay to avoid detection
        time.sleep(random.uniform(5, 10))  # Random delay between 5 and 10 seconds

        # Write the page header to the file
        output_file.write(f"\n--- Page {page} ---\n")

        # Wait until the buttons load
        wait = WebDriverWait(driver, 5)
        try:
            # Wait for the buttons to load
            buttons = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "css-oobyeg")))

            print(f"Found {len(buttons)} buttons.")

            for button in buttons:
                # Extract the title from the button
                title = button.get_attribute("title")
                print(title)  # For debugging purposes

                output_file.write(f"{title}\n")
                
                # Random delay between processing each button
                time.sleep(random.uniform(1, 3))  # Sleep for a random time between 1 and 3 seconds

        except Exception as e:
            print(f"An error occurred on page {page}: {str(e)}")
            output_file.write(f"Error on page {page}: {str(e)}\n")

        addresses = driver.execute_script('return document.querySelectorAll(".css-ee7g92");')
        for element in addresses:  # Changed from forEach to a Python for loop
            address = element.get_attribute('innerHTML')  # Use get_attribute to access innerHTML
            output_file.write(f"{address}\n")
            print(address) 

        # Close the driver after each page to avoid reusing the same session
        driver.quit()

print(f"Data saved to {output_file_path}")
