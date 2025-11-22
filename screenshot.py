import os
import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ---------- CONFIG ----------
HTML_FOLDER = r"D:\HUNG-DUNG\work\BigDataChallenge\hotspots_2025-04-03"
OUTPUT_FOLDER = r"D:\HUNG-DUNG\work\BigDataChallenge\hotspots_2025-04-03_screenshots"
WIDTH = 1920
HEIGHT = 1080
# ----------------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.set_window_size(WIDTH, HEIGHT)

for file in os.listdir(HTML_FOLDER):
    if file.endswith(".html") or file.endswith(".htm"):
        path = os.path.join(HTML_FOLDER, file)
        url = "file:///" + path.replace("\\", "/")
        print("Opening:", url)

        driver.get(url)
        driver.implicitly_wait(1)
        time.sleep(2)
        screenshot_path = os.path.join(OUTPUT_FOLDER, file + ".png")
        driver.save_screenshot(screenshot_path)
        print("Saved:", screenshot_path)

driver.quit()
print("DONE!")
