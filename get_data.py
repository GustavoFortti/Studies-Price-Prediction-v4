from time import sleep
from selenium import webdriver
from bs4 import BeautifulSoup as bs4
import os
from sys import path

driver = webdriver.Chrome()
driver.get('url')
driver.implicitly_wait(100)