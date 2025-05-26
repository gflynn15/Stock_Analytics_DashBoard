Hello and welcome to my Github Repository.
Below is a project I put together to assist me with improving my data pipeline development and analytical skill set. 
From this project I learned how to utilize an API for a datafeed, develop a dashboard to visualize the data, as well 
as how to deploy it to the web for others to see and utilize.

**Project Background:**
- This project's purpose was to assist with improving my knowledge of dashboard development and web based deployment.  

**Project Scope:**
- To deploy a live web based dashboard that provides end users with a simple tool to utilize to visualize company stock pricing information.

**Primary Dependencies Utilized:**
- Dash by Plotly : Dashboard Development Application
- Render : Web Server Host Application
- YFinance : Stock Pricing Information API
- Pandas : Data Manipulation
- BeautifulSoup4 : Webscraping

**Project Description:**
This Github Repository was developed as the file directory to be used for the Render application to access the Dashbaord Server information written within the dashbaord development script. 
Dash by plotly was utilized for the dashboard development because of it's ease of use to create html based graphs and visuals because of it's Java background. The Yfinance Webscrapping Library was 
utilized because it has reliable free data feeds do to it's ability to webscrap Yahoo Finance. This API also has a much higher rate limit tolerance for a free account compared to other API's.

**Link to Dashboard Visual:**
https://stock-analytics-dashboard-j4bx.onrender.com

**Work Flow Steps:**

1. Create a Github Repository
2. Install necessary libraries:
- dash, plotly, pandas, yfinance, requests, beautifulsoup4, python-dotenv, gunicorn, gevent, urllib3==2.0.7
3. Create a requirements.txt file
   - Copy and paste in all of the above libraries
   - Save the file in your github repository
4. Now the python script has to be developed
   1. Webscrap to create a list of stock symbols
   2. Create the instance of the dash application
   3. Develop the html layout
   4. Develop the callback functions
   5. Trigger the yfinance API within the callback function to extract the stock price data
5. Save this as a .py file in the github repository created earlier
6. Create an account with Render
7. Render will ask you to log into your GitHub Repository
8. Navigae back to Render and create a new web service application project
9. Name the web server application
10. Select the closest State relative to where you live
11. Input your github repository URL
12. Enter email for Git Credentials
13. Input (pip install -r requirements.txt) in the first build command field
14. Input (gunicorn Your_.py_File_Name_Here:server --worker-class gevent --workers 1)
15. Then Deploy the application
16. After it is deployed you can select the url below your profile icon in Render 


