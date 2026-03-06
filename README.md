# House_price_prediction

This is my first Machine Learning project which i'm uploading. I built a model that can predict the selling price of a house based on four things — how much it cost to build, how old it is, whether it is in an urban or rural area, and how many rooms it has. I am a complete beginner and I wrote every line of code step by step while learning.

📖 What is this project?
I used a dataset of 1000 houses with their details and selling prices. I trained a Linear Regression model on this data. Linear Regression is the simplest machine learning algorithm — it finds a straight line that best fits all the data points and uses that line to predict prices for new houses.

🧹 Data Cleaning
The dataset had 50 missing values in the selling price column and some outliers — prices that were way too high or way too low. I filled the missing values using the median price because median is not affected by extreme values. I removed the outliers using the IQR method, which finds values that fall too far outside the normal range and caps them.

⚙️ How I Built It
First I converted the text column "Rural" and "Urban" into numbers — Rural became 0 and Urban became 1 — because machine learning models only understand numbers. Then I split the data into 800 houses for training and 200 houses for testing. I scaled all the features so they are on the same range, because making cost is in millions while rooms is just 1 to 8 — without scaling, the model gets confused. Then I trained the Linear Regression model and asked it to predict prices for the 200 test houses it had never seen before.

📊 Results
The model got an R² score of 0.81, which means it can explain 81% of why house prices are different from each other. The average prediction error (MAE) was around ₹6.26 Lakh. For a beginner's first model with just 4 features, I am really happy with this result!

📈 Graphs
I made three simple graphs. The first one is a bar chart showing which feature affects the price the most — making cost turned out to be the biggest factor. The second graph compares the actual prices with what my model predicted for 20 houses. The third graph shows how wrong the model was for each of those 20 houses — bars close to zero mean the prediction was close to the real price.

📁 Files in this Repo
The file house_price_beginner.py has all the code with simple comments explaining every single line. The file house_data.csv is the dataset with 1000 rows. The three PNG files are the graphs I generated.

🛠️ Libraries Used
I used pandas for reading and cleaning the data, numpy for math, matplotlib for drawing graphs, and scikit-learn for the machine learning model, the scaler, and the evaluation metrics.

🚀 How to Run
Install the required libraries by running pip install pandas numpy matplotlib scikit-learn in your terminal. Then place the house_data.csv file in the same folder as the Python file and run python house_price_beginner.py. It will clean the data, train the model, print all the results, and save the three graphs automatically.

💬 A Note from Me
I am just starting my journey in data science and machine learning. This project taught me how to clean messy data, how to train a model, and how to measure whether it is actually good or not. I know there is a lot more to learn but I am proud of building this from scratch. If you have any suggestions, feel free to open an issue.
Thank You
