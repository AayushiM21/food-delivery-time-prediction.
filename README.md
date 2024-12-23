# food-delivery-time-prediction.
A Python project to analyze and predict food delivery times using Random Forest.
Overview
This project analyzes and predicts the time taken for food delivery based on several factors, such as the distance between the restaurant and the delivery location, the age and rating of the delivery person, and other relevant variables. The model leverages machine learning to provide accurate predictions and optimize delivery operations.

Dataset

The dataset includes the following columns:

ID: Unique identifier for each delivery.
Delivery_person_ID: ID of the delivery personnel.
Delivery_person_Age: Age of the delivery person.
Delivery_person_Ratings: Ratings given to the delivery person by customers.
Restaurant_latitude and longitude: Coordinates of the restaurant.
Delivery_location_latitude and longitude: Coordinates of the delivery destination.
Type_of_order: Categories of food ordered (e.g., Veg, Non-Veg, Beverages).
Type_of_vehicle: Mode of delivery (e.g., Bike, Car, Bicycle).
Time_taken (min): Actual time taken for the delivery in minutes (target variable).

Programming Language: Python

Libraries Used:

Data Analysis: Pandas, NumPy
Visualization: Matplotlib, Seaborn, Plotly
Machine Learning: Scikit-learn
