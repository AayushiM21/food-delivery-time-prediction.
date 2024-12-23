#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px


# ### Reading the data from the text file

# In[2]:


data = pd.read_csv("C:/Users/Mahendru/Downloads/Delivery-time/Delivery time/deliverytime.txt")
data.head()


# ### Number of rows

# In[15]:


len(data)


# ### Describing the dataframe

# In[4]:


data.info()


# ### Checking for null values

# In[5]:


data.isnull().sum()


# ### Checking for duplicates if any

# In[6]:


duplicates = data.duplicated()
print(duplicates.sum()) 


# ### Data Consistency - Cleaning categorical columns

# In[7]:


data['Type_of_order'] = data['Type_of_order'].str.strip().str.lower()
data['Type_of_vehicle'] = data['Type_of_vehicle'].str.strip().str.lower()
data.head(1000)


# ### Calculating the distance between restaurant and delivery location

# In[8]:


# Set the Earth's radius (in kilometers)
R = 6371

# Convert degrees to radians
def deg_to_rad(degrees):
    return degrees * (np.pi / 180)

# Function to calculate the distance using the Haversine formula
def calculate_distance(lat1, lon1, lat2, lon2):
    d_lat = deg_to_rad(lat2 - lat1)
    d_lon = deg_to_rad(lon2 - lon1)
    a = np.sin(d_lat / 2) ** 2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Load your dataset
data = pd.read_csv('C:/Users/Mahendru/Downloads/Delivery-time/Delivery time/deliverytime.txt')

# Calculate the 'distance' column
data['distance'] = calculate_distance(
    data['Restaurant_latitude'], 
    data['Restaurant_longitude'], 
    data['Delivery_location_latitude'], 
    data['Delivery_location_longitude']
)
data.head()


# ### Distribution of Delivery Times (data visualisation)

# In[9]:


import plotly.express as px
fig = px.histogram(data, x='Time_taken(min)', nbins=30, title='Distribution of Delivery Times', 
                   marginal="box", opacity=0.7, color_discrete_sequence=['blue'])
fig.update_layout(bargap=0.2, xaxis_title='Time Taken (min)', yaxis_title='Frequency')
fig.show() 


# ### Relationship Between Distance and Time Taken

# In[10]:


figure = px.scatter(data_frame = data, 
                    x="distance",
                    y="Time_taken(min)", 
                    size="Time_taken(min)", 
                    trendline="ols", 
                    title = "Relationship Between Distance and Time Taken")
figure.show()


# ### Delivery Time Distribution by Vehicle Type and Order Type

# In[16]:


fig = px.box(
    data, 
    x="Type_of_vehicle", 
    y="Time_taken(min)", 
    color="Type_of_order", 
    labels={
        "Type_of_vehicle": "Type of Vehicle", 
        "Time_taken(min)": "Time Taken (minutes)", 
        "Type_of_order": "Type of Order"
    }
)

# Add a title to the plot
fig.update_layout(
    title="Delivery Time Distribution by Vehicle Type and Order Type",
    xaxis_title="Type of Vehicle",
    yaxis_title="Time Taken (minutes)",
    template="plotly_white"
)

fig.show()


# ### Relationship Between Time Taken and Age

# In[12]:


figure = px.scatter(data_frame = data, 
                    x="Delivery_person_Age",
                    y="Time_taken(min)", 
                    size="Time_taken(min)", 
                    color = "distance",
                    trendline="ols", 
                    title = "Relationship Between Time Taken and Age")
figure.show()


# ### Predicting the delivery time by taking input of age , rating of delivery partner and distance

# In[14]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Select features and target variable
features = ['Delivery_person_Age', 'Delivery_person_Ratings', 'distance']
target = 'Time_taken(min)'

X = data[features]
y = data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Function to predict delivery time based on user input
def predict_delivery_time(age, rating, distance):
    # Prepare the input
    input_data = np.array([[age, rating, distance]])
    
    # Make prediction
    predicted_time = model.predict(input_data)
    return predicted_time[0]

# Get user input
print("\n--- Predict Delivery Time ---")
user_age = int(input("Enter the age of the delivery person: "))
user_rating = float(input("Enter the rating of the delivery person (e.g., 4.5): "))
user_distance = float(input("Enter the distance between the restaurant and delivery location (in km): "))

# Predict and display the result
predicted_time = predict_delivery_time(user_age, user_rating, user_distance)
print(f"\nPredicted Delivery Time: {predicted_time:.2f} minutes")


# In[ ]:




