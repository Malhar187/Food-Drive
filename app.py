import sklearn
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# Load the dataset with a specified encoding
data = pd.read_csv('data_2024_donation_processed.csv', encoding='latin1')

# Page 1: Dashboard
def dashboard():
    st.image('Logo.png', use_column_width=True)

    st.subheader("💡 Abstract:")

    inspiration = '''
    The Edmonton Food Drive Project \n
    Project Overview: \n
    Given a data set about the food drive in Edmonton and we have to collect , clean, process , use the data to get the information about the volunteers and the food collected in the city. This helps the management in allocating the drivers in the future food collection.
    Problem Statement: \n 
    There are issues with the way Edmonton's existing food donation management system arranges drop-off sites, pick-up procedures, and route planning. In order to avoid logistical complications and guarantee timely donation collection, it is necessary to automate and streamline these operations.

    '''

    st.write(inspiration)

    st.subheader("👨🏻‍💻 What our Project Does?")

    what_it_does = '''
    Our project aims to predict the number of donation bags collected, by getting training from the 2023 Food Drive Dataset, and then making predictions on the 2024 Food Drive Dataset. 
    '''

    st.write(what_it_does)


# Page 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")
    # Rename columns for clarity
    data_cleaned = data.rename(columns={
        'Drop Off Location': 'Location',
        'City': 'City',
        'Stake': 'Stake',
        'Route Number/Name': 'Route',
        '# of Adult Volunteers who participated in this route': '# of Adult Volunteers',
        '# of Youth Volunteers who participated in this route': '# of Youth Volunteers',
        '# of Donation Bags Collected': 'Donation Bags Collected',
        'Time Spent Collecting Donations': 'Time to Complete (min)',
        'TotalRoutes': 'Routes Completed',
        '# of Doors in Route': 'Doors in Route'
    })

    # Visualize the distribution of numerical features using Plotly
    fig = px.histogram(data_cleaned, x='# of Adult Volunteers', nbins=20, labels={'# of Adult Volunteers': 'Adult Volunteers'})
    st.plotly_chart(fig)

    fig = px.histogram(data_cleaned, x='# of Youth Volunteers', nbins=20, labels={'# of Youth Volunteers': 'Youth Volunteers'})
    st.plotly_chart(fig)

    fig = px.histogram(data_cleaned, x='Donation Bags Collected', nbins=20, labels={'Donation Bags Collected': 'Donation Bags Collected'})
    st.plotly_chart(fig)

    fig = px.histogram(data_cleaned, x='Time to Complete (min)', nbins=20, labels={'Time to Complete (min)': 'Time to Complete'})
    st.plotly_chart(fig)

    fig = px.scatter(data_cleaned, x="# of Adult Volunteers", y="Donation Bags Collected", 
                     labels={"# of Adult Volunteers": "Adult Volunteers", "Donation Bags Collected": "Donation Bags"},
                     title="Relationship Between Adult Volunteers and Donation Bags Collected")
    st.plotly_chart(fig)

# Page 3: Machine Learning Modeling
def machine_learning_modeling():
    st.title("Machine Learning Modeling")
    st.write("Enter the details to predict donation bags:")

    # Input fields for user to enter data
    doors_in_route = st.slider("Number of Doors in Route", 10, 500, 100, key="doors_in_route_1")
    routes_completed = st.slider("Routes Completed", 1, 10, 5, key="routes_completed")
    time_spent = st.slider("Time Spent (minutes)", 10, 300, 60, key="time_spent")
    adult_volunteers = st.slider("Number of Adult Volunteers", 1, 50, 10, key="adult_volunteers")
    doors_in_route_2 = st.slider("Number of Doors in Route", 10, 500, 100, key="doors_in_route_2")  # Renamed for clarity
    youth_volunteers = st.slider("Number of Youth Volunteers", 1, 50, 10, key="youth_volunteers")
    data['AvgDoorsPerRoute'] = data['# of Doors in Route'] / data['TotalRoutes']
    avg_doors_per_route = st.slider("Average Doors Per Route", int(data['AvgDoorsPerRoute'].min()), int(data['AvgDoorsPerRoute'].max()), int(data['AvgDoorsPerRoute'].mean()), key="avg_doors_per_route")
    data['VolunteersPerDoor'] = (data['# of Adult Volunteers who participated in this route'] + data['# of Youth Volunteers who participated in this route']) / data['# of Doors in Route']
    data['VolunteersPerDoor'] = data['VolunteersPerDoor'].replace([np.inf, -np.inf], np.nan) #replace the inf values with nan
    data['AvgDoorsPerRoute'] = np.nan_to_num(data['AvgDoorsPerRoute']) 
    data['VolunteersPerDoor'] = np.nan_to_num(data['VolunteersPerDoor']) 
    data['TimePerVolunteer'] = data['Time Spent Collecting Donations'] / (data['# of Adult Volunteers who participated in this route'] + data['# of Youth Volunteers who participated in this route'])
    data['TimePerVolunteer'] = data['TimePerVolunteer'].replace([np.inf, -np.inf], np.nan) #replace the inf values with nan

    data['TimePerVolunteer'] = np.nan_to_num(data['TimePerVolunteer']) 
    volunteers_per_door = st.slider("Volunteers Per Door", float(data['VolunteersPerDoor'].min()), float(data['VolunteersPerDoor'].max()), float(data['VolunteersPerDoor'].mean()), key="volunteers_per_door")
    time_per_volunteer = st.slider("Time Per Volunteer (minutes)", float(data['TimePerVolunteer'].min()), float(data['TimePerVolunteer'].max()), float(data['TimePerVolunteer'].mean()), key="time_per_volunteer")


    # Predict button
    if st.button("Predict"):
        # Load the trained model
        model = joblib.load('trained_donation_model.pkl')

        # Prepare input data for prediction
        input_data = [[time_per_volunteer, volunteers_per_door, avg_doors_per_route, doors_in_route, routes_completed, time_spent, adult_volunteers, doors_in_route, youth_volunteers]]

        # Make prediction
        prediction = model.predict(input_data)

        # Display the prediction
        st.success(f"Predicted Donation Bags: {prediction[0]}")

        # You can add additional information or actions based on the prediction if needed
# Page 4: Neighbourhood Mapping
# Read geospatial data
# geodata = pd.read_csv("Location_data_updated.csv")

# def neighbourhood_mapping():
#     st.title("Neighbourhood Mapping")

#     # Get user input for neighborhood
#     user_neighbourhood = st.text_input("Enter the neighborhood:")

#     # Check if user provided input
#     if user_neighbourhood:
#         # Filter the dataset based on the user input
#         filtered_data = geodata[geodata['Neighbourhood'] == user_neighbourhood]

#         # Check if the filtered data is empty, if so, return a message indicating no data found
#         if filtered_data.empty:
#             st.write("No data found for the specified neighborhood.")
#         else:
#             # Create the map using the filtered data
#             fig = px.scatter_mapbox(filtered_data,
#                                     lat='Latitude',
#                                     lon='Longitude',
#                                     hover_name='Neighbourhood',
#                                     zoom=12)

#             # Update map layout to use OpenStreetMap style
#             fig.update_layout(mapbox_style='open-street-map')

#             # Show the map
#             st.plotly_chart(fig)
#     else:
#         st.write("Please enter a neighborhood to generate the map.")






# Page 5: Data Collection
def data_collection():
    st.title("Data Collection")
    st.write("Please fill out the Google form to contribute to our Food Drive!")
    google_form_url = "https://forms.gle/Sif2hH3zV5fG2Q7P8"#YOUR_GOOGLE_FORM_URL_HERE
    st.markdown(f"[Fill out the form]({google_form_url})")

# Main App Logic
def main():
    st.sidebar.title("Food Drive App")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "EDA", "ML Modeling", "Neighbourhood Mapping", "Data Collection"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "EDA":
        exploratory_data_analysis()
    elif app_page == "ML Modeling":
        machine_learning_modeling()
    # elif app_page == "Neighbourhood Mapping":
    #     neighbourhood_mapping()
    elif app_page == "Data Collection":
        data_collection()

if __name__ == "__main__":
    main()
