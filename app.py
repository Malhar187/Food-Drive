import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Load the dataset with a specified encoding
data = pd.read_csv("data_2024_donation_processed.csv") # Replace with your CSV file name

# Page 1: Dashboard
def dashboard():
    st.image('Logo.PNG', use_column_width=True) # Replace 'Logo.PNG' with your logo file name

    st.subheader("💡 Abstract:")

    inspiration = '''
    The Edmonton Food Drive Project....Talk about the Project here and lessons learned
    '''

    st.write(inspiration)

    st.subheader("👨🏻‍💻 What our Project Does?")

    what_it_does = '''
    Your project description goes here.
    '''

    st.write(what_it_does)


# Page 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")
    # Rename columns for clarity (if needed) - adjust to your column names
    data_cleaned = data.rename(columns={
        'Timestamp': 'Date',
        'Drop Off Location': 'Location',
        'City': 'City',
        'Stake': 'Stake',
        'Route Number/Name': 'Route',
        '# of Adult Volunteers in this route': '# of Adult Volunteers',
        '# of Youth Volunteers in this route': '# of Youth Volunteers',
        '# of Donation Bags Collected/Route': 'Donation Bags Collected',
        'Time to Complete (in minutes) pick up of bags /route': 'Time to Complete (min)',
        'Did you complete more than 1 route?': 'Completed More Than One Route',
        'Number of routes completed': 'Routes Completed',
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

# Page 3: Machine Learning Modeling
def machine_learning_modeling():
    st.title("Machine Learning Modeling")
    st.write("Enter the details to predict donation bags:")

    # Input fields for user to enter data - replace with your actual feature columns
    completed = st.slider("Completed More Than One Route", 0, 1, 0)  
    routes = st.slider("Routes Completed", 1, 10, 5)  
    time = st.slider("Time Spent (minutes)", 10, 300, 60)  
    adults = st.slider("Number of Adult Volunteers", 1, 50, 10)  
    doors = st.slider("Number of Doors in Route", 10, 500, 100)  
    youth = st.slider("Number of Youth Volunteers", 1, 50, 10)  

    # Predict button
    if st.button("Predict"):
        # Load the trained model - replace with your actual model file name
        model = joblib.load('trained_donation_model.pkl')  

        # Prepare input data for prediction
        input_data = [[completed, routes, time, adults, doors, youth]]

        # Make prediction
        prediction = model.predict(input_data)

        # Display the prediction
        st.success(f"Predicted Donation Bags: {prediction[0]}")

# ... (Rest of the code remains the same: Neighbourhood Mapping, Data Collection, Main App Logic)
    # You can add additional information or actions based on the prediction if needed
# Page 4: Neighbourhood Mapping
# Read geospatial data
geodata = pd.read_csv("Location_data_updated.csv")

def neighbourhood_mapping():
    st.title("Neighbourhood Mapping")

    # Get user input for neighborhood
    user_neighbourhood = st.text_input("Enter the neighborhood:")

    # Check if user provided input
    if user_neighbourhood:
        # Filter the dataset based on the user input
        filtered_data = geodata[geodata['Neighbourhood'] == user_neighbourhood]

        # Check if the filtered data is empty, if so, return a message indicating no data found
        if filtered_data.empty:
            st.write("No data found for the specified neighborhood.")
        else:
            # Create the map using the filtered data
            fig = px.scatter_mapbox(filtered_data,
                                    lat='Latitude',
                                    lon='Longitude',
                                    hover_name='Neighbourhood',
                                    zoom=12)

            # Update map layout to use OpenStreetMap style
            fig.update_layout(mapbox_style='open-street-map')

            # Show the map
            st.plotly_chart(fig)
    else:
        st.write("Please enter a neighborhood to generate the map.")






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
    elif app_page == "Neighbourhood Mapping":
        neighbourhood_mapping()
    elif app_page == "Data Collection":
        data_collection()

if __name__ == "__main__":
    main()
