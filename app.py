import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://i.postimg.cc/4xgNnkfX/Untitled-design.png");
background-size: cover;
background-position: center center;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
# Set Page Title and Icon


#Sidebar Navigation
page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "Exploratory Data Analysis", "Modeling", "Make Predictions!" ])

df = pd.read_csv('data/new_train.csv')

# Home Page
if page == "Home":
    st.title("📊 Airline Passenger Satisfaction Dataset Explorer")
    st.subheader("Welcome to my Airline Passenger Satisfaction Dataset explorer app!")
    st.write("This app is designed to make the exploration and analysis of the Airline dataset easy and accessible. Whether you're interested in the distribution of data, relationships between features, or the performance of a machine learning model, this tool provides an interactive and visual platform for your exploration. Enjoy exploring the world of data science and analytics with the Airline dataset")
    st.image ("https://th.bing.com/th/id/OIP.92hE3ZKlVeMizy08tLZFuQHaEK?w=294&h=180&c=7&r=0&o=5&pid=1.7")
    st.write("Use the sidebar to navigate between different sections.")
    
    picture = st.camera_input("Take a picture")
    if picture:
        st.image(picture)
    
    color = st.color_picker('Pick A Color', '#00f900')
    st.write('The current color is', color)


# Data Overview
elif page == "Data Overview":
    st.title("🔢 Data Overview")
    
    st.subheader("About the Data")
    st.write("The dataset contains 103,904 rows and 25 colums with different features relating to passenger satisfaction from customer type, type of travel and flight distance. ")
    st.image("https://www.bing.com/th?id=OIP.29dldVD2w-jSO2-GnKO5SgHaE1&w=197&h=129&c=8&rs=1&qlt=90&o=6&pid=3.1&rm=2")
    st.link_button("Click here to learn more", "https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction?select=train.csv")


    st.subheader("Quick Clance at the Data")

    # Display Dataset
    if st.checkbox("DataFrame"):
        st.dataframe(df)

    # Column List
    if st.checkbox("Column List"):
        st.code(f"Columns: {df.columns.tolist()}")
        if st.toggle("Further breakdown of columns"):
            num_cols = df.select_dtypes(include = 'number').columns.tolist()
            obj_cols = df.select_dtypes(include = 'object').columns.tolist()
            st.code(f"Numerical Columns: {num_cols}\nObject Columns: {obj_cols}")

    # Shape 
    if st.checkbox("Shape"):
        st.write(f"There are {df.shape[0]} rows and {df.shape[1]} columns.")

elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")

    eda_type = st.multiselect(" What type of EDA would you like to view?", ['Histograms', 'Box Plots', 'Scatterplots', 'Count Plots'])    

    obj_cols = df.select_dtypes(include = 'object').columns.tolist()
    num_cols = df.select_dtypes(include = 'number').columns.tolist()



    if 'Histograms' in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column for your histogram:", num_cols, index = None)
        if h_selected_col:
            chart_title = f"Distribution of {' '.join(h_selected_col.split('_')).title()}"
            if st.toggle("Satisfaction Hue on Histogram"):
                st.plotly_chart(px.histogram(df, x=h_selected_col, title = chart_title, color = 'satisfaction', barmode = 'overlay'))
            else:
                st.plotly_chart(px.histogram(df, x=h_selected_col, title = chart_title))

    if 'Box Plots' in eda_type:  
        st.subheader("Box Plots - Visualizing Numerical Distributions")
        b_selected_col = st.selectbox("Select a numerical column for your box plot:", num_cols, index = None)  
        if b_selected_col:
            chart_title = f"Distribution of {' '.join(b_selected_col.split('_')).title()}"
            if st.toggle("Satisfaction  Hue on Box Plot"):
                st.plotly_chart(px.box(df, x=b_selected_col, y = 'satisfaction', title = chart_title, color = 'satisfaction'))
            else:
                st.plotly_chart(px.box(df, x=b_selected_col, title = chart_title))

    
    if 'Scatterplots' in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        selected_col_x = st.selectbox("Select x-axis variable:", num_cols, index = None)
        selected_col_y = st.selectbox("Select y-axis variable:", num_cols, index = None)

        if selected_col_x and selected_col_y:
            hue_toggle = st.toggle("Satisfaction Hue on Scatterplot")
            chart_title = f"{' '.join(selected_col_x.split('_')).title()} vs. {' '.join(selected_col_y.split('_')).title()}"

            if hue_toggle:
                st.plotly_chart(px.scatter(df, x=selected_col_x, y=selected_col_y, color ='satisfaction', title = chart_title))
            else:
                st.plotly_chart(px.scatter(df, x=selected_col_x, y=selected_col_y, title = chart_title ))

    if 'Count Plots' in eda_type:
        st.subheader("Count Plots - Visualizing Categorical Distributions")
        selected_col = st.selectbox("Select a categorical variable:", obj_cols, index = None)
        if selected_col:
            chart_title = f"Distribution of {selected_col.title()}"
            st.plotly_chart(px.histogram(df, x=selected_col, title = chart_title, color ='satisfaction'))            

if page == "Modeling":
    st.title(":gear: Modeling")
    st.markdown("View how well different **machine learning models** make predictions on the Passeneger Satisfaction Dataset on this page!")

    # Set up X and y
    features =['Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking','Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service','Leg room service','Baggage handling', 'Checkin service', 'Inflight service','Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    X = df[features]
    y = df['satisfaction']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

    # Model Selection 
    model_option = st.selectbox("Select a Model", ['KNN', 'Logistic Regression', 'Random Forest'], index = None)

    if model_option:
        #st.write(f"You selected {model_option}")

        if model_option == "KNN":
            k_value = st.slider("Select the number of neighbors (k)", 1, 29, 5, 2)
            model = KNeighborsClassifier(n_neighbors = k_value)
        elif model_option == "Logistic Regression":
            model = LogisticRegression()
        elif model_option == "Random Forest":
            model =RandomForestClassifier()

        if st.button("Lets see the performance"):
            model.fit(X_train, y_train)

            # Display results
            st.subheader(f"{model} Evaluation")
            st.text(f"Training Accuracy: {round(model.score(X_train, y_train)*100, 2)}%")
            st.text(f"Testing Accuracy: {round(model.score(X_test, y_test)*100,2)}%")

            # Confusion Matrix
            st.subheader("Confusion Matrix:")
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap = 'Reds')
            CM_fig = plt.gcf()
            st.pyplot(CM_fig)

if page == "Make Predictions!":
    st.title(":rocket: Make Predictions on Passenger Satisfaction Dataset")

    # Create sliders for user to input data
    st.subheader("Adjust the sliders to input data:")
    
    a = st.slider("Age", 0.01, 90.0, 0.01,0.01)
    f_d = st.slider("Flight Distance", 0.01, 3800.0, 0.01,0.01)
    i_w_s = st. slider("Inflight Wifi Service", 0.01, 5.0, 0.01,0.01)
    d_a_t = st.slider("Departure/Arrival Time Convenient", 0.01, 5.0, 0.01,0.01)
    e_o_b = st.slider("Ease of Online Booking", 0.01, 5.0, 0.01,0.01)
    g_l = st.slider("Gate Location", 0.01, 5.0, 0.01,0.01)
    f_d = st.slider("Food and Drink", 0.01, 5.0, 0.01,0.01)
    o_b = st.slider("Online Boarding", 0.01, 5.0, 0.01,0.01)
    s_c = st.slider("Seat Comfort", 0.01, 5.0, 0.01,0.01)
    i_e = st.slider("Inflight Entertainment", 0.01, 5.0, 0.01,0.01)
    ob_s = st.slider("On-Board Service", 0.01, 5.0, 0.01,0.01)
    l_s = st.slider("Leg Room Service", 0.01, 5.0, 0.01,0.01)
    b_h = st.slider("Baggage Handling", 0.01, 5.0, 0.01,0.01)
    c_s = st.slider("Checkin Service", 0.01, 5.0, 0.01,0.01)
    i_s = st.slider("Inflight Service", 0.01, 5.0, 0.01,0.01)
    c = st.slider("Cleanliness", 0.01, 5.0, 0.01,0.01)
    d_m = st.slider("Departure Delay in Minutes", 0.01, 5.0, 0.01,0.01)
    a_m = st.slider("Arrival delay in Minutes", 0.01, 5.0, 0.01,0.01)

    # Features must be in the same order as the model it was trained on 
    user_input = pd.DataFrame({
            'Age': [a],
            'Flight Distance':[f_d],
            'Inflight wifi service': [ i_w_s],
            'Departure/Arrival time convenient': [d_a_t],
            'Ease of Online booking': [e_o_b],
            'Gate location': [g_l],
            'Food and drink': [f_d],
            'Online boarding': [o_b],
            'Seat comfort': [s_c],
            'Inflight entertainment': [i_e],
            'On-board service': [ob_s],
            'Leg room service': [l_s],
            'Baggage handling': [b_h],
            'Checkin service': [c_s],
            'Inflight service': [i_s],
            'Cleanliness' :[c],
            'Departure Delay in Minutes': [d_m],
            'Arrival_Delay in Minutes': [a_m]

    })


    features =['Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking','Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service','Leg room service','Baggage handling', 'Checkin service', 'Inflight service','Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    X = df[features]
    y = df['satisfaction']

    # Model Selection
    st.write("The predictions are made using RandomForestClassifier as it performed the best out of all of the models.")
    model = RandomForestClassifier()

    if st.button("Make a Prediction!"):
        model.fit(X, y)
        prediction = model.predict(user_input)
        st.write(f"{model} predict this passenger satisfaction is {prediction[0]} satisfaction!")
        


        


