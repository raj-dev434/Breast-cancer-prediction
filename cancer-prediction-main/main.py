import pandas as pd
import matplotlib.pyplot as plt


def get_clean_data():
    df = pd.read_csv('./data/cancer-diagnosis.csv')
    df = df.drop(['Unnamed: 32', 'id'], axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df


def plot_data(df):
    plot = df['diagnosis'].value_counts().plot(
        kind='bar', title="Class distributions \n(0: Benign | 1: Malignant)")
    plot.set_xlabel("Diagnosis")
    plot.set_ylabel("Frequency")
    plt.show()


def get_model(selected_model):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier  # Import C4.5 Equivalent
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler

    df = get_clean_data()

    # Scale predictors and split data
    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis']
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Select Model
    if selected_model == "Logistic Regression":
        model = LogisticRegression()
    elif selected_model == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif selected_model == "SVM":
        model = SVC(probability=True, random_state=42)
    elif selected_model == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif selected_model == "C4.5 Decision Tree":
        model = DecisionTreeClassifier(criterion='entropy', random_state=42)  # C4.5 Equivalent
    else:
        raise ValueError("Invalid model selection")

    # Train the model
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    print(f"Model: {selected_model}")
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
    
    return model, scaler




def create_radar_chart(input_data):

    import plotly.graph_objects as go
    
    input_data = get_scaled_values_dict(input_data)

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
                input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
                input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
                input_data['fractal_dimension_mean']],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Mean'
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
                input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
                input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Standard Error'
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
                input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
                input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
                input_data['fractal_dimension_worst']],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Worst'
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        autosize=True
    )

    return fig


def create_input_form(data):
    import streamlit as st

    st.sidebar.header("Cell Nuclei Details")
    slider_labels = [("Radius (mean)", "radius_mean"), ("Texture (mean)", "texture_mean"),
                     ("Perimeter (mean)", "perimeter_mean"), ("Area (mean)", "area_mean"),
                     ("Smoothness (mean)",
                      "smoothness_mean"), ("Compactness (mean)", "compactness_mean"),
                     ("Concavity (mean)", "concavity_mean"), ("Concave points (mean)",
                                                              "concave points_mean"),
                     ("Symmetry (mean)", "symmetry_mean"), ("Fractal dimension (mean)",
                                                            "fractal_dimension_mean"),
                     ("Radius (se)", "radius_se"), ("Texture (se)",
                                                    "texture_se"), ("Perimeter (se)", "perimeter_se"),
                     ("Area (se)", "area_se"), ("Smoothness (se)", "smoothness_se"),
                     ("Compactness (se)",
                      "compactness_se"), ("Concavity (se)", "concavity_se"),
                     ("Concave points (se)",
                      "concave points_se"), ("Symmetry (se)", "symmetry_se"),
                     ("Fractal dimension (se)",
                      "fractal_dimension_se"), ("Radius (worst)", "radius_worst"),
                     ("Texture (worst)", "texture_worst"), ("Perimeter (worst)",
                                                            "perimeter_worst"),
                     ("Area (worst)", "area_worst"), ("Smoothness (worst)",
                                                      "smoothness_worst"),
                     ("Compactness (worst)",
                      "compactness_worst"), ("Concavity (worst)", "concavity_worst"),
                     ("Concave points (worst)",
                      "concave points_worst"), ("Symmetry (worst)", "symmetry_worst"),
                     ("Fractal dimension (worst)", "fractal_dimension_worst")]
    input_data = {}

    for label, col in slider_labels:
        input_data[col] = st.sidebar.slider(
            label, float(data[col].min()), float(
                data[col].max()), float(data[col].mean())
        )

    return input_data


def get_scaled_values_dict(values_dict):
    # Define a Function to Scale the Values based on the Min and Max of the Predictor in the Training Data
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in values_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict


def display_predictions(input_data, model, scaler):
    import streamlit as st
    import numpy as np

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_data_scaled = scaler.transform(input_array)

    st.subheader('Cell Cluster Prediction')

    # Get the prediction
    prediction = model.predict(input_data_scaled)[0]  
    probabilities = model.predict_proba(input_data_scaled)[0]  # Get raw probability values

    # Display the original output (Benign/Malignant)
    if prediction == 0:
        st.write("<span class='diagnosis bright-green'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis bright-red'>Malignant</span>", unsafe_allow_html=True)

    # Display the modified output message
    st.markdown("---")  # Separator line
    if prediction == 0:
        st.success("✅ **No cancer detected. The diagnosis is Benign.**")
    else:
        st.error("⚠️ **You got cancer! The diagnosis is Malignant.**")

    # Display probabilities without rounding
    st.write(f"**Probability of being Benign:** {probabilities[0]}")
    st.write(f"**Probability of being Malignant:** {probabilities[1]}")

    st.write("This is an AI-based prediction. Please consult a doctor for medical confirmation.")

def upload_and_predict(model, scaler):
    import streamlit as st
    import pandas as pd
    import numpy as np

    st.subheader("📂 Upload CSV File for Batch Prediction")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Ensure the uploaded file has the correct features
        required_features = get_clean_data().drop(columns=['diagnosis']).columns
        if not all(feature in df.columns for feature in required_features):
            st.error("⚠️ CSV file does not contain the required features. Please upload a valid file.")
            return

        # Scale data
        df_scaled = scaler.transform(df[required_features])

        # Predict using the trained model
        predictions = model.predict(df_scaled)
        probabilities = model.predict_proba(df_scaled)

        # Convert to DataFrame and display results
        results = pd.DataFrame({
            "Diagnosis": ["Malignant" if p == 1 else "Benign" for p in predictions],
            "Benign Probability": probabilities[:, 0],
            "Malignant Probability": probabilities[:, 1],
            "Message": ["⚠️ **You got cancer! The diagnosis is Malignant.**" if p == 1 else "✅ **No cancer detected. The diagnosis is Benign.**" for p in predictions]
        })

        st.write("📊 **Prediction Results:**")
        st.dataframe(results)  # Display tabular results

        # Display individual results with radar chart
        for i in range(len(predictions)):
            st.markdown("---")  # Separator line
            st.write(f"**Patient {i+1} Diagnosis:**")

            # Get input data for the radar chart
            input_data = df.iloc[i].to_dict()

            # Generate radar chart
            radar_chart = create_radar_chart(input_data)
            st.plotly_chart(radar_chart, use_container_width=True)

            # Display prediction results
            if predictions[i] == 0:
                st.write("<span class='diagnosis bright-green'>Benign</span>", unsafe_allow_html=True)
                st.success("✅ **No cancer detected. The diagnosis is Benign.**")
            else:
                st.write("<span class='diagnosis bright-red'>Malignant</span>", unsafe_allow_html=True)
                st.error("⚠️ **You got cancer! The diagnosis is Malignant.**")

            st.write(f"**Probability of being Benign:** {probabilities[i][0]}")
            st.write(f"**Probability of being Malignant:** {probabilities[i][1]}")

import streamlit as st

def create_app():
    import streamlit as st

    st.set_page_config(page_title="Breast Cancer Diagnosis", page_icon=":female-doctor:", layout="wide")

    with st.container():
        st.title("🩺 Breast Cancer Diagnosis App")
        st.write("Select an input method: **Manually enter values** or **Upload a CSV file** for batch predictions.")

    # 🔹 Select Model (Including C4.5 Decision Tree)
    selected_model = st.sidebar.selectbox("Choose a Model:", 
                                          ["Logistic Regression", "Random Forest", "SVM", "KNN", "C4.5 Decision Tree"])

    # Load dataset and model
    data = get_clean_data()
    model, scaler = get_model(selected_model)  # ✅ Updated to include C4.5

    # User selection for input type
    input_method = st.radio("Choose Input Method:", ["Manual Entry", "Upload CSV"], horizontal=True)

    if input_method == "Manual Entry":
        # Sidebar manual input form
        input_data = create_input_form(data)

        # Display radar chart
        col1, col2 = st.columns([4, 1])
        with col1:
            radar_chart = create_radar_chart(input_data)
            st.plotly_chart(radar_chart, use_container_width=True)

        with col2:
            display_predictions(input_data, model, scaler)

    elif input_method == "Upload CSV":
        # Handle CSV upload and batch prediction
        upload_and_predict(model, scaler)



def main():
   
    create_app()



if __name__ == '__main__':
    main()
