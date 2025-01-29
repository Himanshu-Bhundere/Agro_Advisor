# 🌾 Smart Agro Advisor

Smart Agro Advisor is a **Streamlit-based Crop & Fertilizer Recommendation System** that provides users with accurate crop and fertilizer suggestions based on input parameters like soil nutrients, temperature, and humidity. The system uses **machine learning models** to enhance agricultural decision-making.

## 🚀 Features

- **Crop Recommendation** 🌱: Predicts the best crop to grow based on soil conditions.
- **Fertilizer Recommendation** 🧪: Suggests suitable fertilizers for optimal growth.
- **Interactive UI** 🎛️: User-friendly design with sliders and dropdowns.
- **Visualizations** 📊: Dynamic bar charts for 'Nutrient Levels' & 'Soil & Environmental Conditions'.
- **Animations** 🎉: Fun rain effects for engagement.

## 📌 Installation & Setup

### 1️⃣ Clone the Repository

```
git clone https://github.com/Himanshu-Bhundere/Agro_Advisor.git
cd Agro_Advisor
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```
streamlit run streamlit_ui.py
```

## 📂 Project Structure

```
├── models/
│   ├── crop_model.pkl            # Trained crop recommendation model
│   ├── fert_model.pkl            # Trained fertilizer recommendation model
│   ├── label_encoders.pkl        # Encoded categorical variables
│   ├── category_mappings.pkl     # Mappings for categorical names
├── streamlit_ui.py               # Streamlit UI file
├── recommendation.py             # ML logic for predictions
├── requirements.txt              # Python dependencies
├── README.md                     # Documentation
```

## ⚡ Usage

1. **Adjust Inputs**: Use the sidebar to modify nutrient levels, temperature, humidity, and other parameters.
2. **Get Recommendations**:
   - Click **“🚜 Recommend Crop”** to see the best crop to plant.
   - Click **“🧪 Recommend Fertilizer”** to get a suitable fertilizer suggestion.
3. **Visualize Data**: View bar charts for nutrient levels.

## 📌 Technologies Used

- **Streamlit**: Interactive UI
- **Scikit-learn**: Machine learning models
- **Pandas & NumPy**: Data handling
- **Plotly**: Data visualization

## 👨‍💻 Developers

Developed with ❤️ by the Himanshu Bhundere 🚀
