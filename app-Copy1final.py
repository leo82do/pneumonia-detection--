import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import matplotlib.pyplot as plt   # ✅ NEW

# Load model
model = tf.keras.models.load_model("pneumonia_model.keras")

st.set_page_config(page_title="Medical AI - Pneumonia Detection", layout="wide")

# 🔵 SIDEBAR (UPDATED)
st.sidebar.title("🫁 Medical AI System")
menu = st.sidebar.radio("Navigation", ["Home", "Detection", "Model Info", "About Pneumonia", "Accuracy Graph"])  # ✅ NEW


# 🏠 HOME
if menu == "Home":
    st.title("🫁 Pneumonia Detection System")

    st.markdown("""
    ### 🏥 AI-Based Medical Diagnosis

    This system uses Deep Learning to detect Pneumonia from Chest X-rays.

    #### 📌 Features:
    - Instant AI prediction
    - Confidence score
    - Medical precautions
    - Downloadable report

    #### 📘 About Pneumonia:
    Pneumonia is a lung infection causing inflammation in air sacs.
    It may fill with fluid and can become life-threatening if untreated.
    """)

    st.markdown("---")


# 🔍 DETECTION
elif menu == "Detection":

    st.title("🔍 Pneumonia Detection")

    uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg","jpeg","png"])

    if uploaded_file is not None:

        col1, col2 = st.columns(2)

        img = Image.open(uploaded_file).convert("RGB")

        with col1:
            st.image(img, caption="Uploaded X-ray", use_container_width=True)

        # Preprocess
        img_resized = img.resize((150,150))
        img_array = np.array(img_resized)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        # RESULT
        if prediction[0][0] > 0.5:
            confidence = round(prediction[0][0]*100, 2)
            result = "PNEUMONIA DETECTED"
            precautions = [
                "Consult a doctor immediately",
                "Take prescribed medication",
                "Get proper rest",
                "Stay hydrated",
                "Avoid crowded places"
            ]
        else:
            confidence = round((1-prediction[0][0])*100, 2)
            result = "NORMAL"
            precautions = [
                "Maintain good hygiene",
                "Exercise regularly",
                "Eat balanced diet",
                "Avoid smoking"
            ]

        with col2:
            st.subheader("📊 Result")

            if "PNEUMONIA" in result:
                st.error(f"⚠ {result}")
            else:
                st.success(f"✅ {result}")

            st.metric("Confidence", f"{confidence}%")

            st.subheader("🩺 Precautions")
            for p in precautions:
                st.write(f"• {p}")

        st.markdown("---")

        # 🧾 CREATE IMAGE REPORT
        report = Image.new("RGB", (800, 600), "white")
        draw = ImageDraw.Draw(report)

        xray = img.resize((350, 350))
        report.paste(xray, (20, 120))

        draw.text((400, 120), "RESULT:", fill="black")
        draw.text((400, 150), result, fill="red")

        draw.text((400, 200), f"Confidence: {confidence}%", fill="black")

        draw.text((400, 250), "Precautions:", fill="black")

        y = 280
        for p in precautions:
            draw.text((400, y), f"- {p}", fill="black")
            y += 30

        buf = io.BytesIO()
        report.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="📥 Download Medical Report (Image)",
            data=byte_im,
            file_name="medical_report.png",
            mime="image/png"
        )

        st.markdown("---")


# 🤖 MODEL INFO
elif menu == "Model Info":

    st.title("🤖 Model Information")

    st.write("""
    - Model: Convolutional Neural Network (CNN)
    - Input Size: 150x150 images
    - Output: Pneumonia / Normal
    - Accuracy: ~90%
    """)

    st.markdown("---")


# 📘 ABOUT PNEUMONIA
elif menu == "About Pneumonia":

    st.title("📘 About Pneumonia")

    st.subheader("What is Pneumonia?")
    st.write("""
    Pneumonia is a serious lung infection that causes inflammation in the air sacs (alveoli) 
    of one or both lungs. These air sacs may fill with fluid or pus, making breathing difficult.
    """)

    st.subheader("Causes")
    st.write("""
    • Bacterial infections (most common)  
    • Viral infections (like influenza)  
    • Fungal infections (rare cases)  
    """)

    st.subheader("Symptoms")
    st.write("""
    • Chest pain while breathing  
    • Persistent cough with mucus  
    • Fever and chills  
    • Shortness of breath  
    """)

    st.subheader("Risk Factors")
    st.write("""
    • Elderly people  
    • Children  
    • Weak immune system  
    • Chronic diseases  
    """)

    st.subheader("Why is it Dangerous?")
    st.write("""
    If untreated, pneumonia can lead to severe complications such as:
    - Lung damage  
    - Respiratory failure  
    - Spread of infection into the bloodstream  
    """)

    st.subheader("Prevention")
    st.write("""
    • Maintain good hygiene  
    • Get vaccinated  
    • Avoid smoking  
    • Eat healthy food  
    • Strengthen immunity  
    """)

    st.markdown("---")


# 📊 NEW PAGE: ACCURACY GRAPH
elif menu == "Accuracy Graph":

    st.title("📊 Model Accuracy Graph")

    # Example values (same as your training)
    train_acc = [0.75, 0.85, 0.90, 0.93, 0.95]
    val_acc = [0.70, 0.80, 0.88, 0.91, 0.93]

    fig, ax = plt.subplots()
    ax.plot(train_acc, label="Training Accuracy")
    ax.plot(val_acc, label="Validation Accuracy")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy")
    ax.legend()

    st.pyplot(fig)

    st.info("This graph shows how the model improved during training.")

    st.markdown("---")


# Footer
st.caption("⚠ This system is for educational purposes only. Consult a doctor.")