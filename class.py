import streamlit as st
from transformers import pipeline
from PIL import Image

# 1. تحميل الموديل اللي بعته من Hugging Face
@st.cache_resource
def load_image_model():
    # هنا حطينا اسم الموديل اللي في الرابط بالظبط
    # استبدل السطر القديم بالسطر ده:
  return pipeline("image-classification", model="google/vit-base-patch16-224")

img_pipe = load_image_model()

st.title("🖼️ Image Classification App")

# 2. واجهة رفع الصور
uploaded_file = st.file_uploader("اختار صورة عشان الموديل يحللها...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة للمستخدم
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة المرفوعة', use_column_width=True)
    
    st.write("جاري التحليل...")
    
    # 3. تشغيل الموديل على الصورة
    results = img_pipe(image)
    
    # 4. عرض النتائج بشكل منظم
    st.subheader("نتائج التصنيف:")
    for result in results:
        label = result['label']
        score = result['score']
        # عرض النتيجة مع شريط تقدم يوضح نسبة الثقة
        st.write(f"**{label}**: {score:.2%}")
        st.progress(score)
