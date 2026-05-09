import streamlit as st
from transformers import pipeline
from PIL import Image # مكتبة للتعامل مع الصور

# تحميل موديل النصوص (زي ما أنت عامل)
@st.cache_resource
def load_text_model():
    return pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# تحميل موديل الصور
@st.cache_resource
def load_image_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

text_pipe = load_text_model()
img_pipe = load_image_model()

st.title("🤖 Multi-AI Analyzer")

# إضافة تبويب الصور للتبويبات اللي عندك
tab1, tab2, tab3 = st.tabs(["📝 Text", "📁 Bulk CSV", "🖼️ Image Classification"])

with tab1:
    # كود تحليل النصوص بتاعك...
    pass

with tab2:
    # كود الملفات بتاعك...
    pass

with tab3:
    st.header("تصنيف الصور")
    uploaded_img = st.file_uploader("ارفع صورة هنا:", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_img:
        image = Image.open(uploaded_img)
        st.image(image, caption='الصورة المرفوعة', use_column_width=True)
        
        if st.button("صنف الصورة"):
            with st.spinner('جاري تحليل الصورة...'):
                results = img_pipe(image)
                # عرض أعلى نتيجة
                top_result = results[0]
                st.success(f"النتيجة: {top_result['label']} (بنسبة ثقة {top_result['score']:.2%})")