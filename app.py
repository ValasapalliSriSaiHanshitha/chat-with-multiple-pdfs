import base64
import io
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
from PIL import Image
import pdf2image
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input,pdf_cotent,prompt):
    model=genai.GenerativeModel('gemini-pro-vision')
    response=model.generate_content([input,pdf_content[0],prompt])
    return response.text

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
    #convert pdf to image
        images=pdf2image.convert_from_bytes(uploaded_file.read())
        
        first_page=images[0]

        #convert to bytes
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type":"image/jpeg",
                "data" : base64.b64encode(img_byte_arr).decode()
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file Uploaded")

## streamlit App

st.set_page_config(page_title=" Resume ExPERT")
st.header("Resume Tracking System")
input_text=st.text_area("Job Description:", key="input")
uploaded_file = st.file_uploader("Upload a folder containing PDFs and DOCXs", type=None, accept_multiple_files=True, help="Please upload a folder containing PDF or DOCX files")
# uploaded_file=st.file_uploader("Upload your resume(PDF)...",type=["pdf"])

if uploaded_file is not None:
    st.write("PDF upleaded Successfully")

submit1=st.button("Tell Me About the Resume")

# submit2=st.button("How can i Improve my Skils")

submit3=st.button("Percentage match")

input_prompt1="""
you are an experienced HR  With Tech Experience  in the field of of any job role from data Science or Full stack Web Development
,Big data Engineering,Devops,Data Analyst,Your task is to review the provided resume against the job 
description for this profiles.please share your professional evaluation on whether the candidate's 
profile aligns with Highlights thestrengths and weakness of the applicant in relation to the specific job mentioned"""

input_prompt3=""" You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of any one job role from Data Science,
 Web development, Big Data Engineering, DEVOPS,Data Analyst and deep ATS functionality, 
 your task is to evaluate the resume against the provided job description. give me the percentage of 
 match if the resume matches the job description. First the output should come as percentage and then keywords missing and last final thoughts"""

if submit1:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response=get_gemini_response(input_prompt1,pdf_content,input_text)
        st.subheader("the response is")
        st.write(response)
    else:
        st.write("please upload the resume")
elif submit3:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response=get_gemini_response(input_prompt3,pdf_content,input_text)
        st.subheader("the response is")
        st.write(response)
    else:
        st.write("please upload the resume")