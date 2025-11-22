import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import pdfplumber
import easyocr
from pdf2image import convert_from_bytes
import io

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: GOOGLE_API_KEY not found in .env file")
else:
    genai.configure(api_key=api_key)

# Try to get available models
AVAILABLE_MODEL = "gemini-2.5-flash"  # Default to the most reliable model
try:
    models = genai.list_models()
    model_names = [m.name for m in models]
    print(f"Available models: {model_names}")
    
    # Find first model that supports generateContent, prefer the latest ones
    preferred_models = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash', 'gemini-pro-latest']
    
    for preferred in preferred_models:
        for model in models:
            model_short_name = model.name.replace('models/', '')
            if preferred == model_short_name and 'generateContent' in model.supported_generation_methods:
                AVAILABLE_MODEL = model_short_name
                print(f"Using model: {AVAILABLE_MODEL}")
                break
        if AVAILABLE_MODEL != "gemini-2.5-flash":  # If we found a preferred model, use it
            break
except Exception as e:
    print(f"Could not list models: {e}")

print(f"Selected model: {AVAILABLE_MODEL}")



def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_bytes = pdf.read()
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf_file:
            for page in pdf_file.pages:
                page_text = page.extract_text() or ""
                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    table_text = "\n".join(["\t".join([str(cell) if cell else "" for cell in row]) for row in table])
                    page_text += "\n" + table_text
                # If page_text is short, do OCR
                if len(page_text.strip()) < 100:  # arbitrary threshold
                    images = convert_from_bytes(pdf_bytes, first_page=page.page_number, last_page=page.page_number)
                    if images:
                        reader = easyocr.Reader(['en'])
                        results = reader.readtext(images[0])
                        ocr_text = " ".join([result[1] for result in results])
                        page_text += "\n" + ocr_text
                text += page_text + "\n"
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    """Generate response using Google Generative AI or fallback to context matching"""
    def answer_question(context, question):
        # If AI model is available, use it
        if AVAILABLE_MODEL:
            try:
                model = genai.GenerativeModel(AVAILABLE_MODEL)
                prompt = f"""Answer the question based on the provided context. If the answer is not in the context, say "answer is not available in the context".

Context:
{context}

Question: {question}

Answer:"""
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"AI Error: {e}")
                # Fall through to basic mode
        
        # Fallback: Basic mode - just extract relevant context
        question_lower = question.lower().split()
        context_lines = context.split('\n')
        relevant_lines = []
        
        for line in context_lines:
            line_lower = line.lower()
            if any(word in line_lower for word in question_lower):
                relevant_lines.append(line)
        
        if relevant_lines:
            answer = '\n'.join(relevant_lines[:5])  # Return top 5 relevant lines
            return f"Based on document context:\n\n{answer}"
        else:
            return "The answer is not available in the provided context."
    
    return answer_question



def user_input(user_question):
    if not os.path.exists("faiss_index"):
        st.error("Please upload and process PDF files first.")
        return
    
    try:
        with st.spinner("Searching documents..."):
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            retriever = new_db.as_retriever()
            docs = retriever.invoke(user_question)
            context = "\n".join([doc.page_content for doc in docs])
        
        with st.spinner("Generating response..."):
            chain = get_conversational_chain()
            response = chain(context=context, question=user_question)

        st.write("**Reply:**")
        st.write(response)
    except Exception as e:
        st.error(f"Error: {str(e)}")
        print(f"Error: {e}")


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with Multiple PDF using Gemini")

    user_question = st.text_input("Ask a Question from Multiple PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()