import replicate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import openai
import langchain
import requests
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from deep_translator import GoogleTranslator
from fpdf import FPDF
import tempfile
from dotenv import load_dotenv
from gtts import gTTS
import os



def main():
    load_dotenv()
    st.set_page_config(page_title="Second Role", page_icon="")
    st.header('We can create story about secondary role ✍🏻')
    pdf = st.file_uploader("Load your story ", type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text)
        embeddings = OpenAIEmbeddings()
        database = FAISS.from_texts(chunks,embeddings)

        user_question = st.text_input("О ком написать историю?")
        if user_question:
            opisanie = st.text_input("В каком стиле хотите получить изображения(prompt)?")
            translatedopisanie = GoogleTranslator(source='auto', target='english').translate(opisanie)
            if opisanie:
                docs = database.similarity_search(user_question)
                llm = langchain.llms.OpenAI()
                chain = load_qa_chain(llm,chain_type="stuff")
                otvet = chain.run(input_documents=docs,question="Детально опиши персонажа " + user_question + " из данного произведения")
                client = openai.OpenAI()
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Вы - блестящий создатель сказок и сюжетов, который создает очень увлекательные и интересные истории с сюжетной линией."},
                        {"role": "user", "content": "Создай длинную историю со всеми частями сюжетной композиции, а именно завязка, развитие действия, кульминация, развязка. Персонаж:  " + otvet + "Каждый элемент должен быть очень проработанным и длинным. "
                                                                                                                                                                                                     "напиши ответ в формате повествования в истории или сказке. В своем ответе не упоминай завязку развязку и тд, также не пиши название произведения. Помни, что главный герой твеого произведения: " + user_question }
                    ]
                )
                plot = response.choices[0].message.content
                parts_of_story = plot.splitlines()
                while '' in parts_of_story:
                    parts_of_story.remove('')
                pdf = FPDF()
                font_path = 'fonts/DejaVuSans.ttf'
                pdf.add_font('DejaVu', '', font_path, uni=True)
                pdf.set_font('DejaVu', '', 12)

                for i in parts_of_story:
                    st.write(i)
                    tts = gTTS(text=i, lang='ru')
                    tts_fp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                    tts.save(tts_fp.name)
                    audio = open(tts_fp.name, 'rb')
                    audio_bytes = audio.read()
                    st.audio(audio_bytes, format='audio/mp3', start_time=0)
                    translated = GoogleTranslator(source='auto', target='english').translate(i)
                    output = replicate.run(
                        "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                        input={"prompt": translated + " , " + translatedopisanie}
                    )
                    image_url = output[0]
                    st.image(image_url)

                    response = requests.get(image_url)
                    if response.status_code == 200:
                        pdf.add_page()

                        content_type = response.headers['Content-Type']
                        suffix = '.jpg' if 'jpeg' in content_type else '.png' if 'png' in content_type else '.tmp'
                        image_data = response.content
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmpfile:
                            tmpfile.write(image_data)
                            tmpfile_path = tmpfile.name

                        pdf.set_font('DejaVu', '', 12)
                        pdf.multi_cell(0, 10, i)
                        pdf.image(tmpfile_path, x=10, y=pdf.get_y() + 10, w=165)
                    else:
                        st.error("Failed to download the image.")

                pdf_output = pdf.output(dest='S').encode('latin1')
                st.download_button(label="Скачайте эту историю в виде PDF", data=pdf_output, file_name="story.pdf",
                                   mime="application/pdf")
if __name__ == "__main__":
    main()
