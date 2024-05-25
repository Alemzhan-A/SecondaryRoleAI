from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import openai
import langchain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


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
            st.write(plot)
            parts_of_story = plot.splitlines()
            while '' in parts_of_story:
                parts_of_story.remove('')
            print(parts_of_story)
if __name__ == "__main__":
    main()
