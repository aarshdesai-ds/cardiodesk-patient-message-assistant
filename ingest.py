from dotenv import load_dotenv
load_dotenv()

import bs4
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document 
import os
from langchain_openai import OpenAIEmbeddings


# Target MedlinePlus high blood pressure URL


SOURCES= {"https://medlineplus.gov/highbloodpressure.html" : "High blood pressure",
"https://medlineplus.gov/heartfailure.html" : "Heart failure",
"https://medlineplus.gov/atrialfibrillation.html" : "Atrial fibrillation",
"https://medlineplus.gov/heartattack.html" : "Heart attack",
"https://medlineplus.gov/cholesterol.html": "Cholesterol",
"https://medlineplus.gov/angina.html": "Angina",
"https://medlineplus.gov/heartvalvediseases.html" : "Heart valve diseases",
"https://medlineplus.gov/bloodthinners.html": "Blood thinners"}


def load_education_pages():
    # Filter to only grab the core health topic content
    loader = WebBaseLoader(
    web_paths=list(SOURCES.keys()),
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(id="topic-summary")
    },
    bs_get_text_kwargs={
        "separator": "\n",
        "strip": True
    })
    docs = loader.load()
    return docs

def clean_text(text):
    cleaned = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        first = line[0]
        if cleaned and first in ",.;:)":
            cleaned[-1] = cleaned[-1] + line
        elif cleaned and first.islower():
            if cleaned[-1].endswith("("):
                cleaned[-1] = cleaned[-1] + line
            else:
                cleaned[-1] = cleaned[-1] + " " + line
        else:
            cleaned.append(line)
    return "\n".join(cleaned)

def split_by_headings(doc):
    topic = SOURCES[doc.metadata['source']]
    lines_so_far = []
    sections = []
    heading = "Overview"
    for line in clean_text(doc.page_content).strip().splitlines():
        if line.endswith("?") and len(line) < 90:
            if lines_so_far:
                sections.append(Document(page_content = "\n".join(lines_so_far), metadata = {"doc_type":"education",
                                                                                             "source": os.path.basename(doc.metadata['source'].removesuffix('.html')),
                                                                                             "topic":topic,
                                                                                             "section":heading,
                                                                                             "source_title": "MedlinePlus: " + topic,
                                                                                             "url":doc.metadata['source']}))
            heading = line
            lines_so_far = []

        else:
            lines_so_far.append(line)
    if lines_so_far:
        sections.append(Document(page_content = "\n".join(lines_so_far), metadata = {"doc_type":"education",
                                                                                     "source": os.path.basename(doc.metadata['source'].removesuffix('.html')),
                                                                                             "topic":topic,
                                                                                             "section":heading,
                                                                                             "source_title": "MedlinePlus: " + topic,
                                                                                             "url":doc.metadata['source']}))
    return sections

def split_long_sections(sections):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(sections)

def add_section(chunks):
    for chunk in chunks:
        headings = []
        title = ""
        for line in chunk.page_content.splitlines():
            
            if line.startswith("## "):
                section = line.removeprefix("## ")
                headings.append(section)
            if line.startswith("# "):
                title = line.removeprefix("# ")

        if headings:
            section = " / ".join(headings)
        elif title:
            section = title
        else:
            section = chunk.metadata['source']

        chunk.metadata['section'] = section
    return chunks

def load_clinic_docs():
    loader = DirectoryLoader(path="kb/clinic", glob="*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "topic"), ("##", "section")])

    chunks = []
    for doc in docs:
        source = os.path.basename(doc.metadata['source']).removesuffix(".md")
        for chunk in splitter.split_text(doc.page_content):
            chunk.metadata['doc_type'] = "clinic_policy"
            chunk.metadata['source'] = source
            chunk.metadata['source_title'] = "Riverbend Heart Clinic: " + chunk.metadata['topic']
            chunk.metadata['url'] = ""
            chunks.append(chunk)
    return chunks

def add_chunk_header(chunks):
    for chunk in chunks:
        header = f"Source: {chunk.metadata['source_title']} | Section: {chunk.metadata['section']}"
        chunk.page_content = header + "\n" + chunk.page_content

    return chunks

def make_ids(chunks):
    counters = {}
    ids = []
    for chunk in chunks:
        source = chunk.metadata['source']
        number = counters.get(source, 0)
        chunk_id = f"{source}-{number}"
        chunk.metadata['chunk_id'] = chunk_id
        ids.append(chunk_id)
        counters[source] = number + 1
    return ids

def build_store(chunks,ids):
    store = Chroma(
        collection_name="cardiodesk",
        embedding_function = OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory="kb_db"
    )
    store.add_documents(chunks, ids=ids)
    return store


if __name__ == "__main__":
    # Education pages
    pages = load_education_pages()
    sections = []
    for page in pages:
        sections.extend(split_by_headings(page))
    education = split_long_sections(sections)

    # Clinic policies
    clinic = load_clinic_docs()

    # Combine, add headers, ids, store
    chunks = add_chunk_header(education + clinic)
    ids = make_ids(chunks)
    print("IDs unique:", len(set(ids)) == len(ids))
    store = build_store(chunks, ids)

    # Summary
    print("Pages loaded:     ", len(pages))
    print("Sections:         ", len(sections))
    print("Education chunks: ", len(education))
    print("Clinic chunks:    ", len(clinic))
    print("Total in store:   ", len(store.get()["ids"]))

    # Is the blood pressure table in one piece?
    whole = [c for c in education
             if "Hypertensive Crisis" in c.page_content and "Higher than 120" in c.page_content]
    print("\nBlood pressure table in one chunk:", len(whole) == 1)
    if whole:
        print(whole[0].metadata["chunk_id"], "|", len(whole[0].page_content), "characters")

    print("\n--- Clinic chunks ---")
    for chunk in clinic:
        print(f"  {chunk.metadata['chunk_id']} | {chunk.metadata['topic']} | {chunk.metadata['section']}")

    print("\n--- Sample education chunk ---")
    print(education[20].metadata)
    print(education[20].page_content[:250])

    print("\n--- Sample clinic chunk ---")
    print(clinic[0].metadata)
    print(clinic[0].page_content[:250])

    # Test searches
    searches = [
        ("symptoms of a heart attack", None),
        ("feeling very tired with AFib", {"doc_type": "education"}),
        ("refill before travelling", {"doc_type": "clinic_policy"}),
    ]
    for query, filt in searches:
        print(f"\n--- Search: {query} | filter: {filt} ---")
        for doc in store.similarity_search(query, k=3, filter=filt):
            print(f"  {doc.metadata['doc_type']} | {doc.metadata['topic']} | {doc.metadata['section']}")