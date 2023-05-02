const fs = require("fs");
const { ChatOpenAI } = require("langchain/chat_models/openai");
const { RetrievalQAChain, LLMChain } = require("langchain/chains");
const { HNSWLib } = require("langchain/vectorstores/hnswlib");
const { OpenAIEmbeddings } = require("langchain/embeddings/openai");
const { RecursiveCharacterTextSplitter } = require ("langchain/text_splitter");
const { PDFLoader } = require("langchain/document_loaders/fs/pdf");
const {
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
  ChatPromptTemplate,
} = require("langchain/prompts");

const dotenv = require("dotenv");
dotenv.config();

const green = "\x1b[32m";
const reset = "\x1b[0m";

async function translateToBostonAccent() {
  const translationPrompt = ChatPromptTemplate.fromPromptMessages([
    SystemMessagePromptTemplate.fromTemplate(
      "You are a helpful assistant that translates {input_language} to English Boston Accent."
    ),
    HumanMessagePromptTemplate.fromTemplate("{text}"),
  ]);

  const chat = new ChatOpenAI({
    model_name:"gpt-3.5-turbo",
  });

  const chain = new LLMChain({
    prompt: translationPrompt,
    llm: chat,
  });

  const response = await chain.call({
    input_language: "English",
    text: "Do you want to visit the Harbor or Harvard Yard",
  });
  
  let answer = JSON.stringify(response.text);
  console.log(`${green}${answer}${reset}`);
  console.log("\n");
}

async function chatWithTxt() {
  // Initialize the LLM to use to answer the question.
  const model = new ChatOpenAI({
    model_name:"gpt-3.5-turbo",
  });
  const text = fs.readFileSync("../data/us-constitution.txt", "utf8");
  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
  const docs = await textSplitter.createDocuments([text]);

  // Create a vector store from the documents.
  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());

  // Create a chain that uses the OpenAI LLM and HNSWLib vector store.
  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());
  const res = await chain.call({
    query: "What is the 2nd ammendment?",
  });

  // Print the answer in green.
  let answer = res.text;
  console.log(`${green}${answer}${reset}`);
  console.log("\n");
};

async function chatWithPDF() {
  // Initialize the LLM to use to answer the question.
  const model = new ChatOpenAI({
    model_name:"gpt-3.5-turbo",
  });

  const loader = new PDFLoader("../data/pkd-metz.pdf", {splitPage: true});
  const docs = await loader.load();

  // Create a vector store from the documents.
  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());

  // Create a chain that uses the OpenAI LLM and HNSWLib vector store.
  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());
  const res = await chain.call({
    query: "are we living in a computer generated simulation",
  });

  // Print the answer in green.
  let answer = res.text;
  console.log(`${green}${answer}${reset}`);
}

translateToBostonAccent();
chatWithTxt();
chatWithPDF();
